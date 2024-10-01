import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from chamfer_distance import ChamferDistance as chamfer_dist
from model.backbone.kpconvfpn import KPConvFPN_Flex
from model.local_global_registration import LocalGlobalRegistration
from model.pmt import PMTBlock
from model.loss import OverallLoss
from model.learnable_sinkhorn import LearnableLogOptimalTransport
from common.node import point_to_node_partition, index_select
from common.node import get_node_correspondences, SuperPointMatching, SuperPointTargetGenerator

from scipy.spatial.transform import Rotation
import random

class PMTR(pl.LightningModule):
    def __init__(self, fine_matcher, cpconv_radius, lr):
        super(PMTR, self).__init__()

        self.lr = lr

        # Initialization
        self.npts_per_node = 128
        self.matching_radius = 0.02
        self.backbone = KPConvFPN_Flex(1, [256, 128], [[64, 128], [128, 256, 256], [256, 512, 512]], fine_matcher)

        self.coarse_matching = SuperPointMatching()
        self.coarse_target = SuperPointTargetGenerator()
        self.fine_matching = LocalGlobalRegistration(
            k=3,
            acceptance_radius=0.1,
            mutual=True,
            confidence_threshold=0.05,
            use_dustbin=False,
            use_global_score=False,
            correspondence_threshold=3,
            correspondence_limit=None,
            num_refinement_steps=5,
        )

        self.optimal_transport = LearnableLogOptimalTransport(num_iterations=100)
        self.training_objective = OverallLoss()

        self.coarse_matcher = PMTBlock([64, 16, 4], [8, 32, 128], [4, 4], 'dist', cpconv_radius)

    def configure_optimizers(self):
        """Build optimizer and lr scheduler."""
        lr = self.lr
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.)
        return optimizer

    def training_step(self, in_dict, batch_idx, optimizer_idx=-1):
        _, loss_dict = self.forward_pass(in_dict, mode='train', optimizer_idx=optimizer_idx)
        return loss_dict['loss']
    
    def validation_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(in_dict, mode='val', optimizer_idx=-1)
        return loss_dict

    def validation_epoch_end(self, outputs):    
        # avg_loss among all data
        losses = {
            f'val/{k}': torch.stack([output[k] for output in outputs])
            for k in outputs[0].keys()
        }
        avg_loss = {k: (v).sum() / v.size(0) for k, v in losses.items()}
        self.log_dict(avg_loss, sync_dist=True)

    def test_step(self, in_dict, batch_idx):
        _, loss_dict = self.forward_pass(in_dict, mode='test', optimizer_idx=-1)
        return loss_dict

    def test_epoch_end(self, outputs):    
        # avg_loss among all data
        losses = {
            f'test/{k}': torch.stack([output[k] for output in outputs])
            for k in outputs[0].keys()
        }
        avg_loss = {k: (v).sum() / v.size(0) for k, v in losses.items()}
        print('; '.join([f'{k}: {v.item():.6f}' for k, v in avg_loss.items()]))
        self.test_results = avg_loss

    def forward_pass(self, in_dict, mode, optimizer_idx):

        out_dict = {}

        # 1. Generate node (superpoint) correspondences
        src_pts, src_nds, src_pt_len, src_nd_len, trg_pts, trg_nds, trg_pt_len, trg_nd_len = self._prepare_input(in_dict)
        _, src_nd_mask, src_nd_knn_ind, src_nd_knn_mask = point_to_node_partition(src_pts, src_nds, self.npts_per_node)
        _, trg_nd_mask, trg_nd_knn_ind, trg_nd_knn_mask = point_to_node_partition(trg_pts, trg_nds, self.npts_per_node)
        
        src_pts_pad = torch.cat([src_pts, torch.zeros_like(src_pts[:1])], dim=0)
        trg_pts_pad = torch.cat([trg_pts, torch.zeros_like(trg_pts[:1])], dim=0)
        src_nd_knn_pts = index_select(src_pts_pad, src_nd_knn_ind, dim=0)
        trg_nd_knn_pts = index_select(trg_pts_pad, trg_nd_knn_ind, dim=0)

        gt_nd_corr_ind, gt_nd_corr_overlap = get_node_correspondences(
            src_nds, trg_nds,
            src_nd_knn_pts, trg_nd_knn_pts,
            in_dict['gt_rotat_inv'], in_dict['gt_trans_inv'],
            self.matching_radius,
            src_nd_mask, trg_nd_mask,
            src_nd_knn_mask, trg_nd_knn_mask
        )
        out_dict['gt_nd_corr_ind'] = gt_nd_corr_ind
        out_dict['gt_nd_corr_overlap'] = gt_nd_corr_overlap

        # 2. Extract KPConvFPN features
        feat = torch.ones_like(torch.cat([src_pts, trg_pts])[:, :1])
        feat = self.backbone(feat, in_dict)
        
        feat_pts = feat[0]
        feat_nds = feat[-1]
        src_feat_pts, trg_feat_pts = feat_pts[:src_pt_len], feat_pts[src_pt_len:]
        src_feat_nds, trg_feat_nds = feat_nds[:src_nd_len], feat_nds[src_nd_len:]
        out_dict['n_pts'], out_dict['n_nds'] = (len(src_feat_pts), len(trg_feat_pts)), (len(src_feat_nds), len(trg_feat_nds))
        
        # 3. Coarse-grained matcher
        src_feat_nds, trg_feat_nds = self.coarse_matcher(
                src_nds.unsqueeze(0), trg_nds.unsqueeze(0),
                src_feat_nds.unsqueeze(0), trg_feat_nds.unsqueeze(0),
        ) 
        src_feat_nds, trg_feat_nds = src_feat_nds.squeeze(0), trg_feat_nds.squeeze(0)
        src_feat_nds_norm = F.normalize(src_feat_nds, p=2, dim=1)
        trg_feat_nds_norm = F.normalize(trg_feat_nds, p=2, dim=1)

        out_dict['src_feat_nds'] = src_feat_nds_norm
        out_dict['trg_feat_nds'] = trg_feat_nds_norm

        # 4. Select topk nearest node correspondences
        with torch.no_grad():
            src_nd_corr_ind, trg_nd_corr_ind, node_corr_scores = self.coarse_matching(src_feat_nds_norm, trg_feat_nds_norm, src_nd_mask, trg_nd_mask)
            
            out_dict['src_nd_corr_ind'] = src_nd_corr_ind
            out_dict['trg_nd_corr_ind'] = trg_nd_corr_ind
            out_dict['node_corr_scores'] = node_corr_scores

            if mode == 'train':
                src_nd_corr_ind, trg_nd_corr_ind, node_corr_scores = self.coarse_target(gt_nd_corr_ind, gt_nd_corr_overlap)

        # 6. Generate matching node points & feats
        src_nd_corr_knn_pts, trg_nd_corr_knn_pts, src_nd_corr_knn_mask, trg_nd_corr_knn_mask, matching_scores, estimated_transform = self._node_match_collection( \
                src_nd_corr_ind, trg_nd_corr_ind, out_dict, src_nd_knn_ind, trg_nd_knn_ind, \
                src_nd_knn_mask, trg_nd_knn_mask, src_nd_knn_pts, trg_nd_knn_pts, \
                src_feat_pts, trg_feat_pts, node_corr_scores)

        out_dict['src_nd_corr_knn_pts'] = src_nd_corr_knn_pts
        out_dict['trg_nd_corr_knn_pts'] = trg_nd_corr_knn_pts
        out_dict['src_nd_corr_knn_mask'] = src_nd_corr_knn_mask
        out_dict['trg_nd_corr_knn_mask'] = trg_nd_corr_knn_mask
        out_dict['matching_scores'] = matching_scores
        out_dict['estimated_rotat'] = estimated_transform[:3, :3].inverse()
        out_dict['estimated_trans'] = -(estimated_transform[:3, :3].inverse() @ -estimated_transform[:3, 3])

        # 7. Training objectives
        loss = self.training_objective(out_dict, in_dict)
        loss = self._compute_orthloss(loss)

        eval_dict = self.evaluate_prediction(in_dict, out_dict)
        loss.update(eval_dict)

        # in training we log for every step
        if mode == 'train' and self.local_rank == 0:
            log_dict = {f'{mode}/{k}': v.item() for k, v in loss.items()}
            data_name = [
                k for k in self.trainer.profiler.recorded_durations.keys()
                if 'prepare_data' in k
            ][0]
            log_dict[f'{mode}/data_time'] = \
                self.trainer.profiler.recorded_durations[data_name][-1]
            self.log_dict(
                log_dict, logger=True, sync_dist=False, rank_zero_only=True)

        return out_dict, loss


    def _compute_orthloss(self, loss, orthloss_scale=10.):
        orth_loss_, zero_loss_ = [], []
        pm_layers = []
        if hasattr(self.backbone, 'pmt'):
            pm_layers += list(self.backbone.pmt[0].pmt) + list(self.backbone.pmt[1].pmt)
        if hasattr(self.coarse_matcher, 'pmt'):
            pm_layers += list(self.coarse_matcher.pmt)

        for layer in pm_layers:
            proxy = layer.proxy
            device = proxy.device
            nhead, in_dim = proxy.size(0), proxy.size(1)
            orth_idx = torch.eye(nhead).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_dim, in_dim).bool().to(device)
            zero_idx = ~orth_idx
            pairwise = torch.einsum('x i d, y j d -> x y i j', proxy, proxy)

            orth_pred = pairwise[orth_idx].view(nhead, in_dim, in_dim)
            zero_pred = pairwise[zero_idx]
            orth_gt = torch.eye(in_dim).unsqueeze(0).repeat(nhead, 1, 1).to(device)
            zero_gt = torch.zeros_like(zero_pred)

            orth_loss = (orth_pred - orth_gt).pow(2).mean()
            zero_loss = (zero_pred - zero_gt).pow(2).mean()

            orth_loss_.append(orth_loss)
            zero_loss_.append(zero_loss)

        loss['orth_loss'] = torch.stack(orth_loss_).mean() * orthloss_scale
        loss['zero_loss'] = torch.stack(zero_loss_).mean() * orthloss_scale
        loss['loss'] = loss['loss'] + loss['orth_loss'] + loss['zero_loss']

        return loss

    def _node_match_collection(self, src_nd_corr_ind, trg_nd_corr_ind, out_dict, \
            src_nd_knn_ind, trg_nd_knn_ind, src_nd_knn_mask, trg_nd_knn_mask, \
            src_nd_knn_pts, trg_nd_knn_pts, src_feat_pts, trg_feat_pts, node_corr_scores):

        src_nd_corr_knn_ind  = src_nd_knn_ind[src_nd_corr_ind]
        trg_nd_corr_knn_ind  = trg_nd_knn_ind[trg_nd_corr_ind]
        src_nd_corr_knn_mask = src_nd_knn_mask[src_nd_corr_ind]
        trg_nd_corr_knn_mask = trg_nd_knn_mask[trg_nd_corr_ind]
        src_nd_corr_knn_pts  = src_nd_knn_pts[src_nd_corr_ind]
        trg_nd_corr_knn_pts  = trg_nd_knn_pts[trg_nd_corr_ind]

        src_feat_pts_pad = torch.cat([src_feat_pts, torch.zeros_like(src_feat_pts[:1])], dim=0)
        trg_feat_pts_pad = torch.cat([trg_feat_pts, torch.zeros_like(trg_feat_pts[:1])], dim=0)
        src_nd_corr_knn_feat = index_select(src_feat_pts_pad, src_nd_corr_knn_ind, dim=0)
        trg_nd_corr_knn_feat = index_select(trg_feat_pts_pad, trg_nd_corr_knn_ind, dim=0)

        # 7. Optimal transport
        matching_scores = torch.einsum('b n d, b m d -> b n m', src_nd_corr_knn_feat, trg_nd_corr_knn_feat)
        matching_scores = matching_scores / src_feat_pts.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, src_nd_corr_knn_mask, trg_nd_corr_knn_mask)
        matching_scores_out = matching_scores

        # 8. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            src_corr_pts, trg_corr_pts, corr_scores, estimated_transform = self.fine_matching(
                src_nd_corr_knn_pts, trg_nd_corr_knn_pts,
                src_nd_corr_knn_mask, trg_nd_corr_knn_mask,
                matching_scores, node_corr_scores
            )

        return src_nd_corr_knn_pts, trg_nd_corr_knn_pts, src_nd_corr_knn_mask, trg_nd_corr_knn_mask, matching_scores_out, estimated_transform

    @torch.no_grad()
    def _prepare_input(self, in_dict):
        points_ext = in_dict['points_ext_t'][list(in_dict['points_ext_t'].keys())[0]]
        lengths_ext = [x.squeeze(0) for x in in_dict['lengths_ext_t'][list(in_dict['lengths_ext_t'].keys())[0]]]
        src_pt_len, trg_pt_len = lengths_ext[0]
        src_nd_len, trg_nd_len = lengths_ext[-1]

        src_pts = points_ext[0][0][:src_pt_len]
        src_nds = points_ext[-1][0][:src_nd_len]
        trg_pts = points_ext[0][0][src_pt_len:]
        trg_nds = points_ext[-1][0][src_nd_len:]

        return src_pts, src_nds, src_pt_len, src_nd_len, trg_pts, trg_nds, trg_pt_len, trg_nd_len

    @torch.no_grad()
    def evaluate_prediction(self, in_dict, out_dict):

        # Init return buffer
        eval_result = {}
        
        pred_relative_trsfm = out_dict['estimated_rotat'], out_dict['estimated_trans'] 
        grtr_relative_trsfm = [x.squeeze(0) for x in in_dict['relative_trsfm']['0-1']]
        src_pcd, trg_pcd = [x.squeeze(0) for x in in_dict['pcd_t']]
        is_trg_larger = self._is_trg_larger(src_pcd, trg_pcd)
        
        # Pairwise shape mating
        assm_pred, pcds_pred = self._pairwise_mating(src_pcd, trg_pcd, pred_relative_trsfm[0], pred_relative_trsfm[1], is_trg_larger)
        assm_grtr, pcds_grtr = self._pairwise_mating(src_pcd, trg_pcd, grtr_relative_trsfm[0], grtr_relative_trsfm[1], is_trg_larger)
        
        # (a) Compute CoRrespondence Distance (CRD) betwween prediction & ground-truth
        eval_result['crd'] = self._correspondence_distance(assm_pred, assm_grtr)

        # (b) Compute CD between prediction & ground-truth
        eval_result['cd'] = self._chamfer_distance(assm_pred, assm_grtr)

        # (c) Compute MSE between prediction & ground-truth for rotation and translation
        eval_result['rrmse'], eval_result['trmse'] = self._transformation_error(pred_relative_trsfm, grtr_relative_trsfm)
        
        return eval_result

    def _correspondence_distance(self, assm1, assm2, scaling=100):
        corr_dist = (assm1 - assm2).norm(dim=-1).mean(dim=-1) * scaling

        return corr_dist

    def _chamfer_distance(self, assm1, assm2, scaling=1000):
        chd = chamfer_dist()
        dist1, dist2, idx1, idx2 = chd(assm1.unsqueeze(0), assm2.unsqueeze(0))
        cd = (dist1.mean(dim=-1) + dist2.mean(dim=-1)) * scaling
        return cd

    def _transformation_error(self, trnsf1, trnsf2, trmse_scaling=100):

        rotat1, trans1 = [trnsf1[0]], [trnsf1[1]]
        rotat2, trans2 = [trnsf2[0]], [trnsf2[1]]
        rrmse, trmse = 0., 0.

        for r1, r2, t1, t2 in zip(rotat1, rotat2, trans1, trans2):
            r1_deg = torch.tensor(Rotation.from_matrix(r1.cpu()).as_euler('xyz', degrees=True))
            r2_deg = torch.tensor(Rotation.from_matrix(r2.cpu()).as_euler('xyz', degrees=True))
            diff1 = (r1_deg - r2_deg).abs()
            diff2 = 360. - (r1_deg - r2_deg).abs()
            diff = torch.minimum(diff1, diff2)
            rrmse += diff.pow(2).mean().pow(0.5)
            trmse += (t1 - t2).pow(2).mean().pow(0.5) * trmse_scaling

        return rrmse, trmse

    def _is_trg_larger(self, src_pcd, trg_pcd):
        src_volume = (src_pcd.max(dim=0)[0] - src_pcd.min(dim=0)[0]).prod(dim=0)
        trg_volume = (trg_pcd.max(dim=0)[0] - trg_pcd.min(dim=0)[0]).prod(dim=0)

        return src_volume < trg_volume

    def _pairwise_mating(self, src_pcd, trg_pcd, rotat, trans, is_trg_larger):
        pcd_t = []
        if is_trg_larger:
            src_pcd_t = self._transform(src_pcd.squeeze(0), rotat, -trans, True)
            pcd_t = [src_pcd_t, trg_pcd.squeeze(0)]
        else:
            trg_pcd_t = self._transform(trg_pcd.squeeze(0), rotat.inverse(), trans, False)
            pcd_t = [src_pcd.squeeze(0), trg_pcd_t]
        return torch.cat(pcd_t, dim=0), pcd_t

    def _transform(self, pcd, rotat=None, trans=None, rotate_first=True):
        if rotat == None: rotat = torch.eye(3, 3)
        if trans == None: trans = torch.zeros(3)

        rotat = rotat.to(pcd.device)
        trans = trans.to(pcd.device)

        if rotate_first:
            return torch.einsum('x y, n y -> n x', rotat, pcd) + trans
        else:
            return torch.einsum('x y, n y -> n x', rotat, pcd + trans)