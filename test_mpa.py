import argparse
import torch
import pytorch_lightning as pl

from model.pmtr import PMTR
from data.dataset import GADataset
from common import utils

import gtsam
import itertools
from scipy.spatial.transform import Rotation
from chamfer_distance import ChamferDistance as chamfer_dist
import numpy as np
import gc
import warnings

warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars", category=RuntimeWarning)

def _transform(pcd, rotat=None, trans=None, rotate_first=True):
    """Apply rotation and translation to a point cloud."""
    if rotat is None:
        rotat = torch.eye(3, 3)
    if trans is None:
        trans = torch.zeros(3)

    rotat = rotat.to(pcd.device)
    trans = trans.to(pcd.device)

    if rotate_first:
        return torch.einsum('x y, n y -> n x', rotat, pcd) + trans
    else:
        return torch.einsum('x y, n y -> n x', rotat, pcd + trans)

def _multi_part_assemble(pcds, rotats, transes):
    """Assemble multiple parts into a single point cloud."""
    transformed_pcds = [
        _transform(pcd.squeeze(0), R.inverse(), t, rotate_first=False)
        for pcd, R, t in zip(pcds, rotats, transes)
    ]
    return torch.cat(transformed_pcds, dim=0), transformed_pcds

def _chamfer_distance(assm1, assm2, scaling=1000):
    """Compute Chamfer distance between two assemblies."""
    chd = chamfer_dist()
    dist1, dist2, _, _ = chd(assm1.unsqueeze(0), assm2.unsqueeze(0))
    cd = (dist1.mean(dim=-1) + dist2.mean(dim=-1)) * scaling
    return cd

def _correspondence_distance(assm1, assm2, scaling=100):
    """Compute mean correspondence distance between two assemblies."""
    return (assm1 - assm2).norm(dim=-1).mean(dim=-1) * scaling

def _transformation_error(rotat1, rotat2, trans1, trans2, rrmse_scaling=100):
    """Compute rotation and translation RMSE between two sets of transformations."""
    rrmse, trmse = 0., 0.
    for r1, r2, t1, t2 in zip(rotat1, rotat2, trans1, trans2):
        r1_deg = torch.tensor(Rotation.from_matrix(r1.cpu()).as_euler('xyz', degrees=True))
        r2_deg = torch.tensor(Rotation.from_matrix(r2.cpu()).as_euler('xyz', degrees=True))
        diff = torch.minimum((r1_deg - r2_deg).abs(), 360. - (r1_deg - r2_deg).abs())
        rrmse += diff.pow(2).mean().sqrt()
        trmse += (t1 - t2).pow(2).mean().sqrt() * rrmse_scaling
    div = len(rotat1)
    return rrmse / div, trmse / div

def _part_accuracy(assm_pts1, assm_pts2, scaling=100):
    """Compute part accuracy using Chamfer distance."""
    success = sum(
        _chamfer_distance(pred_pts, gt_pts, 1) < 0.01
        for pred_pts, gt_pts in zip(assm_pts1, assm_pts2)
    )
    return success / len(assm_pts1)

def _part_accuracy_crd(assm_pts1, assm_pts2, scaling=100):
    """Compute part accuracy using correspondence distance."""
    success = sum(
        _correspondence_distance(pred_pts, gt_pts, 1) < 0.1
        for pred_pts, gt_pts in zip(assm_pts1, assm_pts2)
    )
    return success / len(assm_pts1)

def estimate_poses_given_rot(
        factors: gtsam.BetweenFactorPose3s,
        rotations: gtsam.Values,
        uncertainty,
        anchor_idx
):
    """Estimate Poses from measurements, given rotations. From SfmProblem in shonan.
    Source: https://github.com/Jiaxin-Lu/Jigsaw/blob/41713e196f1b6b294913665945cd22240127c80b/utils/global_alignment/shonan_averaging.py#L6
    Arguments:
        factors -- data structure with many BetweenFactorPose3 factors
        rotations {Values} -- Estimated rotations
    Returns:
        Values -- Estimated Poses
    """

    graph = gtsam.GaussianFactorGraph()
    model = gtsam.noiseModel.Unit.Create(3)

    # Add a factor anchoring t_anchor
    graph.add(anchor_idx, np.eye(3), np.zeros((3,)), model)

    # Add a factor saying t_j - t_i = Ri*t_ij for all edges (i,j)
    for idx in range(len(factors)):
        factor = factors[idx]
        keys = factor.keys()
        i, j, Tij = keys[0], keys[1], factor.measured()
        if i == j:
            continue
        model = gtsam.noiseModel.Diagonal.Variances(
            uncertainty[idx] * (1e-2) * np.ones(3)
        )
        measured = rotations.atRot3(j).inverse().rotate(Tij.translation())
        graph.add(j, np.eye(3), i, -np.eye(3), measured, model)

    # Solve linear system
    translations = graph.optimize()
    # Convert to Values.
    result = gtsam.Values()
    for j in range(rotations.size()):
        tj = translations.at(j)
        result.insert(j, gtsam.Pose3(rotations.atRot3(j), tj))

    return result

@torch.no_grad()
def test(args):
    # Model initialization
    utils.fix_randseed(0)
    model = PMTR(args.fine_matcher, args.cpconv_radius, args.lr, evaluate=False)
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(torch.device('cuda:0'))
    model.eval()

    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category)
    dataloader_val = GADataset.build_dataloader(args.batch_size, args.n_worker, 'test', args.sub_category, args.n_pts, args.subsampling_radius, mpa=True)

    crd_list, cd_list, rrmse_list, trsme_list, pa_list, pa_crd_list = [], [], [], [], [], []

    for idx, in_dict in enumerate(dataloader_val):

        in_dict = utils.to_cuda(in_dict)
        n_frac = in_dict['n_frac']
        pair_indices = list(itertools.permutations(range(n_frac), 2))
        out_dict = {}

        # 1. Forward passes for all pairs
        for pair_idx0, pair_idx1 in pair_indices:
            in_dict_pair = {
                'filepath': in_dict['filepath'],
                'obj_class': in_dict['obj_class'],
                'pcd_t': [in_dict['pcd_t'][pair_idx0], in_dict['pcd_t'][pair_idx1]],
                'pcd': [in_dict['pcd'][pair_idx0], in_dict['pcd'][pair_idx1]],
                'n_frac': 2,
                'gt_trans': [in_dict['gt_trans'][pair_idx0], in_dict['gt_trans'][pair_idx1]],
                'gt_rotat': [in_dict['gt_rotat'][pair_idx0], in_dict['gt_rotat'][pair_idx1]],
                'gt_rotat_inv': [in_dict['gt_rotat_inv'][pair_idx0], in_dict['gt_rotat_inv'][pair_idx1]],
                'gt_trans_inv': [in_dict['gt_trans_inv'][pair_idx0], in_dict['gt_trans_inv'][pair_idx1]],
                'relative_trsfm': {f'{pair_idx0}-{pair_idx1}': in_dict['relative_trsfm'][f'{pair_idx0}-{pair_idx1}']},
                'points_ext_t': {f'{pair_idx0}-{pair_idx1}': in_dict['points_ext_t'][f'{pair_idx0}-{pair_idx1}']},
                'lengths_ext_t': {f'{pair_idx0}-{pair_idx1}': in_dict['lengths_ext_t'][f'{pair_idx0}-{pair_idx1}']},
                'neighbors_ext_t': {f'{pair_idx0}-{pair_idx1}': in_dict['neighbors_ext_t'][f'{pair_idx0}-{pair_idx1}']},
                'subsampling_ext_t': {f'{pair_idx0}-{pair_idx1}': in_dict['subsampling_ext_t'][f'{pair_idx0}-{pair_idx1}']},
                'upsampling_ext_t': {f'{pair_idx0}-{pair_idx1}': in_dict['upsampling_ext_t'][f'{pair_idx0}-{pair_idx1}']}
            }
            out_dict[f'{pair_idx0}-{pair_idx1}'], _ = model.forward_pass(
                in_dict_pair, mode='test', optimizer_idx=-1
            )
            torch.cuda.empty_cache()
            gc.collect()

        # 2. Pose Graph Optimization
        params = gtsam.ShonanAveragingParameters3(gtsam.LevenbergMarquardtParams.CeresDefaults())
        factors = gtsam.BetweenFactorPose3s()
        uncertainty = []

        # 2-1. Add factors
        for pair_idx0, pair_idx1 in pair_indices:
            est_rotat = out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_rotat'].cpu().numpy()
            est_trans = out_dict[f'{pair_idx0}-{pair_idx1}']['estimated_trans'].cpu().numpy()
            relative_rotat = Rotation.from_matrix(est_rotat).as_quat()
            pose = gtsam.Pose3(
                gtsam.Rot3.Quaternion(relative_rotat[3], relative_rotat[0], relative_rotat[1], relative_rotat[2]),
                gtsam.Point3(est_trans)
            )
            score = (torch.pow(out_dict[f'{pair_idx0}-{pair_idx1}']['node_corr_scores'] * 1e5, 2)).detach().cpu().mean()
            info = (1 / score) * np.eye(6)
            factors.append(gtsam.BetweenFactorPose3(pair_idx0, pair_idx1, pose, gtsam.noiseModel.Diagonal.Information(info)))
            uncertainty.append(1 / score)

        # 2-2. Optimize global rotations
        sa3 = gtsam.ShonanAveraging3(factors, params)
        initial = sa3.initializeRandomly()
        pMax = 20
        shonan_fail = False
        while True:
            pMax += 20
            if pMax == 60:
                shonan_fail = True; print("shonan failed")
                break
            try: 
                abs_rotat, _ = sa3.run(initial, 3, pMax)
                break
            except RuntimeError:
                print(f"An error occurred during Shonan::run: with pMax {pMax}")
                continue

        # Align predicted rotation to anchor fracture
        anchor_idx = in_dict['anchor_idx']
        if not shonan_fail:
            aligned_pred_rotat, aligned_pred_trans = [], []

            aligned_pred_rotat2 = []
            abs_anchor_R2 = abs_rotat.atRot3(anchor_idx)

            for j in range(abs_rotat.size()):
                aligned_pred_rotat2.append(torch.tensor(abs_anchor_R2.between(abs_rotat.atRot3(j)).matrix()).to(torch.float32).cuda())

            rel_rotat = gtsam.Values()
            abs_anchor_R = abs_rotat.atRot3(anchor_idx)
            for i in range(abs_rotat.size()):
                if i == anchor_idx:
                    rel_rotat.insert(i, gtsam.Rot3(np.eye(3)))
                else:
                    rel_rotat.insert(i, abs_anchor_R.inverse().compose(abs_rotat.atRot3(i)))

            poses = estimate_poses_given_rot(
                factors, rel_rotat, np.array(uncertainty), anchor_idx
            )
            abs_anchor_T = poses.atPose3(anchor_idx).translation()

            for j in range(poses.size()):
                aligned_pred_rotat.append(
                    torch.tensor(abs_anchor_R.between(abs_rotat.atRot3(j)).matrix()).to(torch.float32).cuda()
                )
                pred_trans = poses.atPose3(j).rotation().rotate(abs_anchor_T + poses.atPose3(j).translation())
                aligned_pred_trans.append(torch.tensor(pred_trans).to(torch.float32).cuda())
        else:
            aligned_pred_rotat, aligned_pred_trans = [], []
            for i in range(0, in_dict['n_frac']):
                if i == anchor_idx: 
                    aligned_pred_rotat.append(torch.eye(3).to(torch.float32).cuda())
                    aligned_pred_trans.append(torch.tensor([0,0,0]).to(torch.float32).cuda())
                else: 
                    aligned_pred_rotat.append(out_dict[f'{anchor_idx}-{i}']['estimated_rotat'].squeeze(0))
                    aligned_pred_trans.append(out_dict[f'{anchor_idx}-{i}']['estimated_trans'])

        # Align GT transformation to anchor fracture
        aligned_gt_rotat, aligned_gt_trans = [], []
        for i in range(0, in_dict['n_frac']):
            if i == anchor_idx:
                aligned_gt_rotat.append(torch.eye(3).to(torch.float32).cuda())
                aligned_gt_trans.append(torch.tensor([0, 0, 0]).to(torch.float32).cuda())
            else:
                aligned_gt_rotat.append(in_dict['relative_trsfm'][f'{anchor_idx}-{i}'][0].squeeze(0))
                aligned_gt_trans.append(in_dict['relative_trsfm'][f'{anchor_idx}-{i}'][1].squeeze(0))

        # Save aligned transformations
        in_dict['aligned_gt_trans'] = aligned_gt_trans
        in_dict['aligned_gt_rotat'] = aligned_gt_rotat
        out_dict['aligned_pred_trans'] = aligned_pred_trans
        out_dict['aligned_pred_rotat'] = aligned_pred_rotat

        assm_pred, pcds_pred = _multi_part_assemble(in_dict['pcd_t'], aligned_pred_rotat, aligned_pred_trans)
        assm_grtr, pcds_grtr = _multi_part_assemble(in_dict['pcd_t'], aligned_gt_rotat, aligned_gt_trans)

        cd = _chamfer_distance(assm_pred, assm_grtr).item()
        crd = _correspondence_distance(assm_pred, assm_grtr).item()
        rrmse, trmse = _transformation_error(aligned_pred_rotat, aligned_gt_rotat, aligned_pred_trans, aligned_gt_trans)
        rrmse, trmse = rrmse.item(), trmse.item()
        pa = _part_accuracy(pcds_pred, pcds_grtr).item()
        pa_crd = _part_accuracy_crd(pcds_pred, pcds_grtr).item()

        crd_list.append(crd)
        cd_list.append(cd)
        rrmse_list.append(rrmse)
        trsme_list.append(trmse)
        pa_list.append(pa)
        pa_crd_list.append(pa_crd)

        print(
            f'{idx}/{len(dataloader_val)} | #-Part: {len(pcds_pred)} | '
            f'CRD: {round(crd,2)} | CD: {round(cd,2)} | RRMSE: {round(rrmse,2)} | '
            f'TRMSE: {round(trmse,2)} | PA(CD): {round(pa,2)} | PA(CRD): {round(pa_crd,2)}'
        )

    num_sample = len(dataloader_val)
    print('====MULTI PART ASSEMBLY RESULTS====')
    print('CRD: ', sum(crd_list) / num_sample)
    print('CD: ', sum(cd_list) / num_sample)
    print('RRMSE: ', sum(rrmse_list) / num_sample)
    print('TRMSE: ', sum(trsme_list) / num_sample)
    print('PA(CRD): ', sum(pa_crd_list) / num_sample)
    print('PA(CD): ', sum(pa_list) / num_sample)

if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='PMTR Inference for multi-part assembly')
    parser.add_argument('--datapath', type=str, default='../../data/bbad_v2')
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--n_pts', type=int, default=5000)
    
    # Model hyperparameters
    parser.add_argument('--cpconv_radius', type=float, default=0.05)
    parser.add_argument('--fine_matcher', type=str, default='pmt', choices=['none', 'pmt'])
    parser.add_argument('--subsampling_radius', type=float, default=0.01)

    args = parser.parse_args()
    test(args)