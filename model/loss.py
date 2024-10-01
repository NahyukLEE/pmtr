import torch.nn.functional as F
import torch.nn as nn
import torch

def circle_loss(
    pos_masks,
    neg_masks,
    feat_dists,
    pos_margin,
    neg_margin,
    pos_optimal,
    neg_optimal,
    log_scale,
):
    # get anchors that have both positive and negative pairs
    row_masks = (torch.gt(pos_masks.sum(-1), 0) & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0) & torch.gt(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (~pos_masks).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights).detach()

    neg_weights = feat_dists + 1e5 * (~neg_masks).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights).detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale

    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss

def weighted_circle_loss(
    pos_masks, neg_masks,
    feat_dists,
    pos_margin, neg_margin,
    pos_optimal, neg_optimal,
    log_scale,
    pos_scales=None, neg_scales=None,
):
    # get anchors that have both positive and negative pairs
    row_masks = (torch.gt(pos_masks.sum(-1), 0) & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0) & torch.gt(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (~pos_masks).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights)
    if pos_scales is not None:
        pos_weights = pos_weights * pos_scales
    pos_weights = pos_weights.detach()

    neg_weights = feat_dists + 1e5 * (~neg_masks).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights)
    if neg_scales is not None:
        neg_weights = neg_weights * neg_scales
    neg_weights = neg_weights.detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale

    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss

class CircleLoss(nn.Module):
    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal, log_scale):
        super(CircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, pos_masks, neg_masks, feat_dists):
        return circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
        )

class WeightedCircleLoss(nn.Module):
    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal, log_scale):
        super(WeightedCircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, pos_masks, neg_masks, feat_dists, pos_scales=None, neg_scales=None):
        return weighted_circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
            pos_scales=pos_scales,
            neg_scales=neg_scales,
        )

class CoarseMatchingLoss(nn.Module):
    def __init__(self):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            0.1,
            1.4,
            0.1,
            1.4,
            24,
        )
        self.positive_overlap = 0.1

    def forward(self, output_dict):
        ref_feats = output_dict['src_feat_nds']
        src_feats = output_dict['trg_feat_nds']
        gt_node_corr_indices = output_dict['gt_nd_corr_ind']
        gt_node_corr_overlaps = output_dict['gt_nd_corr_overlap']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        if ref_feats is None and src_feats is None:
            feat_dists = (2.0 - 2.0 * output_dict['corr']).pow(0.5)
        else:
            feat_dists = (2.0 - 2.0 * torch.einsum('x d, y d -> x y', ref_feats, src_feats)).pow(0.5)

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss

class FineMatchingLoss(nn.Module):
    def __init__(self):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = 0.05

    def forward(self, output_dict, data_dict):
        src_nd_corr_knn_pts = output_dict['src_nd_corr_knn_pts']
        trg_nd_corr_knn_pts = output_dict['trg_nd_corr_knn_pts']
        src_nd_corr_knn_mask = output_dict['src_nd_corr_knn_mask']
        trg_nd_corr_knn_mask = output_dict['trg_nd_corr_knn_mask']
        matching_scores = output_dict['matching_scores']
        src_rotat, trg_rotat = [x.squeeze(0) for x in data_dict['gt_rotat_inv']]
        src_trans, trg_trans = data_dict['gt_trans_inv']

        src_nd_corr_knn_pts = torch.einsum('x y, n k y -> n k x', src_rotat, src_nd_corr_knn_pts) - src_trans
        trg_nd_corr_knn_pts = torch.einsum('x y, n k y -> n k x', trg_rotat, trg_nd_corr_knn_pts) - trg_trans

        dists = torch.cdist(src_nd_corr_knn_pts, trg_nd_corr_knn_pts).pow(2)
        gt_masks = torch.logical_and(src_nd_corr_knn_mask.unsqueeze(2), trg_nd_corr_knn_mask.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), src_nd_corr_knn_mask)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), trg_nd_corr_knn_mask)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss

class OverallLoss(nn.Module):
    def __init__(self):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss()
        self.fine_loss = FineMatchingLoss()
        self.weight_coarse_loss = 1.0
        self.weight_fine_loss = 1.0

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        if coarse_loss != coarse_loss:
            coarse_loss = torch.tensor(0.).cuda()
        fine_loss = self.fine_loss(output_dict, data_dict)
        if fine_loss != fine_loss:
            fine_loss = torch.tensor(0.).cuda()

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
        }