import torch
import torch.nn as nn
import numpy as np

class SuperPointTargetGenerator(nn.Module):
    def __init__(self, num_targets=128, overlap_threshold=0.1):
        super(SuperPointTargetGenerator, self).__init__()
        self.num_targets = num_targets
        self.overlap_threshold = overlap_threshold

    @torch.no_grad()
    def forward(self, gt_corr_indices, gt_corr_overlaps):
        r"""Generate ground truth superpoint (patch) correspondences.

        Randomly select "num_targets" correspondences whose overlap is above "overlap_threshold".

        Args:
            gt_corr_indices (LongTensor): ground truth superpoint correspondences (N, 2)
            gt_corr_overlaps (Tensor): ground truth superpoint correspondences overlap (N,)

        Returns:
            gt_ref_corr_indices (LongTensor): selected superpoints in reference point cloud.
            gt_src_corr_indices (LongTensor): selected superpoints in source point cloud.
            gt_corr_overlaps (LongTensor): overlaps of the selected superpoint correspondences.
        """
        gt_corr_masks = torch.gt(gt_corr_overlaps, self.overlap_threshold)
        if gt_corr_masks.nonzero().size(0) == 0:
            gt_corr_masks = ~gt_corr_masks
        gt_corr_overlaps = gt_corr_overlaps[gt_corr_masks]
        gt_corr_indices = gt_corr_indices[gt_corr_masks]

        if gt_corr_indices.shape[0] > self.num_targets:
            indices = np.arange(gt_corr_indices.shape[0])
            sel_indices = np.random.choice(indices, self.num_targets, replace=False)
            sel_indices = torch.from_numpy(sel_indices).cuda()
            gt_corr_indices = gt_corr_indices[sel_indices]
            gt_corr_overlaps = gt_corr_overlaps[sel_indices]

        gt_ref_corr_indices = gt_corr_indices[:, 0]
        gt_src_corr_indices = gt_corr_indices[:, 1]

        return gt_ref_corr_indices, gt_src_corr_indices, gt_corr_overlaps


class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences=128, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()

        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        # select top-k proposals
        matching_scores = torch.exp(-(2.0 - 2.0 * torch.einsum('x d, y d -> x y', ref_feats, src_feats)))
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = (corr_indices / matching_scores.shape[1]).long()
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]

        return ref_corr_indices, src_corr_indices, corr_scores


def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    r"""Advanced index select.

    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output


@torch.no_grad()
def knn_partition(points: torch.Tensor, nodes: torch.Tensor, k: int, return_distance: bool = False):
    r"""k-NN partition of the point cloud.

    Find the k nearest points for each node.

    Args:
        points: torch.Tensor (num_point, num_channel)
        nodes: torch.Tensor (num_node, num_channel)
        k: int
        return_distance: bool

    Returns:
        knn_indices: torch.Tensor (num_node, k)
        knn_indices: torch.Tensor (num_node, k)
    """
    k = min(k, points.shape[0])
    sq_dist_mat = torch.cdist(nodes, points).pow(2)
    # sq_dist_mat = nodes.pow(2).sum(dim=1).unsqueeze(-1) + points.pow(2).sum(dim=1).unsqueeze(-2) - 2 * torch.einsum('x d, y d -> x y', nodes, points).clamp(min=0)
    knn_sq_distances, knn_indices = sq_dist_mat.topk(dim=1, k=k, largest=False)
    if return_distance:
        knn_distances = torch.sqrt(knn_sq_distances)
        return knn_distances, knn_indices
    else:
        return knn_indices


@torch.no_grad()
def point_to_node_partition(points, nodes, point_limit, return_count=False):
    r"""Point-to-Node partition to the point cloud.

    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`

    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """
    sq_dist_mat = torch.cdist(nodes, points).pow(2)
    # sq_dist_mat = nodes.pow(2).sum(dim=1).unsqueeze(-1) + points.pow(2).sum(dim=1).unsqueeze(-2) - 2 * torch.einsum('x d, y d -> x y', nodes, points).clamp(min=0)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
    node_masks.index_fill_(0, point_to_node, True)
    # for index in point_to_node: node_masks[index] = True
    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(points.shape[0]).cuda()  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)
    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def get_node_correspondences(
    src_nds, trg_nds,
    src_knn_pts, trg_knn_pts,
    gt_rotat_inv, gt_trans_inv,
    pos_radius,
    src_mask, trg_mask,
    src_knn_mask, trg_knn_mask
):
    r"""Generate ground-truth superpoint/patch correspondences.

    Each patch is composed of at most k nearest points of the corresponding superpoint.
    A pair of points match if the distance between them is smaller than `self.pos_radius`.

    Args:
        src_nds: torch.Tensor (M, 3)
        trg_nds: torch.Tensor (N, 3)
        src_knn_pts: torch.Tensor (M, K, 3)
        trg_knn_pts: torch.Tensor (N, K, 3)
        transform: torch.Tensor (4, 4)
        pos_radius: float
        src_mask (optional): torch.BoolTensor (M,) (default: None)
        trg_mask (optional): torch.BoolTensor (N,) (default: None)
        src_knn_mask (optional): torch.BoolTensor (M, K) (default: None)
        trg_knn_mask (optional): torch.BoolTensor (N, K) (default: None)

    Returns:
        corr_indices: torch.LongTensor (C, 2)
        corr_overlaps: torch.Tensor (C,)
    """
    device = src_nds.device

    src_nds = torch.einsum('x y, n y -> n x', gt_rotat_inv[0].squeeze(0), src_nds) - gt_trans_inv[0]
    src_knn_pts = torch.einsum('x y, n k y -> n k x', gt_rotat_inv[0].squeeze(0), src_knn_pts) - gt_trans_inv[0]

    trg_nds = torch.einsum('x y, n y -> n x', gt_rotat_inv[1].squeeze(0), trg_nds) - gt_trans_inv[1]
    trg_knn_pts = torch.einsum('x y, n k y -> n k x', gt_rotat_inv[1].squeeze(0), trg_knn_pts) - gt_trans_inv[1]

    # generate masks
    if src_mask is None:
        src_mask = torch.ones(size=(src_nds.shape[0],), dtype=torch.bool).cuda()
    if trg_mask is None:
        trg_mask = torch.ones(size=(trg_nds.shape[0],), dtype=torch.bool).cuda()
    if src_knn_mask is None:
        src_knn_mask = torch.ones(size=(src_knn_pts.shape[0], src_knn_pts.shape[1]), dtype=torch.bool).cuda()
    if trg_knn_mask is None:
        trg_knn_mask = torch.ones(size=(trg_knn_pts.shape[0], trg_knn_pts.shape[1]), dtype=torch.bool).cuda()

    node_mask_mat = torch.logical_and(src_mask.unsqueeze(1), trg_mask.unsqueeze(0))  # (M, N)

    # filter out non-overlapping patches using enclosing sphere
    src_knn_dist = torch.linalg.norm(src_knn_pts - src_nds.unsqueeze(1), dim=-1)  # (M, K)
    src_knn_dist.masked_fill_(~src_knn_mask, 0.0)
    src_max_dist = src_knn_dist.max(1)[0]  # (M,)

    trg_knn_dist = torch.linalg.norm(trg_knn_pts - trg_nds.unsqueeze(1), dim=-1)  # (N, K)
    trg_knn_dist.masked_fill_(~trg_knn_mask, 0.0)
    trg_max_dist = trg_knn_dist.max(1)[0]  # (N,)

    dist_mat = torch.cdist(src_nds, trg_nds)  # (M, N)
    # dist_mat = src_nds.pow(2).sum(dim=1).unsqueeze(-1) + trg_nds.pow(2).sum(dim=1).unsqueeze(-2) - 2 * torch.einsum('x d, y d -> x y', src_nds, trg_nds).clamp(min=0).pow(0.5)
    intersect_mat = torch.gt(src_max_dist.unsqueeze(1) + trg_max_dist.unsqueeze(0) + pos_radius - dist_mat, 0)
    intersect_mat = torch.logical_and(intersect_mat, node_mask_mat)
    sel_src_ind, sel_trg_ind = torch.nonzero(intersect_mat, as_tuple=True)

    # select potential patch pairs
    src_knn_mask = src_knn_mask[sel_src_ind]  # (B, K)
    trg_knn_mask = trg_knn_mask[sel_trg_ind]  # (B, K)

    src_knn_pts = src_knn_pts[sel_src_ind]  # (B, K, 3)
    trg_knn_pts = trg_knn_pts[sel_trg_ind]  # (B, K, 3)

    point_mask_mat = torch.logical_and(src_knn_mask.unsqueeze(2), trg_knn_mask.unsqueeze(1))  # (B, K, K)

    # compute overlaps
    dist_mat = torch.cdist(src_knn_pts, trg_knn_pts).pow(2)
    # dist_mat = src_knn_pts.pow(2).sum(dim=-1).unsqueeze(-1) + trg_knn_pts.pow(2).sum(dim=-1).unsqueeze(-2) - 2 * torch.einsum('b x d, b y d -> b x y', src_knn_pts, trg_knn_pts).clamp(min=0)
    dist_mat.masked_fill_(~point_mask_mat, 1e12)

    point_overlap_mat = torch.lt(dist_mat, pos_radius ** 2)  # (B, K, K)
    src_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-1), dim=-1).float()  # (B,)
    trg_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-2), dim=-1).float()  # (B,)
    src_overlaps = src_overlap_counts / src_knn_mask.sum(-1).float()  # (B,)
    trg_overlaps = trg_overlap_counts / trg_knn_mask.sum(-1).float()  # (B,)
    overlaps = (src_overlaps + trg_overlaps) / 2  # (B,)

    overlap_masks = torch.gt(overlaps, 0)
    src_corr_ind = sel_src_ind[overlap_masks]
    trg_corr_ind = sel_trg_ind[overlap_masks]
    corr_indices = torch.stack([src_corr_ind, trg_corr_ind], dim=1)
    corr_overlaps = overlaps[overlap_masks]

    if len(corr_indices) == 0:
        corr_indices = torch.tensor([[torch.randint(0, intersect_mat.size(0), size=(1,)), torch.randint(0, intersect_mat.size(1), size=(1,))]]).to(device)
        corr_overlaps = torch.tensor([1.0]).to(device)

    return corr_indices, corr_overlaps
