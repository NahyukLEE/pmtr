import numpy as np
import open3d
import torch

def to_o3d_pcd(pts):
    '''
    From numpy array, make point cloud in open3d format
    :param pts: point cloud (nx3) in numpy array
    :return: pcd: point cloud in open3d format
    '''
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    return pcd

def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    '''
    Give source & target point clouds as well as the relative transformation between them, calculate correspondences according to give threshold
    :param src_pcd: source point cloud
    :param tgt_pcd: target point cloud
    :param trans: relative transformation between source and target point clouds
    :param search_voxel_size: given threshold
    :param K: if k is not none, select top k nearest neighbors from candidate set after radius search
    :return: (m, 2) torch tensor, consisting of m correspondences
    '''
    # src_pcd.transform(trans)
    src_pcd = to_o3d_pcd(np.dot(trans[:3,:3], np.array(src_pcd.points).T).T - trans[:3,3])
    pcd_tree = open3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences
