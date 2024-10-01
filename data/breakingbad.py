import os
from os.path import join

import random
import itertools

import torch
from torch.utils.data import Dataset
import numpy as np

import trimesh
import logging

from scipy.spatial.transform import Rotation as R
from common.sampling import subsampling

class DatasetBreakingBad(Dataset):
    def __init__(self, datapath, data_category, split, sub_category, n_pts, subsampling_radius):
        self.datapath = datapath
        self.data_category = data_category
        self.split = split
        self.sub_category = sub_category
        self.n_pts = n_pts
        self.min_n_pts = 256
        self.min_part = 2
        self.max_part = 2
        self.subsampling_radius = subsampling_radius
        self.anchor_idx = 0

        filepaths = join('./data/data_list', f"{data_category}_{split}.txt")
        with open(filepaths, 'r') as f:
            self.filepaths = [x.strip() for x in f.readlines() if x.strip()]

        self.filepaths = [x for x in self.filepaths if self.min_part <= int(x.split()[0]) <= self.max_part]
        if self.sub_category != 'all': self.filepaths = [x for x in self.filepaths if x.split()[1].split('/')[1] == self.sub_category]

        self.n_frac = [int(x.split()[0]) for x in self.filepaths]
        self.filepaths = [x.split()[1] for x in self.filepaths]

    def __len__(self):
        return len(self.filepaths)

    def _translate(self, mesh, pcd):
        gt_trans = [p.mean(dim=0) for p in pcd]
        pcd_t, mesh_t = [], [m.copy() for m in mesh]
        for idx, trans in enumerate(gt_trans):
            pcd_t.append(pcd[idx] - trans)
            mesh_t[idx].vertices -= trans.numpy()
        return pcd_t, mesh_t, gt_trans

    def _rotate(self, mesh, pcd):
        gt_rotat = [torch.tensor(R.random().as_matrix(), dtype=torch.float) for _ in pcd]
        pcd_t, mesh_t = [], [m.copy() for m in mesh]
        for idx, rotat in enumerate(gt_rotat):
            pcd_t.append(torch.einsum('x y, n y -> n x', rotat, pcd[idx]))
            mesh[idx].vertices = torch.einsum('x y, n y -> n x', rotat, torch.tensor(mesh[idx].vertices).float()).numpy()
        return pcd_t, mesh, gt_rotat

    def _compute_relative_transform(self, trans, rotat):
        permut_relative_transform = {}
        for pair_idx0, pair_idx1 in itertools.permutations(range(len(trans)), 2):
            # Compute relative rotation and translation
            trans0, trans1 = trans[pair_idx0], trans[pair_idx1]
            rotat0, rotat1 = rotat[pair_idx0], rotat[pair_idx1]
            relative_rotat = rotat1 @ rotat0.T
            relative_trans = - (rotat1 @ (trans0 - trans1))

            # Save relative transformation between each pairs
            key = f"{pair_idx0}-{pair_idx1}"
            permut_relative_transform[key] = relative_rotat, relative_trans
        return {'0-1':permut_relative_transform['0-1']}

    def __getitem__(self, idx):
        # Fix randomness
        if self.split == 'val': np.random.seed(idx)

        # Read mesh, point cloud of a fractured object
        logger = logging.getLogger("trimesh")
        logger.setLevel(logging.ERROR)
        mesh, pcd = self.read_obj_data(idx)
        
        # Apply random transformation to sampled points
        pcd_t, mesh_t, gt_trans = self._translate(mesh, pcd)
        pcd_t, mesh_t, gt_rotat = self._rotate(mesh_t, pcd_t)
        gt_relative_trsfm = self._compute_relative_transform(gt_trans, gt_rotat)

        # Subsample for down- & up-sampling point cloud
        subsampled = {'points':{}, 'lengths':{}, 'neighbors':{}, 'subsampling':{}, 'upsampling':{}}
        for pair in gt_relative_trsfm.keys():
            idx0, idx1 = map(int, pair.split('-'))
            subsampled['points'][pair],\
            subsampled['lengths'][pair],\
            subsampled['neighbors'][pair],\
            subsampled['subsampling'][pair],\
            subsampled['upsampling'][pair] = subsampling(torch.cat([pcd_t[idx0], pcd_t[idx1]]), torch.tensor([len(pcd_t[idx0]), len(pcd_t[idx1])]), 3, self.subsampling_radius,  0.125, [35, 32, 34])

        batch = {
                'filepath': self.filepaths[idx],
                'obj_class': self.filepaths[idx].split('/')[1],

                'pcd_t': pcd_t,
                'pcd': pcd,
                'n_frac': self.n_frac[idx],
                'anchor_idx': self.anchor_idx,

                'gt_trans': gt_trans,
                'gt_rotat': gt_rotat,
                'gt_rotat_inv': [R.T for R in gt_rotat],
                'gt_trans_inv': [-t for t in gt_trans],
                'relative_trsfm': gt_relative_trsfm,

                'points_ext_t': subsampled['points'],
                'lengths_ext_t': subsampled['lengths'],
                'neighbors_ext_t': subsampled['neighbors'],
                'subsampling_ext_t': subsampled['subsampling'],
                'upsampling_ext_t': subsampled['upsampling'],
                }

        return batch

    def read_obj_data(self, idx):
        if self.split == 'val': random.seed(idx)
        np.seterr(divide='ignore', invalid='ignore')
        
        filepath = self.filepaths[idx]
        n_frac = self.n_frac[idx]

        # Load N-part meshes and calculate each area
        base_path = join(self.datapath, filepath)
        obj_paths = [join(base_path, x) for x in os.listdir(base_path)]
        mesh_all = [trimesh.load_mesh(x) for x in obj_paths] # N-part meshes
        mesh_areas = [mesh_.area for mesh_ in mesh_all]

        # Set anchor fracture and sum all of areas
        self.anchor_idx, total_area = mesh_areas.index(max(mesh_areas)), sum(mesh_areas)

        # Sample point clouds
        pcd_all = []
        for mesh in mesh_all:
            n_pts = int(self.n_pts * mesh.area / total_area)
            if self.split == 'val': sampled_pts = torch.tensor(trimesh.sample.sample_surface_even(mesh, n_pts, seed=idx)[0]).float()
            else: sampled_pts = torch.tensor(trimesh.sample.sample_surface_even(mesh, n_pts)[0]).float()

            if sampled_pts.size(0) < self.min_n_pts:
                if self.split == 'val': extra_pts, _ = trimesh.sample.sample_surface(mesh, self.min_n_pts - sampled_pts.size(0), seed=idx)
                else: extra_pts, _ = trimesh.sample.sample_surface(mesh, self.min_n_pts - sampled_pts.size(0))
                sampled_pts = torch.cat([sampled_pts, torch.tensor(extra_pts).float()], dim=0)
            
            pcd_all.append(sampled_pts)

        if self.split == 'train' and random.random() > 0.5:
            mesh_all.reverse()
            pcd_all.reverse()
        
        return mesh_all, pcd_all
