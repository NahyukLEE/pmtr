import torch
from torch.utils.data import DataLoader
from data.breakingbad import DatasetBreakingBad

class GADataset:

    @classmethod
    def initialize(cls, datapath, data_category):
        cls.datapath = datapath
        cls.data_category = data_category

    @classmethod
    def build_dataloader(cls, batch_size, nworker, split, sub_category, n_pts, subsampling_radius):
        training = split == 'train'
        shuffle = training

        dataset = DatasetBreakingBad(cls.datapath, cls.data_category, split, sub_category, n_pts, subsampling_radius)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=nworker)

        return dataloader

