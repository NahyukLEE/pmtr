import argparse
import torch
import pytorch_lightning as pl

from model.pmtr import PMTR
from data.dataset import GADataset
from common import utils

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars", category=RuntimeWarning)

@torch.no_grad()
def test(args):
    # Model initialization
    utils.fix_randseed(0)

    model = PMTR(args.fine_matcher, args.cpconv_radius, args.lr)
    model.to(torch.device('cuda:0'))
    model.eval()
    
    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category)
    dataloader_val = GADataset.build_dataloader(args.batch_size, args.n_worker, 'val', args.sub_category, args.n_pts, args.subsampling_radius)

    trainer = pl.Trainer(gpus=[0])
    trainer.test(model, dataloader_val, ckpt_path=args.load)
    results = model.test_results
    results = {k[5:]: v.detach().cpu().numpy() for k, v in results.items()}
    print('Done testing...')

if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Proxy Match TransformeR (PMTR) Pytorch Lightning Implementation')
    parser.add_argument('--datapath', type=str, default='../../data/bbad_v2')
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact', 'other'])
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