import os
import pwd
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from model.pmtr import PMTR
from data.dataset import GADataset

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars", category=RuntimeWarning)

def main(args):

    # Model initialization
    model = PMTR(args.fine_matcher, args.cpconv_radius, args.lr)

    # Dataset initialization
    GADataset.initialize(args.datapath, args.data_category)
    dataloader_trn = GADataset.build_dataloader(args.batch_size, args.n_worker, 'train', args.sub_category, args.n_pts, args.subsampling_radius)
    dataloader_val = GADataset.build_dataloader(args.batch_size, args.n_worker, 'val', args.sub_category, args.n_pts, args.subsampling_radius)

    # Create checkpoint directory
    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    cfg_name = args.logpath
    ckp_dir = os.path.join('checkpoint/', cfg_name, 'models')
    os.makedirs(os.path.dirname(ckp_dir), exist_ok=True)

    CHECKPOINT_DIR = '/checkpoint/'
    if SLURM_JOB_ID and CHECKPOINT_DIR and os.path.isdir(CHECKPOINT_DIR):
        if not os.path.exists(ckp_dir):
            usr = pwd.getpwuid(os.getuid())[0]
            os.system(r'ln -s /checkpoint/{}/{}/ {}'.format(
                usr, SLURM_JOB_ID, ckp_dir))
    else:
        os.makedirs(ckp_dir, exist_ok=True)
    
    preemption = True
    if SLURM_JOB_ID and preemption:
        logger_id = logger_name = f'{cfg_name}-{SLURM_JOB_ID}'
    else:
        logger_name = cfg_name
        logger_id = None
    
    # configure callbacks
    checkpoint_callback_crd = ModelCheckpoint(dirpath=ckp_dir, filename='model-crd-{epoch:03d}', monitor='val/crd', save_top_k=1, mode='min')
    checkpoint_callback_cd = ModelCheckpoint(dirpath=ckp_dir, filename='model-cd-{epoch:03d}', monitor='val/cd', save_top_k=1, mode='min')
    checkpoint_callback_rrmse = ModelCheckpoint(dirpath=ckp_dir, filename='model-rrmse-{epoch:03d}', monitor='val/rrmse', save_top_k=1, mode='min')
    checkpoint_callback_trmse = ModelCheckpoint(dirpath=ckp_dir, filename='model-trmse-{epoch:03d}', monitor='val/trmse', save_top_k=1, mode='min')

    callbacks = [
        LearningRateMonitor('epoch'),
        checkpoint_callback_crd,
        checkpoint_callback_cd,
        checkpoint_callback_rrmse,
        checkpoint_callback_trmse,
    ]

    logger = WandbLogger(
        project='pmtr',
        name=logger_name,
        id=logger_id,
        save_dir=ckp_dir,
    )

    all_gpus = list(args.gpus)

    trainer = pl.Trainer(
        logger=logger,
        gpus=all_gpus,
        strategy=args.parallel_strategy,
        max_epochs=args.epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        profiler='simple',
        precision=32,
    )

    ckp_files = os.listdir(ckp_dir)
    ckp_files = [ckp for ckp in ckp_files if 'model-' in ckp]
    if ckp_files:
        ckp_files = sorted(
            ckp_files,
            key=lambda x: os.path.getmtime(os.path.join(ckp_dir, x)))
        last_ckp = ckp_files[-1]
        print(f'INFO: automatically detect checkpoint {last_ckp}')
        ckp_path = os.path.join(ckp_dir, last_ckp)
    elif args.load != '':
        ckp = torch.load(args.load, map_location='cpu')
        if 'state_dict' in ckp.keys():
            ckp_path = args.load
        else:
            ckp_path = None
            model.load_state_dict(ckp)
    else:
        ckp_path = None

    trainer.fit(model, dataloader_trn, dataloader_val, ckpt_path=ckp_path)
    print('Done training...')


if __name__ == '__main__':
    # arguments parsing
    parser = argparse.ArgumentParser(description='Proxy Match TransformeR (PMTR) Pytorch Lightning Implementation')
    parser.add_argument('--datapath', type=str, default='../../data/bbad_v2')
    parser.add_argument('--data_category', type=str, default='everyday', choices=['everyday', 'artifact', 'other'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--sub_category', type=str, default='all')
    parser.add_argument('--n_pts', type=int, default=5000)
    
    # model hyperparameters
    parser.add_argument('--cpconv_radius', type=float, default=0.05)
    parser.add_argument('--fine_matcher', type=str, default='pmt', choices=['none', 'pmt'])
    parser.add_argument('--subsampling_radius', type=float, default=0.01)

    # DDP settings
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)

    args = parser.parse_args()

    if len(args.gpus) > 1: 
        args.parallel_strategy = 'ddp'
        args.lr = len(args.gpus) * args.lr
    else: args.parallel_strategy = None

    main(args)