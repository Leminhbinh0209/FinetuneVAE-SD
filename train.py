# https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/README.md
from math import e
import os
from omegaconf import OmegaConf
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
from PIL import Image
from argparse import ArgumentParser
from diffusers import  AutoencoderKL
import lpips
from ldm.util import   instantiate_from_config
from ldm.modules.ema import LitEma
from contextlib import contextmanager

torch.cuda.empty_cache()

class FinetuneFaceData(Dataset):
    def __init__(self, data_dir:str, 
                 img_list: list,
                 size:int=384, 
                 ):
        self.data_dir = data_dir
        self.img_list = img_list
        self.size = size
        self.transform =  transforms.Compose(
            [transforms.Resize((size, size)), 
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5]),
             ])
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.img_list[idx])
        image = Image.open(img_name)
        return self.transform(image), self.img_list[idx]
class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, 
                 batch_size=64, 
                 val_size=0.1,
                 size=384):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.size = size
        self.setup('fit')
    def setup(self, stage):
        all_images = sorted([u for u in os.listdir(self.data_dir) if u.endswith(".png") or u.endswith(".jpg")])
        random.shuffle(all_images)
        train_size = int((1-self.val_size)*len(all_images))
        train_images = all_images[:train_size]
        val_images = all_images[train_size:] 
        self.train_ds = FinetuneFaceData(self.data_dir,  train_images, self.size)
        self.val_ds = FinetuneFaceData(self.data_dir,  val_images, self.size)
        print(f"Train size: {len(self.train_ds)}, Val size: {len(self.val_ds)}")
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

class FinetuneVAE(pl.LightningModule):
    def __init__(self, 
                 kl_weight=0.1, 
                 lpips_loss_weight=0.1,
                 lr=1e-4, 
                 momentum=0.9, 
                 weight_decay=5e-4,
                 optim='sgd',
                 vae_config=None,
                 vae_weights=None,
                 device=torch.device('cuda'),
                 ema_decay=0.999,
                 precision=32,
                 log_dir=None):
        super().__init__()
        self.kl_weight = kl_weight
        self.lpips_loss_weight = lpips_loss_weight
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_loss_fn.eval()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optim = optim
        self.encoder = AutoencoderKL()
        self.model =  instantiate_from_config(vae_config)
        self.model.load_state_dict(vae_weights, strict=True)
        self.model.train()
        self.precision = precision
        self.use_ema = use_ema

        self.log_dir = log_dir
        self.log_one_batch = False
        self.use_ema = ema_decay > 0     
        if self.use_ema :    
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Assuming the DataModule is attached to the Trainer and accessible
            self.train_ds = self.trainer.datamodule.train_ds
            self.val_ds = self.trainer.datamodule.val_ds
            print("Warning: The setup method is called")
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        target, _ = batch
        if self.precision == 16:
            target = target.half()
        posterior = self.model.encode(target)
        z = posterior.sample()
        pred = self.model.decode(z)
        # kl_loss = posterior.kl()
        # kl_loss = kl_loss.mean() 
        rec_loss = torch.abs(target.contiguous() - pred.contiguous())
        if self.current_epoch < self.trainer.max_epochs // 3 * 2:
            rec_loss = rec_loss.mean() * rec_loss.size(1)
        else:
            rec_loss = rec_loss.pow(2).mean() * rec_loss.size(1) #
        lpips_loss = self.lpips_loss_fn(pred, target).mean()
        loss = rec_loss + self.lpips_loss_weight * lpips_loss # + self.kl_weight * kl_loss
        self.log('rec_loss', rec_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('lpips_loss', lpips_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        # self.log('kl_loss', kl_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        return loss
    def configure_optimizers(self):
        if self.optim == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        return optimizer
    def validation_step(self, batch, batch_idx):  
        target, name = batch
        if self.precision == 16:
            target = target.half()
        posterior = self.model.encode(target)
        z = posterior.mode()
        pred = self.model.decode(z)
        # kl_loss = posterior.kl()
        # kl_loss = kl_loss.mean() # torch.sum(kl_loss) / kl_loss.shape[0]
        rec_loss = torch.abs(target.contiguous() - pred.contiguous())
        rec_loss = rec_loss.mean() # torch.sum(rec_loss) / (rec_loss.shape[0] *  rec_loss.shape[2] * rec_loss.shape[3])
        lpips_loss = self.lpips_loss_fn(pred, target).mean()
        loss = rec_loss + self.lpips_loss_weight * lpips_loss # + self.kl_weight * kl_loss
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_images(target, pred, name)
        return {'val_loss': loss, "rec_loss": rec_loss, "lpips_loss": lpips_loss}
    def log_images(self, input, output, names):
        if self.log_one_batch: 
            return 
        for img1, img2, name in  zip(input, output, names):
            img1 = img1.cpu().detach().numpy().transpose(1, 2, 0)
            img2 = img2.cpu().detach().numpy().transpose(1, 2, 0)
            img1 = (img1 + 1) / 2
            img2 = (img2 + 1) / 2
            diff = abs(img1 - img2)
            img = np.concatenate([img1, img2, diff], axis=1)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            os.makedirs(self.log_dir + "/" + str(self.current_epoch), exist_ok=True)
            img.save(os.path.join(self.log_dir, str(self.current_epoch), name))
        self.log_one_batch = True
    def train_epoch_end(self, outputs):
        if self.use_ema:
            self.model_ema(self.model)
            self.model_ema.copy_to(self.model)
        if self.current_epoch == self.trainer.max_epochs // 3 * 2:
            self.lpips_loss_weight = self.lpips_loss_weight * 0.1
    def validation_epoch_end(self, validation_step_outputs):
        self.log_one_batch = False
        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        rec_loss = torch.stack([x['rec_loss'] for x in validation_step_outputs]).mean()
        lpips_loss = torch.stack([x['lpips_loss'] for x in validation_step_outputs]).mean()
        # kl_loss = torch.stack([x['kl_loss'] for x in validation_step_outputs]).mean()
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_rec_loss', rec_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_lpips_loss', lpips_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)


def get_vae_weights( input_path):
    pretrained_weights = torch.load(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']
    vae_weight = {}
    for k in pretrained_weights.keys():
        if "first_stage_model" in k:
            vae_weight[k.replace("first_stage_model.", "")] = pretrained_weights[k]
    return vae_weight
def argument_inputs():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='./dataset/',
                        help='The directory that contains the images, including original folder and the emotion folders.')
    parser.add_argument('--use_wandb',  action="store_true",help="Use wandb") 
    parser.add_argument('--use_ema',  action="store_true",help="Use use_ema") 

    parser.add_argument('--precision', type=int, default=16, choices=[16, 32])
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--kl_weight', type=float, default=1.)
    parser.add_argument('--lpips_loss_weight', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--output_dir', type=str, 
                        default='./vae_finetune',)
    parser.add_argument('--note', type=str, 
                        default='',)
    args =  parser.parse_args()
    args.n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    args.devices = [i for i in range(args.n_gpus)]
    args.strategy = "ddp" #"ddp"
    return args
if __name__ == '__main__':
    args = argument_inputs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_names = f"size_({args.image_size})_val({args.val_size})_ema({args.use_ema})_bs({args.batch_size})_lr({args.lr})_epochs({args.num_epochs})_kl({args.kl_weight})_lpips({args.lpips_loss_weight})_{args.note}"
    log_dir = f"{args.output_dir}/{file_names}"
    os.makedirs(log_dir, exist_ok=True)

    config = OmegaConf.load("./configs/config_train_dldm.yaml")
    vae_config = config.model.params.first_stage_config
    input_path = "/sd_model/v1-5-pruned.ckpt"
    vae_weight = get_vae_weights(input_path)
    data_module = DataModule(args.data_dir, 
                             batch_size=args.batch_size, 
                             val_size=0.1,
                             size=args.image_size)
    
    model = FinetuneVAE(vae_config=vae_config, 
                        vae_weights=vae_weight, 
                        kl_weight=args.kl_weight, 
                        lpips_loss_weight=args.kl_weight,
                        lr=args.lr, 
                        device=device,
                        log_dir=log_dir,
                        use_ema=args.use_ema)
    
    wandb_logger = WandbLogger(project='finetune_vae', 
                               name='finetune_vae',
                               entity="ssl2022", config=args, dir=log_dir,
                               ) if args.use_wandb else None
    
    trainer = Trainer(min_epochs=1, 
                          max_epochs=args.num_epochs, 
                        precision=args.precision,
                          strategy=args.strategy, 
                          gpus=args.n_gpus, 
                          num_sanity_val_steps=1 if args.val_size > 0 else 0,
                          logger=wandb_logger,
                          default_root_dir=log_dir,)
    

    trainer.fit(model, datamodule=data_module)
    torch.save(model.model.state_dict(), f"{log_dir}/last_model.pth")