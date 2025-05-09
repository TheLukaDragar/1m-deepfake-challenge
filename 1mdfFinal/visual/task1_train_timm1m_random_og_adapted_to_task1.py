"""
Author: LukaDragar
Date: 01.06.2024
"""
# -*- coding: utf-8 -*-
print("start")
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse
import sys

import json
import random
import numpy as np
from tqdm import tqdm
import time
import torch.nn as nn
from logger import create_logger

# from transforms import build_transforms
from metrics import get_metrics
from torch.utils.data import DataLoader


import os
from torch.utils.data import WeightedRandomSampler
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer,seed_everything

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
#tuner ptl
from pytorch_lightning.tuner.tuning import Tuner

from typing import Generator, List, Set, Dict, Optional, Tuple
import os
import glob
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from torch.utils.data import Dataset

import pandas as pd
import cv2

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2




def build_transforms(height, width, max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406],
                     norm_std=[0.229, 0.224, 0.225], **kwargs):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.E
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
        max_pixel_value (float): max pixel value
    """

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

   
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.3),
        # A.GaussNoise(p=0.1),
        # A.GaussianBlur(p=0.1),
        A.Resize(height, width),
        
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
        ToTensorV2(),
    ])
   
   

    test_transform = A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
        ToTensorV2(),
    ])

   

    return train_transform, test_transform



print("imports done")

def save_network(network, save_filename):
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available():
        network.cuda()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network


def make_weights_for_balanced_classes(train_dataset):
    targets = []

    targets = torch.tensor(train_dataset)

    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)]
    )
    weight = 1.0 / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight

def parse_args():
    parser = argparse.ArgumentParser(description="Training network")
   
    parser.add_argument("--pkl_path", type=str, default="../dataset_helper/all_train.pkl")

    # parser.add_argument('--videos_txt', type=str, default='/ceph/hpc/data/st2207-pgp-users/ldragar/1MDF/dataset/train_videos.txt')


    # parser.add_argument('--save_path', type=str, default='./save_result2_swin')
    parser.add_argument(
        "--save_path", type=str, default="./models_luka_1mdf/"
    )
    # parser.add_argument('--model_name', type=str, default='swin_large_patch4_window12_384_in22k')
    parser.add_argument(
        "--model_name",
        type=str,
        default="eva_giant_patch14_224.clip_ft_in1k",
    )
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--adjust_lr_iteration", type=int, default=30000)
    parser.add_argument("--base_lr", type=float, default=0.00005)
    # parser.add_argument('--base_lr', type=float, default=1.74E-05)
    parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument('--resolution', type=int, default=384) #handled by timm
    # validation_size
    parser.add_argument("--validation_split", type=int, default=0.2)

    # parser.add_argument("--val_batch_size", type=int, default=128)

    # every_n_epochs
    parser.add_argument("--save_checkpoint_every_n_epochs", type=int, default=1)

    # wandb
    # parser.add_argument('--experiment_name', type=str, default='swin_large_patch4_window12_384_in22k_40')
    parser.add_argument("--project_name", type=str, default="1M_df_chall")

    # pytorch lightning
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        default=[0],
        help="Devices to train on.",
    )

    parser.add_argument("--seed", type=int, default=1126)

    #resume_run_id for resuming run 
    parser.add_argument("--resume_run_id", type=str, default="None")

    #args
    parser.add_argument("--auto_lr_find", action="store_true", default=False)

    parser.add_argument("--from_pretrained_cp" , type=str, default="None")

    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR",help="ReduceLROnPlateau or CosineAnnealingLR") 


    args = parser.parse_args()
    print("args", args)
    return args



class MyModel(LightningModule):
    def __init__(self, model_name, num_class=2, learning_rate=0.00005,scheduler="ReduceLROnPlateau",epochs=60):
        super(MyModel, self).__init__()
        self.model = timm.create_model(
            model_name, num_classes=num_class, pretrained=True
        )
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.epochs = epochs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch #TODO optional look into the performance on each database
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
        )
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

        if self.scheduler == "ReduceLROnPlateau":
            scheduler = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.1, patience=2, verbose=True
                ),
                "monitor": "val_loss",
            }
            return [optimizer], [scheduler]

        elif self.scheduler == "CosineAnnealingLR":
            scheduler = {
                "scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs,verbose=True),
                "monitor": "val_loss",
            }
            return [optimizer], [scheduler]

class MyDataModule(LightningDataModule):
    def __init__(
        self, train_dataset, weights_train, batch_size, val_dataset
    ):
        super(MyDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.weights_train = weights_train
        self.batch_size = batch_size

    def train_dataloader(self):
        train_sampler = WeightedRandomSampler(
            self.weights_train, len(self.train_dataset), replacement=True
        )
        # shuflle is false because of WeightedRandomSampler OK
        return DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=64,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=64,
            pin_memory=True,
        )




class MultiVideoFrameLoader(Dataset):

    def __init__(self, video_metadata, transform=None,fps=25.0,phase="train"):
        self.video_metadata = video_metadata
        self.transform = transform
        self.fps = fps
        self.phase = phase
        self.data = self.find_frames()

    def find_frames(self):

        tmp=[]

        for i, video in tqdm(self.video_metadata.iterrows(), total=self.video_metadata.shape[0], desc="Finding frames"):
            
            #if real take num_frames_per_real equally spaced frames from whole video
            if video["label"] == 0:

                tmp.append({
                    "video_path":video["video"],
                    "label":0
                })

            else:

              
                tmp.append({
                    "video_path":video["video"],
                    "label":1
                })

        #save as pickle
        df = pd.DataFrame(tmp)
        df.to_pickle(f"filtered_train_videos_split_per_frame_phase_{self.phase}_random_task1.pkl")

        #add phrased to print
        print(f"Number of all real videos in {self.phase} phase:", len([r for r in tmp if r["label"] == 0]))
        print(f"Number of all fake videos in {self.phase} phase:", len([r for r in tmp if r["label"] == 1]))
        print(f"TOTAL Number of videos in {self.phase} phase:", len(tmp))


        return tmp


    def __getitem__(self, index):
        video_path = self.data[index]["video_path"]
        label = self.data[index]["label"]

        try:

            cap = cv2.VideoCapture(video_path)

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


            #chose random frame from video
            # frame_idx = np.random.randint(0, frame_count)

            #choose first frame 
            frame_idx = 0
        
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
             #check if frame is empty
            if not ret or frame is None:
                raise Exception("Frame is empty or failed to read", video_path, frame_idx)


            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                frame = self.transform(image=frame)["image"]
            return frame, label

        except Exception as e:
            timeStr = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime(time.time()))

            # print(f"{timeStr} Error loading frame {e} {video_path} {frame_idx} {visual_fake_segments} {real_segments_beetween_fake} ,{frame_count}")
            localss = locals()
            print(f"{timeStr} Error loading frame {e}, locals {str(localss)}")

            #save to txt file
            with open("TRAINERROR.txt", "a") as f:
                f.write(f"Error loading frame {e} \n")
            

            #return black picture and mark it as real

            frame = np.zeros((224, 224, 3), dtype=np.uint8)
          

            if self.transform is not None:
                frame = self.transform(image=frame)["image"]
            return frame, 0



    def __len__(self):
        return len(self.data)


    

def main():
    args = parse_args()


    seed_everything(args.seed, workers=True)
    print("set random seed", args.seed)

    wandb_logger = None

    if args.resume_run_id == "None":
        # Logger and Trainer
        wandb_logger = WandbLogger(name=args.model_name, project=args.project_name)

    else:
        print("resuming run id", args.resume_run_id)
        wandb_logger = WandbLogger(name=args.model_name, project=args.project_name,version=args.resume_run_id,resume="must")
        
        

    logger = create_logger(
        output_dir="%s/report" % args.save_path, name=f"{args.model_name}"
    )
    logger.info("Start Training %s" % args.model_name)
    timeStr = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime(time.time()))
    logger.info(timeStr)


    df = pd.read_pickle(arg.pkl_path)


    timeStr = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime(time.time()))
    print(
        f"{timeStr}, All Train videos: all:{len(df)}, real:{len(df[df['label'] == 0])}, fake:{len(df[df['label'] == 1])} ratio(real/fake):{len(df[df['label'] == 0])/len(df[df['label'] == 1])}"
    )

    # ptl
    model = MyModel(
        model_name=args.model_name, num_class=args.num_class, learning_rate=args.base_lr,scheduler=args.scheduler,epochs=args.num_epochs
    )
    # get resolution from model
    resolution = model.model.pretrained_cfg["input_size"][
        1
    ]  # pretrained_cfg since we are using pretrained model

    print("resolution", resolution)

    wandb_logger.log_hyperparams({"resolution": resolution})

    data_cfg = timm.data.resolve_data_config(model.model.pretrained_cfg)
    print("Timm data_cfg", data_cfg)

    # get std
    norm_std = data_cfg["std"]
    print("using norm_std", norm_std)
    norm_mean = data_cfg["mean"]
    print("using norm_mean", norm_mean)

    wandb_logger.log_hyperparams({"norm_std": norm_std})
    wandb_logger.log_hyperparams({"norm_mean": norm_mean})

    # important USE TIMM TRANSFORMS! https://huggingface.co/docs/timm/quickstart
    transform_train, transform_test = build_transforms(
        resolution,
        resolution,
        max_pixel_value=255.0,
        norm_mean=[norm_mean[0], norm_mean[1], norm_mean[2]],
        norm_std=[norm_std[0], norm_std[1], norm_std[2]]
    )

   
    # Split dataset into training and validation sets make sure the labels are balanced
    ratio = args.validation_split

    # get indexes for validation set
    val_indexes = random.sample(range(len(df)), int(len(df) * ratio))

    #check if labels are balanced
    print("first 10 val indexes", val_indexes[:10])

    labels = df["label"].values  # Convert to NumPy array for faster access
    val_mask = np.zeros(len(labels), dtype=bool)
    val_mask[val_indexes] = True  # Create a boolean mask for validation indexes

    # Validation counts
    val_real = np.sum((labels == 0) & val_mask)
    val_fake = np.sum((labels == 1) & val_mask)

    # Training counts
    train_real = np.sum((labels == 0) & ~val_mask)
    train_fake = np.sum((labels == 1) & ~val_mask)

   
    print("train real vids", train_real)
    print("train fake vids", train_fake)

    print("val real vids", val_real)
    print("val fake vids", val_fake)


    split = np.array(['train'] * len(df))

    # Set 'val' at positions indexed by 'val_indexes'
    split[val_indexes] = 'val'

    # Assign this array as a new column in the DataFrame
    df['split'] = split

    
    #save as pickle
    df.to_pickle("filtered_train_videos_split_task1.pkl")


    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]


    train_dataset = MultiVideoFrameLoader(train, transform_train, fps=25.0,phase="train")
    val_dataset = MultiVideoFrameLoader(val, transform_test, fps=25.0,phase="val")



    weights_train = make_weights_for_balanced_classes(train["label"].values)
    # when using WeightedRandomSampler the samples will be drawn from your dataset using the provided weights,
    # so you won’t get a specific order. In that sense your data will be shuffled.
    print("weights_train", weights_train)

    data_sampler = WeightedRandomSampler(weights_train, len(train), replacement=True)

    timeStr = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime(time.time()))
    print(timeStr, "All Train videos Number: %d" % (len(train)))

   

    print("batch_size =", args.batch_size)

    data_module = MyDataModule(train_dataset, weights_train, args.batch_size, val_dataset)
    from torchvision import utils

    #for debugging save first batch as images
    j=0
    for i, batch in enumerate(data_module.train_dataloader()):
        if i < 10:
            images, labels = batch
            print("images", images)
            print("labels", labels)
            for j, img in enumerate(images):

            
             
                utils.save_image(img, f"./debug_train_frames_1m_t1/b_{j}_frame_{j}_{labels[j]}.png",normalize=True)
                print("saved image", j)
                j+=1

        else:
            break
                



    wandb_logger.watch(model, log="all", log_freq=100, log_graph=False)
    wandb_logger.log_hyperparams(args)

    wandb_run_id = str(wandb_logger.version)
    #
    if wandb_run_id == "None":
        print("no wandb run id this is a copy of model with DDP")
        # get pid of process
        # pid = os.getpid()
        # print("pid", pid)

        # #get parent pid
        # parent_pid = os.getppid()
        # print("parent_pid", parent_pid)

        # #get args
        # print("args", args)
        # print("sys.argv", sys.argv)

        # #get from os env variable
        # wandb_run_id = os.environ["MY_WANDB_RUN_ID"]
        # print("got wandb run id from env variable", wandb_run_id)

    else:
        # set OS env variable for wandb run id so other subprocesses can access it
        # make a txt file in tmp dir to save wandb run id
        # print("setting env variable for wandb run id")
        pass
        # add argument to sys.argv
        # TODO this does not work for other subprocesses as far as I tested
        # sys.argv.append("--wandb_run_id")
        # sys.argv.append(wandb_run_id)

        # print("sys.argv", sys.argv)

    print("init trainer")

    # save hyperparameters
    wandb_logger.log_hyperparams(model.hparams)

    # save checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="%s/%s/%s" % (args.save_path, args.model_name, wandb_run_id),
        filename="%s-{epoch:02d}-{val_loss:.2f}-{train_loss:.2f}" % args.model_name,
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=args.save_checkpoint_every_n_epochs,
    )
    # print("checkpoint_callback", checkpoint_callback)

    
    #get last checkpoint path if exists
    ckpt_path = None
    if not args.resume_run_id == "None":
        print("searching for checkpoint")

        if os.path.exists("%s/%s/%s" % (args.save_path, args.model_name, wandb_run_id)):
            #get latest checkpoint from folder
            ckpt_path = max(
                [
                    os.path.join("%s/%s/%s" % (args.save_path, args.model_name, wandb_run_id), f)
                    for f in os.listdir("%s/%s/%s" % (args.save_path, args.model_name, wandb_run_id))
                ],
                key=os.path.getctime,
            )
        
            print("found checkpoint", ckpt_path)
        else:
            print("no checkpoint found")
        
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_epochs=args.num_epochs,
        logger=wandb_logger,
        accelerator="gpu",
        strategy="ddp",
        num_nodes=args.num_nodes,
        devices=args.devices,
        callbacks=[checkpoint_callback,lr_monitor],
    )

    if args.auto_lr_find:

        #THIS DOESENT WORK YET!
        print("running lr finder")
        #base is 0.00005 which is 5e-5
        lr_finder = Tuner(trainer).lr_find(model, min_lr=1e-8, max_lr=1e-4,datamodule=data_module)
        print("lr",lr_finder.results["lr"])
        print("loss",lr_finder.results["loss"])


        # Plot with

        #set lr
        model.learning_rate = lr_finder.suggestion()
        model.hparams.learning_rate = lr_finder.suggestion()
        print("found best lr at ", lr_finder.suggestion())
        wandb_logger.log_hyperparams(model.hparams)



    if trainer.global_rank == 0:
        # make cp save dir savepath _ model name _ wandb run id
        if not os.path.exists(
            "%s/%s/%s" % (args.save_path, args.model_name, wandb_run_id)
        ):
            os.makedirs(
                "%s/%s/%s" % (args.save_path, args.model_name, wandb_run_id),
                exist_ok=True,
            )
            print("made dir %s/%s/%s" % (args.save_path, args.model_name, wandb_run_id))

    print(trainer.global_rank, trainer.world_size, os.environ["SLURM_NTASKS"] if "SLURM_NTASKS" in os.environ else "no slurm")

    if not args.resume_run_id == "None":
        print("using checkpoint", ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    print("saving last checkpoint")
    trainer.save_checkpoint(
        "%s/%s/%s/%s_final.ckpt"
        % (args.save_path, args.model_name, wandb_run_id, args.model_name)
    )


if __name__ == "__main__":
    args = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    if not os.path.exists("%s/%s/" % (args.save_path, args.model_name)):
        os.makedirs("%s/%s/" % (args.save_path, args.model_name), exist_ok=True)
    if not os.path.exists("%s/report" % args.save_path):
        os.makedirs("%s/report" % args.save_path, exist_ok=True)

    main()
