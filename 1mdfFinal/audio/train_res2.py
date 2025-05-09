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

# from transforms import build_transforms
from torch.utils.data import DataLoader

import os
from torch.utils.data import WeightedRandomSampler

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



import shutil
import numpy as np
from modelres_2 import *
from dataset import *
from torch.utils.data import DataLoader
import torch.utils.data.sampler as torch_sampler
from lossres_2 import *
from collections import defaultdict
from tqdm import tqdm, trange
import random
from utils import *
import torch.nn.functional as F
torch.set_default_tensor_type(torch.FloatTensor)

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


######################################################################
# Save model
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


    parser.add_argument("--pkl_path", type=str, default="./train_audio_modifed_and_real_segments_res2_100.pkl")

    parser.add_argument(
        "--save_path", type=str, default="./models_luka_1mdf_audio_res_2/"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="TDL_res2",
    )
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--base_lr", type=float, default=0.0001) 
    # parser.add_argument('--base_lr', type=float, default=1.74E-05)
    parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument('--resolution', type=int, default=384) #handled by timm
    # validation_size
    parser.add_argument("--validation_split", type=int, default=0.2)

    # parser.add_argument("--val_batch_size", type=int, default=128)

    # every_n_epochs
    parser.add_argument("--save_checkpoint_every_n_epochs", type=int, default=1)

    # wandb
    # parser.add_argument('--experiment_name', type=str, default='swin_large_patch4_window12_384_in22k_40')
    parser.add_argument("--project_name", type=str, default="1M_df_chall_audio")

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



    # parser.add_argument("--auto_batch_size", action="store_true", default=False)



    args = parser.parse_args()
    print("args", args)
    return args


def cls_loss(scores, labels):
    '''
    calculate classification loss
    1. dispose label, ensure the sum is 1
    2. calculate topk mean, indicates classification score
    3. calculate loss
    '''
    labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-10)
    clsloss = -torch.mean(torch.sum(labels * F.log_softmax(scores, dim=1), dim=1), dim=0)
    return clsloss

def shuffle(feat, labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    return feat, labels

class MyModel(LightningModule):
    def __init__(self, model_name, learning_rate=0.00005,scheduler="ReduceLROnPlateau",epochs=60,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005):
        super(MyModel, self).__init__()
        self.model =  TDL()
        self.model.train()
    
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.epochs = epochs

        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.criterion = cls_loss
        self.emb_loss = EmbeddingLoss()

        self.val_predictions = []
        self.val_labels = []

        #dont log gradiets od e

    
    def process_and_log_predictions(self,feat_outputs, window_labels, filenames,batch_idx):
        # Get predictions
        score = feat_outputs.squeeze(dim=0).cpu().detach().numpy()  # Remove padding and convert to numpy
        labels = window_labels.squeeze(dim=0).cpu().detach().numpy()  # Remove padding and convert to numpy

        # print("score", score.shape)
        # print("labels", labels.shape)

        score = feat_outputs.cpu().detach().numpy()
        labels = window_labels.cpu().detach().numpy()
        
        yy = labels.astype(float).flatten()
        yp = score.astype(float).flatten()

        self.val_predictions.extend(yp)
        self.val_labels.extend(yy)

        # Log the roc_auc using your logging mechanism (assuming self.log)
        # self.log("roc_auc", roc_auc, prog_bar=True, sync_dist=True, on_epoch=True)

        # # Save predictions and labels to file
        # with open(f"val_predictions/predictions_{batch_idx}.txt", "w") as f:
        #     for i in range(len(filenames)):
        #         for j in range(labels.shape[1]):
        #             f.write(f"{filenames[i]} | {score[i, j]} | {labels[i, j]}\n")



    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        filenames,w2wec,window_labels = batch

        #shuffle
        w2wec, window_labels = shuffle(w2wec, window_labels)

        
        feats = w2wec.transpose(1, 2)

        # print("feats", feats.shape)


        embedding, feat_outputs = self.model(feats)

        # print("embedding", embedding.shape)

        # CE_loss = self.criterion(feat_outputs,window_labels)
        # #we have no padding
        # embedding_loss = self.emb_loss(embedding, 16,window_labels) 
        # loss = CE_loss + embedding_loss
        #NOTE RESULOTION CHNGE FROM 0.08 TO 0.02
        BCE_loss = nn.functional.binary_cross_entropy(feat_outputs, window_labels.float())
        embedding_loss = self.emb_loss(embedding,16*4,window_labels.float())
        loss = BCE_loss + 0.1 * embedding_loss

        #log both losses
        self.log("train_BCE_loss", BCE_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("train_embedding_loss", embedding_loss, prog_bar=True, sync_dist=True, on_epoch=True)
     
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        filenames,w2wec,window_labels = batch

        feats = w2wec.transpose(1, 2)
        embedding, feat_outputs = self.model(feats)

        # CE_loss = self.criterion(feat_outputs,window_labels)
        
        # embedding_loss = self.emb_loss(embedding, 16,window_labels)

        # loss = CE_loss + embedding_loss

        BCE_loss = nn.functional.binary_cross_entropy(feat_outputs, window_labels.float())
        embedding_loss = self.emb_loss(embedding,16*4,window_labels.float())
        loss = BCE_loss + 0.1 * embedding_loss


        self.log("val_BCE_loss", BCE_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_embedding_loss", embedding_loss, prog_bar=True, sync_dist=True, on_epoch=True)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        #save labels and predictions to txt file
        #get predictions 
        # score = feat_outputs
        # score = score.squeeze(dim=0).cpu() # before calculate EER, delete the padding area according to the lenori.
        # score = score.detach().numpy()
        # labels = window_labels
        # labels = labels.squeeze(dim=0).cpu() # before calculate EER, delete the padding area according to the lenori.
        # labels = labels.detach().numpy()

        # print(score.shape, labels.shape)

        self.process_and_log_predictions(feat_outputs, window_labels, filenames,batch_idx)


        return loss

    def on_validation_epoch_end(self):
        yy = np.array(self.val_labels).flatten()
        yp = np.array(self.val_predictions).flatten()

        fpr, tpr, thresholds = roc_curve(yy, yp, pos_label=1)
        roc_auc = auc(fpr, tpr)
        self.log("roc_auc", roc_auc, prog_bar=True, sync_dist=True, on_epoch=True)

        # Clear the accumulators for the next validation epoch
        self.val_predictions = []
        self.val_labels = []

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    
        # #calculate auc
        # #cast both to float
        # yy = labels.astype(float)
        # #multiple arrays to single array
        # yy = np.concatenate(yy)

        # yp = score.astype(float)
        # yp = np.concatenate(yp)

        # fpr, tpr, thresholds = roc_curve(yy, yp, pos_label=1)
        # roc_auc = auc(fpr, tpr)
        # self.log("roc_auc", roc_auc, prog_bar=True, sync_dist=True, on_epoch=True)


       


        # with open("predictions.txt", "w") as f:
        #     for i in range(len(filenames)):
        #         f.write(f"{filenames[i]} | {score[i]} | {labels[i]}")
        #         f.write("\n")
            






    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        
       
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
        # train_sampler = WeightedRandomSampler(
        #     self.weights_train, len(self.train_dataset), replacement=True
        # )
        # shuflle is false because of WeightedRandomSampler OK
        return DataLoader(
            self.train_dataset,
            # sampler=train_sampler,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
        )



def main():
    args = parse_args()


    seed_everything(args.seed, workers=True)
    print("set random seed", args.seed)

    wandb_logger = None

    if args.resume_run_id == "None":
        # Logger and Trainer
        wandb_logger = WandbLogger(name=args.model_name, project=args.project_name,offline=True)

    else:
        print("resuming run id", args.resume_run_id)
        wandb_logger = WandbLogger(name=args.model_name, project=args.project_name,version=args.resume_run_id,resume="must",offline=True)
        
        

    
    timeStr = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime(time.time()))
  

    #/ceph/hpc/data/st2207-pgp-users/ldragar/TDL-ADD/train_audio_modifed_and_1000_real_segments.pkl

    df = pd.read_pickle(args.pkl_path)

    # df_1000_real = df[df['modify_type'] == 'real'].sample(n=1000, random_state=1)
    # #NOLY AUDIO MODIFIED 
    # df = df[df['modify_type'] == 'audio_modified']

    # #and 1000 real
    # df = pd.concat([df, df_1000_real])


    #use only 10 000
    # df = df.sample(n=50000, random_state=1)


    timeStr = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime(time.time()))
    print(
        timeStr,
        "Number of videos in the dataset: %d" % (len(df)),
    )

    # ptl
    model = MyModel(
        model_name=args.model_name, learning_rate=args.base_lr,scheduler=args.scheduler,epochs=args.num_epochs
    )
   

   
    # Split dataset into training and validation sets make sure the labels are balanced
    ratio = args.validation_split

    # get indexes for validation set
    val_indexes = random.sample(range(len(df)), int(len(df) * ratio))

    #check if labels are balanced
    print("first 10 val indexes", val_indexes[:10])


    split = np.array(['train'] * len(df))

    # Set 'val' at positions indexed by 'val_indexes'
    split[val_indexes] = 'val'

    # Assign this array as a new column in the DataFrame
    df['split'] = split

    
    #save as pickle
    df.to_pickle("filtered_train_videos_split_audio.pkl")


    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

   

    train_dataset = DF1M_2(train)
    val_dataset = DF1M_2(val)





    timeStr = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime(time.time()))
    print(timeStr, "All Train videos Number: %d" % (len(train)))

   

    print("batch_size =", args.batch_size)

    data_module = MyDataModule(train_dataset, [], args.batch_size, val_dataset)
   
                

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
        print("setting env variable for wandb run id")
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

    # just in case
    if trainer.global_rank == 0:
        print("saving model 2")



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
