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


# class Transcript:
#     """
#     Represents a transcript of a video.

#     Attributes:
#         word (str): The word in the transcript.
#         start (float): The start time of the word in the video.
#         end (float): The end time of the word in the video.
#     """

#     def __init__(self, word: str, start: float, end: float):
#         self.word = word
#         self.start = start
#         self.end = end

#     def __repr__(self):
#         return f"Transcript(word={self.word}, start={self.start}, end={self.end})"

# class Operation:

#     """
#     Represents an operation applied to a video. can be  {'delete', 'insert', 'replace'}

#     Attributes:
#         operation (str): The operation applied.
#         start (float): The start time of the operation.
#         end (float): The end time of the operation.
#     """


#     def __init__(self, operation: str, **kwargs):
#         self.operation = operation
#         for key, value in kwargs.items():
#             setattr(self, key, value)

      

#     def __repr__(self):
#         return f"Operation(operation={self.operation}, start={self.start}, end={self.end})"


# class Sample:
#     """
#     Represents information about a sample video file including its associated metadata.
    
#     Attributes:
#         path_to_video (str): Path to the video file.
#         path_to_metadata (str): Path to the corresponding metadata file.
#         path_to_audio (str): Path to the extracted audio file.
#         fake_segments (List[Tuple[int, int]]): List of fake segments as (start, end) timestamps.
#         fake_audio_segments (List[Tuple[int, int]]): List of fake audio segments as (start, end) timestamps.
#         fake_visual_segments (List[Tuple[int, int]]): List of fake visual segments as (start, end) timestamps.
#         modify_type (str): Type of modifications (real, visual_modified, audio_modified, both_modified).
#         is_real (bool): True if video is real, False otherwise.
#         audio_model (str): The audio generation model used.
#         operations (List[Dict]): List of operations applied to the video.(Dictionary with keys: 'operation', 'start', 'end' and other optional keys)  operations can be {'delete', 'insert', 'replace'}
#         video_frames (int): Number of frames in the video.
#         audio_frames (int): Number of frames in the audio.
#         transcript (str): Transcript of the video.
#         split (str): Dataset split (train, val, test).
#         original (Optional[str]): Path to the original video if current is fake, else None.
#     """

#     # ['file', 'original', 'split', 'modify_type', 'audio_model', 'fake_segments', 'audio_fake_segments', 'visual_fake_segments', 'video_frames', 'audio_frames', 'operations', 'transcripts'])

#     def __init__(self, path_to_video: str):
#         self.path_to_video = self.__check_path_to_vid(path_to_video)
#         self.path_to_metadata = self.__get_metadata_path(path_to_video)
#         self.path_to_audio: Optional[str] = None
#         self.fake_segments: Optional[List[Tuple[int, int]]] = None
#         self.audio_fake_segments: Optional[List[Tuple[int, int]]] = None
#         self.visual_fake_segments: Optional[List[Tuple[int, int]]] = None
#         self.modify_type: Optional[str] = None
#         self.is_real: Optional[bool] = None
#         self.audio_model: Optional[str] = None
#         self.video_frames: Optional[int] = None
#         self.audio_frames: Optional[int] = None
#         self.split: Optional[str] = None
#         self.original: Optional[str] = None

#         self.__process_metadata()

#     def __check_path_to_vid(self, path_to_video: str) -> str:
#         if not os.path.exists(path_to_video):
#             raise FileNotFoundError(f"Video file '{path_to_video}' not found.")
#         return path_to_video

#     def __get_metadata_path(self, video_path: str) -> str:
#         # Generate the path to metadata file from video path
#         metadata_directory = os.path.dirname(video_path).replace("/dataset/train", "/dataset/train_metadata"). \
#             replace("/dataset/test", "/dataset/test_metadata").replace("/dataset/val", "/dataset/val_metadata")
#         video_name = os.path.basename(video_path)
#         metadata_name = f"{os.path.splitext(video_name)[0]}.json"
#         return os.path.join(metadata_directory, metadata_name)

#     def __process_metadata(self):
#         """Opens metadata file and sets attributes based on its contents."""
#         try:
#             with open(self.path_to_metadata, 'r', errors='ignore') as f:
#                 data = json.load(f)
#                 # print(data.keys())
                
#                 self.fake_segments = data.get('fake_segments')
#                 self.audio_fake_segments = data.get('audio_fake_segments')
#                 self.visual_fake_segments = data.get('visual_fake_segments')
                
#                 self.modify_type = data.get('modify_type')

#                 self.audio_model = data.get('audio_model')

#                 # self.operations = data.get('operations')
#                 self.is_real = self.modify_type == "real"

               
#                 self.video_frames = data.get('video_frames')
#                 self.audio_frames = data.get('audio_frames')

#                 ts = data.get('transcripts')
#                 # self.transcripts = [Transcript(t['word'], t['start'], t['end']) for t in ts] if ts else None


#                 self.split = data.get('split')
#                 self.path_to_audio = self.__get_audio_path()
                
#                 # Additional attributes
#                 self.original = self.path_to_video if self.is_real else data.get('original')

#         except FileNotFoundError:
#             print(f"Metadata file '{self.path_to_metadata}' not found.")

#     def __get_audio_path(self) -> Optional[str]:
#         """Generates path to the audio file based on video path."""
#         if not self.is_real:
#             video_path = self.path_to_video
#             name = "real.wav" if "real_audio" in video_path else "fake.wav"
#             parts = video_path.split('train', 1) if 'train' in video_path else video_path.split('val', 1)
#             audio_path = os.path.join(parts[0], "audio_only", 'train' if 'train' in video_path  else 'val', *parts[1].split(os.sep)[:-1], name)
#             return audio_path
#         return None
    

    

#     def _get_metadata(self) -> dict:
#         """Read and return the metadata from the JSON file."""
#         try:
#             with open(self.path_to_metadata, 'r', errors='ignore') as f:
#                 return json.load(f)
#         except FileNotFoundError:
#             print(f"Metadata file '{self.path_to_metadata}' not found.")
#         except json.JSONDecodeError:
#             print(f"Error decoding JSON from '{self.path_to_metadata}'.")
#         return {}
    
#     @property
#     def operations(self) -> List[Operation]:
#         """Dynamically load operations from metadata."""
#         data = self._get_metadata()
#         operations_data = data.get('operations', [])
#         return [Operation(**op) for op in operations_data]

#     @property
#     def transcripts(self) -> List[Transcript]:
#         """Dynamically load transcripts from metadata."""
#         data = self._get_metadata()
#         transcripts_data = data.get('transcripts', [])
#         return [Transcript(**t) for t in transcripts_data]
        

#     def operations_generator(self) -> Generator[Operation, None, None]:
#         """Yield operations one by one."""
#         self._load_metadata()
#         for op in self._operations_data:
#             yield Operation(**op)

#     def transcripts_generator(self) -> Generator[Transcript, None, None]:
#         """Yield transcripts one by one."""
#         self._load_metadata()
#         for t in self._transcripts_data:
#             yield Transcript(**t)


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

    parser.add_argument("--pkl_path", type=str, default="./only_visual_with_duplicate_so_its_easier_to_load_w_real_segments_beetween_fake_plus_15_percof_real_splited.pkl")

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
            num_workers=6,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )




class MultiVideoFrameLoader(Dataset):

    def __init__(self, video_metadata, transform=None,fps=25.0,phase="train"):
        self.data = video_metadata
        self.transform = transform
        self.fps = fps
        self.phase = phase
        
    def __getitem__(self, index):
        video_path = self.data.iloc[index]["video"]
        # frame_idx = self.data[index]["frame_idx"]
        visual_fake_segments = self.data.iloc[index]["visual_fake_segments"]
        real_segments_beetween_fake = self.data.iloc[index]["real_segments_beetween_fake"]
        label = self.data.iloc[index]["label"]

    



        try:

            cap = cv2.VideoCapture(video_path)

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


            #chose random frame if real
            frame_idx = np.random.randint(0, frame_count)

            #chose random seg and then random frame from the segment if fake
            if visual_fake_segments is not None and label == 1:
                #randomly choose a segment
                start_seconds, end_seconds = random.choice(visual_fake_segments)

                #convert to frame index
                start_frame = int(start_seconds * self.fps)
                end_frame = int(end_seconds * self.fps)

                #fix out of bounds
                start_frame = max(0, start_frame)
                end_frame = min(frame_count, end_frame)

                #randomly choose a frame from the segment
                frame_idx = np.random.randint(start_frame, end_frame)

            elif len(real_segments_beetween_fake) > 0 and label == 0:
                #randomly choose a segment
                start_seconds, end_seconds = random.choice(real_segments_beetween_fake)

                #convert to frame index
                start_frame = int(start_seconds * self.fps)
                end_frame = int(end_seconds * self.fps)

                #fix out of bounds
                start_frame = max(0, start_frame)
                end_frame = min(frame_count, end_frame)

                #randomly choose a frame from the segment
                frame_idx = np.random.randint(start_frame, end_frame)



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
                f.write(f"Error loading frame {e} {video_path} \n")
            

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
        wandb_logger = WandbLogger(name=args.model_name, project=args.project_name,offline=True)

    else:
        print("resuming run id", args.resume_run_id)
        wandb_logger = WandbLogger(name=args.model_name, project=args.project_name,version=args.resume_run_id,resume="must",offline=True)
    


    df = pd.read_pickle(args.pkl_path)
    #NOTE PREPARED AHEAD OF TIME!!

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
            # print("images", images)
            print("labels", labels)
            for j, img in enumerate(images):

            
             
                utils.save_image(img, f"./debug_train_frames_1m/b_{j}_frame_{j}_{labels[j]}.png",normalize=True)
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
        # # add argument to sys.argv
        # # TODO this does not work for other subprocesses as far as I tested
        # sys.argv.append("--wandb_run_id")
        # sys.argv.append(wandb_run_id)

        # print("sys.argv", sys.argv)
        pass

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
