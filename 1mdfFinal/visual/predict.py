"""
Author: LukaDragar
Date: 01.06.2024
"""
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import argparse
import json
import wandb

import numpy as np

# import pandas as pd
import torchvision.models as models
import timm

from train_timm1m_random_og_real_segments_beetween_fake import MyModel

from train_timm1m_random_og_real_segments_beetween_fake import build_transforms

from tqdm import tqdm

import os
from torch.utils.data import Dataset

import pandas as pd
import cv2


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import time

os.environ["WANDB__SERVICE_WAIT"]="60"

import argparse
import os
from urllib.parse import urlparse

#add curent dir to sys
import torch
import os

import os
import cv2
import numpy as np
import pandas as pd
import random
import time

from typing import List
from tqdm import tqdm
import json

# torch.cuda.is_available = lambda : False


# For the CLI, we fallback to saving the output to the current directory


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating network")

    parser.add_argument(
        "--root_path",
        default="../dataset/test/",
        # parser.add_argument('--root_path', default='/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/DFGC-2022/Detection Dataset/10-frames-Private-faces',
        type=str,
        help="path to Evaluating dataset",
    )

    parser.add_argument('--videos_txt', type=str, default='../dataset_helper/test_100.txt') #or val_videos.txt

   
    parser.add_argument(
        "--pre_trained_dir",
        type=str,
        default="./models_luka_1mdf/",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./visual_predictions/" #or predict_1mdf_trained_val_visual_jn41pnlb_final_gpuuu
    )

    parser.add_argument(
        "--latest_only",
        action="store_true",
        default=True,
        help="evaluate the latest model only",
    )

    #run_id
    parser.add_argument("--run_id", type=str, default="jn41pnlb")

    parser.add_argument("--val_batch_size", type=int, default=1)

    parser.add_argument("--from_epoch", type=int, default=0)

    parser.add_argument("--eval_at_epoch", type=int, default=-1)

    #each worker will predfict a subset of videos

    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--worker_id", type=int, default=0)

    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--num_gpus", type=int, default=1)



    args = parser.parse_args()
    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename,map_location=device)["state_dict"])
    return network



def test_model(model, dataloader,gpu_id):
    prediction = np.array([])
    # video_path = np.array([])

    for batch in dataloader:

        face_images = batch

        #to gpu
        face_images = face_images.to(device)
       
        outputs = model(face_images)

        pred_ = torch.nn.functional.softmax(outputs, dim=-1)

        # [:, 1] selects the second column from each row of the array. This corresponds to the probabilities associated with the second class (usually the "positive" class in binary classification). This slicing operation results in a one-dimensional array containing the probability of the positive class for each image in the batch.
        pred_ = pred_.cpu().detach().numpy()[:, 1]

       
        prediction = np.concatenate((prediction, pred_))

        # video_path = np.concatenate((video_path, _video_paths))
       
        #todo video_label is the same for all frames
            


    return prediction



    


import json
import os
from typing import Generator, List, Set, Dict, Optional, Tuple
import os
import glob
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

class Transcript:
    """
    Represents a transcript of a video.

    Attributes:
        word (str): The word in the transcript.
        start (float): The start time of the word in the video.
        end (float): The end time of the word in the video.
    """

    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Transcript(word={self.word}, start={self.start}, end={self.end})"

class Operation:

    """
    Represents an operation applied to a video. can be  {'delete', 'insert', 'replace'}

    Attributes:
        operation (str): The operation applied.
        start (float): The start time of the operation.
        end (float): The end time of the operation.
    """


    def __init__(self, operation: str, **kwargs):
        self.operation = operation
        for key, value in kwargs.items():
            setattr(self, key, value)

      

    def __repr__(self):
        return f"Operation(operation={self.operation}, start={self.start}, end={self.end})"


class Sample_frames_loader(Dataset):

    """
    Dataset class for collecting frames with corresponding labels from a video file.

    Args:
        video_path (str): Path to the video file.
        transform (Optional[Callable]): A function/transform to apply to each frame.

    Returns:
        Tuple[torch.Tensor, NumpyArray, NumpyArray]: A tuple containing the frame image, frame label and video label.
    """

    def __init__(self, video_path, transform=None):
        self.video_path = video_path
    
        face_images,fps = self.extract()

        self.face_images = face_images
        self.fps = fps

        self.transform = transform

    def extract(self):
        face_images = []

        cap_org = cv2.VideoCapture(self.video_path)
        frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_org.get(cv2.CAP_PROP_FPS)
    
        #use every frame
        frame_idxs = np.linspace(0, frame_count_org - 1, frame_count_org, endpoint=True, dtype=np.int64)

       
        # video_label is for both classification and temporal localization task

        for cnt_frame in range(frame_count_org): 
            ret_org, frame_org = cap_org.read()
            height,width=frame_org.shape[:-1]
            if not ret_org:
                raise ValueError(f'Frame read {cnt_frame} Error! : {os.path.basename(org_path)}')
                continue
            
            if cnt_frame not in frame_idxs:
                continue
            
            frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)

            face_images.append(frame)
            

        return face_images,fps

    def __getitem__(self, index):
        image = self.face_images[index]
        # video_path = self.video_path
       

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.face_images)



def run_predict(model,video_files: List[str], transform_test,val_batch_size: int,output_csv,pre_trained_name,gpu_id, root_directory = "/ceph/hpc/data/st2207-pgp-users/ldragar/1MDF/dataset"):

    def process_video(video):
        """
        Function to process a single video, extracting and returning its operations.
        """
        video_path = os.path.join(root_directory, video)

        test_dataset = Sample_frames_loader(video_path, transform=transform_test)


        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        with torch.no_grad():
               
            frame_predictions = test_model(model, test_loader,gpu_id)


              

        # if len(frame_predictions) != 0:
        #     video_prediction = np.mean(frame_predictions, axis=0)
        # else:
        #     video_prediction = 0.5

        sample = {
            "video": video,
            # "video_prediction": video_prediction,
            "frames_prediction": frame_predictions,

            # "fps": 25 #assume 25 fps
        }

        return sample

    


    def save_results(results):
        """
        Function to save the results to disk.
        """
        #crate df with append mode
        df = pd.DataFrame(results)
        # print("computing metrics")
        # print("df", df)
        # compute_metrics(df)
        # print("saving df")

    
        # df.to_csv(output_csv, index=False, mode='a', header=not file_exists)
        # print(f"Data saved to {output_csv} (appended)")

        # output_pickle = 'output.pkl'

        # Check if the file exists
        file_exists = os.path.exists(output_csv)

        if file_exists:
            # Load the existing data
            with open(output_csv, 'rb') as f:
                existing_data =pd.read_pickle(f)
            
            # join the existing df with the new df
            new_data = pd.concat([existing_data, df], ignore_index=True)
            new_data.to_pickle(output_csv)

        else:
            new_data = df
            new_data.to_pickle(output_csv)

        # Save the updated data

        print(f"Data saved to {output_csv}")




    save_every = 200
    metric_every = 100

    # gts= []
    # predictions = []
    

    results = []


    for i, video in enumerate(tqdm(video_files)):
        result = process_video(video)
        results.append(result)
        # gts.append(result["video_label"])
        # predictions.append(result["video_prediction"])

        # if len(results) % metric_every == 0:
        #     # Save results to disk
        #     # print("computing metrics")
        #     # print("gts", gts)
        #     # print("predictions", predictions)
        #     compute_metrics(gts, predictions)

        if len(results) % save_every == 0:
            # Save results to disk
            # print("Saving results to disk")
            # print("results", results)
            save_results(results)
            results = []


    # Save any remaining results
    if len(results) > 0:
        # print("Saving results to disk")
        save_results(results)
        results = []

    print("All videos processed.")

            

def main():
    args = parse_args()
    # test_videos = os.listdir(args.root_path)

    print("running with args", args)  

    vids = open(args.videos_txt, 'r').read().splitlines()
    print("Number of videos: ", len(vids))

    #make them absolute paths
    all_test_videos = [os.path.join(args.root_path, vid) for vid in vids]

    #load metadata

    #depending on the worker_id, each worker will predict a subset of videos

    num_workers = args.num_workers
    worker_id = args.worker_id

    print("worker_id", worker_id)
    print("num_workers", num_workers)

    

    # Distribute the videos across workers
    def distribute_videos(videos, num_workers, worker_id):
        # Ensure the worker_id is within the range
        if worker_id >= num_workers:
            raise ValueError("worker_id must be less than num_workers")
        
        # Distribute videos by slicing the list
        subset_size = len(videos) // num_workers
        remainder = len(videos) % num_workers
        
        # Calculate start and end indices for this worker's subset
        start_idx = worker_id * subset_size + min(worker_id, remainder)
        end_idx = start_idx + subset_size + (1 if worker_id < remainder else 0)
        
        return videos[start_idx:end_idx]

    # Get the subset of videos for this worker
    test_videos = distribute_videos(all_test_videos, num_workers, worker_id)
    
    print("Number of videos for this worker: ", len(test_videos))


    #now distribute againts the gpus
    num_gpus = args.num_gpus
    gpu_id = args.gpu_id

    print("gpu_id", gpu_id)
    print("num_gpus", num_gpus)

    # Distribute the videos across workers
    def distribute_gpus(videos, num_gpus, gpu_id):
        # Ensure the worker_id is within the range
        if gpu_id >= num_gpus:
            raise ValueError("gpu_id must be less than num_gpus")
        
        # Distribute videos by slicing the list
        subset_size = len(videos) // num_gpus
        remainder = len(videos) % num_gpus
        
        # Calculate start and end indices for this worker's subset
        start_idx = gpu_id * subset_size + min(gpu_id, remainder)
        end_idx = start_idx + subset_size + (1 if gpu_id < remainder else 0)
        
        return videos[start_idx:end_idx]

    # Get the subset of videos for this worker

    test_videos = distribute_gpus(test_videos, num_gpus, gpu_id)

    print(f"Number of videos for this worker {worker_id} and gpu {gpu_id}: ", len(test_videos))





    # project_name = "1M_df_chall"
    run_id = args.run_id   

    # Authenticate to wandb
    # wandb.login()

    # # Connect to the specified project
    # api = wandb.Api(timeout=60)
    

    #get details of the run
    # run = api.run(f"{project_name}/{run_id}")
    # model_name = run.config['model_name']
    model_name = "eva_giant_patch14_224.clip_ft_in1k"
    print("wandb run", run_id)
    print("model_name", model_name)


    model = MyModel(model_name=model_name)







    # get resolution from model
    resolution = model.model.pretrained_cfg["input_size"][
        1
    ]  # pretrained_cfg since we are using pretrained model

    print("resolution", resolution)

    data_cfg = timm.data.resolve_data_config(model.model.pretrained_cfg)
    print("Timm data_cfg", data_cfg)

    # get std
    norm_std = data_cfg["std"]
    print("using norm_std", norm_std)
    norm_mean = data_cfg["mean"]
    print("using norm_mean", norm_mean)

    

    # important USE TIMM TRANSFORMS! https://huggingface.co/docs/timm/quickstart
    _, transform_test = build_transforms(
        resolution,
        resolution,
        max_pixel_value=255.0,
        norm_mean=[norm_mean[0], norm_mean[1], norm_mean[2]],
        norm_std=[norm_std[0], norm_std[1], norm_std[2]]
    )

    # load saved model
    trained_models = os.listdir(os.path.join(args.pre_trained_dir, model_name,run_id))
    # print("trained_models", trained_models)
    # trained_models = ['swin_large_patch4_window12_384_in22k_0.pth']
    # trained_models = ["swin_large_patch4_window12_384_in22k_40.pth", "swin_large_patch4_window12_384_in22k.pth"]
    path = os.path.abspath(os.path.join(args.pre_trained_dir, model_name,run_id))
    # keep only .pth files
    files = [os.path.join(path, f) for f in trained_models if f.endswith(".ckpt")]

    files.sort(key=os.path.getctime)
    print("found", files)

    if args.latest_only:
        files = files[-1:]
        print("files", files)


    for trained in files:
        # pre_trained = os.path.join(args.pre_trained_dir, trained)


        if args.eval_at_epoch != -1:
            epoch = trained.split("/")[-1].split("-")[1].split("=")[-1]
            if epoch.isnumeric():
                if int(epoch) != args.eval_at_epoch:
                    print("skipping", epoch)
                    continue
                else:
                    print("evaluating model at epoch", epoch)


        pre_trained = trained
        print("Device:", device)
        print("Model device:", next(model.parameters()).device)
        model = load_network(model, pre_trained).to(device)
        model.train(False)
        model.eval()


    

        n_gpus = torch.cuda.device_count()
        print("Number of GPUs: ", n_gpus)

        #get gpu names
        gpu_name = torch.cuda.get_device_name()
        print("GPU name: ", gpu_name)



        # device_ids = list(range(n_gpus))

        # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()



        output_dir = args.output_dir

        #create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)
            

        # swin_large_patch4_window12_384.ms_in22k_ft_in1k-epoch=15-train_loss=0.03.ckpt

        #check if maybe eva_giant_patch14_224.clip_ft_in1k_final.ckpt

        if "_final" in trained:
            #get number of epoch from wandb
            #umm maybe i save one checkpont to many 9=final maybe when we specify 10 epochs
            epoch = str(69420)

        else:

            epoch = trained.split("/")[-1].split("-")[1].split("=")[-1]
        print("epoch", epoch)

        # epoch = trained.split('_')[-1].split('.')[0]
        if epoch.isnumeric():

            if args.from_epoch:
                if int(epoch) < args.from_epoch:
                    print("skipping", epoch)
                    continue
            
            output_name = f"pred_{args.run_id}_{epoch}_worker_{worker_id}_{num_workers}_gpu_{gpu_id}_{num_gpus}.pkl"
            output_txt = os.path.join(output_dir, output_name)

            #check if the output file already exists if so add a timestamp
            if os.path.exists(output_txt):
                print("output file already exists RESUMING")

                #load df
                df = pd.read_pickle(output_txt)

                #get the videos that have already been processed
                processed_videos = df["video"].values

                #remove the processed videos from the list
                #remove remove the os.path.join(args.root_path, vid) from the list
                #remove root_path from the path

                print("processed_videos", processed_videos[:5])
                print("processed_videos", len(processed_videos))
                print("test_videos", len(test_videos))
                print("test_videos", test_videos[:5])

                #remove the processed videos from the list
                prev_len = len(test_videos)
                test_videos = [vid for vid in test_videos if vid not in processed_videos]

                print("non processed videos", len(test_videos), "from", prev_len)





                # output_txt = output_txt.replace(".pkl", f"_{int(time.time())}.pkl")
                # print("new output file", output_txt)


            run_predict(model,test_videos,transform_test,args.val_batch_size,output_txt,gpu_id,trained)

        


       
        

        
        


        

    




        



if __name__ == "__main__":
    args = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = ["0","1","2","3"]
    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)
    main()
