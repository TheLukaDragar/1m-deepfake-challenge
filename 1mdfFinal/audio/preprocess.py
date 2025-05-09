"""
Author: LukaDragar
Date: 01.06.2024
"""
import raw_dataset
from raw_dataset import MultiAudioFrameLoader
import os
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np


import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--pkl_path", type=str, default="../dataset_helper/train_100.pkl")
    return parser.parse_args()



if __name__ == "__main__":

  
    #THESE RE ONLY REAL AND AUDIO_MODIFIED  
    #sort by num_audio_frames max to min
    # df = df.sort_values(by='num_audio_frames', ascending=False)


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
    

    args = parse_args()

      
    df = pd.read_pickle(args.pkl_path)

    #print columns
    print(df.columns)

    print(df.head())


    #only use audio modified
    df2 = df[df['modify_type'] == 'audio_modified']

    # #add 1000 "modify_type" == "real" videos
    df3 = df[df['modify_type'] == 'real']

    df = pd.concat([df2, df3])

    #ONLY USE REAL AND AUDIO MODIFIED for training

    # #all real
    # df = df[df['modify_type'] == 'real']

    # print(df['modify_type'].value_counts())

    # #dont process visual modified
    # df = df[df['modify_type'] != 'visual_modified']

    # print("after removing visual modified")
    # #print valecount
  

    #     #     yourtts         169065
    #     # vits            150095
    #     # vits_word        45241
    #     # yourtts_word      8516

    # #both modified
    # df = df[df['modify_type'] == 'both_modified']

    # print(df['modify_type'].value_counts())



    # Distribute the videos across workers
    videos = df['video'].unique()

    videos = distribute_videos(videos, args.num_workers, args.worker_id)
    df = df[df['video'].isin(videos)]

    print(f"Worker {args.worker_id} is processing {len(df)} videos")


    #check for each video check if file exists if it does not exist then process it
    for idx, video in enumerate(videos):
        name = video.replace('/1MDF/dataset/train/', '/1MDF/dataset/train_audio/').replace('mp4', 'pt')
        npyy = name.replace('.pt', '_labels_res2.npy')
        if os.path.exists(name) and os.path.exists(npyy):
            df = df[df['video'] != video]



    print(f"Worker {args.worker_id} is actually processing {len(df)} videos")


    ds = MultiAudioFrameLoader(df,resolution=0.02) #0.08/4
    # target_dir = os.path.join("/home/xieyuankun/data/asv2019PS/preprocess_xls-r-300m", part_,
    #                           "xls-r-300m")
    # # mel = wav2vec2_large_CTC()


    #check 



    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")  # best
    # Wav2Vec2FeatureExtractor =Wav2Vec2FeatureExtractor(feature_size=1024)
    # feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1024).from_pretrained("facebook/wav2vec2-base-960h")
    # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").extrcatfeatures.cuda()
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").cuda()
    model.eval()
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)

    # exit()
    # i= 0
    for idx in tqdm(range(len(ds))):
        waveform, filename,labels = ds[idx]

        #create bins or rray of 0 and 1
        # audio_labelss = np.array([int(label[2]) for label in audio_labels])
        # print("Audio labels:", labels)
        # print(waveform.shape, 'waveform')
        # print(filename)

        name = filename.replace('dataset/train/', 'dataset/train_audio/').replace('mp4', 'pt')


        if not os.path.exists(name):

            #create directory if not exists
            os.makedirs(os.path.dirname(name), exist_ok=True)


            waveform = waveform.to(device)
            # print(waveform.shape, 'waveform')
            waveform = waveform.squeeze(dim=0)
            # waveform = pad_dataset(waveform).to('cpu')
            input_values = processor(waveform, sampling_rate=16000,
                                        return_tensors="pt").input_values.cuda()
            with torch.no_grad():
                wav2vec2 = model(input_values).last_hidden_state.cuda()

            torch.save(wav2vec2, name)
            np.save(name.replace('.pt', '_wf.npy'), waveform.shape[0])

        # print(wav2vec2.shape)
        # print(wav2vec2)
        
        # i += 1
        # if i == 10:
        #     break

          #create directory if not exists
        os.makedirs(os.path.dirname(name), exist_ok=True)


        #save labels in same directory they are numpy arrays
        np.save(name.replace('.pt', '_labels_res2.npy'), labels)

        #save wf length

        
        print("saved:", name)
        # torch.save(wav2vec2, os.path.join(target_dir, "%s.pt" % (filename)))
    print("Done!")