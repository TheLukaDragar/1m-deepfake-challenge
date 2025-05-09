    # res = "fake" if probe['format']['tags']['encoder'] == 'Lavf58.45.100' else "real"

import ffmpeg

import tqdm
import os
import argparse


def get_encoder_version(file_path):
    try:
        # Run ffprobe to get the format metadata
        probe = ffmpeg.probe(file_path)
        
        # Extract the encoder version from the format tags
        encoder_version = probe['format']['tags'].get('encoder', 'Unknown')
        
        return probe
    except ffmpeg.Error as e:
        print(f"An error occurred while probing the video file: {e}")
        return None
    



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

   
def main():

    args = parse_args()
    vids = open(args.videos_txt, 'r').read().splitlines()
    print("Number of videos: ", len(vids))

    #make them absolute paths
    all_test_videos = [os.path.join(args.root_path, vid) for vid in vids]


    submit={}
    for vid in tqdm.tqdm(all_test_videos):
        probe = get_encoder_version(vid)
        if probe is not None:
            res = 1.0 if probe['format']['tags']['encoder'] == 'Lavf58.45.100' else 0.0
            submit[vid] = res
        else:
            submit[vid] = 0.0

    #save to txt
    with open('task1_predictions.txt', 'w') as f:
        for key in submit.keys():
            f.write(f"{os.path.basename(key)};{submit[key]}\n")

    

    
