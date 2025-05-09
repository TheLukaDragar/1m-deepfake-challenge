
    
from raw_dataset import MultiAudioFrameLoader, TestMultiAudioFrameLoader
import os
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from train_res2 import MyModel

import argparse


def parse_args():
     #each worker will predfict a subset of videos

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        default="../dataset/test",
        # parser.add_argument('--root_path', default='/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/DFGC-2022/Detection Dataset/10-frames-Private-faces',
        type=str,
        help="path to Evaluating dataset",
    )

    parser.add_argument('--videos_txt', type=str, default='../dataset/test_100.txt')


    parser.add_argument('--output_dir', type=str, default='./audio_predictions/')

    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--worker_id", type=int, default=0)

    # parser.add_argument("--model", type=str, default="/ceph/hpc/data/st2207-pgp-users/models_luka_1mdf_audio_res_2/TDL_res2/er8pbob5/TDL_res2-epoch=07-val_loss=0.03-train_loss=0.03.ckpt")
    parser.add_argument("--model", type=str, default="./models_luka_1mdf_audio_res_2/TDL_res2/er8pbob5/TDL_res2-epoch=31-val_loss=0.02-train_loss=0.02.ckpt")

    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--num_gpus", type=int, default=1)

    return parser.parse_args()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename)["state_dict"])
    return network



def save_results(results, output_pkl):
    """
    Function to save the results to disk.
    """
    #crate df with append mode
    df = pd.DataFrame(results).T
    # print("computing metrics")
    # print("df", df)
    # compute_metrics(df)
    # print("saving df")


    # df.to_csv(output_csv, index=False, mode='a', header=not file_exists)
    # print(f"Data saved to {output_csv} (appended)")

    # output_pickle = 'output.pkl'

    # Check if the file exists
    file_exists = os.path.exists(output_pkl)

    if file_exists:
        # Load the existing data
        with open(output_pkl, 'rb') as f:
            existing_data =pd.read_pickle(f)

        
        # join the existing df with the new df
        new_data = pd.concat([existing_data, df],ignore_index=True)
        new_data.to_pickle(output_pkl)

    else:
        new_data = df
        new_data.to_pickle(output_pkl)

    # Save the updated data

    print(f"Data saved to {output_pkl}")



if __name__ == "__main__":

    args = parse_args()

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


    #make output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


    

    tdl_model = MyModel(
            model_name="TDL",epochs=0
        )

    tdl_model = load_network(tdl_model, args.model)
    tdl_model.train(False)
    tdl_model.eval()

    #to device
    tdl_model = tdl_model.to(device)

    # #only modify_type == "audio_modified"
    # df_tt = df_tt[df_tt["modify_type"] == "audio_modified"]

    # #shuffle df_tt
    # df_tt = df_tt.sample(frac=1, random_state=42).reset_index(drop=True)


    dss = TestMultiAudioFrameLoader(test_videos)
    # target_dir = os.path.join("/home/xieyuankun/data/asv2019PS/preprocess_xls-r-300m", part_,
    #                           "xls-r-300m")
    # # mel = wav2vec2_large_CTC()
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")  # best
    # Wav2Vec2FeatureExtractor =Wav2Vec2FeatureExtractor(feature_size=1024)
    # feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1024).from_pretrained("facebook/wav2vec2-base-960h")
    # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").extrcatfeatures.cuda()
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").cuda()
    model.eval()




    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)

    # output_name = f"pred_{os.path.basename(args.model)}_worker_{worker_id}_{num_workers}.pkl"
    output_name = f"pred_{os.path.basename(args.model)}_worker_{worker_id}_{num_workers}_gpu_{gpu_id}_{num_gpus}.pkl"

    output_pkl = os.path.join(args.output_dir, output_name)

    i= 0

    save_every = 500

    oh_my ={}

    for idx in tqdm(range(len(dss))):
        waveform, filename = dss[idx]

        wf_len = waveform.shape[1]
        # print(waveform.shape, 'waveform')
        # print(audio_labels)
        # print(filename)
        waveform = waveform.to(device)
        # print(waveform.shape, 'waveform')

        waveform = waveform.squeeze(dim=0)
        # waveform = pad_dataset(waveform).to('cpu')
        input_values = processor(waveform, sampling_rate=16000,
                                    return_tensors="pt").input_values.cuda()
        with torch.no_grad():
            wav2vec2 = model(input_values).last_hidden_state.cuda()

            #NOTE A audio WINDOW IS 64 VECTORS LONG AND 0.02 SECONDS LONG
            

            feats = wav2vec2.transpose(1, 2)

            # print(feats.shape,"feats shape")


            # Example tensor of shape [1, 1024, 1641]

            # Calculate the number of full batches
            batch_size = 64
            num_batches = feats.shape[2] // batch_size
            remainder = feats.shape[2] % batch_size

            # print("divided into ", num_batches, "full batches and ", remainder, "remaining vectors")

            # Separate the full batches and the remaining vectors
            full_batches = feats[:, :, :num_batches * batch_size]
            # remaining_vectors = feats[:, :, num_batches * batch_size:]

            # If there are remaining vectors, create a batch of 64 by going backward from the end of the  vectors
            if remainder > 0:

                last_batch = feats[:, :, -batch_size:]
                # print("created last batch of size", last_batch.shape)
            else:
                last_batch = None

            # Reshape the full batches to [num_batches, 1024, 64]
            full_batches_reshaped = full_batches.view(1, 1024, num_batches, batch_size).permute(2, 0, 1, 3).reshape(-1, 1024, 64)

            # Reshape the last batch if it exists
            if last_batch is not None:
                last_batch_reshaped = last_batch.view(1, 1024, 1, batch_size).permute(2, 0, 1, 3).reshape(-1, 1024, 64)
            else:
                last_batch_reshaped = None

            embedding_full, window_prediction_full = tdl_model(full_batches_reshaped)

            #flatten 
            window_prediction_full = window_prediction_full.flatten()

            # print("full batches produced ", window_prediction_full.shape, "predictions")

            # Process the last batch if it exists
            if last_batch_reshaped is not None:
                embedding_last, window_prediction_last = tdl_model(last_batch_reshaped)
                window_prediction_last = window_prediction_last.flatten()

                # print("last batch produced ", window_prediction_last.shape, "predictions")

                # print("together too much ", window_prediction_full.shape + window_prediction_last.shape, "predictions")

                # print("Window prediction last shape:", window_prediction_last.shape)
                # print("Window prediction full shape:", window_prediction_full.shape)
                # print("remainder vectors:", remainder)
                # print("remainder windows:", remainder)

                #NOTE EACH

                remainder_windows = remainder

                
                # Remove the overlapping vectors from the last batch predictions and embeddings
                if remainder > 0 and remainder_windows > 0:


                    # duplocates = window_prediction_last[:remainder_windows]
                    # #check if duplocate predicions mtch in full use 0.5 as threshold
                    # if ([int(p > 0.5) for p in duplocates] == [int(p > 0.5) for p in window_prediction_full[-remainder_windows:]]):
                    #     print("Overlapping vectors match!")
                    # else:
                    #     print("Overlapping vectors don't match!")
                    #     print(window_prediction_last[:remainder_windows])
                    #     print(window_prediction_full[-remainder_windows:])

                    #dont use overlapping vectors so only keep -remainder
                    #before cut
                    # print("full before cut:", window_prediction_full.shape)
                    #be inclusive thatsh why 63-remainder_windows
                    # window_prediction_last = window_prediction_last[:-(63-remainder_windows)]
                    # 
                    # print("window_prediction_last before cut:", window_prediction_last.shape)


                    # #only keep last 64-remainder_windows from window_prediction_last
                    # print("using only last ", 64-remainder_windows, "vectors from last batch")
                    window_prediction_last = window_prediction_last[(64-remainder_windows):]
                    # print("window_prediction_last after cut:", window_prediction_last.shape)

                else:
                    
                    window_prediction_last = window_prediction_last

                # print("Window full before concat:", window_prediction_full.shape)
                # print("Window last before concat:", window_prediction_last.shape)
                # Combine the results extend the array
                window_prediction = torch.cat((window_prediction_full, window_prediction_last), dim=0)
                # print("Window full after concat:", window_prediction_full.shape)

                # embedding = torch.cat((embedding_full, embedding_last), dim=0)
            else:
                window_prediction = window_prediction_full
                #flatten
                window_prediction = window_prediction.flatten()



            # #to cpu
            window_prediction = window_prediction.cpu().detach().numpy()


            

            oh_my[filename] = {"pred_raw": window_prediction,"wf_len": wf_len,"video": filename}
            
        # print(wav2vec2.shape)
        # print(wav2vec2)

        if i % save_every == 0:
            # Save results to disk
            # print("Saving results to disk")
            # print("results", results)
            # save_results(results)
            # results = []
            print("Processed %d files" % i)
            save_results(oh_my, output_pkl)

            #clear
            oh_my = {}

        i+=1

    # Save any remaining results #check if any keys left
    if len(oh_my.keys()) > 0:
        print("Saving results to disk")
        save_results(oh_my, output_pkl)

    print("Done!")

