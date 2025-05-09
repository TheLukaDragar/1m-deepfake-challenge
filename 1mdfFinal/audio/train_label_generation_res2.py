"""
Author: LukaDragar
Date: 01.06.2024
"""
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
from numba import jit
import numpy as np
from numba import jit

import argparse

@jit(nopython=True)
def find_segments(binary_array, n=8):
    results = []
    start = None
    segments = []

    # Find all segments of consecutive 1s
    for i, value in enumerate(binary_array):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            segments.append((start, i-1))
            start = None

    # If the last element in the array is 1, close the last segment
    if start is not None:
        segments.append((start, len(binary_array) - 1))

    # Process each segment
    for start, end in segments:
        segment_length = end - start + 1
        if segment_length >= n:
            values = binary_array[start:start + n]
            results.append((values, start, start + n - 1))
        else:
            extra_needed = n - segment_length
            extended_start = max(0, start - extra_needed // 2)
            extended_end = min(len(binary_array) - 1, end + extra_needed // 2 + extra_needed % 2)

            if extended_end - extended_start + 1 < n:
                extended_start = max(0, extended_end - n + 1)

            # Ensure the segment has exactly n elements
            if extended_end - extended_start + 1 == n:
                values = binary_array[extended_start:extended_start + n]
                results.append((values, extended_start, extended_start + n - 1))
            else:
                # Adjust if still not correct
                if extended_start == 0:
                    values = binary_array[:n]
                    results.append((values, 0, n - 1))
                else:
                    extended_end = extended_start + n - 1
                    values = binary_array[extended_start:extended_end + 1]
                    results.append((values, extended_start, extended_end))

    return results


# Function to process a single row
def process_row(v,n):
    npyy = v.replace('.pt', '_labels_res2.npy')
    wf_meatadata = v.replace('.pt', '_wf.npy')
    name = v
    if not os.path.exists(name) or not os.path.exists(npyy):
        print("Not found:", name, npyy)
        return None, None, None

    try:
        features = torch.load(name, map_location='cpu')
        labels = np.load(npyy)
        audio_windows = [l[2] for l in labels]
    except:
        # print("Error loading:", name)
        return None, None, None
        

    features = torch.load(name, map_location='cpu')
    labels = np.load(npyy)
    audio_windows = [l[2] for l in labels]
    wf_meatadata = np.load(wf_meatadata)

    

    #max to 32 windows

    # print("label",audio_windows)
    # print("wf",wf_meatadata)

    r=[]


    if len(audio_windows) <= n:
        return None, None, None


    vectors = features
    # print("len Vectors:", vectors.shape[1])
    # print("Audio windows:", len(audio_windows))
    # num_vectors = vectors.shape[1]
    # label_resolution=0.16/2
    # sample_rate=16000
    # num_audio_frames = wf_meatadata

    # print(num_vectors, num_audio_frames)

    # # how many audio frames per vector
    # conversion_factor = num_audio_frames / num_vectors
    # print("conver",conversion_factor)

 


    vectors_from_segments = []

    # conversion_factor = int(vectors.shape[1] / len(audio_windows))
    # print("Conversion factor:", conversion_factor)

    conversion_factor = 1 # each vector is 0.0200067s in time duration

    if not vectors.shape[1] == len(audio_windows):
        # print("Not equal", vectors.shape[1], len(audio_windows))
        #it can happen that the audio_windows are for one window longer than the vectors
        #in this case we remove the last element 
        if len(audio_windows) == vectors.shape[1] + 1:
            audio_windows = audio_windows[:-1]
        else:
            print("ErrorRRRR")
            raise ValueError("Not equal")
            return None, None, None
    

        
    #real video
    if np.sum(audio_windows) == 0:
        #get the middle 32
        middle = len(audio_windows) // 2

        #get 32 middle 
        r = audio_windows[middle-n//2:middle+n//2]

        r = [r]

        r = np.array(r)

        start = middle - n // 2
        end = middle + n // 2

        #get that label at start and end index of this segment
        start_vector = int(start * conversion_factor)
        end_vector = int(end * conversion_factor)
        #dont and 1 because we are not using find_segments
        # vectors_from_segments.append(vectors[:, start_vector:end_vector])
        vectors_from_segments.append((start_vector,end_vector))

    else:

        r = find_segments(np.array(audio_windows), n=n)

        #check that all segments are 32 if not print r
        for seg in r:
            if not seg[0].shape[0] == n:
                print("Not n")
                print(seg[0])
                print("Segment:", seg)
                print("Audio windows:", audio_windows)

        
        for seg,start,end in r:

            #get that label at start and end index of this segment
            # print("Start:", start, "End:", end)
            # start_label = labels[start]
            # end_label = labels[end]

            # Start label: [0.16 0.24 0.  ]
            # End label: [1.36 1.44 0.  ]

            # print("Start label:", start_label)
            # print("End label:", end_label)

            # print("len Vectors:", vectors.shape[1])
            # print("Audio windows:", len(audio_windows))

            #get how many vectors are in this segment
            
            # print("conver",conversion_factor)

            end = end + 1  #note find_Segments returns inclusive end indexes so when slicing we need to add 1


            #get the vectors that correspond to this segment
            start_vector = int(start * conversion_factor)
            end_vector = int(end * conversion_factor)

            # print("Start vector:", start_vector)
            # print("End vector:", end_vector)
            # this_segment = vectors[:, start_vector:end_vector]

            # # print("This segment:", this_segment.shape)

            # vectors_from_segments.append(this_segment)

            #return only the start and end indexes
            vectors_from_segments.append((start_vector,end_vector))
            

        #remove tuple
        r = [seg[0] for seg in r]

        r = np.array(r)

    # print("R:", r)


        
    return name,r,vectors_from_segments

def process_row_wrapper(args):
        return process_row(*args)

def generate_labels(pdf, n=16):

    # Convert DataFrame rows to list of dictionaries for multiprocessing
    rows = pdf.to_dict(orient='records')
    #get only vidoes
    rows = [row['video'] for row in rows]

   

    # Create a list of arguments for each row to pass to process_row
    args = [(row, n) for row in rows]

    # # Use Pool to parallelize the process_row function and tqdm for progress
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_row_wrapper, args), total=len(rows)))


    return results
# for arg in args:
#     print(process_row_wrapper(arg))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, default="../dataset_helper/train_100.pkl")
    #otput pkl
    parser.add_argument("--output_pkl", type=str, default="./train_audio_modifed_and_real_segments_res2_100.pkl")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Load and shuffle the DataFrame
    pdf = pd.read_pickle(args.pkl_path)



    #use only  modify_type == 'audio_modified'

    pdf2 = pdf[pdf.modify_type == 'audio_modified']
    pdf3 = pdf[pdf.modify_type == 'real']

    #have real be 25 percent of the data
    # real = len(pdf2) // 4

    pdf = pd.concat([pdf2,pdf3])


    pdf['video'] = pdf['video'].apply(lambda x: x.replace('dataset/train/', 'dataset/train_audio/').replace('mp4', 'pt'))


    n=16*4 #for resolution of 0.08 seconds we used 16 windows for resolution of 0.02 seconds we use 64 windows
    #here windows are labels
    results = generate_labels(pdf, n=n) #use 64 windows around fake segments

    # Filter out None results and accumulate the counts
    total_r = []
    for res in results:
        if res[1] is not None:
            for a in res[1]:
                total_r.append(a)


    print("Total samples:", len(total_r))
    print("Total labels:", len(total_r))

    ones = np.sum([np.sum(a) for a in total_r])
    zeros = len(total_r) * n - ones    

    print("Total fake:", ones)
    print("Total real:", zeros)
    print("Ratio:", ones / (ones + zeros))


    #get how many all real samples i have
    full_real = 0
    for res in total_r:
        if np.sum(res) == 0:
            full_real += 1

    print("Full real:", full_real)
    print("Full real ratio:", full_real / len(total_r))

    #full fale
    full_fake = 0
    for res in total_r:
        if np.sum(res) == n:
            full_fake += 1

    print("Full fake:", full_fake)
    print("Full fake ratio:", full_fake / len(total_r))

    #each window is 1.28 seconds and contains 61 vectors when n=16

    # Initialize a list to collect rows
    rows = []

    # Collect rows using list comprehensions
    for res in results:
        if res[1] is not None:
            rows.extend([(res[0], l, s) for l, s in zip(res[1], res[2])])

    # Convert the list of rows to a DataFrame
    df_res = pd.DataFrame(rows, columns=['video', 'labels', 'segments'])
            

            

    x = df_res.sample(1)
    print(x.labels.values[0])
    print(x.segments.values[0])


    #save the results
    df_res.to_pickle(args.output_pkl)



