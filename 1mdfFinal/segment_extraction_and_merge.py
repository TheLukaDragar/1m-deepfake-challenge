# %%
import pandas as pd
import numpy as np
import os
import decord
import plotly.graph_objects as go
from IPython.display import Video
from decord import VideoReader, cpu
from decord import AudioReader
from IPython.display import Audio, display

# %%
def load_from_worker(path_to_predictions):
    pkls = os.listdir(path_to_predictions)
    #keep only pkl files
    pkls = [pkl for pkl in pkls if pkl.endswith('.pkl')]

    tmp = []


    for pkl in pkls:
        dd = pd.read_pickle(f'{os.path.join(path_to_predictions, pkl)}')
        tmp.append(dd)
        print(f"Loaded {len(dd)} samples from {pkl}")

    tmp = pd.concat(tmp)

    #add another clum that is basename of video
    tmp['video_short'] = tmp['video'].apply(lambda x: os.path.basename(x))

    #sort by video
    tmp = tmp.sort_values(by='video_short')



# %%
#load video predictions 
# audio_pred_path = "audio/audio_predictions"
# audio_df = load_from_worker(audio_pred_path)

# video_pred_path = "visual/visual_predictions"
# video_df = load_from_worker(video_pred_path)


#loading our submited frame_predictions

video_pred_path = "./all_preds_finl/video_df_last_epoch.pkl"
video_old_path = "./all_preds_finl/video_df_epoch39.pkl"
audio_pred_path = "./all_preds_finl/audio_df_epoch36.pkl"
transcripts_path = "./all_preds_finl/transcripts.pkl"




video_df = pd.read_pickle(video_pred_path)
video_df_old = pd.read_pickle(video_old_path)
audio_df = pd.read_pickle(audio_pred_path)
transcripts = pd.read_pickle(transcripts_path)






df = video_df


#rename columns drop video column
video_df_old = video_df_old.rename(columns={'frames_prediction':'frames_prediction_old'})

video_df_old = video_df_old.drop(columns=['video'])

video_df = video_df.merge(video_df_old, on='video_short')


video_df = video_df.drop(columns=['video'])

df = pd.merge(audio_df, video_df, on='video_short')


df = df.merge(transcripts, on='video', how='left')

# %%
#fix the video path
df['video'] = df['video_short'].apply(lambda x: f"./dataset/test/{x}")


# %%
df

# %%
df['avg_frames_prediction'] = df.apply(lambda row: (np.array(row['frames_prediction']) + np.array(row['frames_prediction_old'])) / 2 if len(row['frames_prediction']) == len(row['frames_prediction_old']) else row['frames_prediction'], axis=1)


# %%
def compute_segments_threshold_method_old(preds, threshold=0.98, high_confidence_threshold=0.6,close_distance=2,remove_min=2):


    mask = preds > threshold
    #create segments
    segments = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            segments.append((start, i, np.mean(preds[start:i])))
            start = None

    

    #if start is not none add the last segment
    if start is not None:
        segments.append((start, i, np.mean(preds[start:i])))

    #remove segments with less than 2 frames
    segments = [s for s in segments if s[1] - s[0] > remove_min]

    
    #group segments together if they are close
    new_segments = []
    start = None
    for s in segments:
        if start is None:
            start = s
        else:
            if s[0] - start[1] < close_distance:
                start = (start[0], s[1])
            else:
                new_segments.append(start)
                start = s

    if start is not None:
        new_segments.append(start)

    #compute confidence
    segments = []
    for s in new_segments:
        confidence = np.mean(preds[s[0]:s[1]])
        #if nan set to 0
        if np.isnan(confidence):
            confidence = 0
        segments.append((s[0], s[1], confidence))

    #substract 1/fps from the end and make sure it is not negative
    segments = [(s[0], max(0, s[1]), s[2]) for s in segments]
    


    # # Filter segments with low confidence
    # segments = [seg for seg in segments if seg[2] > high_confidence_threshold]

    # # #return in secods   
    # # segments = [(s[0] / FPS, s[1] / FPS, s[2]) for s in segments]
    # #if still no segments use rptures library
    # if len(segments) == 0:
        
    #     breakpoints = get_sample_segments_ruptures(preds.reshape(-1,1),pen=2)

        
    #     #create segments
    #     for i in range(len(breakpoints) - 1):
           
    #             start = breakpoints[i]
    #             end = breakpoints[i + 1]

    #             segments.append((start, end, np.mean(preds[start:end])))


    #     #filter segments
    #     segments = [seg for seg in segments if seg[1] - seg[0] > remove_min]

    #     #compute confidence

    

    return segments

def compute_segments_threshold_method_old_AUDIO(sample, threshold=0.98, high_confidence_threshold=0.6,close_distance=2,remove_min=2,remove_first_n=1,conf_first_seg=0.75,one_seg_conf=0.5,close_near=13):


    mask = sample.pred_raw > threshold

    mask=np.array([int(m) for m in mask])

    mask[:remove_first_n] = 0
        


   

    #create segments
    segments = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            segments.append((start, i, np.mean(sample.pred_raw[start:i])))
            start = None

    

    #if start is not none add the last segment
    if start is not None:
        segments.append((start, i, np.mean(sample.pred_raw[start:i])))

    #remove segments with less than 2 frames
    segments = [s for s in segments if s[1] - s[0] > remove_min]

    
    #group segments together if they are close
    new_segments = []
    start = None
    for s in segments:
        if start is None:
            start = s
        else:
            if s[0] - start[1] < close_distance:
                start = (start[0], s[1])
            else:
                new_segments.append(start)
                start = s

    if start is not None:
        new_segments.append(start)

    #compute confidence
    segments = []
    for s in new_segments:
        confidence = np.mean(sample.pred_raw[s[0]:s[1]])
        segments.append((s[0], s[1], confidence))

    #substract 1/fps from the end and make sure it is not negative
    # segments = [(s[0], max(0, s[1] - 1), s[2]) for s in segments]
    


    # Filter segments with low confidence
    segments = [seg for seg in segments if seg[2] > high_confidence_threshold]

    # #if the segments starts in first 0.5 seconds aka indexs to 25 make sure the confidence is high and segment si 0.2 seconcd
    # if len(segments) > 0:
    #     if segments[0][0] < 25:
    #         if segments[0][2] < conf_first_seg:
    #             segments = segments[1:]


    # #if its the only segment it has to have high confidence
    # if len(segments) == 1:
    #     if segments[0][2] < one_seg_conf:
    #         segments = []

    # #do merging again if segmetns are close to each other 13
    # if len(segments) > 1:
    #     new_segments = []
    #     start = None
    #     for s in segments:
    #         if start is None:
    #             start = s
    #         else:
    #             if s[0] - start[1] < close_near:
    #                 start = (start[0], s[1],max(start[2],s[2]))
    #             else:
    #                 new_segments.append(start)
    #                 start = s

    #     if start is not None:
    #         new_segments.append(start)

    #     segments = new_segments


    # #generate intermidiate segments and set confidence to 0.5 start at the start
    # new_segments = []

    # #check if we have no segments then 
    # if len(segments) == 0:
    #     return [(0,len(sample.pred_raw),0.1)]

    # for i,s in enumerate(segments):
    #     if i == 0:
    #         if s[0] > 0:
    #             new_segments.append((0,s[0],0.1))
    #     else:
    #         new_segments.append((segments[i-1][1],s[0],0.1))

    #     new_segments.append(s)

    # #add the last segment
    # if len(segments) > 0:
    #     new_segments.append((segments[-1][1],len(sample.pred_raw),0.1))

    # segments = new_segments

    return segments
    


# %%

print("computing segments")


# OG
threshold_video = 0.5
high_confidence_threshold_video = 0.4
close_distance_video = 3
remove_min_video = 1

threshold_audio = 0.1
high_confidence_threshold_audio = 0.01
close_distance_audio = 2
remove_min_audio = 2
remove_first_n_audio = 1
close_near_audio = 0


# threshold_video = 0.3
# high_confidence_threshold_video = 0.3
# close_distance_video = 3
# remove_min_video = 1

# threshold_audio = 0.1
# high_confidence_threshold_audio = 0.01
# close_distance_audio = 2
# remove_min_audio = 2
# remove_first_n_audio = 1
# close_near_audio = 0



df["threshold_segments_video"] = df.apply(lambda x: compute_segments_threshold_method_old(x["avg_frames_prediction"], threshold=threshold_video, high_confidence_threshold=high_confidence_threshold_video,close_distance=close_distance_video,remove_min=remove_min_video), axis=1)
#multiply by 0.04 to get seconds
df["threshold_segments_video"] = df.apply(lambda x: [[s[0] *0.04, s[1] * 0.04, s[2]] for s in x["threshold_segments_video"]], axis=1)

df["threshold_segments_audio"] = df.apply(lambda x: compute_segments_threshold_method_old_AUDIO(x, threshold=threshold_audio, high_confidence_threshold=high_confidence_threshold_audio,close_distance=close_distance_audio,remove_min=remove_min_audio,remove_first_n=remove_first_n_audio,close_near=close_near_audio), axis=1)
#multiply by 0.02 to get seconds
df["threshold_segments_audio"] = df.apply(lambda x: [[s[0] *0.020006814310051108, s[1] * 0.020006814310051108, s[2]] for s in x["threshold_segments_audio"]], axis=1)

# df["threshold_segments_audio"] = df.apply(lambda x: adjust_len_to_multiple_of_zero2(x["threshold_segments_audio"]), axis=1)
#round to nearest multiple of 0.02
df["threshold_segments_audio"] = df.apply(lambda x: [[round(s[0] / 0.02) * 0.02, round(s[1] / 0.02) * 0.02, s[2]] for s in x["threshold_segments_audio"]], axis=1)
# df["fake_segments"] = df.apply(lambda x: x["fake_audio_segments"] + x["fake_visual_segments"], axis=1)
# Create the JSON files
print("finihed computing segmetns")
#video
# df["threshold_segments_video"] = df.apply(lambda x: compute_segments_threshold_method_old(x["frames_prediction"],  threshold=0.9, high_confidence_threshold=0.65,close_distance=4,remove_min=2), axis=1)
def merge_segments(segments):
    if not segments:
        return []

    # Sort segments by start time
    segments.sort(key=lambda x: x[1])

    merged = [segments[0]]
    for current in segments[1:]:
        last = merged[-1]
        if current[1] <= last[2]:  # Overlapping or adjacent
            # Merge segments, using the max confidence of the overlapping segments
            merged[-1] = [max(last[0], current[0]), last[1], max(last[2], current[2])]
        else:
            merged.append(current)
    return merged

def merge_segments2(video_segments, audio_segments):
    if len(video_segments) == 0:
        return audio_segments
    
    if len(audio_segments) == 0:
        return video_segments
    
    #if we have a combination of both then use the locliztion of the audio when merging video segments

    #for each video segment find the audio segment that is closest to it
    #if the audio segment is within the video segment then merge them based on the audio segment

    #if the audio segment is not within the video segment then merge the video segment based on the video segment


    vs =[]

    for video_segment in video_segments:
        for audio_segment in audio_segments:
            #check if any overlap
            if video_segment[0] <= audio_segment[0] <= video_segment[1] or video_segment[0] <= audio_segment[1] <= video_segment[0]:
                #merge based on audio segment loclization
                vs.append([audio_segment[0], audio_segment[1], audio_segment[2]])
                break

        else:
            #no overlap
            vs.append(video_segment)

    #add the audio segments 
    for audio_segment in audio_segments:
        vs.append(audio_segment)

    return vs


def rnk_combine(vis,aud):

    #if vis and aud have intersection lower the confidence of visual by half
    if len(aud)>0:
        lowered_conf_video = [[s[0], s[1], s[2] / 10] for s in vis]
        return aud + lowered_conf_video
    else:
        return vis
    



def combine(x):
    # return x["threshold_segments_audio"] + x["threshold_segments_video"]
    # return merge_segments(x["threshold_segments_audio"] + x["threshold_segments_video"])
    # return merge_segments2(x["threshold_segments_video"], x["threshold_segments_audio"])
    return rnk_combine(x["threshold_segments_video"], x["threshold_segments_audio"])



df["threshold_segments_combined"] = df.apply(lambda x: combine(x), axis=1)



# %%
submit_merged = df[["video_short", "threshold_segments_combined"]].copy()

#reorder threshold_segments_combined so confidence is first
submit_merged["threshold_segments_combined"] = submit_merged["threshold_segments_combined"].apply(lambda x: [[s[2], s[0], s[1]] for s in x])

#set len 0 to [[0,0,0]]
submit_merged["threshold_segments_combined"] = submit_merged["threshold_segments_combined"].apply(lambda x: [[0.0,0.0,0.0]] if len(x) == 0 else x)

#sort by video
submit_merged = submit_merged.sort_values(by="video_short")



# %%
#to json file
finale= {}
for i, row in submit_merged.iterrows():
    finale[row["video_short"]] = row["threshold_segments_combined"]

import json

def convert_data_types(d):
    if isinstance(d, dict):
        return {k: convert_data_types(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_data_types(i) for i in d]
    elif isinstance(d, np.float32):
        return float(d)
    elif isinstance(d, np.int32):
        return int(d)
    else:
        return d

finale = convert_data_types(finale)
with open('submissionmerged_roundedtozero2_avg_frmes_predictions_avgpred_raw.json', 'w') as f:
    json.dump(finale, f)

print("done",os.path.abspath('submissionmerged_roundedtozero2_avg_frmes_predictions_avgpred_raw.json'))

# %%
#load a json file and check if all match
submited = json.load(open('submited_prediction.json'))
submited = convert_data_types(submited)

for k,v in submited.items():
    submited_arr = np.array(v)
    finale_arr = np.array(finale[k])

    for seg in submited_arr:
        if seg not in finale_arr:
            print("not match")
            print(k)
            break

    for seg in finale_arr:
        if seg not in submited_arr:
            print("not match")
            print(k)
            break

print("all match")


