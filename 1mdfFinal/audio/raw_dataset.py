#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import pickle
import os
import librosa
from torch.utils.data.dataloader import default_collate
from typing import Tuple
import soundfile as sf


torch.set_default_tensor_type(torch.FloatTensor)

SampleType = Tuple[Tensor, int, str, str, str]

def torchaudio_load(filepath):
    wave, sr = sf.read(filepath)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]


class ASVspoof2019Raw(Dataset):
    def __init__(self, access_type, path_to_database, path_to_protocol, part='train'):
        super(ASVspoof2019Raw, self).__init__()
        self.access_type = access_type
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.path_to_protocol = path_to_protocol
        if self.part =='train':
            protocol = os.path.join(os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trn.txt'))
        else:
            protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        # if self.part == "eval":
        #     protocol = os.path.join(self.ptd, access_type, 'ASVspoof2019_' + access_type +
        #                             '_cm_protocols/ASVspoof2019.' + access_type + '.cm.' + self.part + '.trl.txt')
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        # # would not work if change data split but this csv is only for feat_len
        # self.csv = pd.read_csv(self.ptf + "Set_csv.csv")

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename + ".flac")
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, tag, label

    def collate_fn(self, samples):
        return default_collate(samples)


class LAVDF(Dataset):
    def __init__(self, path_to_database, path_to_protocol):
        super(LAVDF, self).__init__()
        self.ptd = path_to_database
        self.path_to_audio = os.path.join(self.ptd, '/home/xieyuankun/data/LAV-DF/audio/test/')
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(os.path.join(self.path_to_protocol))
        self.label = {"spoof": 1, "bonafide": 0}
        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename = self.all_info[idx][0]
        filepath = os.path.join(self.path_to_audio, filename + ".wav")
        print(filepath, 'filepath')
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename

    def collate_fn(self, samples):
        return default_collate(samples)


class DF1M(Dataset):
    def __init__(self, path_to_database, path_to_protocol):
        super(DF1M, self).__init__()
        self.ptd = path_to_database
        self.path_to_audio = os.path.join(self.ptd, '/home/xieyuankun/data/LAV-DF/audio/test/')
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(os.path.join(self.path_to_protocol))
        self.label = {"spoof": 1, "bonafide": 0}
        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename = self.all_info[idx][0]
        filepath = os.path.join(self.path_to_audio, filename + ".wav")
        print(filepath, 'filepath')
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename

    def collate_fn(self, samples):
        return default_collate(samples)

from decord import AVReader, VideoReader, AudioReader
from decord import cpu, gpu
import scipy.io.wavfile
import decord

decord.bridge.set_bridge('torch')

class TestMultiAudioFrameLoader(Dataset):
    """
    Dataset class for selectively collecting frames from multiple videos,
    focusing on frames within specified 'visual_fake_segments'.
    
    Each video's metadata is provided as a DataFrame with necessary details.
    
    Args:
        video_metadata (list of pd.DataFrame): Each DataFrame contains metadata for one video. columns=['video', 'num_video_frames, 'visual_fake_segments', 'label']
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        include_non_fake (bool): If True, include frames not in fake segments.
    """

    def __init__(self,videos,phase="train"):
        self.videos = videos
        # self.transform = transfo
      

    


    def __getitem__(self, index):


        video_path = self.videos[index]

        # label = sample["label"]
        # audio_frames = sample["num_audio_frames"]

      
        # Open mp4 file and extract audio using Decord
        ar = AudioReader(video_path, ctx=cpu(0))
        # Get the audio samples
        audio_samples = ar[:]
        # audio_rate = 16000

        # print(audio_samples.shape, audio_rate)

        # print(visual_fake_segments)

        # audio_labels = []
        # window = 0.16 / 2
        # sr = 16000

       

        # gt_audio_fake_segments = visual_fake_segments
        # # print(len(audio_samples), sr, window * sr, len(audio_samples) - int(window * sr))
        # for i in range(0, audio_samples.shape[1] - int(window * sr), int(window * sr)):
        #     start = i / sr
        #     end = (i + window * sr) / sr

        #     # check if there's any intersection with gt labels
        #     is_fake = any(start < seg[1] and end > seg[0] for seg in gt_audio_fake_segments)

        #     audio_labels.append([start, end, is_fake])


        # audio_labels = np.array(audio_labels)

        
       

        buffer = audio_samples.numpy()

        # print(buffer.shape)
    
        # Convert NumPy array to a PyTorch tensor
        waveform = torch.tensor(buffer, dtype=torch.float32)

        # #write
        # scipy.io.wavfile.write("./testt.wav", audio_rate, buffer.T)

        # #check if the same 
        # waveform2, sr = torchaudio_load("./testt.wav")
        # print(waveform2.shape, waveform.shape)
        # print(torch.allclose(waveform, waveform2))

        return waveform,video_path

    
    def __len__(self):
        return len(self.videos)


class MultiAudioFrameLoader(Dataset):
    """
    Dataset class for selectively collecting frames from multiple videos,
    focusing on frames within specified 'visual_fake_segments'.
    
    Each video's metadata is provided as a DataFrame with necessary details.
    
    Args:
        video_metadata (list of pd.DataFrame): Each DataFrame contains metadata for one video. columns=['video', 'num_video_frames, 'visual_fake_segments', 'label']
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        include_non_fake (bool): If True, include frames not in fake segments.
    """

    def __init__(self,video_metadata,phase="train",resolution=0.08):
        self.video_metadata = video_metadata
        # self.transform = transform
        self.phase = phase
        self.data = video_metadata
        self.resolution = resolution

    


    def __getitem__(self, index):

        sample = self.data.iloc[index]

        video_path =sample["video"]
        visual_fake_segments = sample["fake_audio_segments"]
        # label = sample["label"]
        # audio_frames = sample["num_audio_frames"]

      
        # Open mp4 file and extract audio using Decord
        ar = AudioReader(video_path, ctx=cpu(0))
        # Get the audio samples
        audio_samples = ar[:]
        # audio_rate = 16000

        # print(audio_samples.shape, audio_rate)

        # print(visual_fake_segments)

        audio_labels = []
        # window = 0.16 / 2
        window = self.resolution
        sr = 16000

       

        gt_audio_fake_segments = visual_fake_segments
        # print(len(audio_samples), sr, window * sr, len(audio_samples) - int(window * sr))
        for i in range(0, audio_samples.shape[1] - int(window * sr), int(window * sr)):
            start = i / sr
            end = (i + window * sr) / sr

            # check if there's any intersection with gt labels
            is_fake = any(start < seg[1] and end > seg[0] for seg in gt_audio_fake_segments)

            audio_labels.append([start, end, is_fake])


        audio_labels = np.array(audio_labels)

        
       

        buffer = audio_samples.numpy()

        # print(buffer.shape)
    
        # Convert NumPy array to a PyTorch tensor
        waveform = torch.tensor(buffer, dtype=torch.float32)

        # #write
        # scipy.io.wavfile.write("./testt.wav", audio_rate, buffer.T)

        # #check if the same 
        # waveform2, sr = torchaudio_load("./testt.wav")
        # print(waveform2.shape, waveform.shape)
        # print(torch.allclose(waveform, waveform2))

        return waveform,video_path, audio_labels

    
    def __len__(self):
        return len(self.data)



class Labeler(Dataset):
    """
    Dataset class for selectively collecting frames from multiple videos,
    focusing on frames within specified 'visual_fake_segments'.
    
    Each video's metadata is provided as a DataFrame with necessary details.
    
    Args:
        video_metadata (list of pd.DataFrame): Each DataFrame contains metadata for one video. columns=['video', 'num_video_frames, 'visual_fake_segments', 'label']
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        include_non_fake (bool): If True, include frames not in fake segments.
    """

    def __init__(self,video_metadata,phase="train",resolution=0.08):
        self.video_metadata = video_metadata
        # self.transform = transform
        self.phase = phase
        self.data = video_metadata
        self.resolution = resolution

    


    def __getitem__(self, index):

        sample = self.data.iloc[index]

        video_path =sample["video"]
        visual_fake_segments = sample["fake_audio_segments"]
        # label = sample["label"]
        # audio_frames = sample["num_audio_frames"]

      
        # Open mp4 file and extract audio using Decord
        ar = AudioReader(video_path, ctx=cpu(0))
        # Get the audio samples
        audio_samples = ar[:]
        # audio_rate = 16000

        # print(audio_samples.shape, audio_rate)

        # print(visual_fake_segments)

        audio_labels = []
        # window = 0.16 / 2
        window = self.resolution
        sr = 16000

       

        gt_audio_fake_segments = visual_fake_segments
        # print(len(audio_samples), sr, window * sr, len(audio_samples) - int(window * sr))
        for i in range(0, audio_samples.shape[1] - int(window * sr), int(window * sr)):
            start = i / sr
            end = (i + window * sr) / sr

            # check if there's any intersection with gt labels
            is_fake = any(start < seg[1] and end > seg[0] for seg in gt_audio_fake_segments)

            audio_labels.append([start, end, is_fake])


        audio_labels = np.array(audio_labels)

        
       

        buffer = audio_samples.numpy()

        # print(buffer.shape)
    
        # Convert NumPy array to a PyTorch tensor
        waveform = torch.tensor(buffer, dtype=torch.float32)

        # #write
        # scipy.io.wavfile.write("./testt.wav", audio_rate, buffer.T)

        # #check if the same 
        # waveform2, sr = torchaudio_load("./testt.wav")
        # print(waveform2.shape, waveform.shape)
        # print(torch.allclose(waveform, waveform2))

        return waveform,video_path, audio_labels

    
    def __len__(self):
        return len(self.data)





class ASVspoof2019PSRaw(Dataset):
    def __init__(self, path_to_database, path_to_protocol, part='train'):
        super(ASVspoof2019PSRaw, self).__init__()
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, self.part +'/con_wav/')
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(os.path.join("/ceph/hpc/data/st2207-pgp-users/ldragar/TDL-ADD/label/", 'PS_'+ self.part + '_0.16_real1_pad1.txt'))
        # protocol = os.path.join(os.path.join(self.path_to_protocol, 'PartialSpoof.LA.cm.'+ self.part + '.trl.txt'))

        self.label = {"spoof": 1, "bonafide": 0}
        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename = self.all_info[idx][0]
        filepath = os.path.join(self.path_to_audio, filename + ".wav")
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename

    def collate_fn(self, samples):
        return default_collate(samples)


import os
import cv2
import numpy as np
import pandas as pd
import random
import time

from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import json

from typing import List, Tuple, Optional, Generator

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



class Sample:
    """
    Represents information about a sample video file including its associated metadata.
    
    Attributes:
        path_to_video (str): Path to the video file.
        path_to_metadata (str): Path to the corresponding metadata file.
        path_to_audio (str): Path to the extracted audio file.
        fake_segments (List[Tuple[int, int]]): List of fake segments as (start, end) timestamps.
        fake_audio_segments (List[Tuple[int, int]]): List of fake audio segments as (start, end) timestamps.
        fake_visual_segments (List[Tuple[int, int]]): List of fake visual segments as (start, end) timestamps.
        modify_type (str): Type of modifications (real, visual_modified, audio_modified, both_modified).
        is_real (bool): True if video is real, False otherwise.
        audio_model (str): The audio generation model used.
        operations (List[Dict]): List of operations applied to the video.(Dictionary with keys: 'operation', 'start', 'end' and other optional keys)  operations can be {'delete', 'insert', 'replace'}
        video_frames (int): Number of frames in the video.
        audio_frames (int): Number of frames in the audio.
        transcript (str): Transcript of the video.
        split (str): Dataset split (train, val, test).
        original (Optional[str]): Path to the original video if current is fake, else None.
    """

    # ['file', 'original', 'split', 'modify_type', 'audio_model', 'fake_segments', 'audio_fake_segments', 'visual_fake_segments', 'video_frames', 'audio_frames', 'operations', 'transcripts'])

    def __init__(self, path_to_video: str):
        self.path_to_video = self.__check_path_to_vid(path_to_video)
        self.path_to_metadata = self.__get_metadata_path(path_to_video)
        self.path_to_audio: Optional[str] = None
        self.fake_segments: Optional[List[Tuple[int, int]]] = None
        self.audio_fake_segments: Optional[List[Tuple[int, int]]] = None
        self.visual_fake_segments: Optional[List[Tuple[int, int]]] = None
        self.modify_type: Optional[str] = None
        self.is_real: Optional[bool] = None
        self.audio_model: Optional[str] = None
        self.video_frames: Optional[int] = None
        self.audio_frames: Optional[int] = None
        self.split: Optional[str] = None
        self.original: Optional[str] = None

        self.__process_metadata()

    def __check_path_to_vid(self, path_to_video: str) -> str:
        if not os.path.exists(path_to_video):
            raise FileNotFoundError(f"Video file '{path_to_video}' not found.")
        return path_to_video

    def __get_metadata_path(self, video_path: str) -> str:
        # Generate the path to metadata file from video path
        metadata_directory = os.path.dirname(video_path).replace("/dataset/train", "/dataset/train_metadata"). \
            replace("/dataset/test", "/dataset/test_metadata").replace("/dataset/val", "/dataset/val_metadata")
        # metadata_directory = os.path.dirname(video_path).split("dataset")[0] + "dataset/" + os.path.dirname(video_path).split("dataset")[1].split("/")[0] + "_metadata" + os.path.dirname(video_path).split("dataset")[1].split("/")[1:]

        #find the word after dataset/xxx/
        # split = video_path.split("/dataset/")
        # phrase = split[1].split("/")[0]
        # phrase = phrase + "_metadata"

        # metadata_directory = os.path.join(split[0], "dataset", phrase, *split[1].split("/")[1:-1])
       

        

        video_name = os.path.basename(video_path)
        metadata_name = f"{os.path.splitext(video_name)[0]}.json"
        return os.path.join(metadata_directory, metadata_name)

    def __process_metadata(self):
        """Opens metadata file and sets attributes based on its contents."""
        try:
            with open(self.path_to_metadata, 'r', errors='ignore') as f:
                data = json.load(f)
                # print(data.keys())
                
                self.fake_segments = data.get('fake_segments')
                self.audio_fake_segments = data.get('audio_fake_segments')
                self.visual_fake_segments = data.get('visual_fake_segments')
                
                self.modify_type = data.get('modify_type')

                self.audio_model = data.get('audio_model')

                # self.operations = data.get('operations')
                self.is_real = self.modify_type == "real"

               
                self.video_frames = data.get('video_frames')
                self.audio_frames = data.get('audio_frames')

                ts = data.get('transcripts')
                # self.transcripts = [Transcript(t['word'], t['start'], t['end']) for t in ts] if ts else None


                self.split = data.get('split')
                self.path_to_audio = self.__get_audio_path()
                
                # Additional attributes
                self.original = self.path_to_video if self.is_real else data.get('original')

        except FileNotFoundError:
            print(f"Metadata file '{self.path_to_metadata}' not found.")

    def __get_audio_path(self) -> Optional[str]:
        """Generates path to the audio file based on video path."""
        if not self.is_real:
            video_path = self.path_to_video
            name = "real.wav" if "real_audio" in video_path else "fake.wav"
            parts = video_path.split('train', 1) if 'train' in video_path else video_path.split('val', 1)
            audio_path = os.path.join(parts[0], "audio_only", 'train' if 'train' in video_path  else 'val', *parts[1].split(os.sep)[:-1], name)
            return audio_path
        return None
    

    

    def _get_metadata(self) -> dict:
        """Read and return the metadata from the JSON file."""
        try:
            with open(self.path_to_metadata, 'r', errors='ignore') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Metadata file '{self.path_to_metadata}' not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from '{self.path_to_metadata}'.")
        return {}
    
    @property
    def operations(self) -> List[Operation]:
        """Dynamically load operations from metadata."""
        data = self._get_metadata()
        operations_data = data.get('operations', [])
        return [Operation(**op) for op in operations_data]

    @property
    def transcripts(self) -> List[Transcript]:
        """Dynamically load transcripts from metadata."""
        data = self._get_metadata()
        transcripts_data = data.get('transcripts', [])
        return [Transcript(**t) for t in transcripts_data]
        

    def operations_generator(self) -> Generator[Operation, None, None]:
        """Yield operations one by one."""
        self._load_metadata()
        for op in self._operations_data:
            yield Operation(**op)

    def transcripts_generator(self) -> Generator[Transcript, None, None]:
        """Yield transcripts one by one."""
        self._load_metadata()
        for t in self._transcripts_data:
            yield Transcript(**t)


def analyze_concurently(video_files: List[str],root_directory = "/ceph/hpc/data/st2207-pgp-users/ldragar/1MDF/dataset"):

    def process_video(video):
        """
        Function to process a single video, extracting and returning its operations.
        """
        sample = Sample(os.path.join(root_directory, video))
        return sample


  

    number_of_real = 0

    number_of_visual_modified = 0
    number_of_audio_modified = 0
    number_of_both_modified = 0

    results = []


    # The number of workers in ThreadPoolExecutor can be adjusted based on your system's capabilities
    with ThreadPoolExecutor(max_workers=30) as executor:
        # Prepare future tasks
        future_to_video = {executor.submit(process_video, video): video for video in video_files}
        
        # Process as they complete
        for future in tqdm(as_completed(future_to_video), total=len(video_files)):
            try:
                result = future.result()
                operations = [op.operation for op in result.operations]
                if result.modify_type == "real":
                    number_of_real += 1
                elif result.modify_type == "visual_modified":
                    number_of_visual_modified += 1
                elif result.modify_type == "audio_modified":
                    number_of_audio_modified += 1
                elif result.modify_type == "both_modified":
                    number_of_both_modified += 1




                #TODO treat AUDIO MODIFIED and both modified AS REAL VIDEOS !!!!!!!    to get more training data
                # if result.modify_type == "real" or result.modify_type == "audio_modified" or result.modify_type == "both_modified":
                    #for now only
                results.append({
                    "video": result.path_to_video,
                    "num_audio_frames": result.audio_frames,
                    "num_video_frames": result.video_frames,
                    "modify_type": result.modify_type,
                    "audio_model": result.audio_model,
                    "fake_audio_segments": result.audio_fake_segments,
                    "fake_visual_segments": result.visual_fake_segments,
                    "label": 0 if result.modify_type == "real" else 1

                })
            
            except Exception as exc:
                video = future_to_video[future]
                print(f'Video {video} generated an exception: {exc}')

    print("Number of all real videos:", number_of_real)
    print("Number of all fake videos:", len(video_files) - number_of_real)

    print("Number of visual_modified videos:", number_of_visual_modified)
    print("Number of audio_modified videos:", number_of_audio_modified)
    print("Number of both_modified videos:", number_of_both_modified)

    print("Total videos:", len(video_files))
  
    print("filtered real videos (only real)", len([r for r in results if r["label"] == 0]))
    print("filtered fake videos (only visualy modified)", len([r for r in results if r["label"] == 1]))

    return results

if __name__ == "__main__":


    train_vids = open('/ceph/hpc/data/st2207-pgp-users/ldragar/1MDF/dataset/train_videos.txt', 'r').read().splitlines()
    #get audio fakes
    #keep only 100

    if os.path.exists("./filtered_audio_train_videos_with_both.pkl"):
        df = pd.read_pickle("./filtered_audio_train_videos_with_both.pkl")
    else:

        filtered_audio_train_videos = analyze_concurently(train_vids)

        #to df
        df = pd.DataFrame(filtered_audio_train_videos)
        #save as pickle
        df.to_pickle("./filtered_audio_train_videos_with_both.pkl")

    print(df.head())

    #find longest audio
    print(df["num_audio_frames"].max())
    #find shortest audio
    print(df["num_audio_frames"].min())

    #find shortest audio fake
    

    

    # ds = MultiAudioFrameLoader(df)
    # #get first sample
    # x = next(iter(ds))
    # print(x)



