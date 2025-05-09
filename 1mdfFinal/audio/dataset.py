#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import librosa
# from feature_extraction import LFCC
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from decord import VideoReader, cpu
from decord import AudioReader

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import decord
decord.bridge.set_bridge('torch')


class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True,
                 padding='repeat', genuine_only=False):
        super(ASVspoof2019, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8,
                        "A09": 9,
                        "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17,
                        "A18": 18,
                        "A19": 19}
        elif self.access_type == 'PA':
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        else:
            raise ValueError("Access type should be LA or PA!")
        self.label = {"spoof": 1, "bonafide": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        if self.genuine_only:
            assert self.access_type == "LA"
            if self.part in ["train", "dev"]:
                num_bonafide = {"train": 2580, "dev": 2548}
                self.all_files = self.all_files[:num_bonafide[self.part]]
            else:
                res = []
                for item in self.all_files:
                    if "bonafide" in item:
                        res.append(item)
                self.all_files = res
                assert len(self.all_files) == 7355

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 6
        featureTensor = torch.load(filepath)

        if self.feature == "wav2vec2_largeraw":
            # featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        # featureTensor = featureTensor.permute(0, 2, 1)
        filename = "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [padding_Tensor(sample[0], max_len) for sample in samples]
            max_len_label = max([sample[2].shape[1] for sample in samples])
            filename = [sample[1] for sample in samples]
            label = [sample[2] for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(filename), default_collate(label)


class ASVspoof2019PS(Dataset):
    def __init__(self, path_to_features, part='train', feature='W2V2', feat_len=1050, pad_chop=True, padding='repeat'):
        super(ASVspoof2019PS, self).__init__()
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.label = {"spoof": 1, "bonafide": 0}
        self.path = os.path.join(self.ptf, 'xls-r-300m')
        # self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        protocol = os.path.join(os.path.join('/ceph/hpc/data/st2207-pgp-users/ldragar/TDL-ADD/label',
                                             'PS_' + self.part + '_0.16_real1_pad1.txt'))
        with open(protocol, 'r') as f:
            audio_info = [info.strip().split(' ', 2) for info in f.readlines()]

        self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, ori_len, label = self.all_info[idx]
        filepath = os.path.join(self.path, filename + ".pt")
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        featureTensor = torch.load(filepath, map_location='cpu')
        
        if self.feature == "wav2vec2_largeraw":
            # featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        # featureTensor = featureTensor.permute(0, 2, 1)
        label = eval(label)
        label = torch.tensor(label, dtype=torch.float32)
        ori_len = eval(ori_len)
        ori_len = torch.tensor(ori_len, dtype=torch.float32).unsqueeze(dim=0)
        featureTensor = torch.squeeze(featureTensor,dim=0)
        return featureTensor, filename, ori_len, label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [padding_Tensor(sample[0], max_len) for sample in samples]
            filename = [sample[1] for sample in samples]
            max_len_label = max([sample[2].shape[0] for sample in samples])

            label = [pad_tensor(sample[2],max_len_label) for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(filename), default_collate(label)


class DF1M_2(Dataset):

    def __init__(self, df):
        super(DF1M_2, self).__init__()
        self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        video_path = sample["video"]
        segment_labels = sample["labels"]
        segment = sample["segments"]

        # print(video_path)
        # print(segment_labels)
        # print(segment)

        #open video with torch load

        vectors = torch.load(video_path, map_location='cpu')

        #slice the vectors to the segment each segment contains 1.28 seconds of audio
        #or 16 windows of 80ms
        #and includes a bit of real and a bit of fake audio
        start = segment[0]
        end = segment[1]

        waw2v = vectors[:, start:end]

        #  label = torch.tensor(label, dtype=torch.float32)
        # ori_len = eval(ori_len)
        # ori_len = torch.tensor(ori_len, dtype=torch.float32).unsqueeze(dim=0)
        # featureTensor = torch.squeeze(featureTensor,dim=0)

        waw2v = torch.squeeze(waw2v,dim=0)

        segment_labels = torch.tensor(segment_labels, dtype=torch.float32)

        return video_path,waw2v, segment_labels


    







class DF1M(Dataset):

    def __init__(self, video_metadata,feat_model,processor,feat_len=500, pad_chop=True, padding='zero', phase='train'):
        super(DF1M, self).__init__()
      
        self.label = {"spoof": 1, "bonafide": 0}
         # self.transform = transform
        self.phase = phase
        self.data = video_metadata

        # self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")  # best
        self.processor = processor
        # Wav2Vec2FeatureExtractor =Wav2Vec2FeatureExtractor(feature_size=1024)
        # feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1024).from_pretrained("facebook/wav2vec2-base-960h")
        # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").extrcatfeatures.cuda()
        # self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").cpu()
        self.model = feat_model
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feat_len = feat_len
        self.pad_chop = pad_chop
        self.padding = padding
        self.label = {"spoof": 1, "bonafide": 0}

      
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data.iloc[idx]

        video_path =sample["video"]
        audio_fake_segments = sample["fake_audio_segments"]
        label = sample["label"]
        audio_frames = sample["num_audio_frames"]

        #create labels array for all frames
        labels = np.zeros(audio_frames)

        for segment in audio_fake_segments:
            start = segment[0]
            end = segment[1]

            #these are in seconds, convert to frames
            start_frame = int(start * 16000)
            end_frame = int(end * 16000)

            #check out of bounds
            if end_frame > audio_frames:
                end_frame = audio_frames
            


            labels[start_frame:end_frame] = 1


        
        #now we need to have even lenght for all samples lets say max 28 seconds

        #the labels should be collected at every 80ms 

        #create 28/0.08 = 350 labels for 28 seconds

        max_index = min(audio_frames, 16000 * 10)
        labels = labels[:max_index]

        interval_size = 16000 * 0.08  # 80ms interval
        interval_frames = int(interval_size)

        labels_chuncked = np.ones(125)
        
        #how many chunks of 80ms can we get from the audio usfull later when calculating loss so we mask padding
        original_chunked_length =  max_index // interval_frames

        #set everything above original_chunked_length to 0
        labels_chuncked[original_chunked_length:] = 0

        if label == 1:
        #check if more than one of the frames in the interval are labeled as fake
            for i in range(125):
                start_frame = i * interval_frames
                end_frame = (i + 1) * interval_frames

                #check out of bounds
                if end_frame > max_index:
                    end_frame = max_index

            
                interval_labels = labels[start_frame:end_frame]
                if np.sum(interval_labels) > 0:
                    labels_chuncked[i] = 0 #set fake frames to 0 THIS IS BETTER FOR THE MODEL TO LARN
                    #ALSO PADDING IS 0 

        else:
            pass

    

        # Open mp4 file and extract audio using Decord
        ar = AudioReader(video_path, ctx=cpu(0))

        # Get the audio samples
        audio_samples = ar[:]  # Get only the first 28 seconds (16000 samples per second)
        #use max 28 seconds
        length = audio_samples.shape[0]
        if length > 16000 * 10:
            audio_samples = audio_samples[:16000 * 10]
        else:
            pass

        audio_rate = 16000

        # print(audio_samples.shape, audio_rate)



        #padding labels_chuncked is not necesarry 


       

        buffer = audio_samples.numpy()

        #padding the audio samples to 28 seconds
        # if length < 16000 * 28:
        #     buffer = np.pad(buffer, (0, 16000 * 28 - length), 'constant')

        
    
        # Convert NumPy array to a PyTorch tensor
        waveform = torch.tensor(buffer, dtype=torch.float32)

        #get self.model s device
        waveform = waveform.to(self.model.device)
    
        waveform = waveform.squeeze(dim=0)

        #cut to 10s
        waveform = waveform[:16000 * 10]

       

        input_values = self.processor(waveform, sampling_rate=16000,
                                return_tensors="pt").input_values.cuda()


        with torch.no_grad():
            wav2vec2 = self.model(input_values).last_hidden_state.cuda()
        # print(wav2vec2.shape)
        # print(wav2vec2)

        featureTensor = wav2vec2.float()

        this_feat_len = featureTensor.shape[1]

        if self.pad_chop:
            if this_feat_len > self.feat_len:
                #this should not happen since we cut the audio to 28 seconds and that shoud give us max feat len 
                print("this should not happen", this_feat_len)
                # assert False

                #chop
                featureTensor = featureTensor[:, :self.feat_len, :]
                
                # startp = np.random.randint(this_feat_len - self.feat_len)
                # featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                #ussualy the case just pad with zeros
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
        else:
            pass

        labels_chuncked = torch.tensor(labels_chuncked, dtype=torch.float32)

        original_chunked_length = torch.tensor(original_chunked_length, dtype=torch.float32).unsqueeze(dim=0)
        featureTensor = torch.squeeze(featureTensor,dim=0)

        return  featureTensor, video_path, original_chunked_length, labels_chuncked,label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            assert False #not implemented
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [padding_Tensor(sample[0], max_len) for sample in samples]
            filename = [sample[1] for sample in samples]
            max_len_label = max([sample[2].shape[0] for sample in samples])

            label = [pad_tensor(sample[2],max_len_label) for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(filename), default_collate(label)


        


def pad_tensor(a,ref_len):
    tensor = torch.tensor(a)
    padding_length = ref_len - tensor.shape[0]
    padding = torch.zeros(padding_length)
    padded_tensor = torch.cat((tensor, padding))
    return padded_tensor

def padding_Tensor(spec, ref_len):
    _, cur_len, width = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    zero = torch.zeros((1, padd_len, width), dtype=spec.dtype).to(spec.device)
    return torch.cat((spec, zero), 1)


def repeat_padding_Tensor(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul, 1)[:, :ref_len, :]
    return spec


def silence_padding_Tensor(spec, ref_len):
    _, cur_len, width = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((silence_pad_value.repeat(1, padd_len, 1).to(spec.device), spec), 1)


# if __name__ == "__main__":
#     # training_set = ASVspoof2019PS('/home/xieyuankun/data/asv2019PS/preprocess_xls-r-300m', 'train', feature='xls-r-300m'
#     #                               , feat_len=750, pad_chop=False, padding='zero')
#     # trainDataLoader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0,
#     #                              collate_fn=training_set.collate_fn)
#     # feat_mat_batch, filename, label = [d for d in next(iter(trainDataLoader))]

#     # print(feat_mat_batch.shape)
#     # print(filename)
#     # print(label.shape)
#     # print(feat_mat_batch)
#     # feat_mat, filename = training_set[3]
#     # print(feat_mat.shape)


#     #df1m

#     df = pd.read_pickle("/ceph/hpc/data/st2207-pgp-users/ldragar/TDL-ADD/filtered_audio_train_videos.pkl")
#     df = df.sort_values(by='num_audio_frames', ascending=False)

#     #keep only name that contains id00012/_raOc3-IRsw/00110/fake_video_fake_audio.mp4
#     df = df[df['video'].str.contains('_raOc3-IRsw')]



#     ds = DF1M(df,feat_model=Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").cuda(), processor=Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m"))

#     trainDataLoader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0,
#                                     collate_fn=ds.collate_fn)

#     feat_mat_batch, filename, original_chunked_length, labels_chuncked,vid_label = [d for d in next(iter(trainDataLoader))]

#     # print(feat_mat_batch.shape)
#     # print(filename)
#     # print(original_chunked_length)
#     # print(labels_chuncked)

#     #flatten the labels
#     # labels_chuncked = labels_chuncked.flatten()

#     print(labels_chuncked.shape)
#     #print full arrays
#     for i in range(32):
#         print(filename[i])
#         print(vid_label[i])
#         print(labels_chuncked[i])
#         print(labels_chuncked[i][: int(original_chunked_length[i].item())])


    

if __name__ == "__main__":
    # training_set = ASVspoof2019PS('/home/xieyuankun/data/asv2019PS/preprocess_xls-r-300m', 'train', feature='xls-r-300m'
    #                               , feat_len=750, pad_chop=False, padding='zero')
    # trainDataLoader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0,
    #                              collate_fn=training_set.collate_fn)
    # feat_mat_batch, filename, label = [d for d in next(iter(trainDataLoader))]

    # print(feat_mat_batch.shape)
    # print(filename)
    # print(label.shape)
    # print(feat_mat_batch)
    # feat_mat, filename = training_set[3]
    # print(feat_mat.shape)


    #df1m

    #/ceph/hpc/data/st2207-pgp-users/ldragar/TDL-ADD/train_audio_modifed_and_1000_real_segments.pkl
    df = pd.read_pickle("/ceph/hpc/data/st2207-pgp-users/ldragar/TDL-ADD/train_audio_modifed_and_1000_real_segments.pkl")

    print(df.head())

    #dr

    ds = DF1M_2(df)

    trainDataLoader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    waw2v, segment_labels = [d for d in next(iter(trainDataLoader))]
    print(waw2v.shape)
    print(segment_labels.shape)
    print(segment_labels[0])

