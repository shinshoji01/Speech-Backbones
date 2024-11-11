# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np

import torch
import torchaudio as ta

from text import text_to_sequence#, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from params import seed as random_seed

import sys
sys.path.insert(0, 'hifi-gan')
# from meldataset import mel_spectrogram

from matcha.utils.model import normalize

# class TextMelDataset(torch.utils.data.Dataset):
#     def __init__(self, filelist_path, add_blank=True,
#                  n_fft=1024, n_mels=80, sample_rate=22050,
#                  hop_length=256, win_length=1024, f_min=0., f_max=8000):
#         self.filepaths_and_text = parse_filelist(filelist_path)
#         # self.cmudict = cmudict.CMUDict(cmudict_path)
#         self.add_blank = add_blank
#         self.n_fft = n_fft
#         self.n_mels = n_mels
#         self.sample_rate = sample_rate
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.f_min = f_min
#         self.f_max = f_max
#         random.seed(random_seed)
#         random.shuffle(self.filepaths_and_text)

#     def get_pair(self, filepath_and_text):
#         filepath, spk, text = (
#             filepath_and_text[0],
#             int(filepath_and_text[1]),
#             filepath_and_text[2],
#         )
#         filepath, spk, text, _ = filepath_and_text
#         spk = int(spk)-11

#         text = self.get_text(text, add_blank=self.add_blank)
#         mel = self.get_mel(filepath)
#         duration = self.get_duration(filepath)
#         return (text, mel, spk, duration)

#     def get_mel(self, filepath):
#         path = self.preprocessed_dir + f"mel/{filepath[:4]}-mel-" + filepath + ".npy"
#         mel = torch.Tensor(np.load(path)).T
#         mel = normalize(mel, np.array(-5.7727466), np.array(2.1028705))
#         return mel
    
#     def get_duration(self, filepath):
#         path = self.preprocessed_dir + f"duration/{filepath[:4]}-duration-" + filepath + ".npy"
#         duration = torch.Tensor(np.load(path))
#         return duration

#     def get_text(self, text, add_blank=True):
#         text_norm = text_to_sequence(text, ["english_cleaners"])
#         if self.add_blank:
#             text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
#         text_norm = torch.IntTensor(text_norm)
#         return text_norm

#     def __getitem__(self, index):
#         text, mel, duration = self.get_pair(self.filepaths_and_text[index])
#         item = {'y': mel, 'x': text, "duration": duration}
#         return item

#     def __len__(self):
#         return len(self.filepaths_and_text)

#     def sample_test_batch(self, size):
#         idx = np.random.choice(range(len(self)), size=size, replace=False)
#         test_batch = []
#         for index in idx:
#             test_batch.append(self.__getitem__(index))
#         return test_batch


class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_ = item['y'], item['x']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths}


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, preprocessed_dir, ed_name, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000, dataset="esd", vocoder="hifigan", ed_name_list=None, ifmsemotts=False):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.preprocessed_dir = preprocessed_dir
        self.ed_name = ed_name
        # self.cmudict = cmudict.CMUDict(cmudict_path)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank
        random.seed(random_seed)
        random.shuffle(self.filelist)
        
        self.dataset = dataset
        self.vocoder = vocoder
        self.ed_name_list = ed_name_list
        self.emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
        self.emotions.sort()
        self.ifmsemotts = ifmsemotts
        
    def get_pair(self, filepath_and_text):
        filepath, spk, text = (
            filepath_and_text[0],
            filepath_and_text[1],
            filepath_and_text[2],
        )
        filepath, spk, text, _ = filepath_and_text
        if self.dataset=="esd":
            spk = torch.LongTensor([int(spk)-11])
        elif self.dataset=="msp":
            spk = torch.LongTensor([int(spk)])

        emo = torch.LongTensor([self.emotions.index(filepath.split("_")[2])])
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath, spk)
        duration = self.get_duration(filepath, spk)
        ed = self.get_ed(filepath)
        se = self.get_speaker_embedding(filepath)
        return (filepath, text, mel, spk, duration, ed, se, emo)

    def get_mel(self, filepath, spk):
        if self.vocoder=="hifigan":
            dirname = "mel"
            mean = -5.7727466
            std = 2.1028705
        elif self.vocoder=="vocos":
            dirname = "melvocos"
            mean = -1.7079772
            std = 1.9407457
        if self.dataset=="esd":
            path = self.preprocessed_dir + f"{dirname}/{filepath[:4]}-mel-" + filepath + ".npy"
        elif self.dataset=="msp":
            path = self.preprocessed_dir + f"{dirname}/{spk[0]}-mel-" + filepath + ".npy"
            
        mel = torch.Tensor(np.load(path)).T
        mel = normalize(mel, mean, std)
        return mel
    
    def get_duration(self, filepath, spk):
        if self.dataset=="esd":
            path = self.preprocessed_dir + f"duration/{filepath[:4]}-duration-" + filepath + ".npy"
        elif self.dataset=="msp":
            path = self.preprocessed_dir + f"duration/{spk[0]}-duration-" + filepath + ".npy"
        duration = torch.Tensor(np.load(path))
        return duration
    
    def get_ed(self, filepath):
        if "relative-attributes" in self.ed_name:
            en = self.ed_name.split("_")[1]
        else:
            en = self.ed_name.split("_")[0]
        if "-" in en:
            ed_list = []
            for en, ed_name in enumerate(self.ed_name_list):
                if self.dataset=="esd":
                    path = self.preprocessed_dir + f"dnnEDdir/{ed_name}/{filepath}_HED_{ed_name}.npy"
                elif self.dataset=="msp":
                    path = self.preprocessed_dir + f"ED/{filepath}_ED_OpenSMILE_msp.npy"
                ed = np.load(path)
                ed_list += [ed[en*4:(en+1)*4]]
            ed = np.concatenate(ed_list, axis=0)
        else:
            if self.dataset=="esd":
                if self.ifmsemotts:
                    path = self.preprocessed_dir + f"dnnMSdir/{self.ed_name}/{filepath}_HED_{self.ed_name}.npy"
                else:
                    path = self.preprocessed_dir + f"dnnEDdir/{self.ed_name}/{filepath}_HED_{self.ed_name}.npy"
            elif self.dataset=="msp":
                path = self.preprocessed_dir + f"ED/{filepath}_ED_OpenSMILE_msp.npy"
            ed = np.load(path)
        ed = torch.Tensor(ed)
        return ed.T
    
    def get_speaker_embedding(self, filepath):
        if self.dataset=="esd":
            path = self.preprocessed_dir + f"speaker_embedding/{'_'.join(filepath.split('_')[:-2])}_resemblyzer.npy"
        if self.dataset=="msp":
            path = self.preprocessed_dir + f"speaker_embedding/{filepath}_resemblyzer.npy"
            pass
        se = torch.Tensor(np.load(path))
        return se

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, ["english_cleaners"])
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        filepath, text, mel, spk, duration, ed, se, emo = self.get_pair(self.filelist[index])
        item = {'y': mel, 'x': text, "spk": spk, "duration": duration, "ed": ed, "se": se, "filename": filepath, "emo": emo}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        duration = torch.zeros((B, x_max_length), dtype=torch.long)
        ed = torch.zeros((B, x_max_length, 12), dtype=torch.float32)
        se = []
        y_lengths, x_lengths = [], []
        spk = []
        emo = []

        for i, item in enumerate(batch):
            y_, x_, spk_, duration_, ed_, se_, emo_ = item['y'], item['x'], item['spk'], item['duration'], item["ed"], item["se"], item["emo"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            duration[i, :duration_.shape[-1]] = duration_
            ed[i, :ed_.shape[-2], :] = ed_
            se += [se_]
            spk.append(spk_)
            emo.append(emo_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)
        emo = torch.cat(emo, dim=0)
        se = torch.cat(se, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk, 'duration': duration, "ed": ed, "se": se, "emo":emo}
