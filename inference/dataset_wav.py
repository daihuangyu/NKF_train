import torch
from torch.utils.data import DataLoader, Dataset
import multiprocessing
import numpy as np
import random
import os
import soundfile as sf
import pandas as pd
from sklearn import preprocessing

wav_path = '/mnt/users/daihuangyu/AEC_Challenge/AEC-Challenge/datasets/synthetic/'
class MyDataset(Dataset):
    def __init__(self, tp='train'):
        if tp == 'train':
            start = 0
            end = 8000
        elif tp == 'val':
            start = 8000
            end = 9000
        else:
            start = 9000
            end = 10000
            
        nearend_mic_signals = []
        nearend_speechs = []
        farend_speechs = []
        #echo_signals = []
        for i in range(start, end):
            nearend_mic_signals.append(wav_path + 'nearend_mic_signal/' + f'nearend_mic_fileid_{i}.wav')
            nearend_speechs.append(wav_path + 'nearend_speech/' + f'nearend_speech_fileid_{i}.wav')
            farend_speechs.append(wav_path + 'farend_speech/' + f'farend_speech_fileid_{i}.wav')
            #echo_signals.append(wav_path + 'echo_signal/' + f'echo_fileid_{i}.wav')

        self.nearend_mic_signals = nearend_mic_signals
        self.nearend_speechs = nearend_speechs
        self.farend_speechs = farend_speechs
        #self.echo_signals = echo_signals

    def __getitem__(self, index):
        nearend_mic_signal = self.nearend_mic_signals[index]
        nearend_speech= self.nearend_speechs[index]
        farend_speech = self.farend_speechs[index]
        #echo_signal = self.echo_signals[index]
        
        train_wav, _ = sf.read(nearend_mic_signal)
        speech_wav, _ = sf.read(nearend_speech)
        ref_wav, _ = sf.read(farend_speech)
        train_wav  = train_wav[:159900]
        speech_wav = speech_wav[:159900]
        ref_wav = ref_wav[:159900]
        #echo_wav, _ = sf.read(echo_signal)


    

        return train_wav, speech_wav, ref_wav

    def __len__(self):
        return len(self.nearend_mic_signals)


def Mydata_loader(tp='train', batch_size=16, drop_last=True, num_workers=4):
    trainset = MyDataset(tp)
    loader = DataLoader(trainset,
                        batch_size=batch_size,
                        drop_last=drop_last,
                        num_workers=num_workers,
                        shuffle=False,
                        pin_memory=True)

    return loader





