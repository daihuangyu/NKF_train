import torch
from torch.utils.data import DataLoader, Dataset
import multiprocessing
import numpy as np
import random
import os
import soundfile as sf
import pandas as pd
from sklearn import preprocessing

class MyDataset(Dataset):
    def __init__(self, tp='train', wav_path='/mnt/users/daihuangyu/AEC_Challenge/NKF_Data_From_AEC_Challenge/synthetic_B12/'):
        if tp == 'train':
            start = 0
            end = 8000
        elif tp == 'val':
            start = 8600
            end = 9000
        else:
            start = 9000
            end = 10000
            
        nearend_mic_signals = []
        nearend_speechs = []
        farend_speechs = []
        echo_signals = []
        for i in range(start, end):
            nearend_mic_signals.append(wav_path + 'nearend_mic_signal/' + f'nearend_mic_fileid_{i}.wav')
            nearend_speechs.append(wav_path + 'nearend_speech/' + f'nearend_speech_fileid_{i}.wav')
            farend_speechs.append(wav_path + 'farend_speech/' + f'farend_speech_fileid_{i}.wav')
            echo_signals.append(wav_path + 'echo_signal/' + f'echo_fileid_{i}.wav')

        self.nearend_mic_signals = nearend_mic_signals
        self.nearend_speechs = nearend_speechs
        self.farend_speechs = farend_speechs
        self.echo_signals = echo_signals

    def __getitem__(self, index):
        nearend_mic_signal = self.nearend_mic_signals[index]
        nearend_speech= self.nearend_speechs[index]
        farend_speech = self.farend_speechs[index]
        echo_signal = self.echo_signals[index]
        
        train_wav, _ = sf.read(nearend_mic_signal)
        speech_wav, _ = sf.read(nearend_speech)
        ref_wav, _ = sf.read(farend_speech)
        echo_wav, _ = sf.read(echo_signal)


#         train_wav = train_wav[:16000]
#         speech_wav = speech_wav[wav_start:wav_start+16000]
#         ref_wav = ref_wav[wav_start:wav_start+16000]
#         echo_wav = echo_wav[wav_start:wav_start+16000]


    

        return train_wav, speech_wav+0.001*np.random.randn(16000), ref_wav, echo_wav

    def __len__(self):
        return len(self.nearend_mic_signals)


def Mydata_loader(tp='train', wav_path='/mnt/users/daihuangyu/AEC_Challenge/NKF_Data_From_AEC_Challenge/synthetic_B12/',
                  batch_size=16, drop_last=True, num_workers=8):
    trainset = MyDataset(tp, wav_path)
    loader = DataLoader(trainset,
                        batch_size=batch_size,
                        drop_last=drop_last,
                        num_workers=num_workers,
                        shuffle=True,
                        pin_memory=True)

    return loader


def normlization(x):
    x_range = np.max(x, axis=1) - np.min(x, axis=1)
    x_range = np.tile(x_range, (x.shape[1], 1))
    x_min = np.tile(np.min(x, axis=1), (x.shape[1], 1))
    x = (x - np.transpose(x_min)) / np.transpose(x_range)

    return x



def normlization1d(x):
    x_range = np.max(x) - np.min(x)
    print(x_range)
    x_min = np.min(x)
    x = (x - x_min)/x_range

    return x



