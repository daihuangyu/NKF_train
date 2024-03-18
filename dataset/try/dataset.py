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
    def __init__(self, train_csv, speech_csv, ref_csv):
        df_train = pd.read_csv(train_csv)
        df_speech = pd.read_csv(speech_csv)
        df_ref = pd.read_csv(ref_csv)
        train_wavs = []
        speech_wavs = []
        ref_wavs = []
        for i in range(0, len(df_train)):
            train_wavs.append(df_train.iloc[i]['file_path'])
            speech_wavs.append(df_speech.iloc[i]['file_path'])
            ref_wavs.append(df_ref.iloc[i]['file_path'])

        self.train_wavs = train_wavs
        self.speech_wavs = speech_wavs
        self.ref_wavs = ref_wavs

    def __getitem__(self, index):
        train = self.train_wavs[index]
        speech = self.speech_wavs[index]
        ref = self.ref_wavs[index]
        train_wav = np.load(train).squeeze()
        speech_wav = np.load(speech).squeeze()
        ref_wav = np.load(ref).squeeze()
        
        train_wav, speech_wav, ref_wav = normlization2(train_wav, speech_wav, ref_wav)
        # rng = np.random.default_rng()
        # inputs = rng.permutation(inputs)
        # for i in range(0, inputs.shape[0]):
        #    inputs[i,:,:] = preprocessing.scale(inputs[i,:,:])
    

        return train_wav, speech_wav, ref_wav

    def __len__(self):
        return len(self.train_wavs)


def Mydata_loader(train_csv, speech_csv, ref_csv, batch_size=16, drop_last=False, num_workers=8):
    trainset = MyDataset(train_csv, speech_csv, ref_csv)
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



def normlization2(train_wav, speech_wav, ref_wav):
    max_value = np.max(np.max(np.abs(train_wav)))
    train_wav = train_wav/max_value
    speech_wav = speech_wav/max_value
    ref_wav = ref_wav/max_value

    return train_wav, speech_wav, ref_wav



