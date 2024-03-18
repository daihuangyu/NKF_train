import numpy as np
import torch
import librosa
import multiprocessing
import pandas as pd
import os
import sys
import soundfile as sf
import matplotlib.pyplot as plt
import threading
from conv_stft import SgpSTFT
import scipy
import argparse


class STFTExtractor():
    def __init__(self, fs, nfft, hopsize, winlen, device):
        self.fs = fs
        self.nfft = nfft
        self.hopsize = hopsize
        self.winlen = winlen
        self.device = device
        self.stft = SgpSTFT(winlen, hopsize,
                            nfft, win_type='SGP',
                            win_fn="./feature_extract/window.txt",
                            feature_type='complex').to(self.device)

    def stft_tran(self, sig):
        sig = np.expand_dims(sig, 0)
        sig = scipy.signal.lfilter([1, -1], [1, -0.99], sig, axis=1)
        sig = scipy.signal.lfilter([1, -0.9], [1, 0], sig, axis=1)
        sig = torch.from_numpy(sig)
        sig = sig.to(torch.float32)
        sig = sig.to(self.device)
        Px = self.stft(sig)
        Px = Px.cpu().numpy()
        Px_all = np.empty(Px[:, :int(self.nfft/2), :].shape, dtype=np.complex64)
        Px_all.real = Px[:, 0:int(self.nfft/2), :]
        Px_all.imag = Px[:, int(self.nfft/2)+1:self.nfft+1, :]

        return Px_all


def feature_extract(train_csv, ref_csv, speech_csv, feature_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_train_fea = pd.DataFrame(columns=['file_path'])
    df_ref_fea = pd.DataFrame(columns=['file_path'])
    df_speech_fea = pd.DataFrame(columns=['file_path'])

    df_train = pd.read_csv(train_csv)
    df_ref = pd.read_csv(ref_csv)
    df_speech = pd.read_csv(speech_csv)
    for i in range(0, len(df_train)):
        _, dir_name = os.path.split(df_train.iloc[i]['file_path'])
        filename, _ = os.path.splitext(dir_name)
        feature_extractor = STFTExtractor(fs=16000, nfft=512, hopsize=256, winlen=512, device=device)
        train_data, _ = librosa.load(df_train.iloc[i]['file_path'])
        ref_data, _ = librosa.load(df_ref.iloc[i]['file_path'])
        speech_data, _ = librosa.load(df_speech.iloc[i]['file_path'])
 
        train_stft = feature_extractor.stft_tran(train_data)
        ref_stft = feature_extractor.stft_tran(ref_data)
        speech_stft = feature_extractor.stft_tran(speech_data)

        np.save(feature_path+'train/'+filename + '.npy', train_stft)
        np.save(feature_path + 'ref/' + filename + '.npy', ref_stft)
        np.save(feature_path + 'speech/' + filename + '.npy', speech_stft)

        df_train_fea = df_train_fea.append({'file_path': feature_path+'train/'+filename + '.npy'}, ignore_index=True)
        df_ref_fea = df_ref_fea.append({'file_path': feature_path + 'ref/' + filename + '.npy'}, ignore_index=True)
        df_speech_fea = df_speech_fea.append({'file_path': feature_path + 'speech/' + filename + '.npy'}, ignore_index=True)

        print(feature_path+'train/'+filename + '.npy')
    # df = df.sample(frac=1).reset_index(drop=True)
    dir, dir_name = os.path.split(train_csv)
    #dir = os.path.dirname(os.path.normpath(dir))
    filename_train, _ = os.path.splitext(dir_name)
    train_fea_csv = dir + '/fea/' + filename_train + '.csv'

    _, dir_name = os.path.split(ref_csv)
    filename_ref, _ = os.path.splitext(dir_name)
    ref_fea_csv = dir + '/fea/' + filename_ref + '.csv'

    _, dir_name = os.path.split(speech_csv)
    filename_speech, _ = os.path.splitext(dir_name)
    speech_fea_csv = dir + '/fea/' + filename_speech + '.csv'
    df_train_fea.to_csv(train_fea_csv, index=False, sep=',')
    df_ref_fea.to_csv(ref_fea_csv, index=False, sep=',')
    df_speech_fea.to_csv(speech_fea_csv, index=False, sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature_extract')
    parser.add_argument('--index', dest='index', type=int, required=False,
                        default=0, help='index start')
    parser.add_argument('--train_csv', dest='train_csv', type=str, required=False,
                        default='./csv/train.csv', help='train csv')
    parser.add_argument('--ref_csv', dest='ref_csv', type=str, required=False,
                        default='./csv/ref.csv', help='ref csv')
    parser.add_argument('--speech_csv', dest='speech_csv', type=str, required=False,
                        default='./csv/speech.csv', help='speech csv')
    parser.add_argument('--feature_path', dest='feature_path', type=str, required=False,
                        default='', help='feature_path')
    args = parser.parse_args()

    
    dir_path, file_name = os.path.split(args.train_csv)
    file_name, extension = os.path.splitext(file_name)
    train_csv = dir_path + '/train/' + file_name + '_p' + str(args.index) + extension
    print(train_csv)

    dir_path, file_name = os.path.split(args.ref_csv)
    file_name, extension = os.path.splitext(file_name)
    ref_csv = dir_path + '/train/' + file_name + '_p' + str(args.index) + extension
    print(ref_csv)

    dir_path, file_name = os.path.split(args.speech_csv)
    file_name, extension = os.path.splitext(file_name)
    speech_csv = dir_path + '/train/' + file_name + '_p' + str(args.index) + extension
    print(speech_csv)
    
#     if not os.path.exists(dir_path + '/train/' + 'fea'):
#         os.mkdir(dir_path + '/train/' + 'fea')
        

    feature_extract(train_csv, ref_csv, speech_csv, args.feature_path)