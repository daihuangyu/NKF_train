# lst转csv 并进行切割
import sys
import os
import pandas as pd
import librosa 
import numpy as np
import soundfile as sf

wav_length = 16000*4
data_file = '../csv/all_mic_wav.lst'
old_path = '/home/feipeng.pf/train_2020/data/addnoise/add_aec/'
file_path = '/mnt/users/disheng/DNS/data_record/'
save_path = '/mnt/users/daihuangyu/dataset/NKF_train/data/echo/'
df = pd.DataFrame(columns=['file_path'])
with open(data_file, 'r', encoding='UTF-8') as f:
    for line in f.readlines():
        true_path = line.replace(old_path , file_path)
        true_path = true_path.replace('\n','')
        _, file_name = os.path.split(true_path)
        file_name, extension = os.path.splitext(file_name)
        wav, fs = librosa.load(true_path, sr=16000)
        wav_len = np.size(wav)
        #print(wav.shape)
#         import pdb
#         pdb.set_trace()
        for i in range(int(wav_len/wav_length)):
            wav_cut = wav[i*wav_length:(i+1)*wav_length]
            wav_cut_path = save_path + file_name + f'_{i}' + '.wav'
            sf.write(wav_cut_path, wav_cut, 16000)
            print(wav_cut_path)
            df = df.append({'file_path':wav_cut_path}, ignore_index=True)
        
df.to_csv('../csv/echo_sig.csv',index=False,sep=',')