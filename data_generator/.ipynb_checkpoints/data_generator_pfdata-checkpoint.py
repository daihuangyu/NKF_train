import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
import sys
import threading
import argparse
#import gpuRIR

# import multiprocessing

# gpuRIR.activateMixedPrecision(False)
# gpuRIR.activateLUT(True)


def add_noise(sig, noise, snr):
    # snr：生成的语音信噪比
    P_signal = np.sum(abs(sig) ** 2) / len(sig)  # 信号功率
    P_noise_actual = np.sum(abs(noise) ** 2) / len(noise)
    # 噪声功率
    P_noise_need = P_signal / 10 ** (snr / 10.0)  # 噪声需要功率

    return sig + noise * np.sqrt(P_noise_need / P_noise_actual)


def add_echo(sig, echo, ser):
    # ser: 信回比
    # 避免一开始就双讲，回声一直有，语音长度随机
    sig_len = np.random.randint(8000, len(sig)-8000)
    sig_true = np.zeros(len(sig))
    sig_true[sig_len:] = sig[sig_len:] 
    P_signal = np.sum(abs(sig_true) ** 2) / sig_len  # 信号功率
    P_echo_actual = np.sum(abs(echo) ** 2) / len(echo)
    # 回声功率
    P_echo_need = P_signal / 10 ** (ser / 10.0)  # 回声需要功率
    sig_true = np.sqrt(P_echo_actual / P_echo_need )*sig_true


    return sig_true + echo, sig_true


def sig_padding(sig, need_len):
    # 第一次补信号，小于一半后面补0
    # 如果长了就cut
    sig = sig[1500:]
    if sig.shape[0] >= need_len:
        sig_true = sig[:need_len]
    else:
        sig_len = sig.shape[0]
        sig_true = np.zeros(need_len)
        sig_true[:sig_len] = sig
        if (need_len - sig_len) > sig_len:
            sig_true[sig_len:] = sig_len[:need_len - sig_len]
        else:
            sig_true[sig_len:sig_len * 2] = sig_len
    return sig_true


def generate_noise(df_n, need_len):
    noise_index = np.random.randint(0, len(df_n))  # 随机选取一个噪声文件中一段
    data, fs = sf.read(df_n.iloc[noise_index]['file_path'])
    noise_len = data.shape[0]
    noise_begin = np.random.randint(0, noise_len - need_len - 1600)
    return data[noise_begin:noise_begin + need_len]


def data_combine(path, start, end, need_len, echo_low, echo_high, input_sig_csv, echo_csv, speech_csv, ref_csv,
                 train_csv):
    # 合成音频，第一维len，第二维channel
    fs = 16000
    df_sig = pd.read_csv(input_sig_csv)
    df_echo = pd.read_csv(echo_csv)
    df_ref = pd.DataFrame(columns=['file_path'])
    df_speech = pd.DataFrame(columns=['file_path'])
    df_train = pd.DataFrame(columns=['file_path'])
    for i in range(0, 1):
        speech1, _ = librosa.load(df_sig.iloc[i*2]['file_path'], sr=16000)
        speech2, _ = librosa.load(df_sig.iloc[i*2+1]['file_path'], sr=16000)

        speech = np.append(speech1, speech2)
        speech = sig_padding(speech, need_len)

        echo, _ = librosa.load(df_echo.iloc[i]['file_path'], sr=16000)

        ser = np.random.randint(echo_low, echo_high)
        sig_out, speech = add_echo(speech, echo, ser)

        _, wav_name = os.path.split(df_echo.iloc[i]['file_path'])
        speech_wav_file = path + 'speech/' + os.path.splitext(wav_name)[0] + '.wav'
        ref_wav_file = path + 'ref/' + os.path.splitext(wav_name)[0] + '.wav'
        train_wav_file = path + 'train_data/' + os.path.splitext(wav_name)[0] + '.wav'

        sf.write(speech_wav_file, speech, 16000)
        sf.write(train_wav_file, sig_out, 16000)


        df_ref = df_ref.append({'file_path': ref_wav_file}, ignore_index=True)
        df_speech = df_speech.append({'file_path': speech_wav_file}, ignore_index=True)
        df_train = df_train.append({'file_path': train_wav_file}, ignore_index=True)

        print(train_wav_file)

    df_ref.to_csv(ref_csv, index=False, sep=',')
    df_speech.to_csv(speech_csv, index=False, sep=',')
    df_train.to_csv(train_csv, index=False, sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data generate')
    parser.add_argument('--save_path', dest='save_path', type=str, required=False,
                        default='/mnt/users/daihuangyu/dataset/NKF_train/data/', help='path')
    parser.add_argument('--start', dest='start', type=int, required=False,
                        default=0, help='index start')
    parser.add_argument('--thread', dest='thread', type=int, required=False,
                        default=1, help='index start')
    parser.add_argument('--sig_csv', dest='sig_csv', type=str, required=False,
                        default='../csv/vad1mic.csv', help=' ')
    parser.add_argument('--echo_csv', dest='echo_csv', type=str, required=False,
                        default='../csv/echo_sig.csv', help=' ')
    parser.add_argument('--ref_csv', dest='ref_csv', type=str, required=False,
                        default='../csv/ref_sig.csv', help=' ')
    parser.add_argument('--speech_csv', dest='speech_csv', type=str, required=False,
                        default='../csv/speech_csv.csv', help='speech csv')
    parser.add_argument('--train_csv', dest='train_csv', type=str, required=False,
                        default='../csv/train.csv', help='train wav csv')
    args = parser.parse_args()

    
    
    dir_path, file_name = os.path.split(args.speech_csv)
    file_name, extension = os.path.splitext(file_name)
    speech_csv = dir_path + '/train/' + file_name + '_p' + str(args.start) + extension
    print(speech_csv)
    
    
    dir_path, file_name = os.path.split(args.ref_csv)
    file_name, extension = os.path.splitext(file_name)
    ref_csv = dir_path + '/train/' + file_name + '_p' + str(args.start) + extension
    print(ref_csv)

    dir_path, file_name = os.path.split(args.train_csv)
    file_name, extension = os.path.splitext(file_name)
    train_csv = dir_path + '/train/' + file_name + '_p' + str(args.start) + extension
    print(train_csv)

    isExists=os.path.exists(dir_path + '/train/')
    # 判断结果
    if not isExists:
        os.makedirs(dir_path + '/train/') 

    len_dfsig = 50000
    start = int(len_dfsig * ((args.start - 1) / args.thread))
    end = int(len_dfsig * ((args.start) / args.thread))
    fs = 16000
    need_len = fs*4
    echo_low = -10
    echo_high = 5

    data_combine(args.save_path, start, end, need_len, echo_low, echo_high, args.sig_csv, args.echo_csv, speech_csv, ref_csv, train_csv)





