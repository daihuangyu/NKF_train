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

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(True)


def add_noise(sig, noise, snr):
    # snr：生成的语音信噪比
    P_signal = np.sum(abs(sig) ** 2) / len(sig)  # 信号功率
    P_noise_actual = np.sum(abs(noise) ** 2) / len(noise)
    # 噪声功率
    P_noise_need = P_signal / 10 ** (snr / 10.0)  # 噪声需要功率

    return sig + noise * np.sqrt(P_noise_need / P_noise_actual)


def add_echo(sig, echo, ser):
    # ser: 信回比
    print(sig.size)
    P_signal = np.sum(abs(sig) ** 2) / len(sig)  # 信号功率
    P_echo_actual = np.sum(abs(echo) ** 2) / len(echo)
    # 回声功率
    P_echo_need = P_signal / 10 ** (ser / 10.0)  # 回声需要功率
    if(sig.size > echo.size):
        sig = sig[:echo.size]

    return sig + echo * np.sqrt(P_echo_need / P_echo_actual)


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


def data_combine(path, start, end, need_len, echo_low, echo_high, input_sig_csv, input_noise_csv, speech_csv, ref_csv,
                 train_csv):
    # 合成音频，第一维len，第二维channel
    fs = 16000
    df_noise = pd.read_csv(input_noise_csv)
    df_sig = pd.read_csv(input_sig_csv)
    df_ref = pd.DataFrame(columns=['file_path'])
    df_speech = pd.DataFrame(columns=['file_path'])
    df_train = pd.DataFrame(columns=['file_path'])
    for i in range(start, end):
        ref, _ = librosa.load(df_sig.iloc[i]['file_path'], sr=16000)
        #print(ref.shape)
        speech, _ = librosa.load(df_sig.iloc[np.random.randint(0, 21901)]['file_path'], sr=16000)
        rir_index = np.random.randint(0, 10799)
        rir_path = f"./feature_extract/fast_rir/Generated_RIRs/RIR-{rir_index}.wav"
        RIRs, _ = librosa.load(rir_path, sr=16000)
#         import pdb
#         pdb.set_trace()
        speech = sig_padding(speech, need_len)
        #print(speech.shape)
        RIRs = np.expand_dims(RIRs,0)
        RIRs = np.expand_dims(RIRs,0)
        far_echo = gpuRIR.simulateTrajectory(ref, RIRs)
        #print(far_echo.shape)
        far_echo = far_echo.squeeze()[:need_len, ]
        #print(far_echo.shape)
        ser = np.random.randint(echo_low, echo_high)
        sig_out = add_echo(speech, far_echo, ser)
        #noise = generate_noise(df_noise, need_len)
        #snr = np.random.randint(10, 20)
        #output = add_noise(sig_out, noise, snr)
        output = sig_out

        _, wav_name = os.path.split(df_sig.iloc[i]['file_path'])
        speech_wav_file = path + 'speech_data/' + os.path.splitext(wav_name)[0] + '.npy'
        ref_wav_file = path + 'ref/' + os.path.splitext(wav_name)[0] + '.npy'
        train_wav_file = path + 'train_data/' + os.path.splitext(wav_name)[0] + '.npy'
        np.save(ref_wav_file, ref)
        np.save(speech_wav_file, speech)
        np.save(train_wav_file, output)

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
                        default='/mnt/users/daihuangyu/dataset/NKF_train/', help='path')
    parser.add_argument('--start', dest='start', type=int, required=False,
                        default=0, help='index start')
    parser.add_argument('--thread', dest='thread', type=int, required=False,
                        default=1, help='index start')
    parser.add_argument('--block', dest='block', type=int, required=False,
                        default=12, help='block')
    parser.add_argument('--fft_size', dest='fft_size', type=int, required=False,
                        default=512, help='fft point')
    parser.add_argument('--hop_length', dest='hop_length', type=int, required=False,
                        default=256, help='hop_length')
    parser.add_argument('--sig_csv', dest='sig_csv', type=str, required=False,
                        default='./csv/vad1mic.csv', help='sig wav csv')
    parser.add_argument('--noise_csv', dest='noise_csv', type=str, required=False,
                        default='./csv/noise.csv', help='noisy wav csv')
    parser.add_argument('--speech_csv', dest='speech_csv', type=str, required=False,
                        default='./csv/speech.csv', help='speech+noise wav csv')
    parser.add_argument('--ref_csv', dest='ref_csv', type=str, required=False,
                        default='./csv/ref.csv', help='ref wav csv')
    parser.add_argument('--train_csv', dest='train_csv', type=str, required=False,
                        default='./csv/train.csv', help='train wav csv')
    args = parser.parse_args()

    dir_path, file_name = os.path.split(args.speech_csv)
    file_name, extension = os.path.splitext(file_name)
    speech_csv = dir_path + '/' + file_name + '_p' + str(args.start) + extension
    print(speech_csv)
    
    dir_path, file_name = os.path.split(args.ref_csv)
    file_name, extension = os.path.splitext(file_name)
    ref_csv = dir_path + '/' + file_name + '_p' + str(args.start) + extension
    print(ref_csv)

    dir_path, file_name = os.path.split(args.train_csv)
    file_name, extension = os.path.splitext(file_name)
    train_csv = dir_path + '/' + file_name + '_p' + str(args.start) + extension
    print(train_csv)

    df_sig = pd.read_csv(args.sig_csv)
    start = int(len(df_sig) * ((args.start - 1) / args.thread))
    end = int(len(df_sig) * ((args.start) / args.thread))
    fs = 16000
    block = args.block
    fft_size = args.fft_size
    hop_length = args.hop_length
    need_len = fs*2
    echo_low = -10
    echo_high = 20

    data_combine(args.save_path, start, end, need_len, echo_low, echo_high, args.sig_csv, args.noise_csv, speech_csv, ref_csv, train_csv)





