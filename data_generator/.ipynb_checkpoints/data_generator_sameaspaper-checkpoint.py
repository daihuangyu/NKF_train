import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
import sys
import threading
import argparse
import gpuRIR

# import multiprocessing

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(True)

wav_input_path = '/mnt/users/daihuangyu/AEC_Challenge/AEC-Challenge/datasets/synthetic/'
wav_vad_path = '/mnt/users/daihuangyu/AEC_Challenge/NKF_Data_From_AEC_Challenge/synthetic/'
wav_output_path = '/mnt/users/daihuangyu/AEC_Challenge/NKF_Data_From_AEC_Challenge/double_talk_B12_hard/'


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
    
    sig_len = np.random.randint(10000, 16000)
    sig_true = np.zeros(len(sig))
    sig_true[len(sig) - sig_len:] = sig[len(sig) - sig_len:]
    P_signal = np.sum(abs(sig_true) ** 2) / sig_len  # 信号功率
    if P_signal < 0.001:
        return sig_true + echo, sig_true
    else:
        P_echo_actual = np.sum(abs(echo) ** 2) / len(echo)
        # 回声功率
        P_echo_need = P_signal / 10 ** (ser / 10.0)  # 回声需要功率
        sig_true = np.sqrt(P_echo_actual / P_echo_need) * sig_true

        return sig_true + echo, sig_true

def energy_corrector(ref, echo):
    # 回声随机降
    echo = echo/np.random.uniform(low=1, high=2, size=1)
    # 能量矫正器
    P_echo = np.sum(abs(echo) ** 2) / len(echo)
    P_ref = np.sum(abs(ref) ** 2) / len(ref)
    ref = np.sqrt(P_echo / P_ref) * ref / np.random.uniform(low=1, high=2, size=1)
    return ref, echo

def sig_padding(sig, need_len):
    # 第一次补信号，小于一半后面补0
    # 如果长了就cut

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


def data_combine(start, end, need_len, echo_low, echo_high):
    # 合成音频，第一维len，第二维channel

    for i in range(start, end):
        farend_speech, _ = sf.read(wav_input_path + 'farend_speech_vad/' + f'farend_speech_fileid_{i}.wav')
        nearend_speech, _ = sf.read(wav_vad_path + 'nearend_speech/' + f'nearend_speech_fileid_{i}.wav')
        nearend_speech_length = len(nearend_speech)
        if nearend_speech_length > 16000:
            nearend_speech_sample = np.random.randint(0, nearend_speech_length-16000)
            nearend_speech = nearend_speech[nearend_speech_sample:]
        # RIRs, _ = sf.read(f"./feature_extract/fast_rir/Generated_RIRs/RIR-{i}.wav") # 需要进行修改，延时小于150ms
        RIR_sample = np.random.randint(20, 192)
        RIRs = np.random.randn(int(RIR_sample * 16000 / 1000))
        RIRs = RIRs.astype(np.float32)

        #nearend_sample = np.random.randint(16000 * 3, 16000 * 6)
        #nearend_speech = nearend_speech[nearend_sample:nearend_sample + need_len]
        nearend_speech = sig_padding(nearend_speech, need_len)

        RIRs = np.expand_dims(RIRs, 0)
        RIRs = np.expand_dims(RIRs, 0)
        farend_speech = sig_padding(farend_speech, need_len)
        echo_signal = gpuRIR.simulateTrajectory(farend_speech, RIRs)
        echo_signal = echo_signal / np.max(echo_signal)
        # 能量矫正器，解决回声和ref差距过大问题，并随机降低echo_signal能量
        farend_speech, echo_signal = energy_corrector(farend_speech, echo_signal)
        
        echo_signal = echo_signal.squeeze()[:need_len]

        ser = np.random.randint(echo_low, echo_high)
        nearend_mic_signal, nearend_speech = add_echo(nearend_speech, echo_signal, ser)
        nearend_mic_max = np.max(np.abs(nearend_mic_signal))
        if nearend_mic_max > 1:
            nearend_mic_signal = nearend_mic_signal / nearend_mic_max
            echo_signal = echo_signal / nearend_mic_max
            farend_speech = farend_speech / nearend_mic_max
            nearend_speech = nearend_speech / nearend_mic_max

        sf.write(wav_output_path + 'farend_speech/' + f'farend_speech_fileid_{i}.wav', farend_speech, 16000)
        sf.write(wav_output_path + 'nearend_speech/' + f'nearend_speech_fileid_{i}.wav', nearend_speech, 16000)
        sf.write(wav_output_path + 'echo_signal/' + f'echo_fileid_{i}.wav', echo_signal, 16000)
        sf.write(wav_output_path + 'nearend_mic_signal/' + f'nearend_mic_fileid_{i}.wav', nearend_mic_signal, 16000)

        print(f'nearend_mic_fileid_{i}.wav')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data generate')

    parser.add_argument('--index', dest='index', type=int, required=False,
                        default=0, help='index start')
    parser.add_argument('--thread', dest='thread', type=int, required=False,
                        default=8, help='thread start')
    args = parser.parse_args()

    data_number = 10000
    start = int(data_number / 8 * (args.index-1))
    end = int(data_number / 8 * (args.index))
    echo_low = -4
    echo_high = 5

    data_combine(start, end, 16000, echo_low, echo_high)







