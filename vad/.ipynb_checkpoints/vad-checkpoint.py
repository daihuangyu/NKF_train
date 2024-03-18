# vad 批量处理aishell语音
import os
import torch
import pandas as pd
from utils_vad import read_audio, get_speech_timestamps, init_jit_model,save_audio, collect_chunks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = init_jit_model(model_path='./silero_vad.jit',device=torch.device(device))
save_path = '/mnt/users/daihuangyu/AEC_Challenge/NKF_Data_From_AEC_Challenge/synthetic/'
  
if not os.path.exists(save_path + '/nearend_mic_signal_vad/'):
    os.makedirs(save_path + '/nearend_mic_signal_vad/') 
    os.makedirs(save_path + '/nearend_speech_vad/') 
    os.makedirs(save_path + '/echo_signal_vad/') 
    os.makedirs(save_path + '/farend_speech_vad/') 
    
for i in range(2820, 10000):
    wav = read_audio(save_path + 'nearend_mic_signal/' + f'nearend_mic_fileid_{i}.wav', sampling_rate=16000)
    print(f'nearend_mic_fileid_{i}.wav')
    wav_cuda = wav.float()
    wav_cuda = wav_cuda.to(device)
    speech_timestamps = get_speech_timestamps(wav_cuda, model, sampling_rate=16000)
    
    wav2 = read_audio(save_path + 'nearend_speech/' + f'nearend_speech_fileid_{i}.wav', sampling_rate=16000)
    wav3 = read_audio(save_path + 'echo_signal/' + f'echo_fileid_{i}.wav', sampling_rate=16000)
    wav4 = read_audio(save_path + 'farend_speech/' + f'farend_speech_fileid_{i}.wav', sampling_rate=16000)
    
    save_audio(save_path + 'nearend_mic_signal_vad/' + f'nearend_mic_fileid_{i}.wav',
           collect_chunks(speech_timestamps, wav), sampling_rate=16000) 
    save_audio(save_path + 'nearend_speech_vad/' + f'nearend_speech_fileid_{i}.wav',
           collect_chunks(speech_timestamps, wav2), sampling_rate=16000)
    save_audio(save_path + 'echo_signal_vad/' + f'echo_fileid_{i}.wav',
           collect_chunks(speech_timestamps, wav3), sampling_rate=16000)
    save_audio(save_path + 'farend_speech_vad/' + f'farend_speech_fileid_{i}.wav',
           collect_chunks(speech_timestamps, wav4), sampling_rate=16000) 
    
