import torch
import soundfile as sf
import numpy as np
from numpy import *
import sys
sys.path.append('..')
from model.NKF import NKF
from metric import ERLE, PESQ
from torch.backends import cudnn
from utils.modelutils import reload_model_evl
import time


test_path = '/mnt/users/daihuangyu/AEC_Challenge/AEC-Challenge/datasets/synthetic/'
def test_aec(model, device):
    start = time.time()
    reload_model_evl(model, model_dir)
    model.eval()

    fid = open("test_log.txt", "w")

    erle_dbs = []
    pesq_scores = []
    erle_db_tmp = []
    pesq_score_tmp = []
    stft = lambda x: torch.stft(x, n_fft=1024, hop_length=256, win_length=1024,
                                window=torch.hann_window(1024).to(device),
                                return_complex=True)
    istft = lambda X: torch.istft(X, n_fft=1024, hop_length=256, win_length=1024,
                                  window=torch.hann_window(1024).to(device),
                                  return_complex=False)
    with torch.no_grad():
        for j in range(9000, 10000):

            farend_speech, _ = sf.read(test_path+'farend_speech/' + f'farend_speech_fileid_{j}.wav')
            nearend_speech, _ = sf.read(test_path + 'nearend_speech/' + f'nearend_speech_fileid_{j}.wav')
            nearend_mic_signal, _ = sf.read(test_path + 'nearend_mic_signal/' + f'nearend_mic_fileid_{j}.wav')

            
            farend_speech = np.expand_dims(farend_speech, 0)
            nearend_speech = np.expand_dims(nearend_speech, 0)
            nearend_mic_signal = np.expand_dims(nearend_mic_signal, 0)

            farend_speech = torch.from_numpy(farend_speech)
            nearend_speech = torch.from_numpy(nearend_speech)
            nearend_mic_signal = torch.from_numpy(nearend_mic_signal)

            farend_speech, nearend_mic_signal, nearend_speech = farend_speech.float(), nearend_mic_signal.float(), nearend_speech.float()
            farend_speech, nearend_mic_signal, nearend_speech = farend_speech.to(device), nearend_mic_signal.to(device), nearend_speech.to(device)

            echo_hat = model.forward(farend_speech, nearend_mic_signal)
            output_stft = stft(nearend_mic_signal) - echo_hat
            output = istft(output_stft)
       
            output = output.cpu().numpy()
            nearend_mic_signal = istft(stft(nearend_mic_signal))
            nearend_mic_signal = nearend_mic_signal.cpu().numpy()
            
            nearend_speech = istft(stft(nearend_speech))
            nearend_speech = nearend_speech.cpu().numpy()
            
            output = output.squeeze()
            nearend_mic_signal = nearend_mic_signal.squeeze()
            nearend_speech = nearend_speech.squeeze()
            
            pesq_score = PESQ(output, nearend_speech)
            pesq_scores.append(pesq_score)
            pesq_score_tmp.append(pesq_score)
            erle_db = ERLE(output, nearend_mic_signal)
            erle_dbs.append(erle_db)
            erle_db_tmp.append(erle_db)

            if j % 10 == 0:

                pesq_score_avg = mean(pesq_score_tmp)
                erle_db_avg = mean(erle_db_tmp)
                pesq_score_tmp = []
                erle_db_tmp = []
                fid.write('%d time avg pesq score is %.3f\n ' % (j, pesq_score_avg))
                fid.write('%d time avg erle db is %.3f\n' % (j, erle_db_avg))
                fid.flush()


    pesq_score_avg = mean(pesq_score)
    erle_db_avg = mean(erle_db)
    print('finally avg pesq score is %.3f\n' % pesq_score_avg)
    print('finally avg erle db is %.3f\n' % erle_db_avg)
    print('Finished Test! Total cost time: ', time.time() - start)
    fid.write('finally avg pesq score is %.3f\n' % pesq_score_avg)
    fid.write('finally avg erle db is %.3f\n' % erle_db_avg)
    fid.write('Finished Test! Total cost time: ' + str(time.time() - start))
    fid.close()


if __name__ == '__main__':
    model_dir = '../train/model/NKF_B4_5_newdata'
    cudnn.benchmark = True
    model = NKF()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_aec(model, device)

