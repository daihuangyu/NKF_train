import torch
import soundfile as sf
import numpy as np
from numpy import *
import sys
sys.path.append('..')
from speex_dkf import DKF
from speex_mdf import MDF
from metric import ERLE, PESQ
from utils.modelutils import reload_model_evl
import time


test_path = '/mnt/users/daihuangyu/AEC_Challenge/AEC-Challenge/datasets/synthetic/'
def test_aec():
    start = time.time()

    fid = open("test_dkf_log2.txt", "w")

    erle_dbs = []
    pesq_scores = []
    erle_db_tmp = []
    pesq_score_tmp = []

    for j in range(9000, 10000):

        farend_speech, sr = sf.read(test_path+'farend_speech/' + f'farend_speech_fileid_{j}.wav')
        nearend_speech, _ = sf.read(test_path + 'nearend_speech/' + f'nearend_speech_fileid_{j}.wav')
        nearend_mic_signal, _ = sf.read(test_path + 'nearend_mic_signal/' + f'nearend_mic_fileid_{j}.wav')

        min_len = min(len(nearend_mic_signal), len(farend_speech))
   
        nearend_mic_signal = nearend_mic_signal[:min_len]
        farend_speech = farend_speech[:min_len]
    # 64 2048 for 8kHz.
        #processor = MDF(sr, 256, 3072)
        processor = DKF(sr, 256, 3072)
        output, echo_hat = processor.main_loop(farend_speech, nearend_mic_signal)
      
        pesq_score = PESQ(output, nearend_speech)
        pesq_scores.append(pesq_score)
        pesq_score_tmp.append(pesq_score)
        erle_db = ERLE(output, echo_hat)
        erle_dbs.append(erle_db)
        erle_db_tmp.append(erle_db)

        if j % 10 == 0:

            pesq_score_avg = mean(pesq_score_tmp)
            erle_db_avg = mean(erle_db_tmp)
            pesq_score_tmp = []
            erle_db_tmp = []
            print('%d time avg pesq score is %.3f ' % (j, pesq_score_avg))
            print('%d time avg erle db is %.3f' % (j, erle_db_avg))
            fid.write('%d time avg pesq score is %.3f\n' % (j, pesq_score_avg))
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
   
    test_aec()

