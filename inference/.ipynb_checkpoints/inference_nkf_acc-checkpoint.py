import torch
import soundfile as sf
import numpy as np
from numpy import *
import sys
sys.path.append('..')
#from NKF import NKF
from NKF_B12 import NKF
from dataset_wav import Mydata_loader
from metric import ERLE, PESQ
from torch.backends import cudnn
from utils.modelutils import reload_model_evl
import time


#test_path = '/mnt/users/daihuangyu/AEC_Challenge/AEC-Challenge/datasets/synthetic/'


def test_aec(model, device):
    start = time.time()
    #model.load_state_dict(torch.load('./model/nkf_1024_1/checkpoint_99_56949001.pth')['model_state_dict'], strict=True)
    model.load_state_dict(torch.load('./model/nkf_1024_1/checkpoint_99_56949001.pth')['model_state_dict'], strict=True)
    #model.load_state_dict(torch.load('./model/baseline/nkf_epoch70.pt'), strict=True)
    model.eval()
    fid = open("test_log_nkf_b12_1024.txt", "w")
    batch_size = 12
    testloader = Mydata_loader(tp='test', batch_size=batch_size, num_workers=6)
    batch_num = len(testloader)
    print(batch_num)
    erle_dbs = []
    pesq_scores = []

    with torch.no_grad():
        for i, data in enumerate(testloader):
            # 获取输入数据
            train_wav, speech_wav, ref_wav= data
            #print(ref_wav.shape)
            train_wav, speech_wav, ref_wav= train_wav.float(), speech_wav.float(), ref_wav.float()
            train_wav, speech_wav, ref_wav= train_wav.to(device), speech_wav.to(device), ref_wav.to(device)
            ref_wav, train_wav, speech_wav, s_hat, echo_hat = model(ref_wav, train_wav, speech_wav)
            
            ref_wav, train_wav, speech_wav = ref_wav.cpu().numpy(), train_wav.cpu().numpy(), speech_wav.cpu().numpy()
            s_hat, echo_hat = s_hat.cpu().numpy(), echo_hat.cpu().numpy()
            erle_db_tmp = []
            pesq_score_tmp = []
            for j in range(batch_size):
                pesq_score = PESQ(s_hat[j], speech_wav[j])
                pesq_scores.append(pesq_score)
                pesq_score_tmp.append(pesq_score)
                erle_db = ERLE(s_hat[j], echo_hat[j])
                erle_dbs.append(erle_db)
                erle_db_tmp.append(erle_db)  
                
            pesq_score_avg = mean(pesq_score_tmp)
            erle_db_avg = mean(erle_db_tmp)  
            print('%d time avg pesq score is %.3f ' % (i, pesq_score_avg))
            print('%d time avg erle db is %.3f' % (i, erle_db_avg))
            fid.write('%d time avg pesq score is %.3f\n ' % (i, pesq_score_avg))
            fid.write('%d time avg erle db is %.3f\n' % (i, erle_db_avg))
            fid.flush()  
        
            
    pesq_score_avg = mean(pesq_scores)
    erle_db_avg = mean(erle_dbs)

    print('finally avg pesq score is %.3f\n' % pesq_score_avg)
    print('finally avg erle db is %.3f\n' % erle_db_avg)
    print('Finished Test! Total cost time: ', time.time() - start)
    fid.write('finally avg pesq score is %.3f\n' % pesq_score_avg)
    fid.write('finally avg erle db is %.3f\n' % erle_db_avg)
    fid.write('Finished Test! Total cost time: ' + str(time.time() - start))
    fid.close()






if __name__ == '__main__':
    #model_dir = '../train/model/NKF_B4'
    cudnn.benchmark = True
    model = NKF()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_aec(model, device)

