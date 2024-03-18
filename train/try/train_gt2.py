import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import sys
sys.path.append('..')
import argparse
import json
import numpy as np
from dataset.dataset_wav_gt import Mydata_loader
from model.NKF_GT2 import NKF
from utils.modelutils import savecheckpoint, reload_model_val
from torch.backends import cudnn

min_loss = 10000
def train(model, device, config):
    global min_loss
    optimizer = optim.RAdam(model.parameters(), lr=config['model'][0]['lr'], weight_decay=config['model'][0]['weight_decay'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, factor=0.5, patience=2)
    start_ep, sstep, best_loss = reload_model_val(model, optimizer, config['model_dir'])
    min_loss = best_loss
    #val_loss = val(model, device, config)
    trainloader = Mydata_loader(tp='train', batch_size=config['model'][0]['batch_size'],
                                num_workers=8)
    start = time.time()
    batch_num = len(trainloader)
    epochs = config['model'][0]['epoch']
    step = sstep
    fid = open(config['log_dir'] + "/log.txt", "w")

    printFreq = 2
    stft = lambda x: torch.stft(x, n_fft=1024, hop_length=256, win_length=1024,
                                         window=torch.hann_window(1024).to(device),
                                         return_complex=True)
    for epoch in range(start_ep + 1, epochs):
        running_loss = 0.0
        total_loss = 0.0
        model.train()
        stime = time.time()
        for i, data in enumerate(trainloader):
            # 获取输入数据
            train_wav, speech_wav, ref_wav, echo_wav= data
            train_wav, speech_wav, ref_wav, echo_wav = train_wav.float(), speech_wav.float(), ref_wav.float(), echo_wav.float()
            train_wav, speech_wav, ref_wav, echo_wav = train_wav.to(device), speech_wav.to(device), ref_wav.to(device), echo_wav.to(device)    
            
            ref_wav = stft(ref_wav)
            train_wav = stft(train_wav)
            echo_wav = stft(echo_wav)
            speech_wav = stft(speech_wav)
            B, F, T = ref_wav.shape
            BF = B * F
            
            ref_wav = ref_wav.contiguous().view(BF, T)
            train_wav = train_wav.contiguous().view(BF, T)
            echo_wav = echo_wav.contiguous().view(BF, T)
            speech_wav = speech_wav.contiguous().view(BF, T)
            
           
            ref_wav = torch.cat([torch.zeros(B * F, config['n_block']-1, dtype=torch.complex64, device=device), ref_wav], dim=-1)
            #s_hat = torch.zeros(BF, T, dtype=torch.complex64, device=device)
            loss = 0
            optimizer.zero_grad()
            model.init_hidden(BF)
            for t in range(T):         
                s_hat = model(ref_wav[:, t:t + config['n_block']], train_wav[:, t], echo_wav[:, t])
                loss += model.calloss(s_hat, speech_wav[:, t])
                
            loss.backward()                
            optimizer.step()
            running_loss += loss.item()
            total_loss += loss.item()
 
            step += i
            
            if (i + 1) % printFreq == 0:
                # every 200 print
                print('Training epoch %d step %d loss is %.4f' % (
                epoch + 1, i + 1, running_loss / printFreq))
                # fid.write('Training epoch '+ str(epoch+1) + ' loss is ' + str(round(running_loss / printFreq,3))+'\n')
                running_loss = 0.0
        print('Training_Avg epoch %d avg_loss is %.4f  cost time %.3f min ' % (
        epoch + 1, total_loss / batch_num, (time.time() - stime) / 60))

      
        val_loss = val(model, device, config)
        fid.write('Validation_Avg  avg_loss is ' + str(val_loss) + '\n')
        fid.flush()
        if min_loss > val_loss:
            min_loss = val_loss
            savecheckpoint(model, epoch, step, min_loss, optimizer, config['model_dir'])
        scheduler.step(val_loss)

    print('Finished Training! Total cost time: %.3f min' % ((time.time() - start) / 60))
    fid.close()


def val(model, device, config):
    model.eval()
    valloader = Mydata_loader(tp='val', batch_size=config['model'][0]['batch_size'], num_workers=8)
    start = time.time()
    batch_num = len(valloader)
    print(batch_num)
    total_loss = 0.0
    stft = lambda x: torch.stft(x, n_fft=1024, hop_length=256, win_length=1024,
                                         window=torch.hann_window(1024).to(device),
                                         return_complex=True)
    with torch.no_grad():
        for i, data in enumerate(valloader):
            # 获取输入数据
            train_wav, speech_wav, ref_wav, echo_wav= data
            train_wav, speech_wav, ref_wav, echo_wav = train_wav.float(), speech_wav.float(), ref_wav.float(), echo_wav.float()
            train_wav, speech_wav, ref_wav, echo_wav = train_wav.to(device), speech_wav.to(device), ref_wav.to(device), echo_wav.to(device)
            
            ref_wav = stft(ref_wav)
            train_wav = stft(train_wav)
            echo_wav = stft(echo_wav)
            speech_wav = stft(speech_wav)
            B, F, T = ref_wav.shape
            BF = B * F
            
            ref_wav = ref_wav.contiguous().view(BF, T)
            train_wav = train_wav.contiguous().view(BF, T)
            echo_wav = echo_wav.contiguous().view(BF, T)
            speech_wav = speech_wav.contiguous().view(BF, T)
            

           
            ref_wav = torch.cat([torch.zeros(B * F, config['n_block']-1, dtype=torch.complex64, device=device), ref_wav], dim=-1)
            model.init_hidden(BF)
            for t in range(T):
                
                s_hat = model(ref_wav[:, t:t + config['n_block']], train_wav[:, t], echo_wav[:, t])
      
                loss = model.calloss(s_hat, speech_wav[:, t])
                total_loss += loss.item()
            print(total_loss)

    print('Validation_Avg  avg_loss is %.4f ' % (total_loss / batch_num))
    print('Finished Validation! Total cost time: %.3f min ' % ((time.time() - start) / 60.0))

    loss_avg = total_loss / batch_num

    return loss_avg


if __name__ == '__main__':
    # read file
    with open('config.json', 'r') as myfile:
        data = myfile.read()

    # parse file
    config = json.loads(data)
    model_dir = config['model_dir']


    os.makedirs(model_dir, exist_ok=True)
    cudnn.benchmark = True
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NKF(L=config['n_block'], device=device)
    model.initialize()
    model.to(device)
    train(model, device, config)



