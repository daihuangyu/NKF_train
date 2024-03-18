

import torch
import torch.nn as nn
import soundfile as sf
import pandas as pd
import pypesq
import os
import multiprocessing
from utils import gcc_phat
import argparse

ref_path = '/mnt/users/daihuangyu/dataset/NKF_train/data/ref/'
train_path = '/mnt/users/daihuangyu/dataset/NKF_train/data/train_data/'
speech_path = '/mnt/users/daihuangyu/dataset/NKF_train/data/speech/'
class ComplexGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bias=True, dropout=0,
                 bidirectional=False):
        super().__init__()
        self.gru_r = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)
        self.gru_i = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, h_rr=None, h_ir=None, h_ri=None, h_ii=None):
        Frr, h_rr = self.gru_r(x.real, h_rr)
        Fir, h_ir = self.gru_r(x.imag, h_ir)
        Fri, h_ri = self.gru_i(x.real, h_ri)
        Fii, h_ii = self.gru_i(x.imag, h_ii)
        y = torch.complex(Frr - Fii, Fri + Fir)
        return y, h_rr, h_ir, h_ri, h_ii


class ComplexDense(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super().__init__()
        self.linear_real = nn.Linear(in_channel, out_channel, bias=bias)
        self.linear_imag = nn.Linear(in_channel, out_channel, bias=bias)

    def forward(self, x):
        y_real = self.linear_real(x.real)
        y_imag = self.linear_imag(x.imag)
        return torch.complex(y_real, y_imag)


class ComplexPReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        return torch.complex(self.prelu(x.real), self.prelu(x.imag))


class KGNet(nn.Module):
    def __init__(self, L, fc_dim, rnn_layers, rnn_dim):
        super().__init__()
        self.L = L
        self.rnn_layers = rnn_layers
        self.rnn_dim = rnn_dim

        self.fc_in = nn.Sequential(
            ComplexDense(2 * self.L + 1, fc_dim, bias=True),
            ComplexPReLU()
        )

        self.complex_gru = ComplexGRU(fc_dim, rnn_dim, rnn_layers, bidirectional=False)

        self.fc_out = nn.Sequential(
            ComplexDense(rnn_dim, fc_dim, bias=True),
            ComplexPReLU(),
            ComplexDense(fc_dim, self.L, bias=True)
        )

    def init_hidden(self, batch_size, device):
        self.h_rr = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
        self.h_ir = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
        self.h_ri = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
        self.h_ii = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)

    def forward(self, input_feature):
        feat = self.fc_in(input_feature).unsqueeze(1)
        rnn_out, self.h_rr, self.h_ir, self.h_ri, self.h_ii = self.complex_gru(feat, self.h_rr, self.h_ir, self.h_ri, self.h_ii)
        kg = self.fc_out(rnn_out).permute(0, 2, 1)
        return kg


class NKF(nn.Module):
    def __init__(self, L=4):
        super().__init__()
        self.L = L
        self.kg_net = KGNet(L=self.L, fc_dim=18, rnn_layers=1, rnn_dim=18).to("cuda")
        self.stft = lambda x: torch.stft(x, n_fft=1024, hop_length=256, win_length=1024, window=torch.hann_window(1024).to("cuda"),
                                         return_complex=True)
        self.istft = lambda X: torch.istft(X, n_fft=1024, hop_length=256, win_length=1024,
                                           window=torch.hann_window(1024).to("cuda"), return_complex=False)

    def forward(self, x, y):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        x = self.stft(x)
        y = self.stft(y)
        B, F, T = x.shape
        device = x.device
        h_prior = torch.zeros(B * F, self.L, 1, dtype=torch.complex64, device=device)
        h_posterior = torch.zeros(B * F, self.L, 1, dtype=torch.complex64, device=device)
        self.kg_net.init_hidden(B * F, device)

        x = x.contiguous().view(B * F, T)
        y = y.contiguous().view(B * F, T)
        echo_hat = torch.zeros(B * F, T, dtype=torch.complex64, device=device)

        for t in range(T):
            if t < self.L:
                xt = torch.cat([torch.zeros(B * F, self.L - t - 1, dtype=torch.complex64, device=device), x[:, :t + 1]],
                               dim=-1)
            else:
                xt = x[:, t - self.L + 1:t + 1]
            if xt.abs().mean() < 1e-5:
                continue

            dh = h_posterior - h_prior
            h_prior = h_posterior
            e = y[:, t] - torch.matmul(xt.unsqueeze(1), h_prior).squeeze()

            input_feature = torch.cat([xt, e.unsqueeze(1), dh.squeeze()], dim=1)
            kg = self.kg_net(input_feature)
            h_posterior = h_prior + torch.matmul(kg, e.unsqueeze(-1).unsqueeze(-1))

            echo_hat[:, t] = torch.matmul(xt.unsqueeze(1), h_posterior).squeeze()

        s_hat = self.istft(y - echo_hat).squeeze()

        return s_hat

def clean_wav(df, start, end, csv_path):
    model = NKF(L=4)
    model.load_state_dict(torch.load('./nkf_epoch70.pt'), strict=True)
    model.eval()
    df_out = pd.DataFrame(columns=['file_path'])
    for i in range(start, end):
        
        dir_path, file_name = os.path.split(df.iloc[i]['file_path'])
        file_name, extension = os.path.splitext(file_name)
        
        x, sr = sf.read(ref_path + file_name + '.wav')
        y, sr = sf.read(train_path + file_name + '.wav')
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        x = x.to("cuda")
        y = y.to("cuda")

        with torch.no_grad():
            s_hat = model(x, y)
        
        z, _ = sf.read(speech_path + file_name + '.wav')
        
        score = pypesq.pesq(z, s_hat.cpu().numpy())
        print(f"socre of wav {i} is {score}")
        if score > 2.5:
            df_out = df_out.append({'file_path': file_name}, ignore_index=True)
            print(file_name)   
    df_out.to_csv(csv_path, index=False, sep=',')

if __name__ == "__main__":

#   通过开源模型测试，清洗现有nkf算法无法收敛的数据，标准暂时为pesq指标低于3.5


#     ref_path = '/mnt/users/daihuangyu/dataset/NKF_train/data/ref/'
#     train_path = '/mnt/users/daihuangyu/dataset/NKF_train/data/train_data/'
#     speech_path = '/mnt/users/daihuangyu/dataset/NKF_train/data/speech/'
    
    df = pd.read_csv('ref_sig.csv')
    
    csv_names = []
    for i in range(4):
        csv_names.append(f"train_{i+1}.csv")
    
    part_size = int(len(df)/4)
    pool = multiprocessing.Pool(processes=4)
    for i in range(4):
        pool.apply_async(clean_wav, (df, part_size*i, part_size*(i+1), csv_names[i],))
    pool.close()
    pool.join()
  



