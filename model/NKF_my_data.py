'''
Tencent is pleased to support the open source community by making NKF-AEC available.

Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.

Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
in compliance with the License. You may obtain a copy of the License at

https://opensource.org/licenses/BSD-3-Clause

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
'''
import torch
import torch.nn as nn
from torch.nn.functional import dropout
from torch_pesq import PesqLoss
import numpy as np


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


# class ComplexDropout(nn.Module):
#     def __init__(self, dropout=0.3, training=True):
#         super().__init__()
#         self.p = dropout
#         self.training = training

#     def forward(self, x):
#         mask = torch.ones(*x.shape, dtype=torch.float32, device="cuda")
#         mask = dropout(mask, self.p, self.training) * 1 / (1 - self.p)
#         mask.type(x.dtype)
#         return torch.complex(mask * x.real, mask * x.imag)


class ComplexLN(nn.Module):
    def __init__(self, fc_dim):
        super().__init__()
        self.LN = nn.LayerNorm(fc_dim)

    def forward(self, x):
        return torch.complex(self.LN(x.real), self.LN(x.imag))


class ComplexPReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        return torch.complex(self.prelu(x.real), self.prelu(x.imag))


class KGNet(nn.Module):
    def __init__(self, L, fc_dim, rnn_layers, rnn_dim, train):
        super().__init__()
        self.L = L
        self.rnn_layers = rnn_layers
        self.rnn_dim = rnn_dim
       
        self.fc_in = nn.Sequential(
            ComplexDense(2 * self.L + 1, fc_dim, bias=True),
            #ComplexDropout(0.1, training=train),
            # ComplexLN(fc_dim=fc_dim),
            ComplexPReLU()
        )

        self.complex_gru = ComplexGRU(fc_dim, rnn_dim, rnn_layers, bidirectional=False)

        self.fc_out = nn.Sequential(
            ComplexDense(rnn_dim, fc_dim, bias=True),
            #ComplexDropout(0.1, training=train),
            # ComplexLN(fc_dim=fc_dim),
            ComplexPReLU(),
            ComplexDense(fc_dim, self.L, bias=True),
            # ComplexLN(fc_dim=self.L)

        )

    def forward(self, input_feature, h_rr, h_ir, h_ri, h_ii):
        feat = self.fc_in(input_feature).unsqueeze(1)
        rnn_out, h_rr, h_ir, h_ri, h_ii = self.complex_gru(feat, h_rr, h_ir, h_ri, h_ii)
        kg = self.fc_out(rnn_out).permute(0, 2, 1)
        return kg, h_rr, h_ir, h_ri, h_ii


class NKF(nn.Module):
    def __init__(self, L=4, rnn_layers=1, fc_dim=80, rnn_dim=80, train=True, device='cuda'):
        super().__init__()
        self.L = L
        self.rnn_layers = rnn_layers
        self.fc_dim = fc_dim
        self.rnn_dim = rnn_dim
        self.device = device

        self.kg_net = KGNet(L=self.L, fc_dim=self.fc_dim, rnn_layers=self.rnn_layers, rnn_dim=self.rnn_dim, train=train)
        self.stft = lambda x: torch.stft(x, n_fft=1024, hop_length=256, win_length=1024,
                                         window=torch.hann_window(1024).to(self.device),
                                         return_complex=True)
        self.istft = lambda x: torch.istft(x, n_fft=1024, hop_length=256, win_length=1024,
                                           window=torch.hann_window(1024).to(self.device),
                                           return_complex=False)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(f"not init")
                #nn.init.kaiming_uniform_(m.weight.data, a=5)
                #nn.init.kaiming_normal_(m.weight.data)
                #nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.weight.data, 0)
    



    def calloss(self, pre, label):
        label = self.stft(label)
        B, F, T = label.shape
        BF = B*F
        label = label.contiguous().view(BF, T)
    
        criterion1 = nn.MSELoss()
        criterion2 = nn.MSELoss()
        
        weight = torch.cat([1*torch.ones(int(F/4)), 1*torch.ones(F-int(F/4))]).unsqueeze(1)
        weight = weight.repeat(B, T).to(self.device)
        pre = weight * pre
        label = weight * label
        loss1 = criterion1(pre.real, label.real)
        loss2 = criterion2(pre.imag, label.imag)

    
        return loss1 + loss2 
      
    
    def forward(self, x, y):
        x = self.stft(x)
        B, F, T = x.shape
        y = self.stft(y)
        BF = B * F
        x = x.contiguous().view(BF, T)
        y = y.contiguous().view(BF, T)
        device = x.device

        h_prior = torch.zeros(BF, self.L, 1, dtype=torch.complex64, device=device)
        h_posterior = torch.zeros(BF, self.L, 1, dtype=torch.complex64, device=device)
    
        h_rr = torch.zeros(self.rnn_layers, BF, self.rnn_dim).to(device=device)
        h_ir = torch.zeros(self.rnn_layers, BF, self.rnn_dim).to(device=device)
        h_ri = torch.zeros(self.rnn_layers, BF, self.rnn_dim).to(device=device)
        h_ii = torch.zeros(self.rnn_layers, BF, self.rnn_dim).to(device=device)
  
    
        x = torch.cat([torch.zeros(B * F, self.L - 1, dtype=torch.complex64, device=device), x], dim=-1)
    
        echo_hat = torch.zeros(BF, T, dtype=torch.complex64, device=device)
    
        for t in range(T):
            xt = x[:, t:t + self.L]
    
            dh = h_posterior - h_prior
            h_prior = h_posterior
    
            e = y[:, t] - torch.matmul(xt.unsqueeze(1), h_prior).squeeze()
            input_feature = torch.cat([xt, e.unsqueeze(1), dh.squeeze()], dim=1)
    
            kg, h_rr, h_ir, h_ri, h_ii = self.kg_net(input_feature, h_rr, h_ir, h_ri, h_ii)
    
            h_posterior = h_prior + torch.matmul(kg, e.unsqueeze(-1).unsqueeze(-1))
            torch.clamp_(h_posterior.real, -8, 8)
            torch.clamp_(h_posterior.imag, -8, 8)
            echo_hat[:, t] = torch.matmul(xt.unsqueeze(1), h_posterior).squeeze()
    
        s_hat = y - echo_hat
    
        return s_hat

# if __name__ == '__main__':
#     n_block = 4
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = NKF(L=n_block, device=device)
#     numparams = 0
#     for f in model.parameters():
#         numparams += f.numel()
#     print('Total number of parameters: {:,}'.format(numparams))
# 
#     x = torch.randn(2, 160000)
#     y = torch.randn(2, 160000)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 
#     model.initialize()
#     s_hat = model(x, y)
#     print(s_hat.shape)







