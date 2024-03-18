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
from torch.autograd import Variable

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
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        y_real = self.linear_real(x.real)
        y_imag = self.linear_imag(x.imag)
        y_real = self.drop(y_real)
        y_imag = self.drop(y_imag)
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

    #     def initialize(self):
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 nn.init.kaiming_normal_(m.weight.data)

    def forward(self, input_feature, h_rr, h_ir, h_ri, h_ii):
        feat = self.fc_in(input_feature).unsqueeze(1)
        rnn_out, h_rr, h_ir, h_ri, h_ii = self.complex_gru(feat, h_rr, h_ir, h_ri, h_ii)
        kg = self.fc_out(rnn_out).permute(0, 2, 1)
        return kg, h_rr, h_ir, h_ri, h_ii


class NKF(nn.Module):
    def __init__(self, L=4, rnn_layers=1, fc_dim=18, rnn_dim=18, device='cuda'):
        super().__init__()
        self.L = L
        self.rnn_layers = rnn_layers
        self.fc_dim = fc_dim
        self.rnn_dim = rnn_dim
        self.device = device
        self.kg_net = KGNet(L=self.L, fc_dim=self.fc_dim, rnn_layers=self.rnn_layers, rnn_dim=self.rnn_dim).to(
            self.device)
        self.stft = lambda x: torch.stft(x, n_fft=512, hop_length=256, win_length=512,
                                         window=torch.hann_window(512).to(self.device),
                                         return_complex=True)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(f"I have already initial")
                nn.init.kaiming_normal_(m.weight.data)

    def calloss(self, pre, label):

        criterion = nn.MSELoss()
    
        loss1 = criterion(pre.real, label.real) 
        loss2 = criterion(pre.imag, label.imag) 
        return loss1 + loss2
    
    def init_hidden(self, BF):
        self.h_prior = torch.zeros(BF, self.L, 1, dtype=torch.complex64, device=self.device)
        self.h_posterior = torch.zeros(BF, self.L, 1, dtype=torch.complex64, device=self.device)

        self.h_rr = Variable(torch.zeros(1, BF, self.fc_dim).to(device=self.device))
        self.h_ir = Variable(torch.zeros(1, BF, self.fc_dim).to(device=self.device))
        self.h_ri = Variable(torch.zeros(1, BF, self.fc_dim).to(device=self.device))
        self.h_ii = Variable(torch.zeros(1, BF, self.fc_dim).to(device=self.device))

    def forward(self, xt, y, z):

        dh = self.h_posterior - self.h_prior
        self.h_prior = self.h_posterior

        e = y - torch.matmul(xt.unsqueeze(1), self.h_prior).squeeze()
        #e = y - z

        input_feature = torch.cat([xt, e.unsqueeze(1), dh.squeeze()], dim=1)

        kg, self.h_rr, self.h_ir, self.h_ri, self.h_ii = self.kg_net(input_feature, self.h_rr, self.h_ir, self.h_ri, self.h_ii)

        self.h_posterior = self.h_prior + torch.matmul(kg, e.unsqueeze(-1).unsqueeze(-1))
        self.h_posterior.real = torch.clamp_(self.h_posterior.real, -1, 1)
        self.h_posterior.imag = torch.clamp_(self.h_posterior.imag, -1, 1)
        # print(h_posterior)
        echo_hat = torch.matmul(xt.unsqueeze(1), self.h_posterior).squeeze()

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
#     model.initialize()
# 
#     stft = lambda x: torch.stft(x, n_fft=1024, hop_length=256, win_length=1024,
#                                          window=torch.hann_window(1024),
#                                          return_complex=True)
#     x = torch.randn(2, 160000)
#     y = torch.randn(2, 160000)
#     z = torch.randn(2, 160000)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     x = stft(x)
#     y = stft(y)
#     z = stft(z)
#     B, F, T = x.shape
#     BF = B * F
#     x = x.contiguous().view(BF, T)
#     y = y.contiguous().view(BF, T)
#     h_prior = torch.zeros(BF, n_block, 1, dtype=torch.complex64, device=device)
#     h_posterior = torch.zeros(BF, n_block, 1, dtype=torch.complex64, device=device)
# 
#     h_rr = torch.zeros(1, BF, 18).to(device=device)
#     h_ir = torch.zeros(1, BF, 18).to(device=device)
#     h_ri = torch.zeros(1, BF, 18).to(device=device)
#     h_ii = torch.zeros(1, BF, 18).to(device=device)
#     # self.kg_net.init_hidden(BF, device)
# 
#     x = x.contiguous().view(BF, T)
#     y = y.contiguous().view(BF, T)
#     z = z.contiguous().view(BF, T)
#     x = torch.cat([torch.zeros(B * F, n_block-1, dtype=torch.complex64, device=device), x], dim=-1)
# 
#     for t in range(T):
#         h_prior, h_posterior, h_rr, h_ir, h_ri, h_ii, s_hat = model(x[:, t:t + n_block], y[:, t], z[:, t], h_prior, h_posterior, h_rr, h_ir, h_ri, h_ii)
#         print(s_hat.shape)
# 
# 
# 


