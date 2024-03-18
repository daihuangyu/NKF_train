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
        self.kg_net = KGNet(L=self.L, fc_dim=self.fc_dim, rnn_layers=self.rnn_layers, rnn_dim=self.rnn_dim).to(self.device)
        self.stft = lambda x: torch.stft(x, n_fft=512, hop_length=256, win_length=512,
                                         window=torch.hann_window(512).to(self.device),
                                         return_complex=True)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(f"I have already initial")
                
                #nn.init.constant_(m.weight.data, 0)
                nn.init.kaiming_normal_(m.weight.data)

    def calloss(self, pre, label):
        label = self.stft(label)
        B, F, T = label.shape
        BF = B*F
        label = label.contiguous().view(BF, T)
        criterion = nn.MSELoss()
        # print(torch.sum(pre.real-label.real))
        
        loss1 = criterion(pre.real, label.real) * 10
        loss2 = criterion(pre.imag, label.imag) * 10
        return loss1 + loss2

    def forward(self, x, y, z):

        x = self.stft(x)
        B, F, T = x.shape
        y = self.stft(y)
        z = self.stft(z)
        BF = B * F
        x = x.contiguous().view(BF, T)
        y = y.contiguous().view(BF, T)
        z = z.contiguous().view(BF, T)
        device = x.device
        h_prior = torch.zeros(BF, self.L, 1, dtype=torch.complex64, device=device)
        h_posterior = torch.zeros(BF, self.L, 1, dtype=torch.complex64, device=device)

        h_rr = torch.zeros(self.rnn_layers, BF, self.rnn_dim).to(device=device)
        h_ir = torch.zeros(self.rnn_layers, BF, self.rnn_dim).to(device=device)
        h_ri = torch.zeros(self.rnn_layers, BF, self.rnn_dim).to(device=device)
        h_ii = torch.zeros(self.rnn_layers, BF, self.rnn_dim).to(device=device)
        # self.kg_net.init_hidden(BF, device)


        x = torch.cat([torch.zeros(B * F, self.L - 1, dtype=torch.complex64, device=device), x], dim=-1)
        echo_hat = torch.zeros(BF, T, dtype=torch.complex64, device=device)

        for t in range(T):
            xt = x[:, t:t + self.L]
 

            dh = h_posterior - h_prior
            h_prior = h_posterior

            #e = y[:, t] - torch.matmul(xt.unsqueeze(1), h_prior).squeeze()
            e = y[:, t] - z[:, t]
            
            input_feature = torch.cat([xt, e.unsqueeze(1), dh.squeeze()], dim=1)

            kg, h_rr, h_ir, h_ri, h_ii = self.kg_net(input_feature, Variable(h_rr), Variable(h_ir), Variable(h_ri), Variable(h_ii))
            #print(torch.max(kg.real))
#             kg.real = torch.clamp_(kg.real, -5, 5)
#             kg.imag = torch.clamp_(kg.imag, -5, 5)
            h_posterior = h_prior + torch.matmul(kg, e.unsqueeze(-1).unsqueeze(-1))
            h_posterior.real = torch.clamp_(h_posterior.real, -5, 5)
            h_posterior.imag = torch.clamp_(h_posterior.imag, -5, 5)
            # print(h_posterior)
            echo_hat[:, t] = torch.matmul(xt.unsqueeze(1), h_posterior).squeeze()
#         import pdb
#         pdb.set_trace()
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




