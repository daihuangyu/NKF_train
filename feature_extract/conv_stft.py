import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
import struct

import scipy
import pyroomacoustics as pra


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False, window=None):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    elif win_type == 'SGP' or win_type == 'sgp':
        window = window
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__()

        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase


class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConviSTFT, self).__init__()
        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """

        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        # outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[..., self.win_len - self.stride:-(self.win_len - self.stride)]

        return outputs

  
class SgpSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type="SGP", win_fn="../win.txt",feature_type='real', fix=True):
        super(SgpSTFT, self).__init__()

        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        window = np.loadtxt(win_fn)
        window = window.reshape(-1,)
        kernel1, _ = init_kernels(win_len, win_inc, self.fft_len, win_type, False, window[0:win_len,])
        kernel2, _ = init_kernels(win_len, win_inc, self.fft_len, win_type, False, window[win_len:,])   
        
        self.register_buffer('weight1', kernel1)
        self.register_buffer('weight2', kernel2)
        
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        #inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        inputs1 = F.pad(inputs, [0, self.stride*4])[:,:,self.stride*2:]
        inputs2 = F.pad(inputs, [0, self.stride*2])
        
        outputs = F.conv1d(inputs2,self.weight1, stride=self.stride)+F.conv1d(inputs1, self.weight2, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim-1, :]
            imag = outputs[:, dim:-1, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase


class SgpiSTFT(nn.Module):

    def __init__(self, win_len, win_inc, win_fn, fft_len, fix=True):
        super(SgpiSTFT, self).__init__()
        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
            
        f_win = open(win_fn,'rb')
        tmp_data = f_win.read()
        f_win.close()
        data = struct.unpack('f'*(len(tmp_data)//4), tmp_data)
        win = np.array(data, dtype=np.float32)
        win = torch.from_numpy(win)
        win = win[:,None]    
        self.register_buffer('win', win)
       
        self.stride = win_inc
        self.win_len = win_len
        self.fft_len = fft_len

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N//2, T] (complex spec) or [B, N//2, T] (mags)
        phase: [B, N//2, T] (if not none)
        """
        real = inputs[:,0:self.fft_len//2,:]
        imag = inputs[:,self.fft_len//2+1:-1,:]
        inputs = torch.stack((real,imag),1)
        batch, ch, frq, frame = inputs.size()
        x1 = inputs[:,:,1:,:]
        x1 = torch.flip(x1, [2])
        x1 = F.pad(x1, [0,0,1,0])
        
        x1[:,1,:,:] = x1[:,1,:,:] * -1

        t_data = torch.cat([inputs, x1],2)
        t_data = t_data.permute(3, 0, 2, 1)

        t_data = torch.complex(t_data[:,:,:,0], t_data[:,:,:,1])
        ifft_data = torch.real(torch.fft.fft(t_data, dim=-1))
        ifft_data = ifft_data.permute( 1, 2, 0)

        tmp = torch.cat([ifft_data, ifft_data], 1)
        win = self.win.expand( 1024, frame)

        tmp = tmp * win * 256

        tmp1 = F.pad(tmp, [0,3,0,768])
        tmp2 = F.pad(tmp, [1,2,256,512])
        tmp3 = F.pad(tmp, [2,1,512,256])
        tmp4 = F.pad(tmp, [3,0,768,0])
        synbuf = tmp1 + tmp2 + tmp3 + tmp4
        syn = synbuf[:,768:1024, :-4]
        syn = torch.flip(syn, [1])
        batch, f_num, frames = syn.size()
        syn = syn.permute(0, 2, 1)

        frames = f_num*frames  

        syn = syn.reshape(batch, frames)

        return syn


def test_fft():
    torch.manual_seed(20)
    win_len = 320
    win_inc = 160
    fft_len = 512
    inputs = torch.randn([1, 1, 16000 * 4])
    fft = ConvSTFT(win_len, win_inc, fft_len, win_type='hanning', feature_type='real')
    import librosa

    outputs1 = fft(inputs)[0]
    outputs1 = outputs1.numpy()[0]
    np_inputs = inputs.numpy().reshape([-1])
    librosa_stft = librosa.stft(np_inputs, win_length=win_len, n_fft=fft_len, hop_length=win_inc, center=False)
    print(np.mean((outputs1 - np.abs(librosa_stft)) ** 2))


def test_ifft1():
    import soundfile as sf
    N = 400
    inc = 100
    fft_len = 512
    torch.manual_seed(N)
    data = np.random.randn(16000 * 8)[None, None, :]
    #    data = sf.read('../ori.wav')[0]
    inputs = data.reshape([1, 1, -1])
    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    inputs = torch.from_numpy(inputs.astype(np.float32))
    outputs1 = fft(inputs)
    print(outputs1.shape)
    outputs2 = ifft(outputs1)
    sf.write('conv_stft.wav', outputs2.numpy()[0, 0, :], 16000)
    print('wav MSE', torch.mean(torch.abs(inputs[..., :outputs2.size(2)] - outputs2) ** 2))


def test_ifft2():
    N = 400
    inc = 100
    fft_len = 512
    np.random.seed(20)
    torch.manual_seed(20)
    t = np.random.randn(16000 * 4) * 0.001
    t = np.clip(t, -1, 1)
    # input = torch.randn([1,16000*4])
    input = torch.from_numpy(t[None, None, :].astype(np.float32))

    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')

    out1 = fft(input)
    output = ifft(out1)
    print('random MSE', torch.mean(torch.abs(input - output) ** 2))
    import soundfile as sf
    sf.write('zero.wav', output[0, 0].numpy(), 16000)

def test_sgpstft(inwav, outwav):
    stft = SgpSTFT(512, 256, 512, win_type='SGP', win_fn="../res/window.txt", feature_type='complex')
    istft = SgpiSTFT(512, 256,'../res/win.dat', 512)
    
    _, data_4chan = scipy.io.wavfile.read(inwav)
    scipy.io.wavfile.write("../out_ori.wav", 16000, data_4chan[:30*16000,0])
    
    data_4chan = data_4chan.astype('float32')/32768
    
    data_4chan = data_4chan[:30*16000,:]
    noisy = data_4chan[:, 0]
    specs = stft(torch.from_numpy(noisy.reshape(1,-1)))
    out_data = istft(specs).detach().numpy()
    
    data = out_data * 32768
    data = data.astype('short')
    scipy.io.wavfile.write(outwav, 16000, data.T)
    

if __name__ == '__main__':
    #test_fft()
    #test_ifft1()
    #test_ifft2()
    inwav = r'/home/pf/share/apple/tmp/noise_debug/test.wav'
    outwav = r'../out_1.wav'
    test_sgpstft(inwav, outwav)
