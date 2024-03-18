import torch
import sys
sys.path.append('..')
from model.NKF_Res import NKF
from model.NKF_Res_1024_h80 import NKF2
import torch.optim as optim
from utils.modelutils import savecheckpoint, reload_model_val, initialize_config
import copy

def para_state_dict(model1, model2):
    state_dict = copy.deepcopy(model2.state_dict())
    
    loaded_paras = model1.state_dict()
    for key in state_dict:
        if key in loaded_paras and state_dict[key].size() == loaded_paras[key].size():
            print("成功初始化参数:", key)
            state_dict[key] = loaded_paras[key] 
    return state_dict
    

model1 = NKF(L=12, rnn_layers=1, fc_dim=80, rnn_dim=80)
model2 = NKF2(L=12, rnn_layers=1, fc_dim=80, rnn_dim=80)
checkpoint = torch.load('./model/NKF_B12_res_2/checkpoint_139_17340251.pth')
model1.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
epoch = 1
step = checkpoint['step']
loss = checkpoint['loss']
optimizer = optim.RAdam(model1.parameters(), lr=0.001, weight_decay=0.00001)
model2.initialize()
state_dict = para_state_dict(model1, model2)
model2.load_state_dict(state_dict)



savecheckpoint(model2, epoch, step, loss, optimizer, './model/')


# for para in model2.named_parameters():
#     if para[0].startswith("kg_net"):
        
#         print(para[0],'\t',para[1].size())