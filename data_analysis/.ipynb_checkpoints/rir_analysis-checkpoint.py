import numpy as np
import matplotlib.pyplot as plt


rir = np.load("./room_index2_rt300_pos_center_ang240deg.npy")
rir = rir.squeeze(0)
fs = 16000
t = np.arange(rir.shape[1])/fs
plt.plot(t, rir[0])
plt.show()







