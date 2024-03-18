import numpy as np
import pypesq




def ERLE(out, echo_hat):
    echo_energy = np.sum((echo_hat) ** 2) / np.size(echo_hat)
    out_energy = np.sum(out ** 2) / np.size(out)
    return 10 * np.log10(echo_energy/out_energy)

def PESQ(out, nearend_speech):
    return pypesq.pesq(nearend_speech, out)