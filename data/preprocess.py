import numpy as np
from scipy.signal import lfilter

def max_normalization(ecg):
    return ecg / max(np.fabs(np.amin(ecg)), np.fabs(np.amax(ecg)))

# =====================================
# == ecg-af-detection-physionet-2017 ==
# =====================================
def low_pass_filtering(ecg):
    # LPF (1-z^-6)^2/(1-z^-1)^2
    b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
    a = [1, -2, 1]

    # transfer function of LPF
    h_LP = lfilter(b, a, np.append([1], np.zeros(12)))

    ecg2 = np.convolve(ecg, h_LP)
    # cancel delay
    ecg2 = np.roll(ecg2, -6)
    return max_normalization(ecg2)


def high_pass_filtering(ecg):
    # HPF = Allpass-(Lowpass) = z^-16-[(1-z^32)/(1-z^-1)]
    b = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 32, -32, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    a = [1, -1]

    # impulse response iof HPF
    h_HP = lfilter(b, a, np.append([1], np.zeros(32)))
    ecg3 = np.convolve(ecg, h_HP)
    # cancel delay
    ecg3 = np.roll(ecg3, -16)
    return max_normalization(ecg3)


class StandardScaler():
    def __init__(self):
        self.mean = 0
        self.std = 1
        
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
        
    def transform(self, data):
        return (data - self.mean) / self.std