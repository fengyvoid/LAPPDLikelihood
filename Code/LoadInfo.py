import pandas as pd
import uproot
import numpy as np


def load_qe_gain_distribution(qe_file, gain_file):
    # the qe distribution 
    qe_2d = pd.read_csv(qe_file, header=None).values
    gain_2d = pd.read_csv(gain_file, header=None).values
    qe_max = qe_2d.max()
    
    # normalize qe_2d
    if qe_max != 0:  
        qe_2d = qe_2d / qe_max
    
    return qe_2d, gain_2d


# 读取 ROOT 文件中的单光电子模板
def read_spe_pulse_from_root(file_path, hist_name, sPEAmp = 7):
    with uproot.open(file_path) as file:
        hist = file[hist_name]

        # 获取 bin 和 bin 内容
        bins = hist.axis(0).edges()
        values = hist.values()

        # 取负数并计算 sPE factor
        values = -values
        max_value = np.max(values)
        sPE_factor = sPEAmp / max_value

        # 将 bin 值乘以 sPE factor 得到 sPE pulse
        sPE_pulse = values * sPE_factor

        # 将 bin index 除以 1000，得到以 ns 为单位的 pulse time
        pulse_time = bins[:-1] / 1000  # 以 ns 为单位

    return pulse_time, sPE_pulse