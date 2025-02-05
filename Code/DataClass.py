
import json
import numpy as np


class LAPPD_profile:
    def __init__(self, 
                 absorption_wavelengths=None, 
                 absorption_coefficients=None, 
                 qe_2d=None, 
                 gain_2d=None, 
                 QEvsWavelength_lambda=None, 
                 QEvsWavelength_QE=None, 
                 bin_size=None, 
                 CapillaryOpenRatio=None, 
                 grid=None,
                 sPE_pulse_time = None,
                 sPE_pulse=None,
                 LAPPD_stripWidth=None,
                 LAPPD_stripSpace=None):
        self.absorption_wavelengths = absorption_wavelengths
        self.absorption_coefficients = absorption_coefficients
        self.qe_2d = qe_2d    # QE 2D, normalized
        self.gain_2d = gain_2d  # gain distribution 2D, normalized
        self.QEvsWavelength_lambda = QEvsWavelength_lambda   # QE vs wavelength, wavelength array
        self.QEvsWavelength_QE = QEvsWavelength_QE    # QE vs wavelength, QE array
        self.bin_size = bin_size    # wavelength bin size
        self.CapillaryOpenRatio = CapillaryOpenRatio   # capillary open ratio of MCP
        self.grid = grid  # grid position of LAPPDs
        self.sPE_pulse_time = sPE_pulse_time
        self.sPE_pulse = sPE_pulse
        self.LAPPD_stripWidth = LAPPD_stripWidth
        self.LAPPD_stripSpace = LAPPD_stripSpace


import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """
    自定义 JSONEncoder，用来处理 numpy 中的 array 和常见标量 (int64/float64/bool等)。
    """
    def default(self, obj):
        # 1) numpy 数组 -> Python 列表
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # 2) numpy 整型 (如 int64) -> Python int
        if isinstance(obj, np.integer):
            return int(obj)
        
        # 3) numpy 浮点型 (如 float32/float64) -> Python float
        if isinstance(obj, np.floating):
            return float(obj)
        
        # 4) numpy 布尔型 (如 bool_) -> Python bool
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # 5) 其他类型交给默认处理
        return super().default(obj)

