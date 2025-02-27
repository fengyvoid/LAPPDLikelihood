import ROOT
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytz
import csv
import json
import pandas as pd
from natsort import natsorted
import glob



import Projection as proj
import LAPPDGeo as lg
import LoadInfo as li
import DataClass as dc


basePath = '/Users/fengy/ANNIESofts/Analysis/ProjectionComplete/'

beam_data_path = basePath + 'data/'
LAPPD_profile_path = basePath + 'LAPPDProfile/'

absorption_file = LAPPD_profile_path + 'interpolated_water_absorption_data.csv'
absorption_data = pd.read_csv(absorption_file)
absorption_wavelengths = absorption_data['Wavelength (nm)'].values
absorption_coefficients = absorption_data['Absorption Coefficient'].values

# Load QE data for LAPPD 25 and LAPPD 63
qe_file_25 = LAPPD_profile_path + 'LAPPD25_interpolated_photon_energy_qe.csv'
qe_file_63 = LAPPD_profile_path + 'LAPPD63_interpolated_photon_energy_qe.csv'

qe_data_25 = pd.read_csv(qe_file_25)
qe_data_63 = pd.read_csv(qe_file_63)

wavelength25 = qe_data_25['Wavelength (nm)'].values
QE25 = qe_data_25['Average QE'].values

print("wavelength25",wavelength25)
print(QE25)

wavelength63 = qe_data_63['Wavelength (nm)'].values
QE63 = qe_data_63['Average QE'].values

#print("Using doubled QE!")
#print(QE25)

# Create arrays for QE vs Wavelength for 3 LAPPDs
QEvsWavelength_lambda = [wavelength25, wavelength63, wavelength63]
QEvsWavelength_QE = [QE25, QE63, QE63]

qe_file = LAPPD_profile_path + 'interpolated_QE_2d_Position.csv'
gain_file = LAPPD_profile_path + 'interpolated_gain_2d.csv'

qe_2d, gain_2d = li.load_qe_gain_distribution(qe_file, gain_file)

sPE_pulse_time, sPE_pulse = li.read_spe_pulse_from_root(LAPPD_profile_path + "singlePETemplate_LAPPD40.root", "pos10_1_1D", 7)

LAPPD_profile = dc.LAPPD_profile(absorption_wavelengths,absorption_coefficients,qe_2d,gain_2d,QEvsWavelength_lambda,QEvsWavelength_QE,10,1,LAPPD_grids,sPE_pulse_time,sPE_pulse,LAPPD_stripWidth,LAPPD_stripSpace)



root_file_pattern = f'/Users/fengy/ANNIESofts/Analysis/2025.2.4_WCSimReco/gridPoints/shiftYDir/ANNIETree_MC_mu_lr_y+0.04_500.root'

file_list = natsorted(glob.glob(root_file_pattern))
    
for file in file_list:
    
    file = ROOT.TFile(file, "READ")
    tree = file.Get("Event")
    for i in range(0, tree.GetEntries()):
        tree.GetEntry(i)
        
    LHitXs = np.array(tree.LAPPDMCHitX)
    LHitYs = np.array(tree.LAPPDMCHitY)
    LHitZs = np.array(tree.LAPPDMCHitZ)
    LHitTimes = np.array(tree.LAPPDMCHitTime)
    LHitTubeIDs = np.array(tree.LAPPDMCHitTubeIDs)
                

    fitLAPPDTubeID = 1244
    mask = np.where(LHitTubeIDs == fitLAPPDTubeID)
    LHitXs_fit = LHitXs[mask]*100
    LHitYs_fit = LHitYs[mask]*100
    LHitZs_fit = LHitZs[mask]*100
    LHitTimes_fit = LHitTimes[mask]
    LHitTubeIDs_fit = LHitTubeIDs[mask]
                    
                    
LAPPD_MCHit_2D = []
for i in range (1):
    LAPPD_MCHit_2D.append([])
    for j in range(28):
        LAPPD_MCHit_2D[i].append([])

for index, x in enumerate(LHitXs_fit):
    step = int((x - XStart) / LAPPD_gridSize)
    if 0 <= step < 28:
        YPos = LHitYs_fit[index]
        YStep = int((YPos - YStart) / LAPPD_gridSize)
        if 0 <= YStep < 28:
            hit = (YStep, (LHitTimes_fit[index] % 25.0 + 5)*1e-9, 1)
            LAPPD_MCHit_2D[0][step].append(hit)


Data_Waveform = proj.generate_lappd_waveforms(LAPPD_MCHit_2D, sPE_pulse_time, sPE_pulse, LAPPD_stripWidth, LAPPD_stripSpace)
