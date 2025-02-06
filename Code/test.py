from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from tqdm import tqdm
import ROOT
import sys


fileName = '/Users/fengy/ANNIESofts/Analysis/2025.2.4_WCSimReco/tests/ANNIETree_MC_mu_100.root'
fileName = '/Users/fengy/ANNIESofts/Analysis/2025.2.4_WCSimReco/gridPoints/ANNIETree_MC_mu_rl_center_100.root'
fileName = '/Users/fengy/ANNIESofts/Analysis/2025.2.4_WCSimReco/gridPoints/ANNIETree_MC_mu_rl_center_500.root'


file = ROOT.TFile(fileName, "READ")
tree = file.Get("Event")


LMCHitNum = []
LMCHitTimes = []
HitXs = []
HitYs = []
HitZs = []
TubeIDs = []

LselectedID = 1244
LMCHitNum_selectedID = []

HitXs_center = []
HitYs_center = []
HitZs_center = []
HitTimes_center = []

for i in tqdm(range(tree.GetEntries())):
    tree.GetEntry(i)

    LMCHitNum.append(len(tree.LAPPDMCHitCharge))

    LHitTubeIDs = np.array(tree.LAPPDMCHitTubeIDs)
    mask = np.isin(LHitTubeIDs, LselectedID)
    SelecteHits = LHitTubeIDs[mask]
    LMCHitNum_selectedID.append(len(SelecteHits))

    hx = tree.LAPPDMCHitX
    hy = tree.LAPPDMCHitY
    HitXs.append(np.array(hx))
    HitYs.append(np.array(hy))
    
    HitZs.append(np.array(tree.LAPPDMCHitZ))
    TubeIDs.append(np.array(tree.LAPPDMCHitTubeIDs))
    LMCHitTimes.append(np.array(tree.LAPPDMCHitTime))
    
    HitXs_center.append(np.array(hx)[mask])
    HitYs_center.append(np.array(hy)[mask])
    HitZs_center.append(np.array(tree.LAPPDMCHitZ)[mask])
    HitTimes_center.append(np.array(tree.LAPPDMCHitTime)[mask])
    
    
        
fig = plt.figure(figsize=(6, 4))
plt.hist(LMCHitNum, bins = 100, range = (0,500))
#plt.yscale('log')
plt.xlabel('Hit number')
plt.title('Hits(PE) per event')

index = 10
fig = plt.figure(figsize=(6, 4))
plt.scatter(HitXs[index], HitYs[index], s = 5, label = str(index))
plt.title('Hits(PE) in event '+ str(index))


fig = plt.figure(figsize=(6, 4))
plt.hist(LMCHitNum_selectedID, bins = 50, range = (0,150))
plt.title('Hit(PE) number on center LAPPD')

