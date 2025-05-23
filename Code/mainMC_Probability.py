import ROOT
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import glob
from natsort import natsorted
import decimal
from decimal import Decimal, getcontext
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import pytz
import csv
import json
import os
import sys
import h5py


# import other functions
import EventDisplay as ed
import Projection as proj
import LAPPDGeo as lg
import LoadInfo as li
import Optimization as opt
import DataClass as dc


# this will load the MC file and do the projection



# This code will load the root file generated from BeamClusterTree tool chain with ANNIETreeMaker and LAPPD reconstruction enabled
# The LAPPD profile is needed. All file can be generated from the script under GenerateData folder with appropriate input
#   1. LAPPD gain vs position distribution
#   2. LAPPD QE vs position distribution
#   3. LAPPD QE vs wavelength distribution
#   4. Absorption length vs wavelength distribution
#   
# First, loop over the events, based on the cuts applied, select the events pass the cuts
# Second, for events pass the cuts, generate the Cherenkov photons from each muon track in it and project the photons to each LAPPD plane
# Third, save all hits information from the muon tracks in the event to the root tree
# Forth, save the muon Cherenkov light projection into display plots
# Fifth, save the standard event display for this event
# Last, save the root tree to a new root file



if __name__ == "__main__":


    basePath = '/Users/fengy/ANNIESofts/Analysis/ProjectionComplete/'

    beam_data_path = basePath + 'data/'
    LAPPD_profile_path = basePath + 'LAPPDProfile/'
    plot_save_path = basePath + 'MC_plots/'
    save_result_path = basePath + 'OptimizationResults/5.Probability/output/'
    h5filePath = basePath + 'OptimizationResults/5.Probability/output/'
    
    if not os.path.exists(basePath):
        print(f"Error: Base path '{basePath}' does not exist.")
        sys.exit(1)

    for path in [beam_data_path, LAPPD_profile_path, plot_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Create directory: {path}")
            

        
    #root_file_pattern = beam_data_path + 'ANNIETree_MC_*.root'
    
    MCrunNumber = 0
    MCSubRunNumber = 0
    #root_file_pattern = f'/Users/fengy/ANNIESofts/Analysis/MCDataView/MCTrees/ANNIETree_MC_{MCrunNumber}_{MCSubRunNumber}.root'
    
    #root_file_pattern = f'/Users/fengy/ANNIESofts/Analysis/MCDataView/tree_95-99/mergedEventTree_95_0.root'
    root_file_pattern = f'/Users/fengy/ANNIESofts/Analysis/MCDataView/MCTrees/ANNIETree_MC_*.root'
    root_file_pattern = f'/Users/fengy/ANNIESofts/Analysis/2025.2.4_WCSimReco/gridPoints/ANNIETree_MC_mu_lr_500.root'
    #root_file_pattern = f'/Users/fengy/ANNIESofts/Analysis/2025.2.4_WCSimReco/gridPoints/shiftYDir/ANNIETree_MC_mu_lr_y0.03_500.root'
    #root_file_pattern = f'/Users/fengy/ANNIESofts/Analysis/2025.2.4_WCSimReco/gridPoints/shiftYDir/ANNIETree_MC_mu_lr_xDir-0.45_500.root'
    #root_file_pattern = f'/Users/fengy/ANNIESofts/Analysis/2025.2.4_WCSimReco/gridPoints/shiftYDir/ANNIETree_MC_mu_downUp.root'
    #root_file_pattern = f'/Users/fengy/ANNIESofts/Analysis/2025.2.4_WCSimReco/gridPoints/shiftYDir/ANNIETree_MC_mu_lr_y+0.04_500.root'

    root_file_pattern = f'/Users/fengy/ANNIESofts/Analysis/2025.2.4_WCSimReco/innerStructure/ANNIETree_MC_noInnerStructure.root'
    
    root_file_pattern = f'/Users/fengy/ANNIESofts/Analysis/2025.2.4_WCSimReco/rootTree_noInner_10cmStep/ANNIETree_MC_x0_y-2_dirx1_diry1.root'
    
    SelectEntry = False
    entry = 131
    entry_start = entry-5
    entry_end = entry+5
    file_list = natsorted(glob.glob(root_file_pattern))


    processStartEntry = 0
    processEventNumber = 1000

    pdfName = 'EventDisplay.pdf'
    printEventNumber = 0
    printEventMaxNumber = 5
    printPDF = False # True means no event information on the PDf
    if printPDF:
        pdf = PdfPages(plot_save_path + pdfName)

    # cuts to select events
    # for MC data or beam
    '''
    cut_beamOK = True
    cut_clusterExist = True
    cut_clusterCB = True
    cut_clusterCB_value = 0.2
    cut_clusterMaxPE = False
    cut_clusterMaxPE_value = 100
    cut_muonTrack = True
    cut_noVeto = False
    cut_MRDPMTCoinc = True
    cut_ChrenkovCover = False
    cut_ChrenkovCover_nPMT = 4
    cut_ChrenkovCover_PMT_PE = 5
    cut_LAPPDMultip = False
    cut_LAPPDHitAmp = 5
    cut_LAPPDHitNum = 7
    '''
    
    # for mono energy muon
    cut_beamOK = False
    cut_clusterExist = False
    cut_clusterCB = False
    cut_clusterCB_value = 0.2
    cut_clusterMaxPE = False
    cut_clusterMaxPE_value = 100
    cut_muonTrack = False
    cut_noVeto = False
    cut_MRDPMTCoinc = False
    cut_ChrenkovCover = False
    cut_ChrenkovCover_nPMT = 4
    cut_ChrenkovCover_PMT_PE = 5
    cut_LAPPDMultip = False
    cut_LAPPDHitAmp = 5
    cut_LAPPDHitNum = 7


    # for MC root tree
    cut_LAPPDTubeID = True
    TargetTubeID = [1244]
    cut_HitPENumber = True
    HitPENumber = 40
    cut_TotalLAPPDHitPE = True
    TotalLAPPDHitPE = 100


    totalNumOfEvent = 0
    passCutEventNum = 0

    #########################################
    ######## setting for projection     #####
    #########################################
    
    muon_step = 0.01 # in meter # while less than 0.25 cm you need to change the 2000 steps limit in projection function to be higher
    muon_prop_steps = 500 # max number
    muon_start_Z = 1.2 # in meter
    phi_steps = 360 # how many rays on phi direction
    SampleIntergerHitPE = False
    

    #########################################
    ######## start to process the event #####
    #########################################

    PMT_chanKey_40 = [[462,428,406,412]] # for 2022 and 2023 data
    PMT_chanKey_2024 = [[374,377,407,445],[463,411,400,404],[462,428,406,412]] # for 2024 data

    ######   !!!!! this need double check  !!!!
    PMT_chanKey_2023_multi = [[462,428,406,412], [374,377,407,445], [463,411,400,404]]
    ######

    DetectorCenter = [0*100, -0.144649*100, 1.681*100]
    # make some LAPPD grids
    #LAPPD_Centers = [[0,-0.2255,2.951], [-0.898, -0.2255 + 0.5, 2.579], [0.898, -0.2255 - 0.5 , 2.579]]
    #LAPPD_Centers = [[0,-0.2255,2.951]] # center of LAPPD 40
    LAPPD_Centers = [[0, -0.144 , 1.2677543 + 1.681]] # z = 2.948, this is the WCSim position?

    LAPPD_Directions = [[0,0,-1], [1,0,-1], [-1,0,-1]]
    LAPPD_stripWidth = 0.462
    LAPPD_stripSpace = 0.229
    LAPPD_gridSize = LAPPD_stripWidth + LAPPD_stripSpace
    LAPPD_grids = []
    for i in range (len(LAPPD_Centers)):
        LAPPD_grids.append(lg.LAPPD_Grid_position(np.array(LAPPD_Centers[i]), np.array(LAPPD_Directions[i])))

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
    
    #print("wavelength25",wavelength25)
    #print("QE25 is ",QE25)

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



    for fileName in file_list:
        #print('Processing file: ', fileName)

        file = ROOT.TFile(fileName, "READ")
        tree = file.Get("Event")

        SimLAPPDHitMuIndex = ROOT.std.vector('double')()
        SimLAPPDHitID = ROOT.std.vector('double')()
        SimLAPPDHitPE = ROOT.std.vector('double')()
        SimLAPPDHitTime = ROOT.std.vector('double')()

        b_simIndex = tree.Branch("SimMuIndex",SimLAPPDHitMuIndex)
        b_simID = tree.Branch("SimLAPPDHitID",SimLAPPDHitID)
        b_simPE = tree.Branch("SimLAPPDHitPE",SimLAPPDHitPE)
        b_simTime = tree.Branch("SimLAPPDHitTime",SimLAPPDHitTime)

        if processStartEntry >= tree.GetEntries():
            print("processStartEntry is larger than the total number of entries, set to entry number - 1:", tree.GetEntries()-1)
            processStartEntry = tree.GetEntries()-1
            
        tot = tree.GetEntries()
        #print('Total number of events: ', tot)

        #for i in tqdm(range(processStartEntry, tree.GetEntries())):
        for i in range(processStartEntry, tree.GetEntries()):
            tree.GetEntry(i)
            totalNumOfEvent += 1
            if totalNumOfEvent % 10000 == 0:
                print('Processing event: ', totalNumOfEvent)
            ########
            
            
            FinalOptimizationResult = {}
            save_result_Name = save_result_path + f'Result_{MCrunNumber}_{MCSubRunNumber}_Event' + str(totalNumOfEvent) + '_' + str(passCutEventNum) +'.h5'
            
            
            ########

            processThisEntry = True 
            
            if SelectEntry:
                if i < entry_start or i > entry_end:
                    processThisEntry = False

            if cut_clusterExist:
                if tree.numberOfClusters < 1:
                    processThisEntry = False

                ExistClusterInPrompt = False
                for i in tree.clusterTime:
                    if i < 2000:
                        ExistClusterInPrompt = True
                if not ExistClusterInPrompt:
                    processThisEntry = False


            brightest_index = -1
            brightest_PE = -1
            for i in range(len(tree.clusterPE)):
                if tree.clusterPE[i]>brightest_PE:
                    brightest_PE = tree.clusterPE[i]
                    brightest_index = i 

            if cut_clusterMaxPE:
                if brightest_PE < cut_clusterMaxPE_value:
                    processThisEntry = False

            if cut_clusterCB  and brightest_index != -1:
                if tree.clusterChargeBalance[brightest_index] > cut_clusterCB_value:
                    processThisEntry = False

            if cut_muonTrack:
                if all(num <= 0 for num in tree.NumClusterTracks):
                    continue

            if cut_MRDPMTCoinc:
                if tree.TankMRDCoinc != 1:
                    processThisEntry = False

            if cut_noVeto:
                if tree.NoVeto != 1: # if noVeto is not true, skip
                    processThisEntry = False

            if(cut_ChrenkovCover and brightest_index != -1):
                PMT_chanKey = []
                if(tree.runNumber)<4451: # 2023 runs stop at 4450, 2024 runs start from 4763
                    PMT_chanKey = PMT_chanKey_40
                else:
                    PMT_chanKey = PMT_chanKey_2024

                PassChrenkovCoverCut = True
                ids = 0

                nearingPMTID = PMT_chanKey[ids]
                brightClusterKeyArray = tree.Cluster_HitChankey[brightest_index]
                brightClusterPEArray = tree.Cluster_HitPE[brightest_index]
                
                nearingPMTKey = [idx for idx, value in enumerate(brightClusterKeyArray) if value in nearingPMTID]
                if len(nearingPMTKey) == 0:
                    PassChrenkovCoverCut = False
                nearingPMTPE = [brightClusterPEArray[idx] for idx in nearingPMTKey]
                if sum(pe > cut_ChrenkovCover_PMT_PE for pe in nearingPMTPE) < cut_ChrenkovCover_nPMT:
                    PassChrenkovCoverCut = False
                        
                if(not PassChrenkovCoverCut):
                    processThisEntry = False




            #########################################
            ######## cuts for MC data #####
            #########################################
            
            
            LHitXs = np.array(tree.LAPPDMCHitX)
            LHitYs = np.array(tree.LAPPDMCHitY)
            LHitZs = np.array(tree.LAPPDMCHitZ)
            LHitTimes = np.array(tree.LAPPDMCHitTime)
            LHitTubeIDs = np.array(tree.LAPPDMCHitTubeIDs)
            
            if(cut_TotalLAPPDHitPE):
                if(len(LHitXs) < TotalLAPPDHitPE):
                    processThisEntry = False       
                    
            
            # only save the hits with LHitTubeIDs[index] == TargetTubeID
            if(cut_LAPPDTubeID):
                mask = np.isin(LHitTubeIDs, TargetTubeID)
                LHitXs = LHitXs[mask]
                LHitYs = LHitYs[mask]
                LHitZs = LHitZs[mask]
                LHitTimes = LHitTimes[mask]
                LHitTubeIDs = LHitTubeIDs[mask]
    
                
            if(cut_HitPENumber):
                mask = np.where(LHitTubeIDs == TargetTubeID[0])
                if(len(mask[0]) < HitPENumber):
                    processThisEntry = False
                    


                
            #########################################
            ######## start to process the event #####
            #########################################

            a = 0
            if (not processThisEntry):
                a+=1
                #b_simIndex.Fill()
            else:
                # get some basic information
                clusterIndex = brightest_index
                brightestClusterPE = brightest_PE
                clusterTime = tree.clusterTime[clusterIndex]
                
                # Construct the waveform from MCHits
                # select hits on one LAPPD
                fitLAPPDTubeID = 1244
                mask = np.where(LHitTubeIDs == fitLAPPDTubeID)
                LHitXs_fit = LHitXs[mask]*100
                LHitYs_fit = LHitYs[mask]*100
                LHitZs_fit = LHitZs[mask]*100
                LHitTimes_fit = LHitTimes[mask]
                LHitTubeIDs_fit = LHitTubeIDs[mask]
                
                
                ########### generate the waveform from the MC hits for LAPPD 0 at the center
                XStart = LAPPD_Centers[0][0] - 14*LAPPD_gridSize
                #YStart = LAPPD_Centers[0][1] - 14*LAPPD_gridSize
                YStart = 0 - 14*LAPPD_gridSize

                LAPPD_MCHit_2D = []
                for i in range (1):
                    LAPPD_MCHit_2D.append([])
                    for j in range(28):
                        LAPPD_MCHit_2D[i].append([])
                        
                #print("LHitXs_fit:", LHitXs_fit)

                for index, x in enumerate(LHitXs_fit):
                    step = int((x - XStart) / LAPPD_gridSize)
                    if 0 <= step < 28:
                        YPos = LHitYs_fit[index]
                        YStep = int((YPos - YStart) / LAPPD_gridSize)
                        if 0 <= YStep < 28:
                            hit = (YStep, (LHitTimes_fit[index] % 25.0 + 5)*1e-9, 1)
                            LAPPD_MCHit_2D[0][step].append(hit)
            
                MCHitNum = len(LHitXs_fit)
                Data_Waveform = proj.generate_lappd_waveforms(LAPPD_MCHit_2D, sPE_pulse_time, sPE_pulse, LAPPD_stripWidth, LAPPD_stripSpace)
                #Data_Waveform[LAPPD_id][strip number][0=dowm, 1=up][256]
                #print("Data_Waveform:", len(Data_Waveform), len(Data_Waveform[0]), len(Data_Waveform[0][0]), len(Data_Waveform[0][0][0]))
                #print("Data_Waveform:", Data_Waveform[0][10][0])
                
                ##############################################
                ######## project the muon track to LAPPD #####
                ##############################################

                # print("Event number ", totalNumOfEvent)
                
                mu_direction = [tree.trueDirX, tree.trueDirY, tree.trueDirZ]
                #mu_direction = [-0.4, 0, 0.6]
                
                mu_direction = mu_direction/np.linalg.norm(mu_direction)
                muon_fit_start_position = np.array([tree.trueVtxX+DetectorCenter[0], tree.trueVtxY+DetectorCenter[1], tree.trueVtxZ+ DetectorCenter[2]])/100
                
                new_mu_direction = mu_direction
                new_mu_positions = [muon_fit_start_position + (i * new_mu_direction * muon_step) for i in range(muon_prop_steps)]
                new_mu_positions = [pos for pos in new_mu_positions if (pos[2] < 2.95)]
                # !new_mu_positions was only used for tracking muon direction, the actual muon step positions are generaget in the projection function from starting point and direction
                
                #test_position = [muon_fit_start_position[0] + 0.12, muon_fit_start_position[1] + 0.12, muon_fit_start_position[2]]
                test_position = [muon_fit_start_position[0], muon_fit_start_position[1], muon_fit_start_position[2]]
                
                # print("muon_fit_start_position:", muon_fit_start_position)
                # print("test_position input:", test_position)
                
                positionPass = True
                if len(new_mu_positions) == 0:
                    positionPass = False
                    continue
                

                #print("New event, pass the muon position cuts, event number", totalNumOfEvent)
                print(totalNumOfEvent, passCutEventNum, test_position, mu_direction)
                #print("The last muon position: ", new_mu_positions[-1])
                
                
                print("True Muon start at:")
                print("Muon start position: ", test_position)
                print("Muon direction: ", new_mu_direction)
                
                
                # now, prepare to loop
                # for each event, we will save a hdf5 file
                # the hdf5 file name will be the h5filePath + passCutEventNum
                
                # in the loop, we do multiple samplings. For each sampling, we calculate the probability distribution around it
                # the hdf5 file will store the information for each sampling.
                # each entry will be one sampling, contains the center information, and all other informations
                # each "information" is [xStep, yStep, pdf_step, PvsPositions, PLog, shiftTs, logP_shiftTs, best_shiftT]
                # the x and y Step for center is 0,0
                # [sampling number][muon position, muon direction, sampled_2Dhits, logP_center, center information, [shifted information at all positions]]
                
                SampleTimes = 1
                mu_dx = 0.0067
                mu_dy = 0.0067
                mu_dx_step = 6
                mu_dy_step = 6
                LAPPD_profile = dc.LAPPD_profile(absorption_wavelengths,absorption_coefficients,qe_2d,gain_2d,QEvsWavelength_lambda,QEvsWavelength_QE,10,1,LAPPD_grids,sPE_pulse_time,sPE_pulse,LAPPD_stripWidth,LAPPD_stripSpace)
                
                
                with h5py.File(f"{h5filePath}event_{passCutEventNum}.h5", "w") as h5f:
                    filename = f"{h5filePath}event_{passCutEventNum}.h5"
                    
                    # 遍历每个采样
                    for sample_idx in range(SampleTimes):
                        print("Making sample: ", sample_idx)
                    
                        hits_center = opt.DoProjectionForHits(LAPPD_profile, test_position, mu_direction, muon_step =muon_step,phi_steps = phi_steps, weightingBy2D = False)
                        pdf_center = opt.HitsToPDF(hits_center, 0)
                        sampled_hits_center, logP_center = opt.HitsToSampledHits(hits_center)
                        sampled_2DHits = opt.SampledHitsTo2DHits(sampled_hits_center, 0)
                        PvsPosition_center, PLog_center, shiftTs_center, logP_shiftTs_center, best_shiftT_center = opt.Hits2DProbability(pdf_center, sampled_2DHits)

                        samp_group = h5f.create_group("sample_"+str(sample_idx))
                        samp_group.create_dataset("mu_position", data=np.array(test_position))
                        samp_group.create_dataset("mu_direction", data=np.array(mu_direction))
                        samp_group.create_dataset("sampled_2DHits", data=sampled_2DHits)
                        samp_group.create_dataset("logP_center", data=logP_center)
                        
                        # 中心信息组
                        center_group = samp_group.create_group("center_info")
                        center_group.create_dataset("pdf_center", data=pdf_center)
                        center_group.create_dataset("PvsPositions", data=PvsPosition_center)
                        center_group.create_dataset("PLog", data=PLog_center)
                        center_group.create_dataset("shiftTs", data=shiftTs_center)
                        center_group.create_dataset("logP_shiftTs", data=logP_shiftTs_center)
                        center_group.create_dataset("best_shiftT", data=best_shiftT_center)
                        
                        #print("PvsPosition_center: ", PvsPosition_center)
                        print("PLog_center: ", PLog_center)
                        # 偏移信息组
                        shifted_group = samp_group.create_group("shifted_info")
                
                        for xStep in range(1, int(mu_dx_step*2)):
                            for yStep in range(1, int(mu_dy_step*2)):
                                print("Making step: ", xStep, yStep)
                                
                                shiftedPosition = [test_position[0] + (xStep-mu_dx_step)*mu_dx, test_position[1] + (yStep-mu_dy_step)*mu_dy, test_position[2]]
                                hits_step = opt.DoProjectionForHits(LAPPD_profile, shiftedPosition, mu_direction, muon_step =muon_step,phi_steps = phi_steps, weightingBy2D = False)
                                pdf_step = opt.HitsToPDF(hits_step, 0)
                                # don't need to sample ,but calculate the P relative to the center
                                PvsPosition_step, PLog_step, shiftTs_step, logP_step, best_shiftT_step = opt.Hits2DProbability(pdf_step, sampled_2DHits)
                        
                                step_group = shifted_group.create_group(f"x{xStep}_y{yStep}")
                                step_group.create_dataset("xStep", data=(xStep - mu_dx_step)*mu_dx)  # 实际偏移量
                                step_group.create_dataset("yStep", data=(yStep - mu_dy_step)*mu_dy)
                                step_group.create_dataset("pdf_step", data=pdf_step)
                                step_group.create_dataset("PvsPositions", data=PvsPosition_step)
                                step_group.create_dataset("PLog", data=PLog_step)
                                step_group.create_dataset("shiftTs", data=shiftTs_step)
                                step_group.create_dataset("logP_shiftTs", data=logP_step)
                                step_group.create_dataset("best_shiftT", data=best_shiftT_step)  
                            
                    
                
                
                true_vertex_position = muon_fit_start_position
                true_vertex_direction = mu_direction

                passCutEventNum += 1
                if(passCutEventNum > processEventNumber):
                    print("Process the maximum number of events, break")
                    break
                


                






        #file.cd()
        #tree.Write("", ROOT.TObject.kOverwrite)
        #file.Close()
        #print('Processing finished, close the file')

    
    
    print('Total number of events: ', totalNumOfEvent)
    print('Number of events pass the cuts: ', passCutEventNum)

    if printPDF:
        pdf.close()
    print('All done!')

