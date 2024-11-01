import ROOT
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import glob
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

# import other functions
import EventDisplay as ed
import Projection as proj
import LAPPDGeo as lg
import LoadInfo as li


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

    if not os.path.exists(basePath):
        print(f"Error: Base path '{basePath}' does not exist.")
        sys.exit(1)

    for path in [beam_data_path, LAPPD_profile_path, plot_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Create directory: {path}")

        
    root_file_pattern = beam_data_path + 'ANNIETree_MC_withLAPPD.root'

    file_list = np.sort(glob.glob(root_file_pattern))

    processStartEntry = 0
    processEventNumber = 1000

    pdfName = 'EventDisplay.pdf'
    printEventNumber = 0
    printEventMaxNumber = 5
    printPDF = False # True means no event information on the PDf
    if printPDF:
        pdf = PdfPages(plot_save_path + pdfName)

    # cuts to select events
    cut_beamOK = True
    cut_clusterExist = True
    cut_clusterCB = True
    cut_clusterCB_value = 0.2
    cut_clusterMaxPE = False
    cut_clusterMaxPE_value = 100
    cut_muonTrack = True
    cut_noVeto = False
    cut_MRDPMTCoinc = True
    cut_ChrenkovCover = True
    cut_ChrenkovCover_nPMT = 4
    cut_ChrenkovCover_PMT_PE = 5
    cut_LAPPDMultip = True
    cut_LAPPDHitAmp = 5
    cut_LAPPDHitNum = 7


    # for MC root tree
    cut_LAPPDTubeID = True
    TargetTubeID = [1244]
    cut_HitPENumber = True
    HitPENumber = 20
    cut_TotalLAPPDHitPE = True
    TotalLAPPDHitPE = 200


    totalNumOfEvent = 0
    passCutEventNum = 0


    muon_step = 0.01 # in meter
    muon_prop_steps = 300 # max number
    muon_start_Z = 1.2 # in meter

    #########################################
    ######## start to process the event #####
    #########################################

    PMT_chanKey_40 = [[462,428,406,412]] # for 2022 and 2023 data
    PMT_chanKey_2024 = [[374,377,407,445],[463,411,400,404],[462,428,406,412]] # for 2024 data

    ######   !!!!! this need double check  !!!!
    PMT_chanKey_2023_multi = [[462,428,406,412], [374,377,407,445], [463,411,400,404]]
    ######

    # make some LAPPD grids
    #LAPPD_Centers = [[0,-0.2255,2.951], [-0.898, -0.2255 + 0.5, 2.579], [0.898, -0.2255 - 0.5 , 2.579]]
    LAPPD_Centers = [[0,-0.2255,2.951]] # center of LAPPD 40
    LAPPD_Centers = [[0,0, 1.2677543 + 1.681]] # z = 2.948, this is the WCSim position

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

    wavelength63 = qe_data_63['Wavelength (nm)'].values
    QE63 = qe_data_63['Average QE'].values

    # Create arrays for QE vs Wavelength for 3 LAPPDs
    QEvsWavelength_lambda = [wavelength25, wavelength63, wavelength63]
    QEvsWavelength_QE = [QE25, QE63, QE63]

    qe_file = LAPPD_profile_path + 'interpolated_QE_2d_Position.csv'
    gain_file = LAPPD_profile_path + 'interpolated_gain_2d.csv'

    qe_2d, gain_2d = li.load_qe_gain_distribution(qe_file, gain_file)

    sPE_pulse_time, sPE_pulse = li.read_spe_pulse_from_root(LAPPD_profile_path + "singlePETemplate_LAPPD40.root", "pos10_1_1D", 7)


    for fileName in file_list:
        print('Processing file: ', fileName)

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

        for i in tqdm(range(processStartEntry, tree.GetEntries())):
            tree.GetEntry(i)
            totalNumOfEvent += 1
            
            #########

            processThisEntry = True 

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
                    step = int((x - XStart) // LAPPD_gridSize)
                    
                    if 0 <= step < 28:
                        
                        YPos = LHitYs_fit[index]
                        YStep = int((YPos - YStart) // LAPPD_gridSize)
                        if 0 <= YStep < 28:
                            hit = (YStep, (LHitTimes_fit[index] % 25.0 + 5)*1e-9, 1)
                            LAPPD_MCHit_2D[0][step].append(hit)
            
                #Data_Waveform = np.zeros((28, 2, 256))
                #for strip in range(28):
                #    print("LAPPD_MCHit_2D strip:", strip, LAPPD_MCHit_2D[0][strip])
                #print(sPE_pulse_time, sPE_pulse)
                Data_Waveform = proj.generate_lappd_waveforms(LAPPD_MCHit_2D, sPE_pulse_time, sPE_pulse, LAPPD_stripWidth, LAPPD_stripSpace)

                #print("Data_Waveform:", Data_Waveform[0][10][0])
                
                ##############################################
                ######## project the muon track to LAPPD #####
                ##############################################
                
                
                ######## set some test parameters
                x_step = 4
                x_step_size = 0.05
                y_step = 4
                y_step_size = 0.05
                theta_step = 1
                theta_step_size = np.radians(1)  # 转换成弧度
                phi_step = 0
                phi_step_size = np.radians(1)  # 转换成弧度

                
                # testing projection
                print("Event number ", totalNumOfEvent)
                if(totalNumOfEvent==430 or totalNumOfEvent==248):
                    continue
                
                
                mu_direction = [tree.trueDirX, tree.trueDirY, tree.trueDirZ]
                mu_direction = mu_direction/np.linalg.norm(mu_direction)
                muon_fit_start_position = np.array([tree.trueVtxX, tree.trueVtxY, tree.trueVtxZ])/100
                
                print("True Muon start at:")
                print("Muon start position: ", muon_fit_start_position)
                print("Muon direction: ", mu_direction)
                
                ##############################################
                ######## To loop the likelihood, add different muon position and direction here, then make the mu positions
                ##############################################
                
                TotalFitResult = []
                min_waveform_diff = 1e10
                bestResultWaveform = np.zeros((28, 2, 256))
                bestFitHits = []
                
                TotalStepNum = (2*x_step+1)*(2*y_step+1)*(2*theta_step+1)*(2*phi_step+1)
                loop_index = 0

                SimPE = 0
                
                for x_offset in range(-x_step, x_step + 1):
                    for y_offset in range(-y_step, y_step + 1):
                        # 计算新的x, y坐标
                        x_at_z = muon_fit_start_position[0] + x_offset * x_step_size
                        y_at_z = muon_fit_start_position[1] + y_offset * y_step_size - 0.2255 # seems the center of LAPPD 0 in MC is not the actual LAPPD position
                        z_at_z = muon_fit_start_position[2]
                        new_start_position = [x_at_z, y_at_z, z_at_z]

                        # 遍历theta和phi，调整Muon方向
                        for theta_offset in range(-theta_step, theta_step + 1):
                            for phi_offset in range(-phi_step, phi_step + 1):
                                loop_index+=1
                                print("Calculating step x: ", x_offset+x_step, " y: ", y_offset+y_step, ". Total steps: ", loop_index, "/", TotalStepNum)
                                # 计算新的theta和phi
                                theta = theta_offset * theta_step_size
                                phi = phi_offset * phi_step_size
                                
                                # 旋转初始Muon方向
                                new_mu_direction = proj.rotate_vector(mu_direction, theta, phi)
                                new_mu_direction = new_mu_direction / np.linalg.norm(new_mu_direction)  # 归一化
                                
                                
                                mu_positions = [new_start_position + (i * new_mu_direction * muon_step) for i in range(muon_prop_steps)]
                                # select the muon step only if the xyz position is within the tank
                                mu_positions = [pos for pos in mu_positions if (pos[2] < 3)]
                                
                                print("While Looping:")
                                print("Muon start position: ", new_start_position)
                                print("Muon direction: ", new_mu_direction)
                
                            
                                Results = proj.parallel_process_positions(mu_positions, mu_direction, LAPPD_grids)
                                
                                
                                Results_withMuTime = proj.process_results_with_mu_time(Results, muon_step, shiftTime = True)
                                updated_hits_withPE = proj.update_lappd_hit_matrices(
                                    results_with_time=Results_withMuTime,       
                                    absorption_wavelengths = absorption_wavelengths,
                                    absorption_coefficients = absorption_coefficients,
                                    qe_2d=qe_2d,                               # QE 2D, normalized
                                    gain_2d=gain_2d,                           # gain distribution 2D, normlized
                                    QEvsWavelength_lambda=QEvsWavelength_lambda,    # QE vs wavelength, wavelength array
                                    QEvsWavelength_QE=QEvsWavelength_QE,            # QE vs wavelength, QE array
                                    bin_size=10,                                    # wavelength bin size
                                    #CapillaryOpenRatio = 0.64                       # capillary open ratio of MCP
                                    CapillaryOpenRatio = 1                       # capillary open ratio of MCP
                                )
                                # data format: updated_hits_withPE = (LAPPD_index, first_index, second_index, hit_time, photon_distance, weighted_pe)
                                #loop first index gives different y
                                #loop second index gives different x


                                    
                                # use the updated_hits_withPE, and do poisson sampling based on the hit PE in each hit.
                                # do this sampling 5 times, calculate the waveform and min_diff for each time
                                # use the best min_diff to pick the best result, save the best waveform and best fit hits
                                
                                
                                sampleTimes = 5
                                    
                                min_diff_best = float('inf')
                                best_shiftDT = 0
                                
                                    
                                for sample_i in range(sampleTimes):
                                    
                                    sampled_hits_withPE = proj.sample_updatedHits_PE_Poisson(updated_hits_withPE)
                                    LAPPD_Hit_2D_sampled, totalPE = proj.convertToHit_2D(sampled_hits_withPE, number_of_LAPPDs = 1)

                                    Sim_Waveforms_sampled = proj.generate_lappd_waveforms(LAPPD_Hit_2D_sampled, sPE_pulse_time, sPE_pulse, LAPPD_stripWidth, LAPPD_stripSpace)
                                    shiftDT_sampled, min_diff_sampled, waveform_diff_sampled, Sim_Waveform_shifted_sampled = proj.align_waveforms(Sim_Waveforms_sampled[0], Data_Waveform[0])
                                    
                                    print("sample time: ", sample_i, " Min diff: ", min_diff_sampled, " Total PE: ", totalPE)
                                    
                                    if(min_diff_sampled < min_diff_best):
                                        min_diff_best = min_diff_sampled
                                        bestResultWaveform = Sim_Waveform_shifted_sampled
                                        bestFitHits = sampled_hits_withPE
                                        best_totalPE = totalPE
                                        SimPE = totalPE
                                        best_shiftDT = shiftDT_sampled

                                print("Shifted DT: ", best_shiftDT, " Min diff: ", min_diff_best)
                                print("Total PE: ", best_totalPE)
                                '''
                                Sim_Waveforms = proj.generate_lappd_waveforms(LAPPD_Hit_2D, sPE_pulse_time, sPE_pulse, LAPPD_stripWidth, LAPPD_stripSpace)
                                #Sim_Waveforms[LAPPD_id][0=dowm, 1=up][256]
                                
                                shiftDT, min_diff, waveform_diff, Sim_Waveform_shifted = proj.align_waveforms(Sim_Waveforms[0], Data_Waveform[0], SimRange=(0, 100), shiftRange=(100, 150))
                                # Now the Sim_Waveform_shifted is aligned with the Data_Waveform  (not even chi^2 tho)
                                print("Shifted DT: ", shiftDT, " Min diff: ", min_diff)
                                
                                if(min_diff < min_waveform_diff):
                                    min_waveform_diff = min_diff
                                    bestResultWaveform = Sim_Waveform_shifted
                                    bestFitHits = updated_hits_withPE
                                    SimPE = totalPE
                                '''
                                    
                                
                                TotalFitResult.append([new_start_position[0], new_start_position[1], new_start_position[2], new_mu_direction[0], new_mu_direction[1], new_mu_direction[2], min_diff_best, best_totalPE])
                
                #print("Best fit result: ", TotalFitResult)
                output_txtfile = plot_save_path+'Event' + str(totalNumOfEvent) +'_MCoutput.txt'
                with open(output_txtfile, 'w') as filetxt:
                    json.dump(TotalFitResult, filetxt)
                    
                    
                #print("Best fit hits: ", bestFitHits)
                bestFitHits_converted = [[(int(a), int(b), int(c), float(d), float(e), int(f)) for (a, b, c, d, e, f) in sublist] for sublist in bestFitHits]
                output_peTXTFile = plot_save_path+'Event' + str(totalNumOfEvent) +'_MCPEInfo.txt'
                with open(output_peTXTFile, 'w') as filetxt:
                    json.dump(bestFitHits_converted, filetxt)

                
                plotName = plot_save_path + 'Event' + str(totalNumOfEvent) + '_MCwaveform.png'
                ed.plotWaveforms(plotName, Data_Waveform[0], bestResultWaveform, (0,256), SimPE= SimPE, DataPE = len(LHitXs_fit))
                #ed.plotWaveforms(plotName, Data_Waveform[0], Sim_Waveforms[0], (0,256))

                plotName2D = plot_save_path + 'Event' + str(totalNumOfEvent) + '_MC2D.png'
                ed.DisplayHits(plotName2D, bestFitHits, LAPPD_grids)
                
                '''
                '''
                

                passCutEventNum += 1
                if(passCutEventNum > processEventNumber):
                    print("Process the maximum number of events, break")
                    break

                






        #file.cd()
        #tree.Write("", ROOT.TObject.kOverwrite)
        #file.Close()
        print('Processing finished, close the file')


    print('Total number of events: ', totalNumOfEvent)
    print('Number of events pass the cuts: ', passCutEventNum)

    if printPDF:
        pdf.close()
    print('All done!')

