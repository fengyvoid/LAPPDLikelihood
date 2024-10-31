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

# import other functions
import EventDisplay as ed
import Projection as proj
import LAPPDGeo as lg
import LoadInfo as li





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



if __name__ == '__main__':
    
    basePath = '/Users/fengy/ANNIESofts/Analysis/ProjectionComplete/'

    beam_data_path = basePath + 'data/'
    LAPPD_profile_path = basePath + 'LAPPDProfile/'
    plot_save_path = basePath + 'plots/'

    root_file_pattern = beam_data_path + 'ANNIETree_Ped2022_all.root'

    file_list = np.sort(glob.glob(root_file_pattern))

    processStartEntry = 0
    processEventNumber = 15

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

        file = ROOT.TFile(fileName, "UPDATE")
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

            ###########
            ### focus on LAPPD 40 right now, if nothing on ID 0, skip
            ###########
            if not any(x == 0 for x in tree.LAPPD_ID):
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
                for ids in tree.LAPPD_ID:
                    if ids!=0:
                        PassChrenkovCoverCut = False
                        continue
                    if ids==0:
                        PassChrenkovCoverCut = True
                    nearingPMTID = PMT_chanKey[ids]
                    brightClusterKeyArray = tree.Cluster_HitChankey[brightest_index]
                    brightClusterPEArray = tree.Cluster_HitPE[brightest_index]
                    
                    nearingPMTKey = [idx for idx, value in enumerate(brightClusterKeyArray) if value in nearingPMTID]
                    if len(nearingPMTKey) == 0:
                        PassChrenkovCoverCut = False
                        break
                    nearingPMTPE = [brightClusterPEArray[idx] for idx in nearingPMTKey]
                    if sum(pe > cut_ChrenkovCover_PMT_PE for pe in nearingPMTPE) < cut_ChrenkovCover_nPMT:
                        PassChrenkovCoverCut = False
                        break
                        
                if(not PassChrenkovCoverCut):
                    processThisEntry = False

            if(cut_LAPPDMultip):
                PassLAPPDMultipCut = False
                #LAPPDID_Hit[i], LAPPDHitStrip[i], LAPPDHitTime[i], LAPPDHitAmp[i]
                for j, lappd_id in enumerate(tree.LAPPD_ID):
                    if(lappd_id!=0):
                        continue
                    PassHitNum = 0
                    for x, ids in enumerate(tree.LAPPDID_Hit):
                        if(ids == lappd_id and tree.LAPPDHitAmp[x]>cut_LAPPDHitAmp and tree.LAPPDHitTime[x]>80 and tree.LAPPD_PeakTime[x]<120):
                            PassHitNum+=1
                    if(PassHitNum>=cut_LAPPDHitNum):
                        PassLAPPDMultipCut = True

                if(not PassLAPPDMultipCut):
                    processThisEntry = False

            if (totalNumOfEvent == 71 or totalNumOfEvent == 140):
                continue
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
                
                # get the waveform of this event
                # the waveform is a 60 channel * 256 bin array
                # the index of the waveform is board number * 30 + strip number
                # i.e. board 0, strip 12 is at index 12, board 1, strip 12 is at index 42
                #Data_Waveform = []
                Data_Waveform = np.zeros((28, 2, 256))
                for i in range(60):
                    side = int(i/30)
                    strip = i%30
                    if strip == 0 or strip == 29:
                        continue
                    realStrip = strip - 1
                    # re align the baseline from bin 80 to 150
                    wave = np.array(tree.LAPPDWaveform[i])
                    baseline = np.mean(wave[80:150])
                    wave = wave - baseline
                    Data_Waveform[realStrip][side] = np.array(wave)
                    #Data_Waveform.append(np.array(tree.waveform[i]))

                ##############################################
                ######## project the muon track to LAPPD #####
                ##############################################
                
                
                ######## set some test parameters
                x_step = 0
                x_step_size = 0.05
                y_step = 0
                y_step_size = 0.05
                theta_step = 0
                theta_step_size = np.radians(1)  # 转换成弧度
                phi_step = 0
                phi_step_size = np.radians(1)  # 转换成弧度

                
                # testing projection
                print("Event number ", totalNumOfEvent)
                for mu_track_i in range (len(tree.MRDTrackStopX)):
                    #print("Projection muon track number ", mu_track_i, "trackStopX", tree.MRDTrackStopX, "trackStartX", tree.MRDTrackStartX)
                    mu_direction = [tree.MRDTrackStopX[mu_track_i] - tree.MRDTrackStartX[mu_track_i], tree.MRDTrackStopY[mu_track_i] - tree.MRDTrackStartY[mu_track_i], tree.MRDTrackStopZ[mu_track_i] - tree.MRDTrackStartZ[mu_track_i]]
                    mu_direction = mu_direction / np.linalg.norm(mu_direction)
                    # get the muon track information
                    dz = tree.MRDTrackStartZ[mu_track_i] - muon_start_Z
                    scale_factor = dz / mu_direction[2]
                    x_at_z = tree.MRDTrackStartX[mu_track_i] - scale_factor * mu_direction[0]
                    y_at_z = tree.MRDTrackStartY[mu_track_i] - scale_factor * mu_direction[1]
                    muon_fit_start_position = [x_at_z, y_at_z, muon_start_Z]
                    
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

                    
                    for x_offset in range(-x_step, x_step + 1):
                        for y_offset in range(-y_step, y_step + 1):
                            # 计算新的x, y坐标
                            x_at_z = muon_fit_start_position[0] + x_offset * x_step_size
                            y_at_z = muon_fit_start_position[1] + y_offset * y_step_size
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
                    
                                    

                                    #mu_positions = [muon_fit_start_position + (i * mu_direction * muon_step) for i in range(muon_prop_steps)]
                                    # select the muon step only if the xyz position is within the tank
                                    #mu_positions = [pos for pos in mu_positions if (pos[2] < 3)]

                                    Results = proj.parallel_process_positions(mu_positions, mu_direction, LAPPD_grids)
                                    Results_withMuTime = proj.process_results_with_mu_time(Results, muon_step)
                                    updated_hits_withPE = proj.update_lappd_hit_matrices(
                                        results_with_time=Results_withMuTime,       
                                        absorption_wavelengths = absorption_wavelengths,
                                        absorption_coefficients = absorption_coefficients,
                                        qe_2d=qe_2d,                               # QE 2D, normalized
                                        gain_2d=gain_2d,                           # gain distribution 2D, normlized
                                        QEvsWavelength_lambda=QEvsWavelength_lambda,    # QE vs wavelength, wavelength array
                                        QEvsWavelength_QE=QEvsWavelength_QE,            # QE vs wavelength, QE array
                                        bin_size=10,                                    # wavelength bin size
                                        CapillaryOpenRatio = 0.64                       # capillary open ratio of MCP
                                    )
                                    # data format: updated_hits_withPE = (LAPPD_index, first_index, second_index, hit_time, photon_distance, weighted_pe)
                                    #loop first index gives different y
                                    #loop second index gives different x


                                    totalPE = 0

                                    LAPPD_Hit_2D = []
                                    for i in range (1):
                                        LAPPD_Hit_2D.append([])
                                        for j in range(28):
                                            LAPPD_Hit_2D[i].append([])

                                    for i in range (len(updated_hits_withPE)):
                                        # each particle step
                                        for j in range(len(updated_hits_withPE[i])):
                                            # just loop all hits

                                            # for each strip, i.e. same x position but different y position
                                            # each second index is a strip, loop the first index to get all positions on that strip
                                            # each hit is (first_index(y direction), hit time, pe number)
                                            LAPPD_Hit_2D[updated_hits_withPE[i][j][0]][updated_hits_withPE[i][j][2]].append((updated_hits_withPE[i][j][1], updated_hits_withPE[i][j][3], updated_hits_withPE[i][j][5]))
                                            totalPE+=updated_hits_withPE[i][j][5]

                                    print("Total PE: ", totalPE)

                                    Sim_Waveforms = proj.generate_lappd_waveforms(LAPPD_Hit_2D, sPE_pulse_time, sPE_pulse, LAPPD_stripWidth, LAPPD_stripSpace)
                                    #Sim_Waveforms[LAPPD_id][0=dowm, 1=up][256]
                                    
                                    shiftDT, min_diff, waveform_diff, Sim_Waveform_shifted = proj.align_waveforms(Sim_Waveforms[0], Data_Waveform)
                                    # Now the Sim_Waveform_shifted is aligned with the Data_Waveform  (not even chi^2 tho)
                                    print("Shifted DT: ", shiftDT, " Min diff: ", min_diff)
                                    
                                    if(min_diff < min_waveform_diff):
                                        min_waveform_diff = min_diff
                                        bestResultWaveform = Sim_Waveform_shifted
                                        bestFitHits = updated_hits_withPE
                                    
                                    TotalFitResult.append([new_start_position[0], new_start_position[1], new_start_position[2], new_mu_direction[0], new_mu_direction[1], new_mu_direction[2], min_diff, totalPE])
                    
                    print("Best fit result: ", TotalFitResult)
                    output_txtfile = plot_save_path+'Event' + str(totalNumOfEvent) +'output.txt'


                    with open(output_txtfile, 'w') as filetxt:
                        json.dump(TotalFitResult, filetxt)
    
                    
                    plotName = plot_save_path + 'Event' + str(totalNumOfEvent) + '_MuonTrack' + str(mu_track_i) + 'waveform.png'
                    #ed.plotWaveforms(plotName, Data_Waveform, bestResultWaveform, (0,256))
                    plotName2D = plot_save_path + 'Event' + str(totalNumOfEvent) + '_MuonTrack' + str(mu_track_i) + '2D.png'
                    #ed.DisplayHits(plotName2D, bestFitHits, LAPPD_grids)
                    

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

