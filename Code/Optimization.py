import numpy as np
from scipy.spatial import KDTree
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp1d
from scipy import interpolate
import Projection as proj
import DataClass as dc
import random
import math
import os

import json
import h5py

'''
Overall idea:
Start from a muon parameter set gamma = (x,y,z,theta,phi)
    Calculate L.
    Vary gamma on all 5 parameters by a small amount delta gamma.
    Calculate L_n, until L_n - L_n-1 < epsilon (or epsilon * L_n-1).
        For each L, L(gamma'|gamma_D) = Product_alpha S_alpha(gamma'|gamma_D)
        The likelihood is the product of the probobality of all waveforms on all alpha channels.
        The probability is the similarity on each channel, which is the similarity of the waveform get from the selected muon parameter w' and the actuall data wavefrom w_D.
        It is interprelated as the proboability of getting the waveform w_D with the assumption of the muon parameter gamma' so get such a waveform w'.
        Or, assume the parameter gamma' to get waveform w', what's the proboability of getting the waveform w_D.
        P = S(w_D|w') 
        Therefore, maximize the product of P_alpha, will give the best parameter which is the most likely to generate the waveform w_D.
        
            In each P, it's depends on the Poisson sampling for hits on each channel.
            We resample the hits on this channel until getting a better similarity S(w_D|w').
            Then move to other channels and loop over more channels, until the L_this is stable (which means it can't be improved by resampling).
            Then L_now is the L_n for the n_th step with current gamma', then vary gamma to move to next step. 


Optimization procedure:
At L step 0:
    Start from a muon parameter set gamma = (x,y,z,theta,phi), get the hit distribution on LAPPD surface H_0 from this gamma
    Use H_0, do first Poisson sample and construct w'_0
        Calculate L_0_0 between w'_0 and w_D (L_gamma step_Poisson step)
            In all channels, find the channel alpha with the worst similarity S(w_D|w'_0)
                Resample the hits on this channel, construct w'_1_alpha
                Calculate S(w_D|w'_1_alpha) for this channel, if the S is not improved, resample this channel again
                If the S is improved, update w'_1_alpha to w'_1, and get L_0_1
            If the L_0_n was improved, find the new channel alpha with the worst similarity S, repear the resampling process
            If the L_0_n from a better resample for this channel wasn't improved compare with L_0_n-1, find the second worst channel to resample
            
            If the L_0_n was not improved after resampling x_th worst channels, stop the resampling.
            L_0_n is the L for this gamma step.
    
    Vary gamma on five parameters, repeat to find the L_0(gamma_0 + delta gamma) on five directions.
    For the most improved delta gamma, plus it to gamma_0 and get gamma_1
    
Move to L step 1, repeat to get L_1_n(gamma_1)
    Find the L_1(gamma_1 + delta gamma) on five directions, then get gamma_2
    
    
At L step_n, if L_n - L_n-1 < epsilon (or epsilon*L_n-1), stop the iteration.
        

'''
# for one channel, take the waveform from two side as input, and one shift dt value. 
# side 1 is the left part of the combined waveform
# Use the shift dt value to circular shift the waveform, and combine this two waveforms from each t=0, as the t = 256 of the combined waveform
def combineWaveform(w_side1, w_side2, dt):
    """
    Combine two waveforms from side1 and side2 of a channel with a circular shift.

    Parameters:
        w_side1 (numpy array): Waveform from side1 with 256 bins.
        w_side2 (numpy array): Waveform from side2 with 256 bins.
        dt (int): Circular shift value.

    Returns:
        numpy array: Combined waveform with 512 bins.
    """
    # Circular shift both waveforms
    shifted_side1 = np.roll(w_side1, dt)
    shifted_side2 = np.roll(w_side2, dt)

    # Combine the two waveforms
    combined_waveform = np.zeros(512)
    combined_waveform[0:256] = shifted_side1[::-1]  # Reverse side1 and map to 0-255
    combined_waveform[256:512] = shifted_side2      # Map side2 to 256-511

    return combined_waveform
    
# require all signal continously in the window.
# find the first point above the high threshold, the last point below the low threshold. and include 2 ns before and after.
def find_similarity_range(w, high_threshold, low_threshold):
    found_start = 0
    found_end = 0
    for i, value in enumerate(w):
        if value > high_threshold:
            found_start = i
            break
    for i, value in enumerate(w):
        if value < low_threshold:
            found_end = i
    
    found_start = max(0, found_start - 20)
    found_end = min(255, found_end + 20)
    
    return found_start, found_end

# given two combined waveforms, return the FoundRange of the overlapping range, and the similarity normlized by the length of the overlapping range.
# cut_Thres is a threshold for resampling. if the simulated waveform pass the threshold first, means there should be more PEs away from this side. Detail see resampling.
def calculate_section_similarity(start, end, c1, c2, high_thres = 5, low_thres = -3, cut_Thres = 3.5, pe_number = 1):
    """
    Calculate similarity for a section of the waveforms.
    """
    # Extract the waveform section
    w1 = c1[start:end] # simulated from projection
    w2 = c2[start:end] # data
    

    # Find the similarity range for w1 and w2 independently
    bin1s, bin1e = find_similarity_range(w1, high_thres, low_thres)
    bin2s, bin2e = find_similarity_range(w2, high_thres, low_thres)

    #FoundRange = (min(bin1s, bin2s), max(bin1e,bin2e))
    
    FoundRange = (0,255)
    
    #print("FoundRange:", FoundRange)
    
    if(FoundRange[1] - FoundRange[0] < 5):
        #print("FoundRange too small, return 0")
        return 0, FoundRange, 0, 0, 0

    
    part1 = w1[FoundRange[0]:FoundRange[1]] # projected
    part2 = w2[FoundRange[0]:FoundRange[1]] # data
    

    peak1 = part1.max()
    if(peak1 < 0):
        peak1 = 0

    peak2 = part2.max()
    if(peak2 < 0):
        peak2 = 0
            
    t1 = FoundRange[0]*0.1 # projected
    bins_w1 = np.where(part1 >= cut_Thres)[0]
    if(len(bins_w1) > 0 and bins_w1[0]!=0):
        crossing_bin = bins_w1[0]
        y1 = part1[crossing_bin - 1]  # Value before threshold
        y2 = part1[crossing_bin]      # Value after threshold
        x1 = (crossing_bin - 1)
        t1 = (x1 + (cut_Thres - y1) /(y2 - y1)) * 0.1 # in ns
        
    t2 = FoundRange[0]*0.1 # data
    bins_w2 = np.where(part2 >= cut_Thres)[0]
    if(len(bins_w2) > 0 and bins_w2[0]!=0):
        crossing_bin = bins_w2[0]
        y1 = part2[crossing_bin - 1]
        y2 = part2[crossing_bin]
        x1 = (crossing_bin - 1)
        t2 = (x1 + (cut_Thres - y1) /(y2 - y1)) * 0.1 # in ns
    
    ## TODO: maybe need a normalization?
    perfer_t = (t2 - t1)


    # Calculate similarity metric for the overlapping range
    ######################################################################################################
    ########## Obviously we want a better similarity algorithm, here is the position to modify ###########
    
    similarity_positive = 0
    similarity_negative = 0
    
    for i in range(FoundRange[0], FoundRange[1]):
        diff = abs(w1[i] - w2[i])
        #weighting = abs(w2[i])**3
        weighting = 1
        if(w2[i]>0):
            similarity_positive += diff * weighting
        else:
            similarity_negative += diff * weighting
        #weighting = max(abs(w1[i]), abs(w2[i])) #**3 # weighting
        
        #similarity += diff * weighting

    #if(FoundRange[1] - FoundRange[0] != 0):
    #    similarity = similarity/(FoundRange[1]- FoundRange[0])
        
    similarity_positive = similarity_positive#/pe_number
    similarity_negative = similarity_negative#/pe_number
    
    ########## Obviously we want a better similarity algorithm, here is the position to modify ###########
    ######################################################################################################
    
    #return similarity, FoundRange, peak1, peak2, perfer_t,  #, bin1s, bin1e, bin2s, bin2e
    return (similarity_positive, similarity_negative), FoundRange, peak1, peak2, perfer_t,  #, bin1s, bin1e, bin2s, bin2e

# for one microstrip.
# Calculate the similarity between four waveforms, two from simulation with shift, one one each side, and other two from data, one each side.
# w1 is the simulated waveform, w2 is the data waveform without any shift.
def WavefromSimilarity(w1_bottom, w1_top, w2_bottom, w2_top, shift_t1, high_thres = 5, low_thres = -3, sPE_Amp = 7, pe_number = 1):
    
    if(pe_number==0):
        pe_number = 1
    # c1 is simulated waveform
    c1 = combineWaveform(w1_bottom, w1_top, shift_t1) # use bottom side as the left part
    # c2 is data waveform
    c2 = combineWaveform(w2_bottom, w2_top, 0)
    # reverse the combined waveform, and take the later half, which is the original left part of the waveform. (bottom side)
    similarity_bottom, FoundRange_b, peak_bot_s, peak_bot_d, perfer_t_bot = calculate_section_similarity(256, 512 , c1[::-1], c2[::-1], high_thres = high_thres, low_thres = low_thres, pe_number = pe_number)
    # take the normal order later half, which is the original right part of the waveform. (top side)
    similarity_top, FoundRange_t, peak_top_s, peak_top_d, perfer_t_top = calculate_section_similarity(256, 512, c1, c2, high_thres = high_thres, low_thres = low_thres, pe_number = pe_number)
    
    # if use a 1-corr algorithm
    #similarity_bottom = distance_1_minus_xcorr(w1_bottom, w2_bottom)
    #similarity_top = distance_1_minus_xcorr(w1_top, w2_top)
    
    
    range_bottom = (255 - FoundRange_b[1], 255 - FoundRange_b[0])
    range_top = (256 + FoundRange_t[0], 256 + FoundRange_t[1])
    
    # assume 7 mV per PE. (can be changed somewhere, but just leave it there)
    # for resampling, we need around "nPE_DmS" more PEs for resampled result
    nPE_DmS = (peak_bot_d + peak_top_d - peak_bot_s - peak_top_s)/sPE_Amp
    # if perfer_t > 0, means more PEs are needed on the bottom side
    perfer_t = (perfer_t_top - perfer_t_bot)
    
    # Return the total similarity
    return similarity_bottom, similarity_top, range_bottom, range_top, nPE_DmS, perfer_t

def waveform_similarity_pure(w1_bottom, w1_top, w2_bottom, w2_top):
    # only take the four waveforms and only return the similarity, because we don't need to optimize here
    similarity_bottom = distance_1_minus_xcorr(w1_bottom, w2_bottom)
    similarity_top = distance_1_minus_xcorr(w1_top, w2_top)
    
    return (similarity_bottom, similarity_bottom) , (similarity_top, similarity_top)

def distance_1_minus_xcorr(wave1, wave2):
    """
    1 - 最大归一化互相关 (允许最佳时间对齐)
    做互相关求最大值(归一化后)表示相似度，把它转化成距离(=1-相似度)。
    """
    wave1 = np.array(wave1)
    wave2 = np.array(wave2)
    # 去均值
    w1 = wave1 #- np.mean(wave1)
    w2 = wave2 #- np.mean(wave2)
    # 互相关（full 模式可能长度为 511）
    cc = np.correlate(w1, w2, mode='full')
    # 归一化因子
    norm = np.sqrt(np.sum(w1**2)*np.sum(w2**2))
    if norm == 0:
        return 1.0  # 如果其中一个波形全零，直接给个最大距离
    max_corr = np.max(cc) / norm
    # 转换成距离
    return 1 - max_corr

# giving two set of waveforms on all channels, calculate the likelihood between them.
# Input waveforms from multiple LAPPDs should be in the form of:
# Waveforms_simulated[strip number][waveform_bottom, waveform_top]
# = [[waveform_bottom_strip_0, waveform_top_strip_0], [waveform_bottom_strip_1, waveform_top_strip_1], ...]
def Likelihood(Waveforms_simulated, Waveforms_data, high_thres = 5, low_thres = -3, shift_T_range = (0,256), pe_strip_converted = np.zeros(28), PureSimilarity = True):
    
    #L_best = 0  # for probability
    L_best = 1e20  # for distance
    Similarities_best = []
    shift_T_best = shift_T_range[0]
    
    DistanceNorm = 1000
    
    for shift_t in range(shift_T_range[0], shift_T_range[1]):
        Similarities = []
        for stripNum in range(len(Waveforms_simulated)):
            w1_bottom, w1_top = Waveforms_simulated[stripNum]
            w2_bottom, w2_top = Waveforms_data[stripNum]
            pe_number = pe_strip_converted[stripNum]
            #print("check", pe_number, pe_strip_converted[stripNum])
            if not PureSimilarity:
                similarity_bottom, similarity_top, range_bottom, range_top, nPE_DmS, perfer_t = WavefromSimilarity(w1_bottom, w1_top, w2_bottom, w2_top, shift_t, high_thres = high_thres, low_thres = low_thres, pe_number = pe_number)
            else:
                similarity_bottom, similarity_top = waveform_similarity_pure(w1_bottom, w1_top, w2_bottom, w2_top)
                range_bottom = (0,255)
                range_top = (0,255)
                nPE_DmS = 0
                perfer_t = 0
            
            Similarities.append([similarity_bottom, similarity_top, range_bottom, range_top, nPE_DmS, perfer_t])
        
        ######################################################################################################
        ########## To improve the Simularity to likelihood, at here ###########
        #L = 1 # for probability
        L = 0 # for distance
        waveN = 0
        for s in Similarities:
            # [0][0] bottom positive
            # [0][1] bottom negative
            # [1][0] top positive
            # [1][1] top negative
            #L += s[0][0] + s[0][1] + s[1][0] + s[1][1]
            #L += s[0][0] + s[1][0]
            D_bp = s[0][0]
            if(D_bp > DistanceNorm):
                D_bp = DistanceNorm
            D_bn = s[0][1]
            if(D_bn > DistanceNorm):
                D_bn = DistanceNorm
            D_tp = s[1][0]
            if(D_tp > DistanceNorm):
                D_tp = DistanceNorm
            D_tn = s[1][1]
            if(D_tn > DistanceNorm):
                D_tn = DistanceNorm
            
            D_positive = D_bp + D_tp
            D_negative = D_bn + D_tn
            D_total = D_positive + D_negative
            #print("\033[92m Print waveform {} \033[0m".format(waveN))
            #print("D_positive: {:.5f}, D_negative: {:.5f}, D_total: {:.5f}".format(D_positive, D_negative, D_total))
            

            P_positive = (DistanceNorm*2 - D_bp - D_tp)
            P_negative = (DistanceNorm*2 - D_bn - D_bn)
            P_total = (DistanceNorm*4 - D_bp - D_tp - D_bn - D_bn)/(DistanceNorm*4)
            #print("Got P_positive: {:.5f}, P_negative: {:.5f}, P_total: {:.5f}".format(P_positive, P_negative, P_total))
            
            #P_positive = (s[0][0] + s[1][0])/500
            #P_negative = (s[0][1] + s[1][1])/500
            
            L = L + D_total  # for distance
            #if(P_positive > 0):
                
                #L = L * P_total  # for probability
                
            waveN += 1
            #print(f"Calculating sim in Likelihood: {s[0][0]:.5f}, {s[1][0]:.5f}, {s[0][1]:.5f}, {s[1][1]:.5f}")
        ######################################################################################################
        if L < L_best: # for distance
        #if L > L_best: # for probability
            L_best = L
            Similarities_best = Similarities
            shift_T_best = shift_t
            
    print("Found shift_T_best: {}, L_best: {}".format(shift_T_best, L_best))
    
    return L_best, Similarities_best, shift_T_best

# check if the target LAPPD have hits
def HaveHits(sampled_hits_withPE, target_LAPPD_index):
    for step in range(len(sampled_hits_withPE)):
        for hit in sampled_hits_withPE[step]:
            if hit[0] == target_LAPPD_index:
                return True
    return False


def sample_updatedHits_PE_Poisson(hits_withPE):
    sampled_hits_withPE = []
    for step in range(len(hits_withPE)):
        sampled_hits_withPE.append([])
        for hit in hits_withPE[step]:
            LAPPD_index, first_index, second_index, hit_time, photon_distance, weighted_pe = hit
            #sampled_pe = np.random.poisson(weighted_pe)
            sampled_pe = round(weighted_pe)
            new_hit = (LAPPD_index, first_index, second_index, hit_time, photon_distance, weighted_pe, sampled_pe)
            ####
            sampled_hits_withPE[step].append(new_hit)
            
            
            
            
            
    return sampled_hits_withPE

# how to do the resample?
# need more/less hits on top/bottom side

def resampleHits_OneStrip(hits_withPE, re_LAPPD_index, re_strip, perferSide, perferNPE, L_0, LAPPD_profile, data_waveforms, high_thres = 5, low_thres = -3):
    
    resampled_hits_withPE = []
    resampled = False
    hits_tobeResampled = []
    FoundHitNumber = 0
    
    for step in range(len(hits_withPE)):
        resampled_hits_withPE.append([])
        resampled = False
        for hit in hits_withPE[step]:
            new_hit = hit
            if(hit[0] == re_LAPPD_index and hit[1] == re_strip and hit[5]!=0):
                hits_tobeResampled.append((int(step),hit))
                FoundHitNumber+=1
            elif (hit[5]!=0):
                resampled_hits_withPE[step].append(new_hit)

    if(FoundHitNumber != 0):
        resampled = True
    else:
        return hits_withPE, resampled
    
    # now, for selected hits waiting for resampling, do the resampling based on the perfered side and nPE
    # need "perferNPE" more PEs, better to be on "perferSide" side.
    # perferSide > 0 means more PEs on bottom side. In a hit, second index is strip number, first index is position, smaller first index is bottom side.
    # so, larger perferSide>0 means more PEs with smaller first index.
    # hit = (LAPPD_index, first_index, second_index, hit_time, photon_distance, weighted_pe, sampled_pe)
    # hits_tobeResampled = array of [(step, hit)]

    hits_tobeResampled.sort(key=lambda x: x[1][1])
    # reorder the hits in hits_tobeResampled, smaller first index first.
    sampleTimes = 0
    maxSampleTimes = 20

    while(sampleTimes < maxSampleTimes):
        sampleTimes += 1
        # now, do multiple times of resampling by using ReSampleHits
        # this will iteratively change the hits_tobeResampled. After every change, calculate likelihood until reach a upper limit of attempts.
        # Then this will return a result of "this strip was tried multiple times, if work, keep it, if doesn't, change to another strip in outer function"
        hits_afterResampled, changed = ReSampleHits(hits_tobeResampled, perferNPE, perferSide)
        if(len(hits_afterResampled) == 0):
            continue
        # push those hits back to the original array
        for hit in hits_afterResampled:
            resampled_hits_withPE[hit[0]].append(hit[1])
        LAPPD_Hit_2D_resampled, totalPE = proj.convertToHit_2D(resampled_hits_withPE, number_of_LAPPDs = 1, reSampled = True)
        Sim_Waveforms_sampled = proj.generate_lappd_waveforms(LAPPD_Hit_2D_resampled, LAPPD_profile.sPE_pulse_time, LAPPD_profile.sPE_pulse, LAPPD_profile.LAPPD_stripWidth, LAPPD_profile.LAPPD_stripSpace)
        Sim_Waveforms_sampled_converted, Data_waveforms_converted = ConvertWaveform(Sim_Waveforms_sampled, data_waveforms)
        L_this, Similarities_this, shift_T_this = Likelihood(Sim_Waveforms_sampled_converted, Data_waveforms_converted, high_thres = high_thres, low_thres = low_thres)
        if(L_this < L_0 or L_this == 0):
            break
        else:
            resampled_hits_withPE = []
            resampled = False
            hits_tobeResampled = []
            FoundHitNumber = 0
            
            for step in range(len(hits_withPE)):
                resampled_hits_withPE.append([])
                resampled = False
                for hit in hits_withPE[step]:
                    new_hit = hit
                    if(hit[0] == re_LAPPD_index and hit[1] == re_strip and hit[5]!=0):
                        hits_tobeResampled.append((step,hit))
                        FoundHitNumber+=1
                    elif (hit[5]!=0):
                        resampled_hits_withPE[step].append(new_hit)

            if(FoundHitNumber != 0):
                resampled = True
            else:
                return hits_withPE, resampled
                
    if(sampleTimes >= maxSampleTimes):
        resampled = False
        return hits_withPE, resampled
    else:
        return resampled_hits_withPE, resampled
    

def poisson_pmf(k, lam):
    """
    计算泊松分布 P(X=k) = lam^k * e^(-lam) / k!
    为避免直接计算 k! 可能导致浮点下溢/溢出，使用对数形式，然后再用 exp。
    """
    if k < 0 or lam <= 0:
        return 0.0
    # log P(X=k) = k*log(lam) - lam - log(k!)
    # math.lgamma(k+1) = log(k!)
    log_p = k * math.log(lam) - lam - math.lgamma(k+1)
    return math.exp(log_p)

def get_expected_pe(data, i):
    return data[i][1][6]

def get_sampled_pe(data, i):
    """
    data[i] = (step_i, (LAPPD_index, first_index, second_index, hit_time,
                        photon_distance, weighted_pe, sampled_pe))
    weighted_pe = hit[6]
    """
    return data[i][1][6]

def set_sampled_pe(data, i, new_pe):
    step_i, old_hit = data[i]
    hit_list = list(old_hit)
    hit_list[6] = new_pe
    new_hit = tuple(hit_list)
    data[i] = (step_i, new_hit)
    
def ReSampleHits(hits_tobeResampled, perferNPE, perferSide, seed = None):

    if seed is not None:
        random.seed(seed)

    current_total_pe = sum([h[1][6] for h in hits_tobeResampled])
    expected_total_pe = current_total_pe + perferNPE 
    n_hits = len(hits_tobeResampled)
    
    # ============ Case 1: current_total_pe < expected_total_pe ============
    if current_total_pe < expected_total_pe:
        diff_pe = expected_total_pe - current_total_pe
        # find all hits with sampledPE < expectedPE
        cands = []
        for i in range(n_hits):
            exp_i = get_expected_pe(hits_tobeResampled, i)
            samp_i = get_sampled_pe(hits_tobeResampled, i)
            if samp_i <= exp_i:
                cands.append((i, exp_i - samp_i))
        if not cands:
            return hits_tobeResampled, False
        
        i_max = max(cands, key=lambda c: c[1])[0]
        exp_i = get_expected_pe(hits_tobeResampled, i_max)
        set_sampled_pe(hits_tobeResampled, i_max, samp_i + 1)
        return hits_tobeResampled, True
        
    # ============ Case 2: current_total_pe == expected_total_pe ============
    elif current_total_pe == expected_total_pe:
        bigger_cands = []
        for i in range(n_hits):
            exp_i = get_expected_pe(hits_tobeResampled, i)
            samp_i = get_sampled_pe(hits_tobeResampled, i)
            if samp_i > exp_i:
                bigger_cands.append((i, samp_i - exp_i))
        
        if not bigger_cands:
            # all < exp_pe -> randomly choose one hit to +1
            idx = random.randrange(n_hits)
            pe_i = get_sampled_pe(hits_tobeResampled, idx)
            set_sampled_pe(hits_tobeResampled, idx, pe_i + 1)
            return hits_tobeResampled, True
        
        # sort by (samp_i - exp_i) in descending order
        bigger_cands.sort(key=lambda x:x[1], reverse=True)
        
        chosen_i = None
        for (i_idx, diff_val) in bigger_cands:
            if i_idx == 0 or i_idx == n_hits - 1:
                continue
            chosen_i = i_idx
            break
        
        if chosen_i is None:
            idx = random.randrange(n_hits)
            pe_i = get_sampled_pe(hits_tobeResampled, idx)
            set_sampled_pe(hits_tobeResampled, idx, pe_i + 1)
            return hits_tobeResampled, True
        else:
            if perferSide < 0:
                j = random.randrange(0,chosen_i)
                x_chosen = get_sampled_pe(hits_tobeResampled, chosen_i)
                x_j = get_sampled_pe(hits_tobeResampled, j)
                set_sampled_pe(hits_tobeResampled, chosen_i, x_chosen - 1)
                set_sampled_pe(hits_tobeResampled, j, x_j + 1)
            else:
                j = random.randrange(chosen_i+1, n_hits)
                x_chosen = get_sampled_pe(hits_tobeResampled, chosen_i)
                x_j = get_sampled_pe(hits_tobeResampled, j)
                set_sampled_pe(hits_tobeResampled, chosen_i, x_chosen - 1)
                set_sampled_pe(hits_tobeResampled, j, x_j + 1)
            return hits_tobeResampled, True
        
    # ============ Case 3: current_total_pe > expected_total_pe ============
    else:
        # find hits with sampledPE > expectedPE
        cand_indices = []
        for i in range(n_hits):
            exp_i = get_expected_pe(hits_tobeResampled, i)
            samp_i = get_sampled_pe(hits_tobeResampled, i)
            if samp_i > exp_i:
                cand_indices.append(i)
                
        if not cand_indices:
            idx = random.randrange(n_hits)
            pe_i = get_sampled_pe(hits_tobeResampled, idx)
            if pe_i > 0:
                set_sampled_pe(hits_tobeResampled, idx, pe_i - 1)
            return hits_tobeResampled, True
        
        # find the hits with minimum probility in cand_indices
        pmfs = []
        for i_idx in cand_indices:
            exp_i = get_expected_pe(hits_tobeResampled, i_idx)
            samp_i = get_sampled_pe(hits_tobeResampled, i_idx)
            pmfs.append((i_idx,poisson_pmf(samp_i, exp_i)))
        pmfs.sort(key=lambda x:x[1])
        
        chosen_i = None
        for (i_idx, pmf_val) in pmfs:
            if i_idx == 0 or i_idx == n_hits - 1:
                continue
            chosen_i = i_idx
            break
        
        if chosen_i is None:
            idx = random.randrange(n_hits)
            pe_i = get_sampled_pe(hits_tobeResampled, idx)
            if pe_i > 0:
                set_sampled_pe(hits_tobeResampled, idx, pe_i - 1)
            return hits_tobeResampled, True
        else:
            if perferSide < 0:
                j = random.randrange(0,chosen_i)
                x_chosen = get_sampled_pe(hits_tobeResampled, chosen_i)
                x_j = get_sampled_pe(hits_tobeResampled, j)
                set_sampled_pe(hits_tobeResampled, chosen_i, x_chosen - 1)
                set_sampled_pe(hits_tobeResampled, j, x_j + 1)
            else:
                j = random.randrange(chosen_i+1, n_hits)
                x_chosen = get_sampled_pe(hits_tobeResampled, chosen_i)
                x_j = get_sampled_pe(hits_tobeResampled, j)
                set_sampled_pe(hits_tobeResampled, chosen_i, x_chosen - 1)
                set_sampled_pe(hits_tobeResampled, j, x_j + 1)
            return hits_tobeResampled, True
                
        
        

# convert:
    # From:
    # Simulated waveform [LAPPD id][strip number][side][256] 
    # Data waveform [LAPPD id][strip number][side][256]
    # To:
    # waveform [LAPPD id * 28 + strip number][side][256]
def ConvertWaveform(Sim_Waveforms, Data_waveforms, pe_strip_samples):
    # First, check each LAPPD id, if the strip number is not 28, return error.
    Converted_Sim_Waveforms = []
    Converted_Data_waveforms = []
    Converted_stripPE = []
    for i in range(len(Sim_Waveforms)):
        if len(Sim_Waveforms[i]) != 28:
            print("ConvertWaveform Error: In Simulated waveform, LAPPD id {} has {} strips".format(i, len(Sim_Waveforms[i])))
            return Converted_Sim_Waveforms, Converted_Data_waveforms, Converted_stripPE
    for i in range(len(Data_waveforms)):
        if len(Data_waveforms[i]) != 28:
            print("ConvertWaveform Error: In Data waveform, LAPPD id {} has {} strips".format(i, len(Data_waveforms[i])))
            return Converted_Sim_Waveforms, Converted_Data_waveforms, Converted_stripPE
    for i in range(len(pe_strip_samples)):
        if len(pe_strip_samples[i]) != 28:
            print("ConvertWaveform Error: In PE strip samples, LAPPD id {} has {} strips".format(i, len(pe_strip_samples[i])))
            return Converted_Sim_Waveforms, Converted_Data_waveforms, Converted_stripPE
        
    for i in range(len(Sim_Waveforms)):
        for j in range(28):
            Converted_Sim_Waveforms.append([Sim_Waveforms[i][j][0], Sim_Waveforms[i][j][1]])
            Converted_Data_waveforms.append([Data_waveforms[i][j][0], Data_waveforms[i][j][1]])
            Converted_stripPE.append(pe_strip_samples[i][j])
            
    #print("ConvertWaveform: ",Converted_stripPE)
            
    return Converted_Sim_Waveforms, Converted_Data_waveforms, Converted_stripPE
    

def L_expected(sampled_hits_withPE, data_waveforms, LAPPD_profile, high_thres = 5, low_thres = -3, saving = False, sampling = True):
        
    # using the input hits, do a first calculation for L and similarities
    pe_list = []
    probability_list = []
    hitNum_list = []
    #LAPPD_Hit_2D_exp, pe_list, probability_list, hitNum_list = proj.convertToHit_2D_exped(sampled_hits_withPE, number_of_LAPPDs = 1, usingExpectedPEForEachHit = True)
    #LAPPD_Hit_2D_exp, pe_list, probability_list, hitNum_list = proj.convertToHit_2D_perStepSlice_exp(sampled_hits_withPE, number_of_LAPPDs = 1, usingExpectedPEForEachHit = not sampling)
    LAPPD_Hit_2D_exp, pe_list, probability_list, hitNum_list = proj.convertToHit_2D_perTimeSlice_exp(sampled_hits_withPE, number_of_LAPPDs = 1, usingExpectedPEForEachHit = not sampling)
    

    print("L_exp, get assigned PE list: ", pe_list)
    print("Probability_list: ", probability_list)
    print("hitNum_list: ", hitNum_list)
    #print("LAPPD_Hit_2D_exp[id0][strip0] print:", LAPPD_Hit_2D_exp[0][0])
    
    Info = [pe_list, probability_list, hitNum_list]
    
    Sim_Waveforms_sampled, pe_strip_samples = proj.generate_lappd_waveforms(LAPPD_Hit_2D_exp, LAPPD_profile.sPE_pulse_time, LAPPD_profile.sPE_pulse, LAPPD_profile.LAPPD_stripWidth, LAPPD_profile.LAPPD_stripSpace,  generatePE = True)
    Sim_Waveforms_sampled_converted, Data_waveforms_converted, pe_strip_converted = ConvertWaveform(Sim_Waveforms_sampled, data_waveforms, pe_strip_samples)
    L_final, Similarities_final, shift_T_final = Likelihood(Sim_Waveforms_sampled_converted, Data_waveforms_converted, high_thres = high_thres, low_thres = low_thres, pe_strip_converted = pe_strip_converted, PureSimilarity = False)

    printTest = saving
    if(printTest):
        save_directory = "/Users/fengy/ANNIESofts/Analysis/ProjectionComplete/OptimizationResults/3.SamplingTest"
        base_filename  = "WithSampling_"
        if not sampling:
            base_filename  = "NoSampling_"
        extension      = ".txt"

        file_index = 0
        save_path = os.path.join(save_directory, f"{base_filename}{file_index}{extension}")
        while os.path.exists(save_path):
            file_index += 1
            save_path = os.path.join(save_directory, f"{base_filename}{file_index}{extension}")
            
        if file_index > 5000:
            return L_final, Similarities_final, shift_T_final, pe_list, LAPPD_Hit_2D_exp, Info
        with open(save_path, 'w') as f:
            for i in range(28):  
                w1_bottom, w1_top = Sim_Waveforms_sampled_converted[i]
                w2_bottom, w2_top = Data_waveforms_converted[i]
                c1 = np.array(combineWaveform(w1_bottom, w1_top, shift_T_final))
                c2 = np.array(combineWaveform(w2_bottom, w2_top, 0))
                c1_str = ", ".join(f"{x:.2f}" for x in c1)
                c2_str = ", ".join(f"{x:.2f}" for x in c2)
                f.write(c1_str + "\n")
                f.write(c2_str + "\n")
        print("Saved to ", save_path)   
        
    return L_final, Similarities_final, shift_T_final, pe_list, LAPPD_Hit_2D_exp, Info



def OptimizeL(sampled_hits_withPE, re_LAPPD_index, data_waveforms, LAPPD_profile, max_resample_strip_num = 10, max_resample_total = 10, high_thres = 5, low_thres = -3):
    
    resampledHits_withPE_final = sampled_hits_withPE
    
    opt_time = 0
    opt_limit = max_resample_total #
    oneStrip_opt_limit = max_resample_strip_num
    
    
    # using the input hits, do a first calculation for L and similarities
    LAPPD_Hit_2D_resampled, totalPE = proj.convertToHit_2D(sampled_hits_withPE, number_of_LAPPDs = 1, reSampled = True)
    Sim_Waveforms_sampled = proj.generate_lappd_waveforms(LAPPD_Hit_2D_resampled, LAPPD_profile.sPE_pulse_time, LAPPD_profile.sPE_pulse, LAPPD_profile.LAPPD_stripWidth, LAPPD_profile.LAPPD_stripSpace)
    Sim_Waveforms_sampled_converted, Data_waveforms_converted = ConvertWaveform(Sim_Waveforms_sampled, data_waveforms)
    L_final, Similarities_final, shift_T_final = Likelihood(Sim_Waveforms_sampled_converted, Data_waveforms_converted, high_thres = high_thres, low_thres = low_thres)
    L_0 = L_final
    Similarities_0 = Similarities_final
    shift_T_0 = shift_T_final
    
    reached_opt_limit = False
    # based on the first calculation, optimize the sampling on each strip.
    # first, find the worst strip, 
    while(opt_time < opt_limit and (not reached_opt_limit)):
        print(f"\033[93mOptimizeL, opt_time: {opt_time}, L: {L_final}\033[0m")
        
        printDegubInfo = False
        if(printDegubInfo):
            print("Printing range and Similarities_final")
            #Similarities.append([similarity_bottom, similarity_top, range_bottom, range_top, nPE_DmS, perfer_t])
            simBot = [i[0] for i in Similarities_final]
            simTop = [i[1] for i in Similarities_final]
            rangeBot = [i[2]for i in Similarities_final]
            rangeTop = [i[3] for i in Similarities_final]
            PENum = []
            
            for i in range(len(Similarities_final)):
                num = 0
                for hits in resampledHits_withPE_final:
                    for hit in hits:
                        if hit[0] == re_LAPPD_index and hit[2] == i:
                            num += hit[6]
                PENum.append(num)
            
            for i in range(len(Similarities_final)):
                simBot_i = simBot[i]
                simTop_i = simTop[i]
                rangeBot_i = rangeBot[i]
                rangeTop_i = rangeTop[i]
                peN = PENum[i]
                print("Strip: {}, pe number: {}, simBot: {}, simTop: {}, rangeBot: {}, rangeTop: {}".format(i, peN, simBot_i, simTop_i, rangeBot_i, rangeTop_i))
            
        # find the worst strip
        Similarities = [i[0]+i[1] for i in Similarities_final]
        sorted_indices = sorted(
            range(len(Similarities_final)),
            key=lambda idx: Similarities_final[idx][0]+Similarities_final[idx][1],
            reverse=True
        )
        
        #loop 
        for i in range(len(sorted_indices)):
            
            if(i >= oneStrip_opt_limit):
                reached_opt_limit = True
                break
            
            opt_strip = sorted_indices[i]
        
            #[similarity_bottom, similarity_top, range_bottom, range_top, nPE_DmS, perfer_t]
            perferSide = Similarities_final[opt_strip][5]
            perferNPE = Similarities_final[opt_strip][4]
            
            print("OptimizeL, resampling strip:", opt_strip, "perferSide:", perferSide, "perferNPE:", perferNPE)
            
            
            # do resample for this strip
            resampled_hits_withPE, reSmapled = resampleHits_OneStrip(resampledHits_withPE_final, re_LAPPD_index, opt_strip, perferSide, perferNPE, L_final, LAPPD_profile, data_waveforms, high_thres = 5, low_thres = -3)
            if (not reSmapled):
                continue
            LAPPD_Hit_2D_resampled, totalPE = proj.convertToHit_2D(resampled_hits_withPE, number_of_LAPPDs = 1, reSampled = True)
            Sim_Waveforms_sampled = proj.generate_lappd_waveforms(LAPPD_Hit_2D_resampled, LAPPD_profile.sPE_pulse_time, LAPPD_profile.sPE_pulse, LAPPD_profile.LAPPD_stripWidth, LAPPD_profile.LAPPD_stripSpace)
            Sim_Waveforms_sampled_converted, Data_waveforms_converted = ConvertWaveform(Sim_Waveforms_sampled, data_waveforms)
            L_this, Similarities_this, shift_T_this = Likelihood(Sim_Waveforms_sampled_converted, Data_waveforms_converted, high_thres = high_thres, low_thres = low_thres)
            
            if(L_this < L_final and L_this!= 0):
                resampledHits_withPE_final = resampled_hits_withPE
                L_final = L_this
                Similarities_final = Similarities_this
                shift_T_final = shift_T_this
                break
            

        opt_time += 1
        
    if(reached_opt_limit):
        print("OptimizeL reached opt limit.")
        return sampled_hits_withPE, L_0, Similarities_0, shift_T_0, False
    else:
        return resampledHits_withPE_final, L_final, Similarities_final, shift_T_final, True


# TO BE DELETED
# previous function, not needed anymore
# do resmapling for this projection step, until the likelihood can't be improved or the max resample time is reached.
def OptimizeLInThisStep(sampled_hits_withPE, L_0, Similarities_0, re_LAPPD_index, data_waveforms, LAPPD_profile, max_resample_strip_num = 5, max_resample_one_strip = 10, high_thres = 5, low_thres = -3):
    
    # Similarities_0 = array of [similarity_bottom, similarity_top, range_bottom, range_top, nPE_DmS, perfer_t]
    # the converted waveform index is [LAPPD id * 28 + strip number][side][256]
    # the similarities index is also [LAPPD id * 28 + strip number]
    # initialize the final result
    resampled_hits_withPE_final = []
    L_final = L_0
    Similarities_final = Similarities_0
    shift_T_final = 0
    
    # initialize the resample counter
    not_improve_resampled_strip_num = 0
    totalResampleTime = 0
    lastStepWasImproved = True
    CheckedStrips = []
    
    L_prev = L_0
    L_this = L_prev
    print("Start L: {}".format(L_prev))
    reSampleTime = 0
    FirstResample = True
    L_stable_check = 0
    
    while (not_improve_resampled_strip_num < max_resample_strip_num and totalResampleTime < 100):
        print("Resample Time: {}".format(totalResampleTime))
        
        
        Similarities_this = []
        reSampleTime = 0
        
        if(lastStepWasImproved):
            print("Improved, clean the checked strips", totalResampleTime)
            CheckedStrips = []
            
        while (FirstResample or L_this>L_prev or (reSampleTime < max_resample_one_strip)):
            FirstResample = False
            Similarities = [i[0]+i[1] for i in Similarities_0]
            #worst_strip, info = max(enumerate(Similarities), key=lambda x: x[1])
            if(len(CheckedStrips) >= max_resample_strip_num):
                break
            worst_strip, info = max(
                ((i, val) for i, val in enumerate(Similarities) if i not in CheckedStrips),
                key=lambda x: x[1]
            )
            resampled_hits_withPE, reSmapled = resampleHits_OneStrip(sampled_hits_withPE, re_LAPPD_index, worst_strip, L_0, LAPPD_profile, data_waveforms)
            if(not reSmapled):
                break
            LAPPD_Hit_2D_resampled, totalPE = proj.convertToHit_2D(resampled_hits_withPE, number_of_LAPPDs = 1, reSampled = True)
            Sim_Waveforms_sampled = proj.generate_lappd_waveforms(LAPPD_Hit_2D_resampled, LAPPD_profile.sPE_pulse_time, LAPPD_profile.sPE_pulse, LAPPD_profile.LAPPD_stripWidth, LAPPD_profile.LAPPD_stripSpace)
            Sim_Waveforms_sampled_converted, Data_waveforms_converted = ConvertWaveform(Sim_Waveforms_sampled, data_waveforms)
            L_this, Similarities_this, shift_T_this = Likelihood(Sim_Waveforms_sampled_converted, Data_waveforms_converted, high_thres = high_thres, low_thres = low_thres)
            print("Resample Strip: {}".format(worst_strip),", reSampleTime: {}".format(reSampleTime), ", L_this: {}".format(L_this), ", shift_T_this: {}".format(shift_T_this))
            
            '''    
            LHit2D, tPE = proj.convertToHit_2D(sampled_hits_withPE, number_of_LAPPDs = 1)
            Sim_s = proj.generate_lappd_waveforms(LHit2D, LAPPD_profile.sPE_pulse_time, LAPPD_profile.sPE_pulse, LAPPD_profile.LAPPD_stripWidth, LAPPD_profile.LAPPD_stripSpace)
            Sim_s_converted, Data_waveforms_converted = ConvertWaveform(Sim_s, data_waveforms)
            L_this_0, Similarities_this_0, shift_T_this_0 = Likelihood(Sim_s_converted, Data_waveforms_converted, high_thres = high_thres, low_thres = low_thres)
            print("L_this_0: {}".format(L_this_0), ", shift_T_this_0: {}".format(shift_T_this_0))
            '''  
            reSampleTime += 1
            
            if(L_this < L_prev):
                L_final = L_this
                L_prev = L_this
                Similarities_final = Similarities_this
                sampled_hits_withPE = resampled_hits_withPE
                resampled_hits_withPE_final = resampled_hits_withPE
                shift_T_final = shift_T_this
                lastStepWasImproved = True
                print("Improved")
            else:
                lastStepWasImproved = False
                CheckedStrips.append(worst_strip)
                print("Not improved, CheckedStrips: ", CheckedStrips)
                L_stable_check += 1
            
            
        totalResampleTime += 1
        #print(reSampleTime, max_resample_strip_num, lastStepWasImproved)
        
        if(reSampleTime >= max_resample_strip_num and not lastStepWasImproved):
            CheckedStrips.append(worst_strip)
            not_improve_resampled_strip_num += 1
            
        if(L_stable_check >= 5):
            break
            
        
    return resampled_hits_withPE_final, L_final, Similarities_final, shift_T_final, totalResampleTime
        
def MuonOptimization_expected(LAPPD_profile, mu_input, data_waveforms, dx, dy, dz, dtheta, dphi, maxIterStep_xyz = 10, shrinkStepThreshold = 0.005, shrinkStepRatio = 0.8, high_thres = 5, low_thres = -3, makeGridL = False, mu_step = 0.01, phi_steps = 360, sampling = True):
    mu_optimization_chain = []
    Final_hits = []
    

    print("Start Optimization")
    mu_position, mu_direction = mu_input
    sampled_hits_withPE = DoProjection(LAPPD_profile, mu_position, mu_direction, muon_step = mu_step, phi_steps=phi_steps)
    L_0, Similarities_0, shift_T_0, pe_list_0, LAPPD_Hit_2D_exp_0, Info =  L_expected(sampled_hits_withPE, data_waveforms, LAPPD_profile, saving = False, sampling = sampling)
    mu_optimization_chain.append([mu_position, mu_direction, L_0, Similarities_0, shift_T_0, [dx, dy, dz,0,0], Info, LAPPD_Hit_2D_exp_0])
    
    best_L = L_0
    best_Similarities = Similarities_0
    bset_shiftT = shift_T_0
    
    if (not makeGridL):
        print(f"Start Optimization with L_0 = {best_L}")
        
        # now, start from mu_position to optimize
        axes = ["x","y"]
        steps = [dx, dy]
        for iter_l in range(maxIterStep_xyz):
            print(f"\n=== Iteration {iter_l} ===")
            improved_global = False
            
            dtheta = 0
            dphi = 0   

            for ax_i, ax_name in enumerate(axes):
                step_val = steps[ax_i]
                if abs(step_val) < shrinkStepThreshold:
                    continue
                
                base_x, base_y, base_z = mu_position[0], mu_position[1], mu_position[2]
                base_L = L_0
                base_theta, base_phi = (0,0)
                
                # try positive direction
                if ax_name == "x":
                    cand_x, cand_y, cand_z = base_x + step_val, base_y, base_z
                elif ax_name == "y":
                    cand_x, cand_y, cand_z = base_x, base_y + step_val, base_z
                else:  # "z"
                    cand_x, cand_y, cand_z = base_x, base_y, base_z + step_val
                    
                # find the best phi and theta
                theta_plus, phi_plus, L_plus, Similarities_plus, shiftT_plus, hit2D_plus = step_search_theta_phi_expected(cand_x, cand_y, cand_z, mu_direction, LAPPD_profile, data_waveforms, max_iter = 1)
                    
                if ax_name == "x":
                    cand_x2, cand_y2, cand_z2 = base_x - step_val, base_y, base_z
                elif ax_name == "y":
                    cand_x2, cand_y2, cand_z2 = base_x, base_y - step_val, base_z
                else:  # "z"
                    cand_x2, cand_y2, cand_z2 = base_x, base_y, base_z - step_val
                    
                theta_minus, phi_minus, L_minus, Similarities_minus, shiftT_minus, hit2D_minus = step_search_theta_phi_expected(cand_x2, cand_y2, cand_z2, mu_direction, LAPPD_profile, data_waveforms, max_iter = 1)
                
                new_positions = [
                    (cand_x, cand_y, cand_z, theta_plus, phi_plus, L_plus, Similarities_plus, shiftT_plus, step_val),
                    (cand_x2, cand_y2, cand_z2, theta_minus, phi_minus, L_minus, Similarities_minus, shiftT_minus, -step_val)
                ]
                #new_positions.sort(key=lambda item: item[5], reverse=True) 
                new_positions.sort(key=lambda item: item[5]) 

                best_candidate = new_positions[0]  # find the smallest L
                cand_L = best_candidate[5]
                if cand_L == 0:
                    best_candidate = new_positions[1]
                    cand_L = best_candidate[5]
                    
                cand_Sim = best_candidate[6]
                cand_shiftT = best_candidate[7]
                cand_theta = best_candidate[3]
                dtheta = cand_theta
                cand_phi = best_candidate[4]
                dphi = cand_phi
                
                if cand_L > best_L and cand_L != 0:
                    mu_position = (best_candidate[0], best_candidate[1], best_candidate[2])
                    mu_direction = proj.rotate_vector(mu_direction, cand_theta, cand_phi)
                    best_L = cand_L
                    best_Similarities = cand_Sim
                    bset_shiftT = cand_shiftT
                    improved_global = True
                    delta = best_candidate[8]
                    #print(f"  Axis {ax_name}, step=+/-{step_val} found better L={best_L:.3f}")
                    print(f"\033[92m  Axis {ax_name}, delta = {delta:.4f}, step=+/-{step_val:.4f} found better L={best_L:.8f}, dtheta = {cand_theta:.3f}, dphi = {cand_phi:.3f}\033[0m")

                else:
                    steps[ax_i] = step_val * shrinkStepRatio
                    #print(f"  Axis {ax_name}, step=+/-{step_val} not improved, shrink step to {steps[ax_i]}")
                    print(f"\033[91m  Axis {ax_name}, step=+/-{step_val:.4f} not improved, L = {cand_L:.4f}, shrink step to {steps[ax_i]:.4f}, dtheta = {cand_theta:.3f}, dphi = {cand_phi:.3f}\033[0m")

            mu_optimization_chain.append([mu_position, mu_direction, best_L, best_Similarities, bset_shiftT, [dx, dy, dz, dtheta, dphi]])
            
            print("Iteration", iter_l, "best L:", best_L)
            #print("mu_optimization_chain:", mu_optimization_chain)
            
            if all(abs(step) < shrinkStepThreshold for step in steps):
                print("All steps are smaller than ", shrinkStepThreshold, ", break.")
                break
            
    else:
        # make a grid of likelihood
        gridStep = 0.03
        stepNumber = 0.5
        
        offset = 0
        
        mu_position, mu_direction = mu_input
        
        
        print("Start generating grid with center at ", mu_position)
        
        for xStep in range(0,int(stepNumber*2)):
            for yStep in range(0, int(stepNumber*2)):
                print("xStep: ", xStep, "yStep: ", yStep)
                shiftedPosition = [mu_position[0] + offset + (xStep - stepNumber) * gridStep, mu_position[1] + offset + (yStep - stepNumber) * gridStep, mu_position[2]]
                sampled_hits_withPE = DoProjection(LAPPD_profile, shiftedPosition, mu_direction, phi_steps=phi_steps)
                L_xy, Similarities_0, shift_T_0, pe_list_0, LAPPD_Hit_2D_exp_0, Info =  L_expected(sampled_hits_withPE, data_waveforms, LAPPD_profile, saving = True, sampling = sampling)
                mu_optimization_chain.append([mu_position, mu_direction, L_xy, Similarities_0, shift_T_0, [dx, dy, dz,0,0], Info, LAPPD_Hit_2D_exp_0])
                
            #print("this x finished, L in last step: ", L_xy)
            
        improved_global = True

    return mu_optimization_chain, improved_global





# new optimization function.
# at each x and y, optimize the theta and phi first to find the best theta and phi.
# Use OptimizeL
def MuonOptimization(LAPPD_profile, mu_input, data_waveforms, dx, dy, dz, dtheta, dphi, maxIterStep_xyz = 20, shrinkStepThreshold = 0.005, shrinkStepRatio = 0.6, high_thres = 5, low_thres = -3):
    mu_optimization_chain = []
    Final_hits = []
    
    print("Start Optimization")
    mu_position, mu_direction = mu_input
    sampled_hits_withPE = DoProjection(LAPPD_profile, mu_position, mu_direction)
    resampledHits_withPE_0, L_0, Similarities_0, shift_T_0, ResampleCondition =  OptimizeL(sampled_hits_withPE, 0, data_waveforms, LAPPD_profile)
    mu_optimization_chain.append([mu_position, mu_direction, L_0, Similarities_0, shift_T_0, [dx, dy, dz,0,0]])
    best_L = L_0
    best_Similarities = Similarities_0
    bset_shiftT = shift_T_0
    best_hits = resampledHits_withPE_0
    
    
    # now, start from mu_position to optimize
    axes = ["x","y"]
    steps = [dx, dy]
    for iter_l in range(maxIterStep_xyz):
        print(f"\n=== Iteration {iter_l} ===")
        improved_global = False
        
        dtheta = 0
        dphi = 0
        
        for ax_i, ax_name in enumerate(axes):
            step_val = steps[ax_i]
            if abs(step_val) < shrinkStepThreshold:
                continue
            
            base_x, base_y, base_z = mu_position[0], mu_position[1], mu_position[2]
            base_L = L_0
            base_theta, base_phi = (0,0)
            
            # try positive direction
            if ax_name == "x":
                cand_x, cand_y, cand_z = base_x + step_val, base_y, base_z
            elif ax_name == "y":
                cand_x, cand_y, cand_z = base_x, base_y + step_val, base_z
            else:  # "z"
                cand_x, cand_y, cand_z = base_x, base_y, base_z + step_val
                
            # find the best phi and theta
            theta_plus, phi_plus, L_plus, Similarities_plus, shiftT_plus, resampledHits_withPE_plus = step_search_theta_phi(cand_x, cand_y, cand_z, mu_direction, LAPPD_profile, data_waveforms, max_iter = 5)
            
            if ax_name == "x":
                cand_x2, cand_y2, cand_z2 = base_x - step_val, base_y, base_z
            elif ax_name == "y":
                cand_x2, cand_y2, cand_z2 = base_x, base_y - step_val, base_z
            else:  # "z"
                cand_x2, cand_y2, cand_z2 = base_x, base_y, base_z - step_val
                
            theta_minus, phi_minus, L_minus, Similarities_minus, shiftT_minus, resampledHits_withPE_minus = step_search_theta_phi(cand_x2, cand_y2, cand_z2, mu_direction, LAPPD_profile, data_waveforms, max_iter = 5)
            
            new_positions = [
                (cand_x, cand_y, cand_z, theta_plus, phi_plus, L_plus, Similarities_plus, shiftT_plus, resampledHits_withPE_plus),
                (cand_x2, cand_y2, cand_z2, theta_minus, phi_minus, L_minus, Similarities_minus, shiftT_minus, resampledHits_withPE_minus)
            ]
            new_positions.sort(key=lambda item: item[5]) 

            best_candidate = new_positions[0]  # find the smallest L
            cand_L = best_candidate[5]
            if cand_L == 0:
                best_candidate = new_positions[1]
                cand_L = best_candidate[5]
                
            cand_Sim = best_candidate[6]
            cand_shiftT = best_candidate[7]
            cand_theta = best_candidate[3]
            dtheta = cand_theta
            cand_phi = best_candidate[4]
            dphi = cand_phi
            cand_hits = best_candidate[8]
            
            if cand_L < best_L and cand_L != 0:
                mu_position = (best_candidate[0], best_candidate[1], best_candidate[2])
                mu_direction = proj.rotate_vector(mu_direction, cand_theta, cand_phi)
                best_L = cand_L
                best_Similarities = cand_Sim
                bset_shiftT = cand_shiftT
                best_hits = cand_hits
                improved_global = True
                #print(f"  Axis {ax_name}, step=+/-{step_val} found better L={best_L:.3f}")
                print(f"\033[92m  Axis {ax_name}, step=+/-{step_val} found better L={best_L:.3f}, dtheta = {cand_theta:.3f}, dphi = {cand_phi:.3f}\033[0m")

            else:
                steps[ax_i] = step_val * shrinkStepRatio
                #print(f"  Axis {ax_name}, step=+/-{step_val} not improved, shrink step to {steps[ax_i]}")
                print(f"\033[91m  Axis {ax_name}, step=+/-{step_val} not improved, shrink step to {steps[ax_i]}, dtheta = {cand_theta:.3f}, dphi = {cand_phi:.3f}\033[0m")

        mu_optimization_chain.append([mu_position, mu_direction, best_L, best_Similarities, bset_shiftT, [dx, dy, dz, dtheta, dphi]])
        
        print("Iteration", iter_l, "best L:", best_L)
        #print("mu_optimization_chain:", mu_optimization_chain)
        
        if all(abs(step) < shrinkStepThreshold for step in steps):
            print("All steps are smaller than ", shrinkStepThreshold, ", break.")
            break
        
    return mu_optimization_chain, best_hits, improved_global
                
def step_search_theta_phi_expected(x, y, z, mu_direction, LAPPD_profile, data_waveforms, step_theta = 1.0, step_phi = 1.0, min_angle = 0.05, angleStepShrinkRatio = 0.5, max_iter = 5):
    best_theta = 0
    best_phi = 0

    mu_position = (x, y, z)
    start_hits_withPE = DoProjection(LAPPD_profile, mu_position, mu_direction)
    L_0, Similarities_0, shift_T_0, pe_list_0, LAPPD_Hit_2D_exp_0, Info =  L_expected(start_hits_withPE, data_waveforms, LAPPD_profile)
    
    iteration = 0
    best_L = L_0
    best_Sim = Similarities_0
    best_shiftT = shift_T_0   
    best_hits_2D = []
    
    #print(f"Search tf, find best L {best_L}")
    
    return best_theta, best_phi, best_L, best_Sim, best_shiftT, best_hits_2D
    
    print("searching theta and phi")
    
    while step_theta > min_angle and step_phi > min_angle and iteration < max_iter:
        iteration += 1
        improved = False
        
        candidates = []

        mu_direction_theta_plus = proj.rotate_vector(mu_direction, step_theta, 0)
        hits_theta_plus = DoProjection(LAPPD_profile, mu_position, mu_direction_theta_plus)
        hasHits = HaveHits(hits_theta_plus, 0)
        if(hasHits):
            L_theta_plus, Similarities_theta_plus, shift_T_theta_plus, pe_list_theta_plus, LAPPD_Hit_2D_exp_theta_plus =  L_expected(hits_theta_plus, data_waveforms, LAPPD_profile)
            candidates.append((best_theta + step_theta, best_phi, L_theta_plus, Similarities_theta_plus, shift_T_theta_plus, LAPPD_Hit_2D_exp_theta_plus))
        else:
            candidates.append((best_theta + step_theta, best_phi, 1e20, [], 0, []))  
            
        mu_direction_theta_minus = proj.rotate_vector(mu_direction, -step_theta, 0)
        hits_theta_minus = DoProjection(LAPPD_profile, mu_position, mu_direction_theta_minus)
        hasHits = HaveHits(hits_theta_minus, 0)
        if(hasHits):
            L_theta_minus, Similarities_theta_minus, shift_T_theta_minus, pe_list_theta_minus, LAPPD_Hit_2D_exp_theta_minus =  L_expected(hits_theta_minus, data_waveforms, LAPPD_profile)
            candidates.append((best_theta - step_theta, best_phi, L_theta_minus, Similarities_theta_minus, shift_T_theta_minus, LAPPD_Hit_2D_exp_theta_minus))
        else:
            candidates.append((best_theta - step_theta, best_phi, 1e20, [], 0, []))   
            
        mu_direction_phi_plus = proj.rotate_vector(mu_direction, 0, step_phi)
        hits_phi_plus = DoProjection(LAPPD_profile, mu_position, mu_direction_phi_plus)
        hasHits = HaveHits(hits_phi_plus, 0)
        if(hasHits):
            L_phi_plus, Similarities_phi_plus, shift_T_phi_plus, pe_list_phi_plus, LAPPD_Hit_2D_exp_phi_plus =  L_expected(hits_phi_plus, data_waveforms, LAPPD_profile)
            candidates.append((best_theta, best_phi + step_phi, L_phi_plus, Similarities_phi_plus, shift_T_phi_plus, LAPPD_Hit_2D_exp_phi_plus))
        else:
            candidates.append((best_theta, best_phi + step_phi, 1e20, [], 0, []))
            
        mu_direction_phi_minus = proj.rotate_vector(mu_direction, 0, -step_phi)
        hits_phi_minus = DoProjection(LAPPD_profile, mu_position, mu_direction_phi_minus)
        hasHits = HaveHits(hits_phi_minus, 0)
        if(hasHits):
            L_phi_minus, Similarities_phi_minus, shift_T_phi_minus, pe_list_phi_minus, LAPPD_Hit_2D_exp_phi_minus =  L_expected(hits_phi_minus, data_waveforms, LAPPD_profile)
            candidates.append((best_theta, best_phi - step_phi, L_phi_minus, Similarities_phi_minus, shift_T_phi_minus, LAPPD_Hit_2D_exp_phi_minus))
        else:
            candidates.append((best_theta, best_phi - step_phi, 1e20, [], 0, []))
        
        
        candidates.sort(key=lambda item: item[2])
        cand_theta, cand_phi, cand_L, cand_Sim, cand_shiftT, cand_hits_2D = candidates[0]
        
        if cand_L < best_L and cand_L != 0:
            best_theta = cand_theta
            best_phi = cand_phi
            best_L = cand_L
            best_Sim = cand_Sim
            best_shiftT = cand_shiftT
            best_hits_2D = cand_hits_2D
            improved = True
            print(f"  Iter {iteration}, theta={best_theta:.1f}, phi={best_phi:.1f}, L={best_L:.3f}, L updated")
        else: 
            step_theta *= angleStepShrinkRatio
            step_phi *= angleStepShrinkRatio
            print(f"  Iter {iteration}, theta={best_theta:.1f}, phi={best_phi:.1f}, L={best_L:.3f}, shrink step to {step_theta:.3f}")
            
    return best_theta, best_phi, best_L, best_Sim, best_shiftT, best_hits_2D


                
# given xyz and start mu direction, rotate the muon track to step search a best angle of theta and phi.        
def step_search_theta_phi(x, y, z, mu_direction, LAPPD_profile, data_waveforms, step_theta = 1.0, step_phi = 1.0, min_angle = 0.05, angleStepShrinkRatio = 0.5, max_iter = 3):
    best_theta = 0
    best_phi = 0
    
    mu_position = (x, y, z)
    start_hits_withPE = DoProjection(LAPPD_profile, mu_position, mu_direction)
    resampledHits_withPE_0, L_0, Similarities_0, shift_T_0, ResampleCondition =  OptimizeL(start_hits_withPE, 0, data_waveforms, LAPPD_profile)

    iteration = 0
    
    best_L = L_0
    best_hits = resampledHits_withPE_0
    best_Sim = Similarities_0
    best_shiftT = shift_T_0
    
    while step_theta > min_angle and step_phi > min_angle and iteration < max_iter:
        iteration += 1
        improved = False
        #new_mu_direction = proj.rotate_vector(mu_direction, theta, phi)
        
        candidates = []
        
        # theta + step angle
        mu_direction_theta_plus = proj.rotate_vector(mu_direction, step_theta, 0)
        hits_theta_plus = DoProjection(LAPPD_profile, mu_position, mu_direction_theta_plus)
        hasHits = HaveHits(hits_theta_plus, 0)
        if(hasHits):
            resampledHits_withPE_theta_plus, L_theta_plus, Similarities_theta_plus, shift_T_theta_plus, Resample_tplus =  OptimizeL(hits_theta_plus, 0, data_waveforms, LAPPD_profile)
            candidates.append((best_theta + step_theta, best_phi, L_theta_plus, Similarities_theta_plus, shift_T_theta_plus, resampledHits_withPE_theta_plus))
        else:
            candidates.append((best_theta + step_theta, best_phi, 1e20, [], 0, []))
            
            
        # theta - step angle
        mu_direction_theta_minus = proj.rotate_vector(mu_direction, -step_theta, 0)
        hits_theta_minus = DoProjection(LAPPD_profile, mu_position, mu_direction_theta_minus)
        hasHits = HaveHits(hits_theta_minus, 0)
        if(hasHits):
            resampledHits_withPE_theta_minus, L_theta_minus, Similarities_theta_minus, shift_T_theta_minus, Resample_tminus =  OptimizeL(hits_theta_minus, 0, data_waveforms, LAPPD_profile)
            candidates.append((best_theta - step_theta, best_phi, L_theta_minus, Similarities_theta_minus, shift_T_theta_minus, resampledHits_withPE_theta_minus))
        else:
            candidates.append((best_theta - step_theta, best_phi, 1e20, [], 0, []))
            
        
        # phi + step angle
        mu_direction_phi_plus = proj.rotate_vector(mu_direction, 0, step_phi)
        hits_phi_plus = DoProjection(LAPPD_profile, mu_position, mu_direction_phi_plus)
        hasHits = HaveHits(hits_phi_plus, 0)
        if(hasHits):
            resampledHits_withPE_phi_plus, L_phi_plus, Similarities_phi_plus, shift_T_phi_plus, Resample_pplus =  OptimizeL(hits_phi_plus, 0, data_waveforms, LAPPD_profile)
            candidates.append((best_theta, best_phi + step_phi, L_phi_plus, Similarities_phi_plus, shift_T_phi_plus, resampledHits_withPE_phi_plus))
        else:
            candidates.append((best_theta, best_phi + step_phi, 1e20, [], 0, []))
            
        # phi - step angle
        mu_direction_phi_minus = proj.rotate_vector(mu_direction, 0, -step_phi)
        hits_phi_minus = DoProjection(LAPPD_profile, mu_position, mu_direction_phi_minus)
        hasHits = HaveHits(hits_phi_minus, 0)
        if(hasHits):
            resampledHits_withPE_phi_minus, L_phi_minus, Similarities_phi_minus, shift_T_phi_minus, Resample_pminus =  OptimizeL(hits_phi_minus, 0, data_waveforms, LAPPD_profile)
            candidates.append((best_theta, best_phi - step_phi, L_phi_minus, Similarities_phi_minus, shift_T_phi_minus, resampledHits_withPE_phi_minus))
        else:
            candidates.append((best_theta, best_phi - step_phi, 1e20, [], 0, []))
        
        
        candidates.sort(key=lambda item: item[2])
        cand_theta, cand_phi, cand_L, cand_Sim, cand_shiftT, cand_hits = candidates[0]
        
        if cand_L < best_L and cand_L != 0:
            best_theta = cand_theta
            best_phi = cand_phi
            best_L = cand_L
            best_Sim = cand_Sim
            best_shiftT = cand_shiftT
            best_hits = cand_hits
            improved = True
            print(f"  Iter {iteration}, theta={best_theta:.1f}, phi={best_phi:.1f}, L={best_L:.3f}")
        else: 
            step_theta *= angleStepShrinkRatio
            step_phi *= angleStepShrinkRatio
            
    return best_theta, best_phi, best_L, best_Sim, best_shiftT, best_hits
    

# the most outer loop of the optimization.
# take the seed of muon parameter, and the data waveforms as input.
# return the optimization parameter chain, each step contains the muon parameter and the likelihood value.
# also return the last result of projected hits on LAPPD surface.
# LAPPD_profile = dc.LAPPD_profile(absorption_wavelengths,absorption_coefficients,qe_2d,gain_2d,QEvsWavelength_lambda,QEvsWavelength_QE,10,1,LAPPD_grids,sPE_pulse_time,sPE_pulse,LAPPD_stripWidth,LAPPD_stripSpace)
def MuonOptimization_old(LAPPD_profile, mu_input, data_waveforms, dx, dy, dz, dtheta, dphi, epsilon, maxStepNumber, shrinkStepThreshold = 0.3, shrinkStepRatio = 0.5, high_thres = 5, low_thres = -3):
    mu_optimization_chain = []
    Final_hits = []
    
    DeltaX = np.array([0, 0, 0, 0, 0], dtype=float)
    
    # do step L_0
    print("Start L_0")
    mu_position, mu_direction = mu_input
    sampled_hits_withPE = DoProjection(LAPPD_profile, mu_position, mu_direction)
    resampledHits_withPE_0, L_0_n, Similarities_0, shift_T_0, LWasOptimized = OptimizeL(sampled_hits_withPE, 0, data_waveforms, LAPPD_profile, high_thres = high_thres, low_thres = low_thres)
    #sampled_hits_withPE[muon step][hit], hit = (LAPPD_index, first_index, second_index, hit_time, photon_distance, weighted_pe, sampled_pe)
    #LAPPD_Hit_2D_sampled, totalPE = proj.convertToHit_2D(sampled_hits_withPE, number_of_LAPPDs = 1, reSampled = True)
    #Sim_Waveforms_sampled = proj.generate_lappd_waveforms(LAPPD_Hit_2D_sampled, LAPPD_profile.sPE_pulse_time, LAPPD_profile.sPE_pulse, LAPPD_profile.LAPPD_stripWidth, LAPPD_profile.LAPPD_stripSpace)
    #Sim_Waveforms_sampled_converted, Data_waveforms_converted = ConvertWaveform(Sim_Waveforms_sampled, data_waveforms)
    #L_0_0, Similarities_0, shift_T_0 = Likelihood(Sim_Waveforms_sampled_converted, Data_waveforms_converted, high_thres = high_thres, low_thres = low_thres)
    #resampled_hits_withPE_step, L_0_n, Similarities_step, shift_T, totalResampleTime = OptimizeLInThisStep(sampled_hits_withPE, L_0_0, Similarities_0, 0, data_waveforms, LAPPD_profile)
    mu_optimization_chain.append([mu_position, mu_direction, L_0_n])
    print("Start L_0_n: {}".format(L_0_n), ", shift_T_0: {}".format(shift_T_0))
    
    ####
    # each second index is a strip, loop the first index to get all positions on that strip
    printTest = True
    if(printTest):
        LAPPD_Hit_2D_sampled, totalPE = proj.convertToHit_2D(resampledHits_withPE_0, number_of_LAPPDs = 1, reSampled = True)
        Sim_Waveforms_sampled = proj.generate_lappd_waveforms(LAPPD_Hit_2D_sampled, LAPPD_profile.sPE_pulse_time, LAPPD_profile.sPE_pulse, LAPPD_profile.LAPPD_stripWidth, LAPPD_profile.LAPPD_stripSpace)
        Sim_Waveforms_sampled_converted, Data_waveforms_converted = ConvertWaveform(Sim_Waveforms_sampled, data_waveforms)
        print("Plotting the test waveform on strip 15:")
        w1_bottom, w1_top = Sim_Waveforms_sampled_converted[15]
        w2_bottom, w2_top = Data_waveforms_converted[15]
        c1 = np.array(combineWaveform(w1_bottom, w1_top, shift_T_0))
        c2 = np.array(combineWaveform(w2_bottom, w2_top, 0))
        print("c1 =", repr(c1.tolist()))
        print("c2 =", repr(c2.tolist()))
        similarity_bottom, FoundRange_b = calculate_section_similarity(256, 512 , c1[::-1], c2[::-1])
        # take the normal order later half, which is the left part of the waveform. (top side)
        similarity_top, FoundRange_t = calculate_section_similarity(256, 512, c1, c2)
        range_bottom = (255 - FoundRange_b[1], 255 - FoundRange_b[0])
        range_top = (255 + FoundRange_t[0], 255 + FoundRange_t[1])
        print("Result test plot", similarity_bottom, similarity_top, range_bottom, range_top)
        
    ###
    
    # now, start optimization
    # for each xyz position, optimize the theta and phi first. 
    L_previous = L_0_n
    L_n_n = L_0_n
    LStepNumber = 0
    new_mu_position = mu_position 
    new_mu_direction = mu_direction
    
    #(L_n_n - L_previous > epsilon*L_previous) and
    
    
    while ( LStepNumber < maxStepNumber):
        print("Start L_{}".format(LStepNumber))
        LStepNumber += 1
        
        loop = [dx, dy, dz, dtheta, dphi]
        L_n_opt = []
        
        print(L_n_opt, loop, mu_position, mu_direction)
        
        for axis_i in range(5):
            if(axis_i == 0):
                new_mu_position = mu_position + np.array([dx, 0, 0])
            elif(axis_i == 1):
                new_mu_position = mu_position + np.array([0, dy, 0])
            elif(axis_i == 2):
                new_mu_position = mu_position + np.array([0, 0, dz])
            elif(axis_i == 3):
                new_mu_direction = proj.rotate_vector(mu_direction, dtheta, 0)
            elif(axis_i == 4):
                new_mu_direction = proj.rotate_vector(mu_direction, 0, dphi)
                
            print("Searching on axis: ", axis_i, new_mu_position, new_mu_direction)
        
            sampled_hits_withPE_n = DoProjection(LAPPD_profile, new_mu_position, new_mu_direction)
            LAPPD_Hit_2D_sampled_n, totalPE_n = proj.convertToHit_2D(sampled_hits_withPE_n, number_of_LAPPDs = 1, reSampled = True)
            if(totalPE_n <= 0.001):
                L_n_opt.append(1e20)
                print("no PE found in this position, skip")
                continue
            Sim_Waveforms_sampled_n = proj.generate_lappd_waveforms(LAPPD_Hit_2D_sampled_n, LAPPD_profile.sPE_pulse_time, LAPPD_profile.sPE_pulse, LAPPD_profile.LAPPD_stripWidth, LAPPD_profile.LAPPD_stripSpace)
            Sim_Waveforms_sampled_converted_n, Data_waveforms_converted_n = ConvertWaveform(Sim_Waveforms_sampled_n, data_waveforms)
            L_n_0, Similarities_n, shift_T_n = Likelihood(Sim_Waveforms_sampled_converted_n, Data_waveforms_converted_n, high_thres = high_thres, low_thres = low_thres)
            resampled_hits_withPE_step_n, L_n_n, Similarities_step_n, shift_T_n, totalResampleTime_n = OptimizeLInThisStep(sampled_hits_withPE_n, L_n_0, Similarities_n, 0, data_waveforms, LAPPD_profile)
            L_n_n_normed = L_n_n/totalPE_n
            mu_optimization_chain.append([new_mu_position, new_mu_direction, L_n_n, L_n_n_normed])
            Final_hits = resampled_hits_withPE_step_n
        
            print("In this L step, L previous = {}, L_n_n = {}".format(L_previous, L_n_n))
            print("dx = {}, dy = {}, dz = {}, dtheta = {}, dphi = {}".format(dx, dy, dz, dtheta, dphi))
            
            if L_n_n <= 0.001:
                L_n_n_normed = 1e20
            L_n_opt.append(L_n_n_normed)
        
        print("L_n_opt: ", L_n_opt)
        valid_enumerated = [(i, val) for i, val in enumerate(L_n_opt) if val < 1e10]
        result = min(valid_enumerated, key=lambda x: x[1], default=None)

        if result is None:
            move_axis_ind = None
            print("No valid L_value found, half the steps")
            dx = dx * shrinkStepRatio
            dy = dy * shrinkStepRatio
            dz = dz * shrinkStepRatio
            dtheta = dtheta * shrinkStepRatio
            dphi = dphi * shrinkStepRatio
        else:
            move_axis_ind, L_value = result
            print(f"Found move_axis_ind: {move_axis_ind}, L_value: {L_value}")
        
        if(move_axis_ind == 0):
            mu_position = mu_position + np.array([dx, 0, 0])
            DeltaX += np.array([dx, 0, 0, 0, 0])
        elif(move_axis_ind == 1):
            mu_position = mu_position + np.array([0, dy, 0])
            DeltaX += np.array([0, dy, 0, 0, 0])
        elif(move_axis_ind == 2):
            mu_position = mu_position + np.array([0, 0, dz])
            DeltaX += np.array([0, 0, dz, 0, 0])
        elif(move_axis_ind == 3):
            mu_direction = proj.rotate_vector(mu_direction, dtheta, 0)
            DeltaX += np.array([0, 0, 0, dtheta, 0])
        elif(move_axis_ind == 4):
            mu_direction = proj.rotate_vector(mu_direction, 0, dphi)
            DeltaX += np.array([0, 0, 0, 0, dphi])
            
        if(L_n_n - L_previous > shrinkStepThreshold):
            if(move_axis_ind == 0):
                dx = dx * shrinkStepRatio
            if(move_axis_ind == 1):
                dy = dy * shrinkStepRatio
            if(move_axis_ind == 2):
                dz = dz * shrinkStepRatio
            if(move_axis_ind == 3):
                dtheta = dtheta * shrinkStepRatio
            if(move_axis_ind == 4):
                dphi = dphi * shrinkStepRatio
    
        L_previous = L_value

    print("#################################################")
    print("#############Final optimization result: ", DeltaX)
    print("#################################################")

    return mu_optimization_chain, Final_hits


def DoProjection(LAPPD_profile, mu_position, mu_direction, muon_step = 0.01, muon_prop_steps = 2000, phi_steps = 360):
    
    mu_positions = [mu_position + (i * mu_direction * muon_step) for i in range(muon_prop_steps)]
    mu_positions = [pos for pos in mu_positions if (pos[2] < 2.948)]
    Results = proj.parallel_process_positions(mu_positions, mu_direction, LAPPD_profile.grid, phi_steps = phi_steps)
    Results_withMuTime = proj.process_results_with_mu_time(Results, muon_step)
    updated_hits_withPE = proj.update_lappd_hit_matrices(
        results_with_time=Results_withMuTime,       
        absorption_wavelengths = LAPPD_profile.absorption_wavelengths,
        absorption_coefficients = LAPPD_profile.absorption_coefficients,
        qe_2d=LAPPD_profile.qe_2d,                               # QE 2D, normalized
        gain_2d=LAPPD_profile.gain_2d,                           # gain distribution 2D, normlized
        QEvsWavelength_lambda=LAPPD_profile.QEvsWavelength_lambda,    # QE vs wavelength, wavelength array
        QEvsWavelength_QE=LAPPD_profile.QEvsWavelength_QE,            # QE vs wavelength, QE array
        bin_size=LAPPD_profile.bin_size,                                    # wavelength bin size
        #CapillaryOpenRatio = 0.64                       # capillary open ratio of MCP
        CapillaryOpenRatio = LAPPD_profile.CapillaryOpenRatio,                 # capillary open ratio of MCP
        phi_steps = phi_steps,
        muon_step = muon_step
    )
    sampled_hits_withPE = sample_updatedHits_PE_Poisson(updated_hits_withPE)
    
    return sampled_hits_withPE



def store_results_to_hdf5(output_hdf5_file, all_results):
    print("Writing results to HDF5 file...")
    
    with h5py.File(output_hdf5_file, 'w') as hf:
        result_group = hf.create_group("result")
        for event_id, data_dict in all_results.items():
            print("Writing event", event_id)
            event_group = result_group.create_group(str(event_id))
            
            mu_chain_str = json.dumps(data_dict["mu_optimization_chain"], cls=dc.NumpyEncoder)
            #print("mu_chain_str:", mu_chain_str)
            event_group.create_dataset("mu_optimization_chain", data=mu_chain_str)
            
            #best_hits_str = json.dumps(data_dict["best_hits"], cls=dc.NumpyEncoder)
            #event_group.create_dataset("best_hits", data=best_hits_str)
            
            vpos_str = json.dumps(data_dict["true_vertex_position"], cls=dc.NumpyEncoder)
            event_group.create_dataset("true_vertex_position", data=vpos_str)
            
            vdir_str = json.dumps(data_dict["true_vertex_direction"], cls=dc.NumpyEncoder)
            event_group.create_dataset("true_vertex_direction", data=vdir_str)

            #event_group.create_dataset("improved_global", data=int(data_dict["improved_global"]))
            
            pe_str = json.dumps(data_dict["PE_on_LAPPD0"], cls=dc.NumpyEncoder)
            event_group.create_dataset("PE_on_LAPPD0", data=pe_str)

    print("Saving Done!")






                