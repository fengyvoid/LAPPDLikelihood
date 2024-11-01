import numpy as np
from scipy.spatial import KDTree
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp1d
from scipy import interpolate

def process_photon_hits(particle_position, particle_direction, LAPPD_grids, 
                        tolerance=0.007, PMTPosition=[(0.26489847,0.01038878,2.72413225, 0.254/2), (-0.2666204,0.00990167,2.72809739, 0.254/2), 
                                                      (0.27324662,-0.3994107,2.79745653, 0.2032/2), (-0.2689693,-0.4085338,2.79763283, 0.2032/2)], 
                        speed_of_light=2.98e8):
    
    KDTree_list = [KDTree(LAPPD_grid.reshape(-1, 3)) for LAPPD_grid in LAPPD_grids]
    
    min_tStep = (tolerance / 1.5) / speed_of_light

    theta = np.deg2rad(42)
    phi_angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    
    particle_direction = np.array(particle_direction)
    particle_direction /= np.linalg.norm(particle_direction)

    if not np.allclose(particle_direction[:2], 0):
        arbitrary_vector = np.array([0, 0, 1])
    else:
        arbitrary_vector = np.array([1, 0, 0])
    
    perpendicular_vector = np.cross(particle_direction, arbitrary_vector)
    perpendicular_vector /= np.linalg.norm(perpendicular_vector)
    perpendicular_vector_2 = np.cross(particle_direction, perpendicular_vector)
    
    photon_vectors = []
    for phi in phi_angles:
        rotation_vector = np.cos(phi) * perpendicular_vector + np.sin(phi) * perpendicular_vector_2
        vector = np.cos(theta) * particle_direction + np.sin(theta) * rotation_vector
        vector /= np.linalg.norm(vector)
        photon_vectors.append(vector)
    
    hit_results = []

    for vector in photon_vectors:
        min_distance = min(tree.query(particle_position, k=1)[0] for tree in KDTree_list)
        t = 0
        
        Hitted = False
        while not Hitted:
            P = t * vector * speed_of_light + particle_position

            # only search hit within some tank volume
            if P[2] > 3 or (P[0] > 1.1 or P[0] < -1.1) or (P[1] > 1.3 or P[1] < -1.6):
                break  

            best_hit = None 
            for LAPPD_index, tree in enumerate(KDTree_list):
                distance, idx = tree.query(P, k=1)
                if distance < min_distance:
                    hit_position = LAPPD_grids[LAPPD_index].reshape(-1, 3)[idx]
                    vector_to_hit = hit_position - particle_position
                    vector_to_hit /= np.linalg.norm(vector_to_hit)  
                    min_distance = distance  
                    propagation_distance = np.linalg.norm(hit_position - particle_position)
                    propagation_time = propagation_distance / speed_of_light
                    first_index, second_index = np.unravel_index(idx, LAPPD_grids[LAPPD_index].shape[:2])
                    best_hit = [LAPPD_index, first_index, second_index, propagation_time, propagation_distance]
            
            t += min(0.2, max(min_distance / (4 * speed_of_light), min_tStep))

            if min_distance <= tolerance:
                skip_hit = False

                for PMT in PMTPosition:
                    PMT_center = np.array(PMT[:3])
                    PMT_radius = PMT[3]

                    vec_to_pmt = PMT_center - particle_position
                    proj_length = np.dot(vec_to_pmt, vector)  # projection length
                    if proj_length < 0:
                        continue
                    proj_point = particle_position + proj_length * vector  # projection point
                    min_dist_to_pmt = np.linalg.norm(proj_point - PMT_center)  # min distance to PMT center

                    if min_dist_to_pmt < PMT_radius:  
                        skip_hit = True
                        break

                if not skip_hit and best_hit is not None:
                    hit_results.append(best_hit)
                    Hitted = True

    return hit_results

# processing multiple particle position
def process_particle_position(args):
    particle_position, particle_direction, LAPPD_grids = args

    return process_photon_hits(particle_position, particle_direction, LAPPD_grids)

def parallel_process_positions(particle_positions, particle_direction, LAPPD_grids):
    with Pool(8) as pool:  # use 8 "cores", multiple threads in one core may not be better because of the 
        #results = list(tqdm(pool.imap_unordered(
        #    process_particle_position, [(pos, particle_direction, LAPPD_grids) for pos in particle_positions]
        #), total=len(particle_positions), desc="Processing particles", miniters=1))
        results = list(pool.imap_unordered(
            process_particle_position, [(pos, particle_direction, LAPPD_grids) for pos in particle_positions]
        ))
             
    return results




# change the hit time from photon propogation to photon + muon propogation time
def process_results_with_mu_time(results, step_size, speed_of_light = 2.998e8, shiftTime = True):
    Results_withMuTime = []
    
    hitTimes = []
    for i, hits in enumerate(results):
        particle_propagation_time = i * step_size / speed_of_light
        for hit in hits:
            LAPPD_index, first_index, second_index, propagation_time, propagation_distance = hit
            # 添加 Muon 传播时间
            t = propagation_time + particle_propagation_time
            hitTimes.append(t)
    
    if(len(hitTimes) != 0):
        shiftT = min(hitTimes)
    if not shiftTime:
        shiftTime = 0
        
    #print("hitTimes: ", hitTimes)
    #print("shiftT: ", shiftT)
    
    for i, hits in enumerate(results):
        particle_propagation_time = i * step_size / speed_of_light
        new_hits = []
        for hit in hits:
            LAPPD_index, first_index, second_index, propagation_time, propagation_distance = hit
            # 添加 Muon 传播时间
            total_time = propagation_time + particle_propagation_time - shiftT
            new_hits.append((LAPPD_index, first_index, second_index, total_time, propagation_distance))
        Results_withMuTime.append(new_hits)
    return Results_withMuTime



# calculate the PE number of each hit
def update_lappd_hit_matrices(results_with_time, absorption_wavelengths, absorption_coefficients , qe_2d, gain_2d, QEvsWavelength_lambda, QEvsWavelength_QE, bin_size=10, CapillaryOpenRatio = 0.6):
    # Define wavelength bins and constants
    wavelengths = np.arange(200 + bin_size / 2, 601 + bin_size / 2, bin_size)
    
    updated_results = []

    # Process hits
    for hits in results_with_time:
        updated_results_this_particle_step = []
        for hit in hits:
            LAPPD_index, first_index, second_index, hit_time, photon_distance = hit
            # Interpolate absorption coefficient for the photon's distance
            absorption_function = interpolate.interp1d(absorption_wavelengths, absorption_coefficients, kind='linear')
            absorption_at_distance = absorption_function(wavelengths)
            
            # Calculate after-absorption photon count
            initial_photons = 0.4513 * 370 / 360 #sin theta_C ^2 * 370 constant / 360 degree
            after_absorption_photons = initial_photons * np.exp(-absorption_at_distance * photon_distance)

            # Sum of photoelectrons (PE) for the current hit across wavelength bins
            # need to consider Capillary Open Area Ratio 
            total_PE = 0
            for i, wavelength in enumerate(wavelengths):
                #photon_energy = calculate_photon_energy(wavelength)

                energy_minus = 1241.55 / (wavelength - bin_size / 2)
                energy_plus = 1241.55 / (wavelength + bin_size / 2)
                dE = abs(energy_minus - energy_plus)

                # Get corresponding QE for the current LAPPD
                QE_function = interpolate.interp1d(QEvsWavelength_lambda[LAPPD_index], QEvsWavelength_QE[LAPPD_index], kind='linear', fill_value="extrapolate")
                QE_at_wavelength = QE_function(wavelength)
                PE_number = after_absorption_photons[i] * dE * QE_at_wavelength
                total_PE += PE_number 
            
            total_PE = total_PE * CapillaryOpenRatio

            # Update the PE and time matrices for the hit
            # qe2d is symmetric, so which is first doesn't matter
            weighted_pe = qe_2d[first_index, second_index] * gain_2d[first_index, second_index] * total_PE

            hit_withPE = (LAPPD_index, first_index, second_index, hit_time, photon_distance, weighted_pe)
            updated_results_this_particle_step.append(hit_withPE)

        updated_results.append(updated_results_this_particle_step)


    return updated_results

# generate Gaussian distribution, using the pulse_start_time_ns + delay as the mean, use the sigma as the sigma, use the amplitude as the amplitude
def generate_crossTalk_for_strip(hit_list, delay, amplitude, sigma, LAPPD_gridSize, strip_direction, speed_of_light = 2.998e8):
    waveform = np.zeros(256)
    
    for hit in hit_list:
        hit_y, hit_time, hit_pe = hit
        y_pos = (hit_y+0.5) * LAPPD_gridSize  # hit_y 为 strip index，需要转换为坐标
        if hit_time is None or hit_time ==0:
            continue
        # 计算 pulse 起始时间（以 ns 为单位）
        if strip_direction == "down":
            pulse_start_time_ns = (y_pos ) / (0.567 * speed_of_light) + hit_time * 1e9
        elif strip_direction == "up":
            pulse_start_time_ns = (28*LAPPD_gridSize - y_pos ) / (0.567 * speed_of_light) + hit_time * 1e9

        
        # 将 pulse 映射到 waveform 的时间轴，0.1ns 为一个 bin
        binStartTime = int((pulse_start_time_ns + delay) * 10 - 1)
        #still need to make the whole wavefosm - -.
        for i in range(256):
            bin_time_ns = i * 0.1  # 每个 bin 代表 0.1ns
            pulse_time_ns = bin_time_ns - pulse_start_time_ns - delay  # 相对于 pulse 起始时间的时间差

            if i < binStartTime or i > binStartTime+100:
                continue
            
            # 计算该 bin 对应的 pulse amplitude（使用线性插值）
            pulse_amplitude = amplitude * np.exp(-0.5*(pulse_time_ns/sigma)**2)
            
            # 将 pulse_amplitude 乘以 hit_pe 加入 waveform 中
            waveform[i] -= hit_pe * pulse_amplitude

    return waveform

def weightedGain(pe_num):
    # Calculate the total gain corresponding to the number of PEs using an empirically estimated formula and a constant gain.
    # The basic assumption is:
    # 1. One pore can only discharge once for inserted PEs within ~ns time scale
    # 2. 1e5 pores per 0.7*0.7cm^2 pixel (25um pore size)
    # 3. All PEs that hit the same pixel will be considered using those pores in that pixel for calculation
    # 4. Ignore the charge sharing. The actual charge sharing is ~ 10% for one nearing strip
    # 5. The recharging of pores in different pixels are independent (questionalble)
    # the experimental proof of the idea is from the paper: https://pubs.aip.org/aip/rsi/article/89/7/073301/358380/Extending-the-dynamic-range-of-microchannel-plate
    # It use the measured current to calculate the insered electron number.
    
    Gained_PE = 0
    
    # take an integral for the gain per PE * number of PEs
    
    Gained_PE = pe_num
    
    
    return Gained_PE
    
def sample_PE_Poisson(LAPPD_Hit_2D):
    LAPPD_Hit_2D_sampled = []
    for id in range(len(LAPPD_Hit_2D)):
        LAPPD_Hit_2D_sampled.append([])
        for strip in range(len(LAPPD_Hit_2D[id])):
            LAPPD_Hit_2D_sampled[id].append([])
            for hit in LAPPD_Hit_2D[id][strip]:
                hit_y, hit_time, hit_pe = hit
                hit_pe = np.random.poisson(hit_pe)
                if(hit_pe != 0 ):
                    LAPPD_Hit_2D_sampled[id][strip].append((hit_y, hit_time, hit_pe))
    return LAPPD_Hit_2D_sampled


def sample_updatedHits_PE_Poisson(hits_withPE):
    sampled_hits_withPE = []
    for step in range(len(hits_withPE)):
        sampled_hits_withPE.append([])
        for hit in hits_withPE[step]:
            LAPPD_index, first_index, second_index, hit_time, photon_distance, weighted_pe = hit
            sampled_pe = np.random.poisson(weighted_pe)
            if(sampled_pe != 0):
                sampled_hits_withPE[step].append((LAPPD_index, first_index, second_index, hit_time, photon_distance, sampled_pe))
    return sampled_hits_withPE

def convertToHit_2D(hits_withPE, number_of_LAPPDs = 1):
    LAPPD_Hit_2D = []
    totalPE = 0
    
    for i in range (number_of_LAPPDs):
        LAPPD_Hit_2D.append([])
        for j in range(28):
            LAPPD_Hit_2D[i].append([])

    for i in range (len(hits_withPE)):
        # each particle step
        for j in range(len(hits_withPE[i])):
            # just loop all hits
            # for each strip, i.e. same x position but different y position
            # each second index is a strip, loop the first index to get all positions on that strip
            LAPPD_Hit_2D[hits_withPE[i][j][0]][hits_withPE[i][j][2]].append((hits_withPE[i][j][1], hits_withPE[i][j][3], hits_withPE[i][j][5]))
            totalPE+=hits_withPE[i][j][5]
            
    return LAPPD_Hit_2D, totalPE


# 使用sPE和hit info来生成waveform
def generate_waveform_for_strip(hit_list, pulse_time, sPE_pulse, LAPPD_gridSize, speed_of_light, strip_direction):
    waveform = np.zeros(256)
    
    # 创建插值函数，从 pulse_time 和 sPE_pulse 生成 pulse amplitude
    sPE_interp = interp1d(pulse_time, sPE_pulse, kind='linear', bounds_error=False, fill_value=0)



    for hit in hit_list:
        hit_y, hit_time, hit_pe = hit
        #hit_pe = weightedGain(hit_pe)
        y_pos = (hit_y+0.5) * LAPPD_gridSize  # hit_y 为 strip index，需要转换为坐标
        if hit_time is None or hit_time ==0:
            continue
        # 计算 pulse 起始时间（以 ns 为单位）

        if strip_direction == "down":
            pulse_start_time_ns = ((y_pos ) / (0.567 * speed_of_light)/100 + hit_time) * 1e9
            #print("y_pos: ", y_pos, "hit_time: ", hit_time, "pulse_start_time_ns: ", pulse_start_time_ns, "speed_of_light: ", speed_of_light, "prop time", (y_pos ) / (0.567 * speed_of_light))
        elif strip_direction == "up":
            pulse_start_time_ns = ((28*LAPPD_gridSize - y_pos ) / (0.567 * speed_of_light)/100 + hit_time )* 1e9

        
        # 将 pulse 映射到 waveform 的时间轴，0.1ns 为一个 bin
        for i in range(256):
            bin_time_ns = i * 0.1  # 每个 bin 代表 0.1ns
            pulse_time_ns = bin_time_ns - pulse_start_time_ns  # 相对于 pulse 起始时间的时间差
            
            # 计算该 bin 对应的 pulse amplitude（使用线性插值）
            pulse_amplitude = sPE_interp(pulse_time_ns)
            
            # 将 pulse_amplitude 乘以 hit_pe 加入 waveform 中
            waveform[i] += hit_pe * pulse_amplitude

    return waveform


def generate_lappd_waveforms(LAPPD_Hits_2D, sPEPulseTime, sPEPulseAmp,LAPPD_stripWidth, LAPPD_stripSpace):
    LAPPD_waveforms_AllLAPPDs = []
    LAPPD_gridSize = LAPPD_stripWidth + LAPPD_stripSpace
    speed_of_light = 2.998e8  # 光速

    #just assume a fixed number 
    crossTalk_Amp = 0.045 # percentage
    crossTalk_TimeDelay = 0.09 #ns


    index = np.argmax(sPEPulseAmp)
    max_pulse_time = sPEPulseTime[index]
    

    for lappd_hits in LAPPD_Hits_2D:
        
        LAPPD_waveforms = np.zeros((28, 2, 256))  # 28 strips, 2 waveforms (down, up), 256 bins each

        for x in range(28):

            hits_here = lappd_hits[x]
            #print("hits_here: ", x, hits_here)
            # a list of all hits with (hit_y, hit_time, hit_pe)

            if len(hits_here) > 0:
                waveform_down = generate_waveform_for_strip(hits_here, sPEPulseTime, sPEPulseAmp, LAPPD_gridSize, speed_of_light,  "down")
                waveform_up = generate_waveform_for_strip(hits_here, sPEPulseTime, sPEPulseAmp, LAPPD_gridSize, speed_of_light, "up")
                
                #print("waveform_down: ", waveform_down)
                
                LAPPD_waveforms[x][0] += waveform_down
                LAPPD_waveforms[x][1] += waveform_up


                for cros in range(28):
                    if cros == x:
                        continue

                    dStrip = abs(cros-x)

                    delay = max_pulse_time + dStrip * crossTalk_TimeDelay # in ns
                    amplitude = max(sPEPulseAmp)*crossTalk_Amp # in mV
                    sigma = 0.48+0.036*dStrip # in ns

                    crossTalk_down = generate_crossTalk_for_strip(hits_here, delay, amplitude, sigma, LAPPD_gridSize, "down")
                    crossTalk_up = generate_crossTalk_for_strip(hits_here, delay, amplitude, sigma, LAPPD_gridSize, "up")

                    LAPPD_waveforms[cros][0] += crossTalk_down
                    LAPPD_waveforms[cros][1] += crossTalk_up

        # now, the first index of LAPPD_waveforms is the x index of LAPPD_grids
        # but, the grid is in corrdinate direction, not the real view direction
        # so, the x index need to be flipped
        # link here: https://annie-docdb.fnal.gov/cgi-bin/sso/ShowDocument?docid=5779
        flipped_LAPPD_waveforms = np.flip(LAPPD_waveforms, axis=0)


        LAPPD_waveforms_AllLAPPDs.append(flipped_LAPPD_waveforms)

    return LAPPD_waveforms_AllLAPPDs




def align_waveforms(Sim_Waveform, Data_Waveform, SimRange=(50, 150), shiftRange=(80, 130)):
    # Cut the Sim_Waveform according to SimRange
    Sim_Waveform_cut = Sim_Waveform[:, :, SimRange[0]:SimRange[1]]
    
    # Initialize an array to store the chi2 differences for each shift
    waveform_diff = np.zeros(shiftRange[1] - shiftRange[0] + 1)

    # Loop over each shift value in the range
    for shift in range(shiftRange[0], shiftRange[1] + 1):
        total_diff = 0
        
        # Loop over each strip and side
        for strip in range(28):
            for side in range(2):
                # Get the Sim_Waveform_cut for the current strip and side
                sim_cut_waveform = Sim_Waveform_cut[strip, side]
                
                # Get the corresponding Data_Waveform based on the current shift
                start_idx = SimRange[0] + shift
                end_idx = SimRange[1] + shift

                if end_idx >= 256:
                    data_waveform_to_compare = np.concatenate((Data_Waveform[strip, side, start_idx:256], Data_Waveform[strip, side, :end_idx % 256]))
                else:
                    data_waveform_to_compare = Data_Waveform[strip, side, start_idx:end_idx]

                # Calculate the sum of absolute differences
                diff = 0
                for i in range(len(data_waveform_to_compare)):
                    scaling = data_waveform_to_compare[i]**3
                    #scaling = 1
                    diff += np.abs(data_waveform_to_compare[i] - sim_cut_waveform[i]) * np.abs(scaling)

                
                # Accumulate the differences
                total_diff += diff

        # Store the total difference for this shift
        waveform_diff[shift - shiftRange[0]] = total_diff

    # Find the shift with the minimum total difference
    min_shift_index = np.argmin(waveform_diff)
    min_shift = shiftRange[0] + min_shift_index
    min_diff = waveform_diff[min_shift_index]
    
    # Shift Sim_Waveform by min_shift to create Sim_Waveform_shifted
    Sim_Waveform_shifted = np.zeros_like(Sim_Waveform)
    
    for strip in range(28):
        for side in range(2):
            for bin in range(256):
                # Apply the shift and wrap around using modulo operation
                Sim_Waveform_shifted[strip, side, (bin + min_shift) % 256] = Sim_Waveform[strip, side, bin]

    # Print the minimum shift and return the results
    # print(f"Minimum shift: {min_shift}")
    return min_shift, min_diff, waveform_diff, Sim_Waveform_shifted


def rotate_vector(vector, theta, phi):
    """旋转向量 vector, theta 是绕y轴旋转的角度, phi 是绕x轴旋转的角度"""
    # 绕y轴旋转 (theta)
    rot_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    # 绕x轴旋转 (phi)
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(phi), -np.sin(phi)],
                      [0, np.sin(phi), np.cos(phi)]])
    
    # 先绕y轴旋转，再绕x轴旋转
    rotated_vector = rot_x @ (rot_y @ vector)
    return rotated_vector