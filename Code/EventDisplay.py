from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from decimal import Decimal, getcontext
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import pandas as pd
import datetime
import pytz
import matplotlib.cm as cm

# event display Pure, without text and stat

# Load geometry and defind LAPPD geometry here
# all in cm

# tank center position in X, Y, Z
TC = [0,-14.4649,168.1]
TRadius = 152
THeight = 396
TPMTRadius = 100
TPMTHeight = 145*2



# LAPPD position
# center, lower left, upper right
LAPPD_Centers = [[0,-0.2255,2.951], [-0.898, -0.2255 - 0.5, 2.579], [0.898, -0.2255 + 0.5 , 2.579]]
for i in range(len(LAPPD_Centers)):
    LAPPD_Centers[i] = np.array(LAPPD_Centers[i])*100

LAPPD_Center = LAPPD_Centers
LAPPD_Direction = [[0,0,-1], [1,0,-1], [-1,0,-1]]
LAPPD_stripWidth = 0.462
LAPPD_stripSpace = 0.229
LAPPD_H_width = 20
LAPPD_V_height = 20


# dead PMT position
deadPMTs = [
    [-0.2476846, -0.1882051, 0.14804416, 0.34590855, -0.3860695, 0.47238569, -0.1523917, -1.0413511, 0.26391394, -0.5454545, 0.54989219],  #X
    [-1.6939623, -1.6939623, -1.6939623, -1.6986855, -1.6939623, -1.6939623, 1.32865366, 0.86344388, -1.2980369, -1.2864868, -0.8586892],  #Y
    [1.6037927, 0.91420765, 1.20706832, 1.00870613,1.11256984, 1.52414908,2.4965238, 1.40794291, 2.72564427, 2.60815843, 2.6087467],  #Z
    [333, 342, 343, 345, 346, 349, 352, 416, 431, 444, 445]   #channel key
]

deadPMTs = [[i - TC[0] for i in deadPMTs[0]], [i - TC[1]/100 for i in deadPMTs[1]], [i - TC[2]/100 for i in deadPMTs[2]], deadPMTs[3]]


'''
333,333,Bottom,0,-0.2476846,-1.6939623,1.6037927,0,1,0,LUX,N2,OFF,6,2,6,0,1,2,11,2,-999,-999,-999,-9999,ok (noisy)
342,342,Bottom,0,-0.1882051,-1.6956896,0.91420765,0,1,0,LUX,N5,OFF,6,11,6,0,10,2,14,3,-999,-999,-999,-9999,ok (low rate)
343,343,Bottom,0,0.14804416,-1.6939623,1.20706832,0,1,0,LUX,W3,ON,6,12,6,0,11,2,14,4,-999,-999,-999,-9999,ok (noisy)
345,345,Bottom,0,0.34590855,-1.6986855,1.00870613,0,1,0,LUX,E5,OFF,7,2,6,0,13,2,15,2,-999,-999,-999,-9999,no signal
346,346,Bottom,0,-0.3860695,-1.6939623,1.11256984,0,1,0,LUX,E1,OFF,7,3,6,0,14,2,15,3,-999,-999,-999,-9999,ok (noisy)
349,349,Bottom,0,0.47238569,-1.6939623,1.52414908,0,1,0,LUX,S3,OFF,7,6,6,1,1,2,18,2,-999,-999,-999,-9999,ok
352,352,Top,9,-0.1523917,1.32865366,2.4965238,0,-1,0,ETEL,116,OFF,5,1,6,2,0,2,6,1,-999,-999,-999,-9999,ok
*396,396,Barrel,2,-1.1172646,-0.4138866,1.94262871,0.92372753,0.00187934,-0.3830443,Hamamatsu,SQ0367,ON,12,3,6,5,8,3,13,1,-999,-999,-999,-9999,ok
416,416,Barrel,3,-1.0413511,0.86344388,1.40794291,0.9222563,-0.0025943,0.3865435,Watchboy,3,ON,1,1,5,7,0,1,4,1,-999,-999,-999,-9999,ok
*423,423,Barrel,2,-1.0470203,0.86292047,1.93673598,0.92372753,0.00187934,-0.3830443,Watchboy,4,ON,1,8,5,7,7,1,5,4,-999,-999,-999,-9999,ok
431,431,Barrel,8,0.26391394,-1.2980369,2.72564427,-0.3773816,0.00035279,-0.9260398,Watchboy,36,OFF,4,4,6,7,15,2,3,4,-999,-999,-999,-9999,no signal HV channel only ramps up to 250V
444,444,Barrel,1,-0.5454545,-1.2864868,2.60815843,0.38530308,0.00451611,-0.9227789,Watchboy,31,OFF,4,5,6,8,12,2,4,1,-999,-999,-999,-9999,no signal HV channel only ramps up to 700V
445,445,Barrel,8,0.54989219,-0.8586892,2.6087467,-0.3773816,0.00035279,-0.9260398,Watchboy,42,OFF,4,6,6,8,13,2,4,2,-999,-999,-999,-9999,ok
'''

# MRD position
MRDFile = '/Users/fengy/Documents/GitHub/ToolAnalysis/configfiles/LoadGeometry/FullMRDGeometry_09_29_20.csv'
with open(MRDFile, 'r') as file:
    lines = file.readlines()
df = pd.read_csv(MRDFile, skiprows=0)
MRDGeo = df.set_index('channel_num').T.to_dict()
class MRDGeoAccessor:
    def __init__(self, data):
        self.data = data
    
    def __getattr__(self, attr):
        def get_by_channel(channel_num):
            return self.data[channel_num][attr]
        return get_by_channel
MRDGeoAccessor = MRDGeoAccessor(MRDGeo)

#channel_num = 82
#y_center_value = MRDGeoAccessor.y_center(channel_num)
#print(f'Loading MRD channel test: Channel {channel_num} y_center: {y_center_value}')


def process_pmt_hits(PMTHits):
    hitX, hitY, hitZ, hitT, hitPE, hitChanKey = PMTHits
    
    # Convert lists to numpy arrays for efficient operations
    hitT = np.array(hitT)
    hitChanKey = np.array(hitChanKey)
    
    # Find the minimum hitT
    startHitT = np.min(hitT)
    
    # Create a mask for hits within the time window
    time_mask = hitT < startHitT + 50
    
    # Get unique channel keys and their first occurrences
    unique_keys, key_indices = np.unique(hitChanKey, return_index=True)
    
    # Initialize the merged hits array
    PMTHits_merged = [[] for _ in range(5)]
    
    # Process each unique channel key
    for key_index in key_indices:
        # Find all hits for this channel key
        channel_mask = hitChanKey == hitChanKey[key_index]
        
        # Combine channel mask with time mask
        combined_mask = channel_mask & time_mask
        
        if np.any(combined_mask):
            # Find the index of the hit with minimum hitT for this channel
            min_t_index = np.argmin(hitT[combined_mask])
            
            # Get the actual index in the original arrays
            hit_index = np.where(combined_mask)[0][min_t_index]
            
            # Add this hit to the merged array
            PMTHits_merged[0].append(hitX[hit_index])
            PMTHits_merged[1].append(hitY[hit_index])
            PMTHits_merged[2].append(hitZ[hit_index])
            PMTHits_merged[3].append(hitT[hit_index])
            PMTHits_merged[4].append(hitPE[hit_index])
    
    return PMTHits_merged

# Example usage:
# PMTHits = [hitX, hitY, hitZ, hitT, hitPE, hitChanKey]
# PMTHits_merged = process_pmt_hits(PMTHits)

# event display 

def EventDisplay(pdf, EventInfo, PMTClusterInfo, PMTHits_raw, ExtendedCluster, MRDHits, MRDTracks, LAPPDHits, hasVeto):
    # print this event into a pdf
    
    '''
    input information:
    select and pass the right information to this function 
    PMTHits = [hitX, hitY, hitZ, hitT, hitPE, hitChanKey]
    MRDHits = [MRDhitChankey, MRDhitT]
    MRDTracks = [MRDTrackStartX, MRDTrackStopX, MRDTrackStartY, MRDTrackStopY, MRDTrackStartZ, MRDTrackStopZ]
    LAPPDHits = [LAPPDID, LAPPDHitStrip, LAPPDHitTime, LAPPDHitAmp]
    '''
    PMTHits_raw_4_array = np.array(PMTHits_raw[4])
    valid_indices = np.isfinite(PMTHits_raw_4_array)
    for ind in range (len(PMTHits_raw)):
        HitArray = np.array(PMTHits_raw[ind])
        filtered_PMTHits = HitArray[valid_indices]
        PMTHits_raw[ind] = filtered_PMTHits.tolist()


    
    PMTHits = process_pmt_hits(PMTHits_raw)
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 16))
    for ax in fig.get_axes():
        ax.remove()
    
    ax_tank = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax_tank.axis('off')
    
    axmuyz = plt.subplot2grid((3, 3), (2, 0)) #muon y-z
    axmuyz.axis('off')
    axmuxz = plt.subplot2grid((3, 3), (2, 1)) #muon x-z
    axmuxz.axis('off')
    ax_l0 = plt.subplot2grid((3, 3), (0, 2)) #LAPPD 0 hit time
    ax_l1 = plt.subplot2grid((3, 3), (1, 2)) #LAPPD 1 hit time
    ax_l2 = plt.subplot2grid((3, 3), (2, 2)) #LAPPD 2 hit time
    
    
    inset_ax =  plt.subplot2grid((15, 5), (1, 2),colspan = 1, rowspan = 3)
    cmap = plt.cm.viridis  # 获取 'viridis' 颜色映射
    new_cmap = ListedColormap(cmap(np.linspace(0, 1, cmap.N)))  # 创建新的映射
    new_cmap.set_under('white')  # 设置低于vmin部分为白色
    h = inset_ax.hist2d(PMTHits_raw[3], PMTHits_raw[4], bins=[30, 30], cmap=new_cmap, vmin=0.1)
    pos = inset_ax.get_position()

    inset_cbar = plt.colorbar(h[3], ax=inset_ax)
    inset_cbar.set_label('Counts', fontsize=10)
    inset_ax.set_ylabel('PMT Hit PE', fontsize=10)
    inset_ax.set_xlabel('Hit Time (ns)', fontsize=10)
    inset_ax.set_title('PMT Hit PE vs Time', fontsize=12)
    
    if(len(ExtendedCluster[0])>0):
        ext_ax =  plt.subplot2grid((21, 5), (11, 2),colspan = 1, rowspan = 2)
        ext = ext_ax.scatter(ExtendedCluster[0], ExtendedCluster[1], s = 20)
        ext_ax.set_ylabel('Extended Cluster PE', fontsize=10)
        ext_ax.set_xlabel('Extended Cluster Time (ns)', fontsize=10)
        ext_ax.set_title('Extended Cluster', fontsize=12)
        ext_ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ext_ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        
        
        dPE = max(ExtendedCluster[1]) - min(ExtendedCluster[1])
        dT = max(ExtendedCluster[0]) - min(ExtendedCluster[0])
        if(dPE != 0):
            ext_ax.set_ylim(min(ExtendedCluster[1]) - 0.1*dPE, max(ExtendedCluster[1]) + 0.1*dPE)
        else:
            ext_ax.set_ylim(min(ExtendedCluster[1]) - 5, min(ExtendedCluster[1]) + 5) 
        ext_ax.set_xlim(2000, max(ExtendedCluster[0]) + 0.1*dT)

    
    ####################################################
    #########        plot tank PMTs            #########
    ####################################################
    Hit_2DX = []
    Hit_2DY = []
    Hit_PE = []
    
    maxPE = max(PMTHits[4])
    secondMaxPE = max([x for x in PMTHits[4] if x != maxPE])
    normMax = maxPE
    if(maxPE > 2*secondMaxPE):
        normMax = secondMaxPE
    
    for h in range(len(PMTHits[0])):
        if(PMTHits[4][h]<2):
            continue
            
        plotThis = True
        if(PMTHits[4][h]<0.2*normMax):
            plotThis = False
            ##check if there is a hit within 0.6m has PE > 10%
            for n in range (len(PMTHits[0])):
                if(PMTHits[4][n]>0.2*normMax):
                    distance = np.sqrt((PMTHits[0][n] - PMTHits[0][h])**2 + (PMTHits[1][n] - PMTHits[1][h])**2 + (PMTHits[2][n] - PMTHits[2][h])**2 )
                    if(distance<0.6):
                        plotThis = True
                        break
                        
        if(plotThis != True):
            continue
            
        Hit_PE.append(PMTHits[4][h])
        if(PMTHits[1][h]> 1.3):    # top
            Hit_2DX.append(PMTHits[0][h])
            Hit_2DY.append(TC[1]/100 + (THeight/2 + TRadius)/100 - (PMTHits[2][h]))
        elif(PMTHits[1][h]< -1.4):  # bottom
            Hit_2DX.append(PMTHits[0][h])
            Hit_2DY.append(TC[1]/100 - (THeight/2 + TRadius)/100 + (PMTHits[2][h]))
        else:
            Hit_2DY.append(PMTHits[1][h])
            dX = PMTHits[0][h]
            dZ = PMTHits[2][h]
            phi = abs(np.arctan(dZ/dX)/np.pi)
            distance_X = (phi+0.5)*np.pi*(TRadius/100)
            if (dX>0 and dZ>0):
                distance_X = ((0.5-phi)*np.pi*(TRadius/100))
            elif (dX<0 and dZ>0):
                distance_X = -((0.5-phi)*np.pi*(TRadius/100))
            elif (dX<0 and dZ<0):
                distance_X = -((0.5+phi)*np.pi*(TRadius/100))
            Hit_2DX.append(distance_X)
            
    Hit_2DX = [hit*100 for hit in Hit_2DX]
    Hit_2DY = [hit*100 for hit in Hit_2DY]
    
    # dead PMTs
    Hit_2DX_dead = []
    Hit_2DY_dead = []
    
    for h in range(len(deadPMTs[0])):
        if(deadPMTs[1][h]> 1.3):    # top
            Hit_2DX_dead.append(deadPMTs[0][h])
            Hit_2DY_dead.append(TC[1]/100 + (THeight/2 + TRadius)/100 - (deadPMTs[2][h]))
        elif(deadPMTs[1][h]< -1.4):  # bottom
            Hit_2DX_dead.append(deadPMTs[0][h])
            Hit_2DY_dead.append(TC[1]/100 - (THeight/2 + TRadius)/100 + (deadPMTs[2][h]))
        else:
            Hit_2DY_dead.append(deadPMTs[1][h])
            dX = deadPMTs[0][h]
            dZ = deadPMTs[2][h]
            phi = abs(np.arctan(dZ/dX)/np.pi)
            distance_X = (phi+0.5)*np.pi*(TRadius/100)
            if (dX>0 and dZ>0):
                distance_X = ((0.5-phi)*np.pi*(TRadius/100))
            elif (dX<0 and dZ>0):
                distance_X = -((0.5-phi)*np.pi*(TRadius/100))
            elif (dX<0 and dZ<0):
                distance_X = -((0.5+phi)*np.pi*(TRadius/100))
            Hit_2DX_dead.append(distance_X)
            
    
    Hit_2DX_dead = [hit*100 for hit in Hit_2DX_dead]
    Hit_2DY_dead = [hit*100 for hit in Hit_2DY_dead]

    
    dead = ax_tank.scatter(Hit_2DX_dead, Hit_2DY_dead, color = 'grey', alpha = 0.5, s = 200)

        
    #sc = ax_tank.scatter(Hit_2DX, Hit_2DY, c=Hit_PE, cmap='inferno', norm=plt.Normalize(vmin=0, vmax=normMax), s = 200)
    sc = ax_tank.scatter(Hit_2DX, Hit_2DY, c=Hit_PE, cmap='viridis', norm=plt.Normalize(vmin=0, vmax=normMax), s = 200)
    #cbar = plt.colorbar(sc, ax=ax_tank)
    #cbar.set_label('PE', fontsize = 20)
    #cbar.ax.tick_params(labelsize=15)
    #cbar.ax.set_position([0.85, 0.2, 0.2, 0.2])

    #divider = make_axes_locatable(ax_tank)
    #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    #cax = divider.append_axes("right", size="3%", pad=0)  # width 1%, pad 0.1
    #pos = cax.get_position()  # 获取当前 color bar 的位置
    #cax.set_position([pos.x0, pos.y0 + pos.height * 0.35, pos.width, pos.height * 0.3])  # x0不变，高度30%，居中
    # 创建 colorbar
    cbar = plt.colorbar(sc, ax=ax_tank, shrink = 0.3, anchor=(0.0, 0.3), pad = 0, fraction = 0.1)
    cbar.set_label('PMT Hit PE', fontsize=15)
    cbar.ax.tick_params(labelsize=15)


    rect = plt.Rectangle((-np.pi*TRadius, TC[1] - THeight/2), 2*np.pi*TRadius, THeight, linewidth=2, edgecolor='black', facecolor='none')
    ax_tank.add_patch(rect)
    circle1 = plt.Circle((TC[0], TC[1] + THeight/2 + TRadius), TRadius, linewidth=2, edgecolor='black', facecolor='none')
    ax_tank.add_patch(circle1)
    circle2 = plt.Circle((TC[0], TC[1] - THeight/2 - TRadius), TRadius, linewidth=2, edgecolor='black', facecolor='none')
    ax_tank.add_patch(circle2)
    rect_mrd = patches.Rectangle((TC[0]-152.25, TC[1]-131.8), 152.2*2, 131.8 * 2, linewidth=1, edgecolor='black', facecolor='none', linestyle='dashed')
    ax_tank.add_patch(rect_mrd) 
    
    LAPPDsquare = patches.Rectangle((TC[0]-10, TC[1]-10), 20, 20, linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
    ax_tank.add_patch(LAPPDsquare)

    #ax_tank.set_title('Tank PMT Hits ')
    #ax_tank.set_xlabel('X Coordinate')
    #ax_tank.set_ylabel('Z Coordinate')
    ax_tank.set_xlim(-600, 600)
    ax_tank.set_ylim(-600, 1000)
    
    pmt_text_x = -550
    pmt_text_y1 = 840
    
    pmt_text_fontsize = 12
    pmt_text_foutspace = 40
    ax_tank.text(pmt_text_x, 900, 'ANNIE Phase II', fontsize=20, color='purple', fontweight='bold', ha='left')

    for text_n in range(len(EventInfo)):
        ax_tank.text(pmt_text_x, 840 - pmt_text_foutspace*text_n, EventInfo[text_n], fontsize=pmt_text_fontsize, color='black', ha='left')

    beam_trigger_time_ns = int(EventInfo[-1])  
    beam_trigger_time_s = beam_trigger_time_ns / 1e9  
    utc_time = datetime.datetime.utcfromtimestamp(beam_trigger_time_s)
    central_time = utc_time.astimezone(pytz.timezone('US/Central'))
    formatted_time = central_time.strftime('%Y-%m-%d %H:%M:%S')
    ax_tank.text(pmt_text_x, 840 - pmt_text_foutspace*(len(EventInfo)), formatted_time, fontsize=pmt_text_fontsize, color='black', ha='left')
    

    for text_n in range (len(PMTClusterInfo)):
        ax_tank.text(pmt_text_x, -300 - pmt_text_foutspace*text_n, PMTClusterInfo[text_n], fontsize=pmt_text_fontsize, color='black', ha='left')
    ax_tank.text(pmt_text_x, -300 - pmt_text_foutspace*(len(PMTClusterInfo)), 'PMT plot max PE: {:.2f}'.format(normMax), fontsize=pmt_text_fontsize, color='black', ha='left')
    ax_tank.text(pmt_text_x, -300 - pmt_text_foutspace*(len(PMTClusterInfo)+1), 'PMT plot threshold: 20%', fontsize=pmt_text_fontsize, color='black', ha='left')
    


    
    ####################################################
    #########           plot muons             #########
    ####################################################
    #MRDHits = [MRDhitChankey, MRDhitT]
    #MRDTracks = [MRDTrackStartX, MRDTrackStopX, MRDTrackStartY, MRDTrackStopY, MRDTrackStartZ, MRDTrackStopZ]
    MRD_YZ_Plot_Side = [[],[],[]] # Z is x axis, Y is y axis
    MRD_XZ_Plot_Top = [[],[],[]]  # X is x axis, Z is y axis
    
    for h in range (len(MRDHits[0])):
        if(MRDGeoAccessor.orientation(MRDHits[0][h])==0):
            MRD_YZ_Plot_Side[0].append(MRDGeoAccessor.z_center(MRDHits[0][h]))
            MRD_YZ_Plot_Side[1].append(MRDGeoAccessor.y_center(MRDHits[0][h]))
            MRD_YZ_Plot_Side[2].append(MRDHits[1][h])
        elif (MRDGeoAccessor.orientation(MRDHits[0][h])==1):
            MRD_XZ_Plot_Top[0].append(MRDGeoAccessor.x_center(MRDHits[0][h]))
            MRD_XZ_Plot_Top[1].append(MRDGeoAccessor.z_center(MRDHits[0][h]))
            MRD_XZ_Plot_Top[2].append(MRDHits[1][h])

    # MRD Side view
    if(len(MRD_YZ_Plot_Side[0])>0):
        myz = axmuyz.scatter(MRD_YZ_Plot_Side[0], MRD_YZ_Plot_Side[1], c = MRD_YZ_Plot_Side[2], cmap='viridis', norm=plt.Normalize(vmin=np.min(MRD_YZ_Plot_Side[2]), vmax=np.max(MRD_YZ_Plot_Side[2])), marker='s', s = 40)

        axmuyz.set_xlabel('Z Coordinate')
        axmuyz.set_ylabel('Y Coordinate')
        axmuyz.set_xlim(-150, 600)
        axmuyz.set_ylim(-250, 210)
        axmuyz.set_title('MRD Hit Side View')
        

        tank_side = patches.Rectangle((TC[2]- TRadius, TC[1] - THeight/2), TRadius*2, THeight, linewidth=2, edgecolor='black', facecolor='none')
        axmuyz.add_patch(tank_side) 
        rect_yz = patches.Rectangle((325.5, TC[1]-131.8), 140.49, 131.8 * 2, linewidth=2, edgecolor='black', facecolor='none')
        axmuyz.add_patch(rect_yz)  
        rect_yz_f = patches.Rectangle(( TC[2]-TPMTRadius ,TC[1] - TPMTHeight/2)*100, TPMTRadius*2, TPMTHeight, linewidth=1, edgecolor='blue', facecolor='none', linestyle='dashed')
        axmuyz.add_patch(rect_yz_f)  
        
        cbar1 = plt.colorbar(myz, ax=axmuyz, orientation='horizontal', shrink=0.8, pad = 0)
        cbar1.set_label('MRD Hit Time (ns)', fontsize=8)
        cbar1.ax.tick_params(labelsize=8)
        cbar1.locator = MaxNLocator(integer=True)
        cbar1.update_ticks()

        axmuyz.plot([LAPPD_Center[0][2], LAPPD_Center[0][2]], [LAPPD_Center[0][1] - LAPPD_V_height/2, LAPPD_Center[0][1] + LAPPD_V_height/2], color='red', linewidth=2)
        #print(MRDTracks)
        for t in range (len(MRDTracks[0])):            

            delta_z = MRDTracks[5][t]*100 - MRDTracks[4][t]*100
            delta_y = MRDTracks[3][t]*100 - MRDTracks[2][t]*100
            if hasVeto:
                z_start = -200
            else:
                z_start = 150
            extended_y_start = MRDTracks[2][t]*100 + (z_start - MRDTracks[4][t]*100) * delta_y / delta_z
            extended_y_end = MRDTracks[3][t]*100 + (600 - MRDTracks[5][t]*100) * delta_y / delta_z
            axmuyz.plot([z_start, 600], [extended_y_start, extended_y_end], 'b--', linewidth=1)
            
            axmuyz.plot([MRDTracks[4][t]*100, MRDTracks[5][t]*100], [MRDTracks[2][t]*100, MRDTracks[3][t]*100], 'g-', linewidth=2)

    
    # MRD Top view
    if(len(MRD_XZ_Plot_Top[0])>0):
        mxz = axmuxz.scatter(MRD_XZ_Plot_Top[0], MRD_XZ_Plot_Top[1], c = MRD_XZ_Plot_Top[2], cmap='viridis', norm=plt.Normalize(vmin=np.min(MRD_XZ_Plot_Top[2]), vmax=np.max(MRD_XZ_Plot_Top[2])), marker='s', s = 40)
        
        axmuxz.set_xlabel('X Coordinate')
        axmuxz.set_ylabel('Z Coordinate')
        axmuxz.set_xlim(-300, 300)
        axmuxz.set_ylim(-50, 500)
        axmuxz.set_title('MRD Hit Top View')
        
        circle1 = plt.Circle((TC[0], TC[2]), TRadius, linewidth=2, edgecolor='black', facecolor='none')
        axmuxz.add_patch(circle1)
        circle2 = plt.Circle((TC[0], TC[2]), TPMTRadius, linewidth=1, edgecolor='blue', facecolor='none', linestyle='dashed')
        axmuxz.add_patch(circle2)        
        
        rect_xz = patches.Rectangle((-152.25, 325.5), 152.25*2, 140.49, linewidth=2, edgecolor='black', facecolor='none')
        axmuxz.add_patch(rect_xz)  
        cbar2 = plt.colorbar(mxz, ax=axmuxz, orientation='horizontal', shrink=0.8, pad = 0)
        cbar2.set_label('MRD Hit Time (ns)', fontsize=8)
        cbar2.ax.tick_params(labelsize=8)
        cbar2.locator = MaxNLocator(integer=True)
        cbar2.update_ticks()
        
        axmuxz.plot([LAPPD_Center[0][0] - LAPPD_H_width/2, LAPPD_Center[0][0] + LAPPD_H_width/2], [LAPPD_Center[0][2], LAPPD_Center[0][2]], color='red', linewidth=2)

        for t in range (len(MRDTracks[0])):
            
            delta_x = MRDTracks[1][t]*100 - MRDTracks[0][t]*100
            delta_z = MRDTracks[5][t]*100 - MRDTracks[4][t]*100
            if hasVeto:
                z_start = -100
            else:
                z_start = 150
            extended_x_start = MRDTracks[0][t]*100 + (z_start - MRDTracks[4][t]*100) * delta_x / delta_z
            extended_x_end = MRDTracks[1][t]*100 + (550 - MRDTracks[5][t]*100) * delta_x / delta_z
            axmuxz.plot([extended_x_start, extended_x_end], [z_start, 550], 'b--', linewidth=1)
            
            axmuxz.plot([MRDTracks[0][t]*100, MRDTracks[1][t]*100], [MRDTracks[4][t]*100, MRDTracks[5][t]*100], 'g-', linewidth=2)
         
            
    ####################################################
    #########         plot LAPPD hits          #########
    ####################################################   
    # LAPPDHits = [LAPPDID, LAPPDHitStrip, LAPPDHitTime, LAPPDHitAmp]
    maxPlotTime = 13
    minPlotTime = 8
    
    LHit = [[[],[],[]],[[],[],[]],[[],[],[]]]
    # Convert hit time from bin to ns
    for h in range(len(LAPPDHits[0])):
        LHit[LAPPDHits[0][h]][0].append(LAPPDHits[1][h])  # flip x axis
        LHit[LAPPDHits[0][h]][1].append(LAPPDHits[2][h] / 10)  # time in ns
        LHit[LAPPDHits[0][h]][2].append(LAPPDHits[3][h])       # amp
    # LHit[id][info], where info = strip, time, amp
    
    if len(LHit[0][2]) > 0:
        # Convert LHit times to NumPy array for element-wise comparison
        LHit_strips = np.array(LHit[0][0])  # Strip numbers
        LHit_times = np.array(LHit[0][1])   # Hit times
        LHit_amplitudes = np.array(LHit[0][2])  # Amplitudes
        
        # Apply the mask to filter the times and associated data
        mask = (LHit_times >= minPlotTime) & (LHit_times <= maxPlotTime)
        selected_times = LHit_times[mask]         # Filtered hit times
        minTime = min(selected_times)
        mask2 = (LHit_times >= minTime) & (LHit_times <= minTime+2)

        # Now use the mask to select filtered data
        selected_times = [i - minTime for i in LHit_times[mask2]]      # Filtered hit times
        selected_strips = LHit_strips[mask2]       # Strip numbers corresponding to the time mask
        selected_amplitudes = LHit_amplitudes[mask2]  # Filtered amplitudes        

        if len(selected_times) > 0:
            min_val = np.min(selected_times)
            max_val = np.max(selected_times)
            dT = max_val - min_val
    
            # Set Y-axis limits with a buffer
            y_min = min_val - 0.1 * dT
            y_max = max_val + 0.1 * dT
            ax_l0.set_ylim(y_min, y_max)
    
          
            # Create the scatter plot using amplitude data for colors
            l0 = ax_l0.scatter(selected_strips, selected_times,
                               c=selected_amplitudes, cmap = 'YlGnBu', norm=plt.Normalize(vmin=0, vmax=np.max(selected_amplitudes)),
                               s=60, marker='o',
                               edgecolors='face', linewidths=5, alpha=0.8)
            
            # Add colorbar
            cbar0 = plt.colorbar(l0, ax=ax_l0, shrink=0.8)
            cbar0.set_label('Amplitude (mV)')
    
            # Draw a black dashed line at y=0
            ax_l0.axhline(min(selected_times), color='black', linestyle='--', linewidth=1)
    
            # Set axis labels and title
            ax_l0.set_title('Hits on LAPPD ID=0')
            ax_l0.set_xlabel('Strip Number')
            ax_l0.set_ylabel('Hit Time (ns)')
            ax_l0.set_xlim(0, 30)

            dT = max(selected_times) - min(selected_times)

            if(dT>0):
                ax_l0.set_ylim(min(selected_times) - 0.1*dT, max(selected_times) + 0.1*dT)
            else:
                ax_l0.set_ylim(min(selected_times) - 0.5, min(selected_times) + 0.5)

            
            min_time = min(selected_times)
            max_time = max(selected_times)
            dT = max_time - min_time
            

            y_min = - 0.2
            y_max = max_time - min_time + 0.2
            
            # 设置 y 轴范围
            ax_l0.set_ylim(y_min, y_max)
            
            # 生成以 0.2 为间隔的 y 轴刻度，并确保包含 0
            yti = -0.5
            yticks = []
            while(yti<y_max+0.5):
                yticks.append(yti)
                yti += 0.5
            
            # 根据 min_time 生成新的标签，min_time 对应的刻度为 0
            new_yticklabels = [f"{(ytick):.2f}" for ytick in yticks]
            
            # 设置新的 y 轴刻度和标签
            ax_l0.set_yticks(yticks + min_time)
            ax_l0.set_yticklabels(new_yticklabels)

            #print(min_time, max_time, dT, y_min, y_max, yticks, new_yticklabels)
            
    
            # Adjust the position of the plot
            pos = ax_l0.get_position()
            new_pos0 = [pos.x0, pos.y0 + 0.05, pos.width, pos.height * 0.6]
            ax_l0.set_position(new_pos0)
            
        else:
            ax_l0.axis('off')


        
    if(len(LHit[1][2])>0):
        l1 = ax_l1.scatter(LHit[1][0], LHit[1][1], c = LHit[1][2], cmap = 'YlGnBu', norm=plt.Normalize(vmin=0, vmax=np.max(LHit[1][2])),s = 50, marker='o')
        cbar1 = plt.colorbar(l1, ax=ax_l1, shrink=0.8)
        cbar1.set_label('Amplitude (mV)') 
        ax_l1.set_title('Hits on LAPPD ID=1')
        ax_l1.set_xlabel('Strip Number')
        ax_l1.set_ylabel('Hit Time (ns)')
        ax_l1.set_xlim(0, 30)
        ax_l1.set_ylim(8, 13)
        pos = ax_l1.get_position()
        new_pos1 = [pos.x0, pos.y0 + 0.05, pos.width, pos.height * 0.6]
        ax_l1.set_position(new_pos1)
    else:
        ax_l1.axis('off')
        
    if(len(LHit[2][2])>0):
        l2 = ax_l2.scatter(LHit[2][0], LHit[2][1], c = LHit[2][2], cmap = 'YlGnBu', norm=plt.Normalize(vmin=0, vmax=np.max(LHit[2][2])),s = 50, marker='o')
        cbar2 = plt.colorbar(l2, ax=ax_l2, shrink=0.8)
        cbar2.set_label('Amplitude (mV)')
        ax_l2.set_title('Hits on LAPPD ID=2')
        ax_l2.set_xlabel('Strip Number')
        ax_l2.set_ylabel('Hit Time (ns)')
        ax_l2.set_xlim(0, 30)
        ax_l2.set_ylim(8, 13)
        pos = ax_l2.get_position()
        new_pos2 = [pos.x0, pos.y0 + 0.05, pos.width, pos.height * 0.6]
        ax_l2.set_position(new_pos2)   
    else:
        ax_l2.axis('off')                  

                  
    pdf.savefig(fig, bbox_inches = 'tight')
    plt.close(fig)
    # print this event into a pdf
    
    '''
    input information:
    select and pass the right information to this function 
    PMTHits = [hitX, hitY, hitZ, hitT, hitPE, hitChanKey]
    MRDHits = [MRDhitChankey, MRDhitT]
    MRDTracks = [MRDTrackStartX, MRDTrackStopX, MRDTrackStartY, MRDTrackStopY, MRDTrackStartZ, MRDTrackStopZ]
    LAPPDHits = [LAPPDID, LAPPDHitStrip, LAPPDHitTime, LAPPDHitAmp]
    '''
    PMTHits_raw_4_array = np.array(PMTHits_raw[4])
    valid_indices = np.isfinite(PMTHits_raw_4_array)
    for ind in range (len(PMTHits_raw)):
        HitArray = np.array(PMTHits_raw[ind])
        filtered_PMTHits = HitArray[valid_indices]
        PMTHits_raw[ind] = filtered_PMTHits.tolist()


    
    PMTHits = process_pmt_hits(PMTHits_raw)
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 16))
    for ax in fig.get_axes():
        ax.remove()
    
    ax_tank = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax_tank.axis('off')
    
    axmuyz = plt.subplot2grid((3, 3), (2, 0)) #muon y-z
    axmuyz.axis('off')
    axmuxz = plt.subplot2grid((3, 3), (2, 1)) #muon x-z
    axmuxz.axis('off')
    
    ax_l0 = plt.subplot2grid((3, 3), (0, 2)) #LAPPD 0 hit time
    ax_l1 = plt.subplot2grid((3, 3), (1, 2)) #LAPPD 1 hit time
    ax_l2 = plt.subplot2grid((3, 3), (2, 2)) #LAPPD 2 hit time
    
    


    
    ####################################################
    #########        plot tank PMTs            #########
    ####################################################
    Hit_2DX = []
    Hit_2DY = []
    Hit_PE = []
    
    maxPE = max(PMTHits[4])
    secondMaxPE = max([x for x in PMTHits[4] if x != maxPE])
    normMax = maxPE
    if(maxPE > 2*secondMaxPE):
        normMax = secondMaxPE

    firstHitTime = min(PMTHits[3])
    timeCut = 20
    
    for h in range(len(PMTHits[0])):
        if(PMTHits[4][h]<2):
            continue

        if(PMTHits[3][h]>firstHitTime+timeCut):
            continue
            
        plotThis = True
        if(PMTHits[4][h]<0.2*normMax):
            plotThis = False
            ##check if there is a hit within 0.6m has PE > 10%
            for n in range (len(PMTHits[0])):
                if(PMTHits[4][n]>0.2*normMax):
                    distance = np.sqrt((PMTHits[0][n] - PMTHits[0][h])**2 + (PMTHits[1][n] - PMTHits[1][h])**2 + (PMTHits[2][n] - PMTHits[2][h])**2 )
                    if(distance<0.6):
                        plotThis = True
                        break
                        
        if(plotThis != True):
            continue
            
        Hit_PE.append(PMTHits[4][h])
        if(PMTHits[1][h]> 1.3):    # top
            Hit_2DX.append(PMTHits[0][h])
            Hit_2DY.append(TC[1]/100 + (THeight/2 + TRadius)/100 - (PMTHits[2][h]))
        elif(PMTHits[1][h]< -1.4):  # bottom
            Hit_2DX.append(PMTHits[0][h])
            Hit_2DY.append(TC[1]/100 - (THeight/2 + TRadius)/100 + (PMTHits[2][h]))
        else:
            Hit_2DY.append(PMTHits[1][h])
            dX = PMTHits[0][h]
            dZ = PMTHits[2][h]
            phi = abs(np.arctan(dZ/dX)/np.pi)
            distance_X = (phi+0.5)*np.pi*(TRadius/100)
            if (dX>0 and dZ>0):
                distance_X = ((0.5-phi)*np.pi*(TRadius/100))
            elif (dX<0 and dZ>0):
                distance_X = -((0.5-phi)*np.pi*(TRadius/100))
            elif (dX<0 and dZ<0):
                distance_X = -((0.5+phi)*np.pi*(TRadius/100))
            Hit_2DX.append(distance_X)
            
    Hit_2DX = [hit*100 for hit in Hit_2DX]
    Hit_2DY = [hit*100 for hit in Hit_2DY]
    
    # dead PMTs
    Hit_2DX_dead = []
    Hit_2DY_dead = []
    
    for h in range(len(deadPMTs[0])):
        if(deadPMTs[1][h]> 1.3):    # top
            Hit_2DX_dead.append(deadPMTs[0][h])
            Hit_2DY_dead.append(TC[1]/100 + (THeight/2 + TRadius)/100 - (deadPMTs[2][h]))
        elif(deadPMTs[1][h]< -1.4):  # bottom
            Hit_2DX_dead.append(deadPMTs[0][h])
            Hit_2DY_dead.append(TC[1]/100 - (THeight/2 + TRadius)/100 + (deadPMTs[2][h]))
        else:
            Hit_2DY_dead.append(deadPMTs[1][h])
            dX = deadPMTs[0][h]
            dZ = deadPMTs[2][h]
            phi = abs(np.arctan(dZ/dX)/np.pi)
            distance_X = (phi+0.5)*np.pi*(TRadius/100)
            if (dX>0 and dZ>0):
                distance_X = ((0.5-phi)*np.pi*(TRadius/100))
            elif (dX<0 and dZ>0):
                distance_X = -((0.5-phi)*np.pi*(TRadius/100))
            elif (dX<0 and dZ<0):
                distance_X = -((0.5+phi)*np.pi*(TRadius/100))
            Hit_2DX_dead.append(distance_X)
            
    
    Hit_2DX_dead = [hit*100 for hit in Hit_2DX_dead]
    Hit_2DY_dead = [hit*100 for hit in Hit_2DY_dead]

    
    dead = ax_tank.scatter(Hit_2DX_dead, Hit_2DY_dead, color = 'grey', alpha = 0.5, s = 200)

        
    #sc = ax_tank.scatter(Hit_2DX, Hit_2DY, c=Hit_PE, cmap='inferno', norm=plt.Normalize(vmin=0, vmax=normMax), s = 200)
    sc = ax_tank.scatter(Hit_2DX, Hit_2DY, c=Hit_PE, cmap='viridis', norm=plt.Normalize(vmin=0, vmax=normMax), s = 200)
    #cbar = plt.colorbar(sc, ax=ax_tank)
    #cbar.set_label('PE', fontsize = 20)
    #cbar.ax.tick_params(labelsize=15)
    #cbar.ax.set_position([0.85, 0.2, 0.2, 0.2])

    #divider = make_axes_locatable(ax_tank)
    #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    #cax = divider.append_axes("right", size="3%", pad=0)  # width 1%, pad 0.1
    #pos = cax.get_position()  # 获取当前 color bar 的位置
    #cax.set_position([pos.x0, pos.y0 + pos.height * 0.35, pos.width, pos.height * 0.3])  # x0不变，高度30%，居中
    # 创建 colorbar
    cbar = plt.colorbar(sc, ax=ax_tank, shrink = 0.3, anchor=(0.0, 0.3), pad = 0, fraction = 0.1)
    cbar.set_label('PMT Hit PE', fontsize=15)
    cbar.ax.tick_params(labelsize=15)


    rect = plt.Rectangle((-np.pi*TRadius, TC[1] - THeight/2), 2*np.pi*TRadius, THeight, linewidth=2, edgecolor='black', facecolor='none')
    ax_tank.add_patch(rect)
    circle1 = plt.Circle((TC[0], TC[1] + THeight/2 + TRadius), TRadius, linewidth=2, edgecolor='black', facecolor='none')
    ax_tank.add_patch(circle1)
    circle2 = plt.Circle((TC[0], TC[1] - THeight/2 - TRadius), TRadius, linewidth=2, edgecolor='black', facecolor='none')
    ax_tank.add_patch(circle2)
    rect_mrd = patches.Rectangle((TC[0]-152.25, TC[1]-131.8), 152.2*2, 131.8 * 2, linewidth=1, edgecolor='black', facecolor='none', linestyle='dashed')
    ax_tank.add_patch(rect_mrd) 
    
    LAPPDsquare = patches.Rectangle((TC[0]-10, TC[1]-10), 20, 20, linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
    ax_tank.add_patch(LAPPDsquare)

    #ax_tank.set_title('Tank PMT Hits ')
    #ax_tank.set_xlabel('X Coordinate')
    #ax_tank.set_ylabel('Z Coordinate')
    ax_tank.set_xlim(-600, 600)
    ax_tank.set_ylim(-600, 1000)
    
    pmt_text_x = -550
    pmt_text_y1 = 840
    
    pmt_text_fontsize = 12
    pmt_text_foutspace = 40



    
    ####################################################
    #########           plot muons             #########
    ####################################################
    #MRDHits = [MRDhitChankey, MRDhitT]
    #MRDTracks = [MRDTrackStartX, MRDTrackStopX, MRDTrackStartY, MRDTrackStopY, MRDTrackStartZ, MRDTrackStopZ]
    MRD_YZ_Plot_Side = [[],[],[]] # Z is x axis, Y is y axis
    MRD_XZ_Plot_Top = [[],[],[]]  # X is x axis, Z is y axis
    
    for h in range (len(MRDHits[0])):
        if(MRDGeoAccessor.orientation(MRDHits[0][h])==0):
            MRD_YZ_Plot_Side[0].append(MRDGeoAccessor.z_center(MRDHits[0][h]))
            MRD_YZ_Plot_Side[1].append(MRDGeoAccessor.y_center(MRDHits[0][h]))
            MRD_YZ_Plot_Side[2].append(MRDHits[1][h])
        elif (MRDGeoAccessor.orientation(MRDHits[0][h])==1):
            MRD_XZ_Plot_Top[0].append(MRDGeoAccessor.x_center(MRDHits[0][h]))
            MRD_XZ_Plot_Top[1].append(MRDGeoAccessor.z_center(MRDHits[0][h]))
            MRD_XZ_Plot_Top[2].append(MRDHits[1][h])

    # MRD Side view
    if(len(MRD_YZ_Plot_Side[0])>0):
        myz = axmuyz.scatter(MRD_YZ_Plot_Side[0], MRD_YZ_Plot_Side[1], c = MRD_YZ_Plot_Side[2], cmap='viridis', norm=plt.Normalize(vmin=np.min(MRD_YZ_Plot_Side[2]), vmax=np.max(MRD_YZ_Plot_Side[2])), marker='s', s = 40)

        axmuyz.set_xlabel('Z Coordinate')
        axmuyz.set_ylabel('Y Coordinate')
        axmuyz.set_xlim(-150, 600)
        axmuyz.set_ylim(-250, 210)
        axmuyz.set_title('MRD Hit Side View')
        

        tank_side = patches.Rectangle((TC[2]- TRadius, TC[1] - THeight/2), TRadius*2, THeight, linewidth=2, edgecolor='black', facecolor='none')
        axmuyz.add_patch(tank_side) 
        rect_yz = patches.Rectangle((325.5, TC[1]-131.8), 140.49, 131.8 * 2, linewidth=2, edgecolor='black', facecolor='none')
        axmuyz.add_patch(rect_yz)  
        rect_yz_f = patches.Rectangle(( TC[2]-TPMTRadius ,TC[1] - TPMTHeight/2)*100, TPMTRadius*2, TPMTHeight, linewidth=1, edgecolor='blue', facecolor='none', linestyle='dashed')
        axmuyz.add_patch(rect_yz_f)  
        
        cbar1 = plt.colorbar(myz, ax=axmuyz, orientation='horizontal', shrink=0.8, pad = 0)
        cbar1.set_label('MRD Hit Time (ns)', fontsize=8)
        cbar1.ax.tick_params(labelsize=8)
        cbar1.locator = MaxNLocator(integer=True)
        cbar1.update_ticks()

        axmuyz.plot([LAPPD_Center[0][2], LAPPD_Center[0][2]], [LAPPD_Center[0][1] - LAPPD_V_height/2, LAPPD_Center[0][1] + LAPPD_V_height/2], color='red', linewidth=2)
        #print(MRDTracks)
        for t in range (len(MRDTracks[0])):            

            delta_z = MRDTracks[5][t]*100 - MRDTracks[4][t]*100
            delta_y = MRDTracks[3][t]*100 - MRDTracks[2][t]*100
            if hasVeto:
                z_start = -200
            else:
                z_start = 150
            extended_y_start = MRDTracks[2][t]*100 + (z_start - MRDTracks[4][t]*100) * delta_y / delta_z
            extended_y_end = MRDTracks[3][t]*100 + (600 - MRDTracks[5][t]*100) * delta_y / delta_z
            axmuyz.plot([z_start, 600], [extended_y_start, extended_y_end], 'b--', linewidth=1)
            
            axmuyz.plot([MRDTracks[4][t]*100, MRDTracks[5][t]*100], [MRDTracks[2][t]*100, MRDTracks[3][t]*100], 'g-', linewidth=2)

    
    # MRD Top view
    if(len(MRD_XZ_Plot_Top[0])>0):
        mxz = axmuxz.scatter(MRD_XZ_Plot_Top[0], MRD_XZ_Plot_Top[1], c = MRD_XZ_Plot_Top[2], cmap='viridis', norm=plt.Normalize(vmin=np.min(MRD_XZ_Plot_Top[2]), vmax=np.max(MRD_XZ_Plot_Top[2])), marker='s', s = 40)
        
        axmuxz.set_xlabel('X Coordinate')
        axmuxz.set_ylabel('Z Coordinate')
        axmuxz.set_xlim(-300, 300)
        axmuxz.set_ylim(-50, 500)
        axmuxz.set_title('MRD Hit Top View')
        
        circle1 = plt.Circle((TC[0], TC[2]), TRadius, linewidth=2, edgecolor='black', facecolor='none')
        axmuxz.add_patch(circle1)
        circle2 = plt.Circle((TC[0], TC[2]), TPMTRadius, linewidth=1, edgecolor='blue', facecolor='none', linestyle='dashed')
        axmuxz.add_patch(circle2)        
        
        rect_xz = patches.Rectangle((-152.25, 325.5), 152.25*2, 140.49, linewidth=2, edgecolor='black', facecolor='none')
        axmuxz.add_patch(rect_xz)  
        cbar2 = plt.colorbar(mxz, ax=axmuxz, orientation='horizontal', shrink=0.8, pad = 0)
        cbar2.set_label('MRD Hit Time (ns)', fontsize=8)
        cbar2.ax.tick_params(labelsize=8)
        cbar2.locator = MaxNLocator(integer=True)
        cbar2.update_ticks()
        
        axmuxz.plot([LAPPD_Center[0][0] - LAPPD_H_width/2, LAPPD_Center[0][0] + LAPPD_H_width/2], [LAPPD_Center[0][2], LAPPD_Center[0][2]], color='red', linewidth=2)

        for t in range (len(MRDTracks[0])):
            
            delta_x = MRDTracks[1][t]*100 - MRDTracks[0][t]*100
            delta_z = MRDTracks[5][t]*100 - MRDTracks[4][t]*100
            if hasVeto:
                z_start = -100
            else:
                z_start = 150
            extended_x_start = MRDTracks[0][t]*100 + (z_start - MRDTracks[4][t]*100) * delta_x / delta_z
            extended_x_end = MRDTracks[1][t]*100 + (550 - MRDTracks[5][t]*100) * delta_x / delta_z
            axmuxz.plot([extended_x_start, extended_x_end], [z_start, 550], 'b--', linewidth=1)
            
            axmuxz.plot([MRDTracks[0][t]*100, MRDTracks[1][t]*100], [MRDTracks[4][t]*100, MRDTracks[5][t]*100], 'g-', linewidth=2)
         
            
    ####################################################
    #########         plot LAPPD hits          #########
    ####################################################   
    # LAPPDHits = [LAPPDID, LAPPDHitStrip, LAPPDHitTime, LAPPDHitAmp]
    maxPlotTime = 13
    minPlotTime = 8
    
    LHit = [[[],[],[]],[[],[],[]],[[],[],[]]]
    # Convert hit time from bin to ns
    for h in range(len(LAPPDHits[0])):
        LHit[LAPPDHits[0][h]][0].append(LAPPDHits[1][h])  # flip x axis
        LHit[LAPPDHits[0][h]][1].append(LAPPDHits[2][h] / 10)  # time in ns
        LHit[LAPPDHits[0][h]][2].append(LAPPDHits[3][h])       # amp
    # LHit[id][info], where info = strip, time, amp
    
    if len(LHit[0][2]) > 0:
        # Convert LHit times to NumPy array for element-wise comparison
        LHit_strips = np.array(LHit[0][0])  # Strip numbers
        LHit_times = np.array(LHit[0][1])   # Hit times
        LHit_amplitudes = np.array(LHit[0][2])  # Amplitudes
        
        # Apply the mask to filter the times and associated data
        mask = (LHit_times >= minPlotTime) & (LHit_times <= maxPlotTime)
        selected_times = LHit_times[mask]         # Filtered hit times
        minTime = min(selected_times)
        mask2 = (LHit_times >= minTime) & (LHit_times <= minTime+2)

        # Now use the mask to select filtered data
        selected_times = [i - minTime for i in LHit_times[mask2]]      # Filtered hit times
        selected_strips = LHit_strips[mask2]       # Strip numbers corresponding to the time mask
        selected_amplitudes = LHit_amplitudes[mask2]  # Filtered amplitudes        

        if len(selected_times) > 0:
            min_val = np.min(selected_times)
            max_val = np.max(selected_times)
            dT = max_val - min_val
    
            # Set Y-axis limits with a buffer
            y_min = min_val - 0.1 * dT
            y_max = max_val + 0.1 * dT
            ax_l0.set_ylim(y_min, y_max)
    
          
            # Create the scatter plot using amplitude data for colors
            l0 = ax_l0.scatter(selected_strips, selected_times,
                               c=selected_amplitudes, cmap = 'YlGnBu', norm=plt.Normalize(vmin=0, vmax=np.max(selected_amplitudes)),
                               s=60, marker='o',
                               edgecolors='face', linewidths=5, alpha=0.8)
            
            # Add colorbar
            cbar0 = plt.colorbar(l0, ax=ax_l0, shrink=0.8)
            cbar0.set_label('Amplitude (mV)')
    
            # Draw a black dashed line at y=0
            ax_l0.axhline(min(selected_times), color='black', linestyle='--', linewidth=1)
    
            # Set axis labels and title
            ax_l0.set_title('Hits on LAPPD ID=0')
            ax_l0.set_xlabel('Strip Number')
            ax_l0.set_ylabel('Hit Time (ns)')
            ax_l0.set_xlim(0, 30)

            dT = max(selected_times) - min(selected_times)

            if(dT>0):
                ax_l0.set_ylim(min(selected_times) - 0.1*dT, max(selected_times) + 0.1*dT)
            else:
                ax_l0.set_ylim(min(selected_times) - 0.5, min(selected_times) + 0.5)

            
            min_time = min(selected_times)
            max_time = max(selected_times)
            dT = max_time - min_time
            

            y_min = - 0.2
            y_max = max_time - min_time + 0.2
            
            ax_l0.set_ylim(y_min, y_max)
            
            yti = -0.5
            yticks = []
            while(yti<y_max+0.5):
                yticks.append(yti)
                yti += 0.5
            
            new_yticklabels = [f"{(ytick):.2f}" for ytick in yticks]
            
            ax_l0.set_yticks(yticks + min_time)
            ax_l0.set_yticklabels(new_yticklabels)

            #print(min_time, max_time, dT, y_min, y_max, yticks, new_yticklabels)
            
    
            # Adjust the position of the plot
            pos = ax_l0.get_position()
            new_pos0 = [pos.x0, pos.y0 + 0.05, pos.width, pos.height * 0.6]
            ax_l0.set_position(new_pos0)
            
        else:
            ax_l0.axis('off')


        
    if(len(LHit[1][2])>0):
        l1 = ax_l1.scatter(LHit[1][0], LHit[1][1], c = LHit[1][2], cmap = 'YlGnBu', norm=plt.Normalize(vmin=0, vmax=np.max(LHit[1][2])),s = 50, marker='o')
        cbar1 = plt.colorbar(l1, ax=ax_l1, shrink=0.8)
        cbar1.set_label('Amplitude (mV)') 
        ax_l1.set_title('Hits on LAPPD ID=1')
        ax_l1.set_xlabel('Strip Number')
        ax_l1.set_ylabel('Hit Time (ns)')
        ax_l1.set_xlim(0, 30)
        ax_l1.set_ylim(8, 13)
        pos = ax_l1.get_position()
        new_pos1 = [pos.x0, pos.y0 + 0.05, pos.width, pos.height * 0.6]
        ax_l1.set_position(new_pos1)
    else:
        ax_l1.axis('off')
        
    if(len(LHit[2][2])>0):
        l2 = ax_l2.scatter(LHit[2][0], LHit[2][1], c = LHit[2][2], cmap = 'YlGnBu', norm=plt.Normalize(vmin=0, vmax=np.max(LHit[2][2])),s = 50, marker='o')
        cbar2 = plt.colorbar(l2, ax=ax_l2, shrink=0.8)
        cbar2.set_label('Amplitude (mV)')
        ax_l2.set_title('Hits on LAPPD ID=2')
        ax_l2.set_xlabel('Strip Number')
        ax_l2.set_ylabel('Hit Time (ns)')
        ax_l2.set_xlim(0, 30)
        ax_l2.set_ylim(8, 13)
        pos = ax_l2.get_position()
        new_pos2 = [pos.x0, pos.y0 + 0.05, pos.width, pos.height * 0.6]
        ax_l2.set_position(new_pos2)   
    else:
        ax_l2.axis('off')                  

                  
    pdf.savefig(fig, bbox_inches = 'tight')
    plt.close(fig)


def DisplayHits(savePath, hits, LAPPD_grids):
    num_LAPPDs = len(LAPPD_grids)
    hit_pe_matrices = [np.zeros(LAPPD_grids[i].shape[:2]) for i in range(num_LAPPDs)]
    hit_time_matrices = [np.zeros(LAPPD_grids[i].shape[:2]) for i in range(num_LAPPDs)]
    hit_counts = [np.zeros(LAPPD_grids[i].shape[:2]) for i in range(num_LAPPDs)]

    tick_fontsize = 18
    label_fontsize = 18
    text_fontsize = 24

    for p in range(len(hits)):
        for hit in hits[p]:
            LAPPD_index, first_index, second_index, hit_time, photon_distance, weighted_pe = hit
            #print(LAPPD_index, first_index, second_index, hit_time, photon_distance, weighted_pe)
            hit_pe_matrices[LAPPD_index][27-first_index, second_index] += weighted_pe
            #print(hit_pe_matrices[LAPPD_index][28-first_index, second_index])
            hit_time_matrices[LAPPD_index][27-first_index, second_index] = (hit_time + hit_time_matrices[LAPPD_index][27-first_index, second_index])/2

    xSize = 10 * num_LAPPDs
    fig, axis = plt.subplots(2, num_LAPPDs, figsize=(xSize, 20), squeeze=False)
    for i in range(num_LAPPDs):
        impe = axis[0, i].imshow(np.ma.masked_equal(hit_pe_matrices[i], 0), cmap='viridis', interpolation='nearest')
        axis[0, i].set_title(f'LAPPD {i} PE', fontsize=label_fontsize)
        axis[0, i].set_xlabel('Strip Number', fontsize=label_fontsize)
        axis[0, i].set_ylabel('Position on Strip', fontsize=label_fontsize)
        axis[0, i].tick_params(axis='both', labelsize=tick_fontsize)
        #axis[0, i].set_xlim(0, 28)
        #axis[0, i].set_ylim(0, 28)
        plt.colorbar(impe, ax=axis[0, i])

        imtime = axis[1, i].imshow(np.ma.masked_equal(hit_time_matrices[i], 0), cmap='coolwarm', interpolation='nearest')
        axis[1, i].set_title(f'LAPPD {i} Hit Time', fontsize=label_fontsize)
        axis[1, i].set_xlabel('Strip Number', fontsize=label_fontsize)
        axis[1, i].set_ylabel('Position on Strip', fontsize=label_fontsize)
        axis[1, i].tick_params(axis='both', labelsize=tick_fontsize)
        #axis[1, i].set_xlim(0, 28)
        #axis[1, i].set_ylim(0, 28)
        plt.colorbar(imtime, ax=axis[1, i])

    plt.tight_layout()
    plt.savefig(savePath, transparent=True)
    plt.close(fig)


def plotWaveforms(SavePath, DataWaveform, SimWaveform, xlim, SimPE = -1, DataPE = -1, LAPPD_stripWidth = 0.462, LAPPD_stripSpace = 0.229):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    colors = cm.rainbow(np.linspace(0, 1, 30))

    tick_fontsize = 18
    label_fontsize = 18
    text_fontsize = 24
    
    LAPPD_gridSize = LAPPD_stripWidth + LAPPD_stripSpace

    # side 0 is bottom, side 1 is top
    
    # plot gradient
    Data_PulseStart = []
    Sim_PulseStart = []
    Data_RecoHit = []
    Sim_RecoHit = []
    
    for strip in range(28):
        DWaveform_bottom = DataWaveform[strip, 0]
        DWaveform_top = DataWaveform[strip, 1]
        SWaveform_bottom = SimWaveform[strip][0]
        SWaveform_top = SimWaveform[strip][1]
        
        D_bot_t = find_first_time_bin_above_threshold(DWaveform_bottom)
        D_top_t = find_first_time_bin_above_threshold(DWaveform_top)
        S_bot_t = find_first_time_bin_above_threshold(SWaveform_bottom, cut = (0,256))
        S_top_t = find_first_time_bin_above_threshold(SWaveform_top, cut = (0,256))
        
        #if D_bot_t != none:
        if(D_bot_t != None and D_top_t != None):
            Data_T = (strip, D_top_t , D_bot_t)
            Data_PulseStart.append(Data_T)
            hit_t = calculate_gradient_hit_time(Data_T, LAPPD_gridSize)
            Data_RecoHit.append((strip, hit_t))
            
        if(S_bot_t != None and S_top_t != None):
            Sim_T = (strip, S_top_t, S_bot_t)
            Sim_PulseStart.append(Sim_T)
            hit_t = calculate_gradient_hit_time(Sim_T, LAPPD_gridSize)
            Sim_RecoHit.append((strip, hit_t))
            
    print(" ")
    print("Data_RecoHit=", Data_RecoHit)
    print("Sim_RecoHit=", Sim_RecoHit)
    print("    ")
    

    # Plotting DataWaveform, side 0
    for strip in range(28):
        axes[0, 0].plot(DataWaveform[strip, 0], label=f'Strip {strip}', color=colors[strip])
    axes[0, 0].set_title('Data - Side 0', fontsize=label_fontsize)
    axes[0, 0].set_xlim(xlim)
    axes[0, 0].set_xlabel('Time (0.1ns)', fontsize=label_fontsize)
    axes[0, 0].set_ylabel('Amplitude (mV)', fontsize=label_fontsize)
    axes[0, 0].tick_params(axis='both', labelsize=tick_fontsize)  # 设置tick的字号
    axes[0, 0].legend(loc='upper left')
    if DataPE != -1:
        axes[0, 0].text(0.05, 0.05, f"Total PE: {DataPE}", transform=axes[0, 0].transAxes, 
                        fontsize=text_fontsize, verticalalignment='bottom')

    # Plotting DataWaveform, side 1
    for strip in range(28):
        axes[0, 1].plot(DataWaveform[strip, 1], label=f'Strip {strip}', color=colors[strip])
    axes[0, 1].set_title('Data - Side 1', fontsize=label_fontsize)
    axes[0, 1].set_xlim(xlim)
    axes[0, 1].set_xlabel('Time (0.1ns)', fontsize=label_fontsize)
    axes[0, 1].set_ylabel('Amplitude (mV)', fontsize=label_fontsize)
    axes[0, 1].tick_params(axis='both', labelsize=tick_fontsize)  # 设置tick的字号
    axes[0, 1].legend(loc='upper left')
    if DataPE != -1:
        axes[0, 1].text(0.05, 0.05, f"Total PE: {DataPE}", transform=axes[0, 1].transAxes, 
                        fontsize=text_fontsize, verticalalignment='bottom') 

    # Plotting SimWaveform, side 0
    for strip in range(28):
        axes[1, 0].plot(SimWaveform[strip][0], label=f'Strip {strip+1}', color=colors[strip])
    axes[1, 0].set_title('Sim - Side 0', fontsize=label_fontsize)
    axes[1, 0].set_xlim(xlim)
    axes[1, 0].set_xlabel('Time (0.1ns)', fontsize=label_fontsize)
    axes[1, 0].set_ylabel('Amplitude (mV)', fontsize=label_fontsize)
    axes[1, 0].tick_params(axis='both', labelsize=tick_fontsize)  # 设置tick的字号
    axes[1, 0].legend(loc='upper left')
    if SimPE != -1:
        axes[1, 0].text(0.05, 0.05, f"Total PE: {SimPE:.1f}", transform=axes[1, 0].transAxes, 
                        fontsize=text_fontsize, verticalalignment='bottom')
    # Plotting SimWaveform, side 1
    for strip in range(28):
        axes[1, 1].plot(SimWaveform[strip][1], label=f'Strip {strip+1}', color=colors[strip])
    axes[1, 1].set_title('Sim - Side 1', fontsize=label_fontsize)
    axes[1, 1].set_xlim(xlim)
    axes[1, 1].set_xlabel('Time (0.1ns)', fontsize=label_fontsize)
    axes[1, 1].set_ylabel('Amplitude (mV)', fontsize=label_fontsize)
    axes[1, 1].tick_params(axis='both', labelsize=tick_fontsize)  # 设置tick的字号
    axes[1, 1].legend(loc='upper left')
    if SimPE != -1:
        axes[1, 1].text(0.05, 0.05, f"Total PE: {SimPE:.1f}", transform=axes[1, 1].transAxes, 
                        fontsize=text_fontsize, verticalalignment='bottom')
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(SavePath, transparent=True)
    plt.close(fig)



# find the time bin where the waveform first exceeds a threshold, up to 10 ps level
def find_first_time_bin_above_threshold(waveform, threshold=7, cut = (150,250)):
    waveform = waveform[cut[0]:cut[1]]
    for i in range(1, len(waveform)):
        if waveform[i] >= threshold:
            # 对 bin[i-1] 和 bin[i] 之间进行线性插值
            y1, y2 = waveform[i-1], waveform[i]
            x1, x2 = (i-1) * 0.1, i * 0.1  # 每个 bin 对应 0.1 ns
            if y2 != y1:  
                interpolated_time_ns = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
            else:
                interpolated_time_ns = x1  # 如果两个点相同，直接取x1
                
            #print(f"Interpolated time: {interpolated_time_ns}")
            return interpolated_time_ns+cut[0]*0.1
    return None 

def calculate_gradient_hit_time(pulse_Ts, LAPPD_gridSize):
    strip, top_t, bot_t = pulse_Ts
    reco_time = 0.5 * ((top_t + bot_t) - (28*LAPPD_gridSize/(0.567*(2.998e8))))
    return reco_time
    