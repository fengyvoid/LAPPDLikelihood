import ROOT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from tqdm import tqdm
import sys

# 从命令行参数读取阈值和事件索引
hit_threshold = 200
event_index = 16

# 打开ROOT文件
fileName = '/Users/fengy/ANNIESofts/Analysis/WCSimFile/ANNIETree_MC_withLAPPD.root'
file = ROOT.TFile(fileName, "READ")
tree = file.Get("Event")

# 初始化计数变量和存储数组
count = 0
HitXs, HitYs, HitZs, LMCHitTimes, TubeIDs = [], [], [], [], []
trueFLV = [[],[],[]]
trueFLM = [[],[],[]]

truePos = [[],[],[]]
trueDir = [[],[],[]]

track = [[],[],[],[],[],[]]

trueTrackLengthInWater = []

# 遍历所有事件并统计满足条件的事件数量
for i in tqdm(range(tree.GetEntries())):
    tree.GetEntry(i)
    
    if tree.TankMRDCoinc==0:
        continue

    #if tree.trueFSLPdg != 13:
    #    continue

    if tree.trueMuonEnergy < 0:
        continue

    if tree.trueCC != 1:
        continue
        
    if len(tree.LAPPDMCHitCharge) > hit_threshold:
        count += 1
        print("Got event",i," with count", count)
        
        HitXs.append(np.array(tree.LAPPDMCHitX))
        HitYs.append(np.array(tree.LAPPDMCHitY))
        HitZs.append(np.array(tree.LAPPDMCHitZ))
        LMCHitTimes.append(np.array(tree.LAPPDMCHitTime))
        TubeIDs.append(np.array(tree.LAPPDMCHitTubeIDs))

        trueFLV[0].append(tree.trueFSLVtx_X)
        trueFLV[1].append(tree.trueFSLVtx_Y)
        trueFLV[2].append(tree.trueFSLVtx_Z)
        trueFLM[0].append(tree.trueFSLMomentum_X)
        trueFLM[1].append(tree.trueFSLMomentum_Y)
        trueFLM[2].append(tree.trueFSLMomentum_Z)

        truePos[0].append(tree.trueVtxX)
        truePos[1].append(tree.trueVtxY)
        truePos[2].append(tree.trueVtxZ)
        trueDir[0].append(tree.trueDirX)
        trueDir[1].append(tree.trueDirY)
        trueDir[2].append(tree.trueDirZ)
        

        track[0].append(tree.MRDTrackStartX)
        track[1].append(tree.MRDTrackStartY)
        track[2].append(tree.MRDTrackStartZ)
        track[3].append(tree.MRDTrackStopX)
        track[4].append(tree.MRDTrackStopY)
        track[5].append(tree.MRDTrackStopZ)

        trueTrackLengthInWater.append(tree.trueTrackLengthInWater)


# 输出满足阈值条件的事件数量
print(f"Number of events with hits greater than {hit_threshold}: {count}")

# 检查事件索引是否有效
if event_index >= count:
    print(f"Error: Event index {event_index} is out of range. Only {count} events meet the hit threshold.")

# 绘制3D散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 使用 rainbow 颜色映射，并将时间归一化到 [0, 50]
norm = mcolors.Normalize(vmin=0, vmax=50)
sc = ax.scatter(HitXs[event_index], HitYs[event_index], HitZs[event_index], c=LMCHitTimes[event_index], cmap='rainbow', norm=norm, s=10)

# 添加颜色条
cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=8)
cbar.set_label('Time (ns)')

# 设置标签和标题
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
plt.title(f'3D Visualization of Hits for Event {event_index} with Time as Color')

# 设置初始视角
ax.view_init(elev=-15, azim=-90)

# 计算箭头的起点（位置除以100）和方向（归一化为单位向量）
start_point = np.array([
    truePos[0][event_index] / 100,
    truePos[1][event_index] / 100,
    truePos[2][event_index] / 100
])

momentum_vector = np.array([
    trueDir[0][event_index],
    trueDir[1][event_index],
    trueDir[2][event_index]
])

# 将方向向量归一化为单位向量
unit_vector = momentum_vector / np.linalg.norm(momentum_vector)

# 在3D图中添加箭头
ax.quiver(
    start_point[0], start_point[1], start_point[2],
    unit_vector[0], unit_vector[1], unit_vector[2],
    length=trueTrackLengthInWater[event_index], color='red', arrow_length_ratio=0.1
)

plt.show()


print(truePos[0][event_index], truePos[1][event_index], truePos[2][event_index])
print(trueDir[0][event_index], trueDir[1][event_index], trueDir[2][event_index])
print(trueTrackLengthInWater[event_index])
