import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm



input_folder = "/Users/fengy/ANNIESofts/Analysis/ProjectionComplete/OptimizationResults/5.Probability/output/"
output_folder = os.path.join(input_folder, "plot_withPECorrection")
#output_folder = os.path.join(input_folder, "plot")

os.makedirs(output_folder, exist_ok=True)

file_paths = sorted(glob.glob(os.path.join(input_folder, "event_*.h5")), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

if file_paths:
    file_paths.pop()  

max_points = []
all_x_values, all_y_values = [], []

plotEvent = 5
eventnum = 0

dStep = 0.0067
halfBin = dStep / 2

for file_path in file_paths:

    
    event_name = os.path.basename(file_path).split('.')[0]
    
    
    with h5py.File(file_path, "r") as h5f:
        #sample_idx = 0
        
        for sample_idx in range(1):
            shifted_info_group = h5f[f"sample_{sample_idx}/shifted_info"]
            hit2D = h5f[f"sample_{sample_idx}/sampled_2DHits"][()]
            nPESampled = 0
            for line in hit2D:
                for hit in line:
                    nPESampled+=hit[0]
                    
            if nPESampled < 1:
                continue

            x_values = []
            y_values = []
            PLog_values = []
            pdfs = []
            totalPEs = []
            deltaPEs = []
            

            for step_name in shifted_info_group.keys():
                step_group = shifted_info_group[step_name]
                x_values.append(step_group["xStep"][()])  
                y_values.append(step_group["yStep"][()])  

                pdf = step_group["pdf_step"][()]
                pdfs.append(pdf)

                totalPE = 0
                pes = np.zeros((28,28))
                for i in range(28):
                    for j in range(28):
                        pes[i,j] = pdf[i,j][0]
                        totalPE += pdf[i,j][0]
                totalPEs.append(totalPE)
                deltaPEs.append(totalPE - nPESampled)

                density = totalPE / (28 * 28)
                logPerPE = poisson.logpmf(1, density)  

                
                pLog = step_group["PLog"][()]
                deltaPLog = abs((totalPE - nPESampled) * logPerPE)
                PLog_values.append(pLog - deltaPLog)  
                #PLog_values.append(pLog)  
            
            max_PLog = max(PLog_values)
            if max_PLog == -np.inf:
                print(f"Skipping event {event_name}, sample {sample_idx} due to max PLog being -inf")
                continue  
            print("Event: ", event_name, "Sample: ", sample_idx, "Max PLog: ", max(PLog_values))
            maxIndex = np.argmax(PLog_values)
            max_points.append((x_values[maxIndex], y_values[maxIndex], PLog_values[maxIndex]))
            x_values = np.array(x_values)
            y_values = np.array(y_values)
            
            PLog_values = np.array(PLog_values)
            totalPEs = np.array(totalPEs)
            x_unique = np.unique(x_values)
            y_unique = np.unique(y_values)
            X, Y = np.meshgrid(x_unique, y_unique)

            PLog_grid = np.full(X.shape, np.nan)
            for i in range(len(x_values)):
                xi = np.where(x_unique == x_values[i])[0][0]
                yi = np.where(y_unique == y_values[i])[0][0]
                PLog_grid[yi, xi] = PLog_values[i]
                


            plt.figure(figsize=(8, 6))
            plt.imshow(PLog_grid, origin="lower", extent=[x_unique.min()-halfBin, x_unique.max()+halfBin, y_unique.min()-halfBin, y_unique.max()+halfBin],
                    aspect="auto", cmap="viridis")
            plt.colorbar(label="ln(P)")
            plt.xlabel(f"$\Delta$ X (m)")
            plt.ylabel(f"$\Delta$ Y (m)")
            plt.title("2D Heatmap of ln(P)")
            plt.grid(False)
            plt.savefig(os.path.join(output_folder, f"{event_name}_2D.png"))
            plt.close()
            

            PLog_norm = [i - max(PLog_values) for i in PLog_values]
            PLog_grid_norm = np.full(X.shape, np.nan)
            for i in range(len(x_values)):
                xi = np.where(x_unique == x_values[i])[0][0]
                yi = np.where(y_unique == y_values[i])[0][0]
                PLog_grid_norm[yi, xi] = PLog_norm[i]
            
            plt.figure(figsize=(8, 6))
            plt.imshow(PLog_grid_norm, origin="lower", extent=[x_unique.min()-halfBin, x_unique.max()+halfBin, y_unique.min()-halfBin, y_unique.max()+halfBin],
                    aspect="auto", cmap="viridis")
            plt.colorbar(label="ln(P) (relative)")
            plt.xlabel(f"$\Delta$ X (m)")
            plt.ylabel(f"$\Delta$ Y (m)")
            plt.title("2D Heatmap of ln(P) (relative)")
            plt.grid(False)
            plt.savefig(os.path.join(output_folder, f"{event_name}_{sample_idx}_2D_normed.png"))
            plt.close()

plt.figure(figsize=(8, 6))
maxXData = [i[0] for i in max_points]
maxYData = [i[1] for i in max_points]
plt.hist2d(maxXData, maxYData, bins =[len(x_unique), len(y_unique)], range=[(x_unique.min()-0.025, x_unique.max()+0.025), (y_unique.min()-0.025, y_unique.max()+0.025)], cmap='Blues')
plt.colorbar(label='Counts')
plt.xlabel(f"$\Delta$ X (m)")
plt.ylabel(f"$\Delta$ Y (m)")
plt.title("Delta position of the most probable solution for all events")
plt.savefig(os.path.join(output_folder, f"stat_hist2D.png"))
plt.close()

unique = y_unique

plt.figure(figsize=(8, 5))
counts, bins, _ = plt.hist(maxXData, bins=len(unique), range=(unique.min()-halfBin, unique.max()+halfBin), 
                            alpha=0.7, color='blue', density=True)
mu, sigma = norm.fit(maxXData)
x = np.linspace(bins.min(), bins.max(), 300)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'r-', lw=2, label='Gaussian Fit')
text_str = f"Mean: {mu:.3f}\nSigma: {sigma:.3f}"
plt.text(0.95, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))
#plt.figure(figsize=(8, 6))
#plt.hist(maxXData, bins = len(x_unique),range=(x_unique.min()-0.025, x_unique.max()+0.025), alpha=0.7, color='blue')
plt.xlabel(f"$\Delta$ X (m)")
plt.ylabel("")
plt.title("Delta position of the most probable solution")
plt.savefig(os.path.join(output_folder, f"stat_histX.png"))
plt.close()


plt.figure(figsize=(8, 5))
counts, bins, _ = plt.hist(maxYData, bins=len(unique), range=(unique.min()-halfBin, unique.max()+halfBin), 
                            alpha=0.7, color='red', density=True)
mu, sigma = norm.fit(maxYData)
x = np.linspace(bins.min(), bins.max(), 300)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y, 'r-', lw=2, label='Gaussian Fit')
text_str = f"Mean: {mu:.3f}\nSigma: {sigma:.3f}"
plt.text(0.95, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))
#plt.figure(figsize=(8, 6))
#plt.hist(maxYData, bins = len(y_unique),range=(y_unique.min()-0.025, y_unique.max()+0.025), alpha=0.7, color='blue')
plt.xlabel(f"$\Delta$ Y (m)")
plt.ylabel("")
plt.title("Delta position of the most probable solution")
plt.savefig(os.path.join(output_folder, f"stat_histY.png"))
plt.close()
