


def makePlotsBar(path, plotName, peak, plotPath, thr=15, plotOpt=True, r1=12, r2=16, whis=1.5, fs = 18):
    all_files = [f for f in os.listdir(path) if f.endswith('.hdf5') and 'WCSim_WithSimShift' in f]
    all_files.sort(key=natural_key)

    threshold = thr
    wc_thres_all, sim_thres_all, wc_avg_all = [], [], []

    for filename in all_files:
        full_path = os.path.join(path, filename)

        with h5py.File(full_path, 'r') as hf:
            p_data = hf['p'][:]
            wave_data = hf['wave'][:]

            shift_bins = int(p_data[0])

            wc_thres = np.zeros((28, 2), dtype=float)
            sim_thres = np.zeros((28, 2), dtype=float)
            wc_averaged = np.zeros(28, dtype=float)

            for i in range(28):
                wc_side0, wc_side1 = wave_data[0][i][0], wave_data[0][i][1]
                sim_side0_rolled = np.roll(wave_data[1][i][0], shift_bins)
                sim_side1_rolled = np.roll(wave_data[1][i][1], shift_bins)

                if peak:
                    wc_thres[i][0] = find_waveform_peak(wc_side0)
                    sim_thres[i][0] = find_waveform_peak(sim_side0_rolled)
                    wc_thres[i][1] = find_waveform_peak(wc_side1)
                    sim_thres[i][1] = find_waveform_peak(sim_side1_rolled)
                    wc_averaged[i] = (wc_thres[i][0] + wc_thres[i][1]) / 2
                else:
                    wc_thres[i][0] = find_threshold_crossing(wc_side0, threshold)
                    sim_thres[i][0] = find_threshold_crossing(sim_side0_rolled, threshold)
                    wc_thres[i][1] = find_threshold_crossing(wc_side1, threshold)
                    sim_thres[i][1] = find_threshold_crossing(sim_side1_rolled, threshold)
                    wc_averaged[i] = (wc_thres[i][0] + wc_thres[i][1]) / 2

            wc_thres_all.append(wc_thres)
            sim_thres_all.append(sim_thres)
            wc_avg_all.append(wc_averaged)

    wcResult, simResult, wcStd, simStd, wcavgResult, wcavgStd = [], [], [], [], [], []

    for side in range(2):
        wcVals, simVals, wcaVals = [[] for _ in range(28)], [[] for _ in range(28)], [[] for _ in range(28)]

        for event in range(len(wc_thres_all)):
            wc, sim, wca = wc_thres_all[event], sim_thres_all[event], wc_avg_all[event]
            for s in range(28):
                wc_val, sim_val, wca_val = wc[s][side] / 10.0, sim[s][side] / 10.0, wca[s] / 10.0

                if wc_val > 0:
                    wcVals[s].append(wc_val)
                if sim_val > 0:
                    simVals[s].append(sim_val)
                if wca_val > 0:
                    wcaVals[s].append(wca_val)

        wcavg, simavg, wcaavg = np.zeros(28), np.zeros(28), np.zeros(28)
        wcsem, simsem, wcasem = np.zeros(28), np.zeros(28), np.zeros(28)

        for s in range(28):
            wcavg[s], wcsem[s] = np.mean(wcVals[s]), np.std(wcVals[s], ddof=1) / np.sqrt(len(wcVals[s]))
            simavg[s], simsem[s] = np.mean(simVals[s]), np.std(simVals[s], ddof=1) / np.sqrt(len(simVals[s]))
            wcaavg[s], wcasem[s] = np.mean(wcaVals[s]), np.std(wcaVals[s], ddof=1) / np.sqrt(len(wcaVals[s]))

        wcResult.append(wcavg)
        simResult.append(simavg)
        wcStd.append(wcsem)
        simStd.append(simsem)
        wcavgResult.append(wcaavg)
        wcavgStd.append(wcasem)

        if plotOpt:
            positions = np.arange(28)

            plt.figure(figsize=(12, 6))
            plt.boxplot(wcVals, positions=positions, widths=0.2, whis=whis, label = 'Hit start time on this side',
                        patch_artist=True, boxprops=dict(facecolor="green"), medianprops=dict(color="black"))
            plt.errorbar(range(28), wcavgResult[0], yerr=wcavgStd, xerr=0.5, fmt='o', capsize=3, label='Averaged Hit Arrival Time' , color = 'red')


            plt.xlabel('Strip number', fontsize=fs)
            plt.ylabel('Time (ns)', fontsize=fs)
            plt.title(f'{plotName} side {side}', fontsize=fs)
            plt.legend()
            plt.ylim(r1, r2)
            plt.xticks(ticks=range(0, 28, 2), labels=range(0, 28, 2), fontsize=fs)
            plt.yticks(fontsize=fs)
            plt.legend(fontsize=fs-4)
            plt.grid(True)
            plt.savefig(f"{plotPath}_{side}.png", transparent=True)
            print('making plot to ', f"{plotPath}_{side}.png")

            plt.show()

    return wcResult, simResult, wcStd, simStd, wcavgResult, wcavgStd


