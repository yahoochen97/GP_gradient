import numpy as np
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
sys.path.append("./utility")

def main(args):
    N = int(args["num_units"])
    T = int(args["num_times"])
    MAXSEED = int(args["seed"])
    MODELS = ["2FE", "2RE", "GPR", "FULLBAYES"]
    LABELS = ["2FE", "2RE", "GPR", "FBGPR"]
    MEASURES = ["RMSE_x1", "RMSE_x2", "COVERAGE_x1", "COVERAGE_x2", "LL_x1",  "LL_x2"]

    column_names = ['mean','std', 'model', 'measure']
    df = pd.DataFrame(columns = column_names)
    data = pd.read_csv("./results/summary_N" + str(N) + "_T" + \
                       str(T) + "_SEED" + str(MAXSEED) + ".csv", index_col=[0])
    for j in range(len(MODELS)):
        MODEL = MODELS[j]
        for i in range(len(MEASURES)):
            tmp = data[data["MODEL"]==MODEL][MEASURES[i]]
            df.loc[len(df.index)] = [np.mean(tmp), np.std(tmp), MODEL, MEASURES[i]]

    data = pd.read_csv("./results/EVIDENCE_N" + str(N) + "_T" + str(T) + "_SEED" + str(MAXSEED) + ".csv")
    for j in range(len(MODELS)):
        MODEL = MODELS[j]
        tmp = data[MODEL]/N/T
        df.loc[len(df.index)] = [np.mean(tmp), np.std(tmp), MODEL, "EVIDENCE"]

    df.to_csv("./results/figure_N" + str(N) + "_T" + str(T) + "_SEED" + str(MAXSEED) + ".csv")

    MEASURES.append("EVIDENCE")
    fig = plt.figure(figsize=(10,6))
    xshifts = [-0.3,-0.1,0.1, 0.3]
    COLORS = [ [132/255, 105/255, 127/255], [67/255, 71/255, 91/255],[255/255, 172/255, 28/255],[177/255, 83/255, 42/255]]
    for j in range(len(MODELS)):
        # for i in range(len(MEASURES)):
        tmp = df[df.model==MODELS[j]]
        plt.errorbar(x=np.arange(len(MEASURES))+xshifts[j],\
                         y=tmp["mean"],\
                         yerr=tmp["std"],ls='none',\
                         label=LABELS[j],capsize=8,fmt="o", ecolor=COLORS[j],\
                         mfc=COLORS[j], mec=COLORS[j], ms=4, mew=1)
    plt.xticks(range(len(MEASURES)),MEASURES)
    plt.legend(loc=0, fontsize="20")
    plt.savefig("./results/figure_N" + str(N) + "_T" + str(T) +\
                 "_SEED" + str(MAXSEED) + ".png", dpi=300, bbox_inches='tight')
    
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(h_pad=0,w_pad=2)
    xshifts = [-0.3,-0.1,0.1,0.3]
    handles = []
    for i in range(len(MEASURES)):
        for j in range(len(MODELS)):
            ax_i = (i//2)//2
            ax_j = (i//2)%2
            tmp = df[(df.model==MODELS[j]) & (df.measure==MEASURES[i])]
            parts = MEASURES[i].split("_")
            LABEL = None
            if len(parts) == 2:
                LABEL = parts[1]
                x_base = int(LABEL[-1])
            else:
                x_base = 0
            l = axs[ax_i, ax_j].errorbar(x=x_base+xshifts[j],\
                            y=tmp["mean"],\
                            yerr=tmp["std"],ls='none',\
                            label=MODELS[j],capsize=8,fmt="o", ecolor=COLORS[j],\
                            mfc=COLORS[j], mec=COLORS[j], ms=6, mew=1)
            axs[ax_i, ax_j].set_ylabel(parts[0],fontsize="10")
            if ax_i==1 and ax_j==1:
                axs[ax_i, ax_j].set_xticks([0],["model"],fontsize="10")
                axs[ax_i, ax_j].set_xlim([-0.5,0.5])
            else:
                axs[ax_i, ax_j].set_xticks([1,2],["x1","x2"],fontsize="10")
                axs[ax_i, ax_j].set_xlim([0.5,2.5])
            if len(handles)<=len(MODELS):
                handles.append(l)
    fig.legend(handles, LABELS, bbox_to_anchor=(1.18, 0.5),loc = 'center right', fontsize="12")
    plt.savefig("./results/figure_N" + str(N) + "_T" + str(T) +\
                 "_SEED" + str(MAXSEED) + "_panel.pdf", dpi=300, bbox_inches='tight')
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='-N num_units -T num_times -s seed')
    parser.add_argument('-N','--num_units', help='number of units', required=True)
    parser.add_argument('-T','--num_times', help='number of times', required=True)
    parser.add_argument('-s','--seed', help='random seed', required=True)
    args = vars(parser.parse_args())
    main(args)