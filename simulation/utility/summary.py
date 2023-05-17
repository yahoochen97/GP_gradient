import numpy as np
import pandas as pd
import argparse
import sys
sys.path.append("./utility")

def main(args):
    N = int(args["num_units"])
    T = int(args["num_times"])
    MAXSEED = int(args["seed"])
    MODELS = ["2FE", "2RE", "GPR"]

    results = np.zeros((len(MODELS),2+2+2,MAXSEED))
    for SEED in range(1,MAXSEED+1):
        for j in range(len(MODELS)):
            MODEL = MODELS[j]
            result_filename = "./results/"+ MODEL + "_N" + str(N) \
                + "_T" + str(T) + "_SEED" + str(SEED) + ".csv"
            data = pd.read_csv(result_filename)
            dx1 = data["dx1"].to_numpy()
            dx1_mu = data["dx1_mu"].to_numpy()
            dx1_std = data["dx1_std"].to_numpy()
            dx2 = data["dx2"].to_numpy()
            dx2_mu = data["dx2_mu"].to_numpy()
            dx2_std = data["dx2_std"].to_numpy()
            RMSE_x1 = np.sqrt(np.mean((dx1_mu-dx1)**2))
            COVERAGE_x1 = np.mean(np.logical_and((dx1_mu-2*dx1_std)<=dx1, dx1<=(dx1_mu+2*dx1_std)))
            LL_x1 = -np.log(2*np.pi)/2+np.mean(-np.log(dx1_std)-(dx1_mu-dx1)**2)
            RMSE_x2 = np.sqrt(np.mean((dx2_mu-dx2)**2))
            COVERAGE_x2 = np.mean(np.logical_and((dx2_mu-2*dx2_std)<=dx2, dx2<=(dx2_mu+2*dx2_std)))
            LL_x2 = -np.log(2*np.pi)/2+np.mean(-np.log(dx2_std)-(dx2_mu-dx2)**2)
            results[j,:,SEED-1] = [RMSE_x1,COVERAGE_x1, LL_x1, RMSE_x2,COVERAGE_x2, LL_x2]
    
    column_names = ['RMSE_x1','COVERAGE_x1', 'LL_x1',\
                                            'RMSE_x2','COVERAGE_x2', 'LL_x2']
    df = pd.DataFrame(columns = column_names)
    for j in range(len(MODELS)):
        df = df.append(pd.DataFrame(np.transpose(results[j]),columns=column_names))

    df['MODEL'] = np.repeat(MODELS,MAXSEED)
    
    df.to_csv("./results/summary_N" + str(N) + "_T" + str(T) + "_SEED" + str(MAXSEED) + ".csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='-N num_units -T num_times -s seed')
    parser.add_argument('-N','--num_units', help='number of units', required=True)
    parser.add_argument('-T','--num_times', help='number of times', required=True)
    parser.add_argument('-s','--seed', help='random seed', required=True)
    args = vars(parser.parse_args())
    main(args)