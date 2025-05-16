import numpy as np
import matplotlib.pyplot as plt
from language_distance_metrics import cltad_distance, l2v_distance
import multiprocessing as mp
import pickle
import scipy
import seaborn as sns
from sklearn.linear_model import LinearRegression

def pkldump(obj, file):
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)

def pklload(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)

allScores=[
# fullfrompretrained below
"""
hy,cs,33.39
hy,ro,33.15
hy,ru,33.43
hy,hu,33.42
hy,tr,33.17
hy,hr,33.10
hy,ko,33.07
""",
"""
az,cs,23.50
az,ro,23.75
az,ru,23.76
az,hu,23.59
az,tr,23.96
az,hr,23.54
az,ko,23.67
""",
"""
ka,cs,31.12
ka,ro,31.40
ka,ru,31.23
ka,hu,31.25
ka,tr,31.09
ka,hr,31.27
ka,ko,30.72
""",
"""
be,cs,36.37
be,ro,35.61
be,ru,35.86
be,hu,35.56
be,tr,35.79
be,hr,35.63
be,ko,35.65
""",
"""
gl,cs,41.83
gl,ro,42.43
gl,ru,41.85
gl,hu,41.89
gl,tr,41.59
gl,hr,41.88
gl,ko,41.95
""",
#fullfromscratch8 below
"""
hy,cs,13.82
hy,ro,14.77
hy,ru,13.33
hy,hu,13.01
hy,tr,13.49
hy,hr,14.84
hy,ko,12.35
""",
"""
az,cs,4.01
az,ro,4.54
az,ru,4.08
az,hu,3.86
az,tr,3.96
az,hr,4.78
az,ko,3.29
""",
"""
ka,cs,9.30
ka,ro,10.30
ka,ru,9.20
ka,hu,10.51
ka,tr,9.05
ka,hr,9.81
ka,ko,8.13
""",
"""
be,cs,6.27
be,ro,6.20
be,ru,5.50
be,hu,5.90
be,tr,4.88
be,hr,6.29
be,ko,3.48
""",
"""
gl,cs,15.78
gl,ro,18.67
gl,ru,16.15
gl,hu,15.35
gl,tr,14.35
gl,hr,16.87
gl,ko,14.19
"""
]

def calculate_correlation(args, plot=False):
    shift_lambda, shift_exponent = args[0], args[1]

    info_for_plot = []

    correlations = []
    mses = []
    for score in allScores:
        lines = [ i.strip().split(",") for i in score.strip().split("\n") ]

        bleus = []
        dists = []
        langs = []

        for line in lines:
            bleus.append(float(line[-1]))
            dists.append(cltad_distance(line[0], line[1], shift_lambda=shift_lambda, shift_exponent=shift_exponent))
            langs.append(line[1])

        if any(np.array(dists) < 1e-100):
            correlations.append(0)
        else:
            correlations.append(np.corrcoef(bleus, np.log(dists))[0,1])
            reg = LinearRegression().fit(np.log(dists).reshape(-1,1), bleus)
            mse = np.mean((reg.predict(np.log(dists).reshape(-1,1)) - bleus)**2)
            mses.append(mse)
            if plot:
                sns.set()
                for d, b, l in zip(np.log(dists), bleus, langs):
                    md = np.max(dists) - np.min(dists)
                    mb = np.max(bleus) - np.min(bleus)
                    plt.text(d + 0.01*md, b + 0.01*mb, l)
                plt.scatter(dists, bleus)
                corr = correlations[-1]
                plt.legend([], title=f"Correlation: {corr}", framealpha=0, markerscale=0, title_fontproperties={"weight": "bold"})
                plt.title(f"Main language: {lines[0][0]}")
                plt.xlabel("CLTAD distances")
                plt.ylabel("BLEU")
                plt.tight_layout()
                plt.show()
            
        info_for_plot.append([bleus, np.log(dists), lines[0][0]])
        #print(f"Correlation found for language {lines[0][0]}: {correlations[-1]}")

    print(f"{args[0]:.2f}-{args[1]:.2f}: Average correlation and MSE: {np.mean(correlations)}, {np.mean(mse)}")
    
    return np.mean(correlations), np.mean(mse), info_for_plot

def calculate_correlation2(lambdas=np.array([1]*33), plot=False):
    correlations = []
    for score in allScores:
        lines = [ i.strip().split(",") for i in score.strip().split("\n") ]
        if lines[0][0] in ['be', 'gl']:
            continue

        bleus = []
        dists = []
        langs = []

        for line in lines:
            if line[1] == 'hr':
                continue
            bleus.append(float(line[-1]))
            dists.append(l2v_distance(line[0], line[1], lambdas=lambdas))
            langs.append(line[1])

        if any(np.array(dists) < 1e-100):
            correlations.append(0)
        else:
            correlations.append(np.corrcoef(bleus, dists)[0,1])
            if plot:
                sns.set()
                for d, b, l in zip(dists, bleus, langs):
                    md = np.max(dists) - np.min(dists)
                    mb = np.max(bleus) - np.min(bleus)
                    plt.text(d + 0.01*md, b + 0.01*mb, l)
                plt.scatter(dists, bleus)
                corr = correlations[-1]
                plt.legend([], title=f"Correlation: {corr}", framealpha=0, markerscale=0, title_fontproperties={"weight": "bold"})
                plt.title(f"Main language: {lines[0][0]}")
                plt.xlabel("Lang2vec syntactic distances")
                plt.ylabel("BLEU")
                plt.tight_layout()
                plt.show()

    print(f"Average correlation: {np.mean(correlations)}")
    
    return np.mean(correlations)

# if __name__ == "__main__":
#     retval = scipy.optimize.minimize(calculate_correlation2, np.array([0.001]*33), bounds=[(0, 1)]*33)

#     print("Lambdas:", retval.x)
#     print("Average BLEU x distance correlation:", retval.fun)

#     l2v_distance("hy", "ru", lambdas=retval.x, verbose=True)


allArgs = []
for shift_lambda in [1/3, 1/2, 1/1.5, 1/1.2, 1, 1.2, 1.5, 2, 3]:
    for shift_exponent in [1/3, 1/2, 1/1.5, 1/1.2, 1, 1.2, 1.5, 2, 3]:
        allArgs.append([shift_lambda, shift_exponent])

if __name__ == "__main__":
    with mp.Pool(8) as p:
        correlations = p.map(calculate_correlation, allArgs, chunksize=1)

    # correlations = [ calculate_correlation(arg, plot=False) for arg in allArgs ]

    for args, corr in zip(allArgs, correlations):
        print(f"lambda = {args[0]:.3f}\texponent = {args[1]:.3f}\tavg. correlation = {corr[0]}\tMSE = {corr[1]}")

    pkldump({'args': allArgs, 'corr': correlations}, "optimize-metric-results.pickle")