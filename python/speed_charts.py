#######################################
# Copyright (C) 2020 Otmar Ertl.      #
# All rights reserved.                #
#######################################

import os
import csv
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import color_defs

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='cm10')
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

dataDir = 'data/'
resultFilePrefix = 'performance_test_result_'

def readData():
    result = []
    for file in os.listdir(dataDir):
        if file.startswith(resultFilePrefix):
            filename = os.path.join(dataDir, file)
            with open(filename, 'r') as file:
                reader = csv.reader(file, skipinitialspace=True, delimiter=';')
                for r in reader:
                    result.append(r)

    return result


def drawChart(ax, data, hashSize, mode, isLastRow, isFirstCol):

    r = {}
    algorithms = set()
    for d in data:
        if int(d[2]) != hashSize or d[5] != mode:
            continue
        algorithm = d[0]
        algorithms.add(algorithm)
        dataSize = int(d[3])
        avgCalcTime = float(d[4])
        r.setdefault(dataSize, {}).setdefault(algorithm, []).append(avgCalcTime)

    algorithms = sorted(algorithms)
    if (mode == "exp(1)"):
        title = r"$m=" + str(hashSize) + r"\quad w(d)\sim\text{Exp}(1)$"
    elif (mode == "exp(1E30)"):
        title = r"$m=" + str(hashSize) + r"\quad w(d)\sim\text{Exp}(10^{30})$"
    elif (mode == "exp(1E-30)"):
        title = r"$m=" + str(hashSize) + r"\quad w(d)\sim\text{Exp}(10^{-30})$"
    elif (mode == "exp(n)"):
        title = r"$m=" + str(hashSize) + r"\quad w(d)\sim\text{Exp}(n)$"
    elif (mode == "exp(n*1E-6)"):
        title = r"$m=" + str(hashSize) + r"\quad w(d)\sim\text{Exp}(n\cdot 10^{-6})$"
    elif (mode == "exp(n*1E6)"):
        title = r"$m=" + str(hashSize) + r"\quad w(d)\sim\text{Exp}(n\cdot 10^{6})$"
    elif (mode == "weibull(0.1,1)"):
        title = r"$m=" + str(hashSize) + r"\quad w(d)\sim\text{Weibull}(0.1,1)$"
    else:
        assert(False)

    # assert(sorted(algorithms) == sorted(sortedAlgorithms))

    # algorithms = sortedAlgorithms

    dataSizes = list(r.keys())
    dataSizes.sort()


    ax.set_xscale("log", basex=10)
    ax.set_yscale("log", basey=10)
    if isFirstCol:
        ax.set_ylabel(r"calculation time (s)")
    if isLastRow:
        ax.set_xlabel(r"$n$")

    ax.set_title(title, fontsize=10)

    #for k in r.keys():
    #    assert(len(r[k]) == len(algorithms))
    for algorithm in algorithms:
        if algorithm != "ICWS":
            y = [r[dataSize][algorithm] for dataSize in dataSizes]
            x = dataSizes
            #ax.plot(dataSizes, y, marker='.', label=algorithm.translate(str.maketrans({"_":  r"\_"})), color=color_defs.colors[algorithm], linewidth=1)
        else:
            x = dataSizes[:13]
            y = [r[dataSize][algorithm] for dataSize in dataSizes[:13]]
        ax.plot(x, y, marker='.', label=algorithm.translate(str.maketrans({"_":  r"\_"})), color=color_defs.colors[algorithm], linewidth=1)

    handles, labels = ax.get_legend_handles_labels()

    # while len(handles) < 12:
    #     handles.append(plt.Line2D([],[], alpha=0))
    #     labels.append('')

    # order = [1,2,3,4,5, 6, 0, 7,8,9,10,11]

    leg = ax.legend(loc=2, numpoints=1)

    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor('none')

hashSizes = [256, 1024, 4096]
modes = ["exp(1)","exp(1E30)","exp(1E-30)","exp(n)", "exp(n*1E6)", "exp(n*1E-6)"]

data = readData()

fig, ax = plt.subplots(3, len(modes), sharex=True, sharey="row")
fig.set_size_inches(24, 9)

for i in range(0, len(hashSizes)):
    for j in range(0, len(modes)):
        drawChart(ax[i][j], data, hashSizes[i], modes[j], i + 1 == len(hashSizes), j == 0)

fig.subplots_adjust(left=0.025, bottom=0.045, right=0.994, top=0.975, wspace=0.05, hspace=0.15)
fig.savefig("paper/speed_charts.pdf", format='pdf', dpi=1200, metadata={'creationDate': None})
fig.savefig("paper/speed_charts.svg", format='svg', dpi=1200, metadata={'creationDate': None})
plt.close(fig)
