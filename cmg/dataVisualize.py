import sys
from glob import glob
from parse import parse
from statistics import mean, stdev
from pathlib import Path
import matplotlib.pyplot as plt

lineFormat = "{}: {}"
stemFormat = "{}_Sampling_Data"
outputFormat = "{}\{}.png"
size = (10,10)

class PlotGroup:
    def __init__(self, name, yaxis, index):
        self.primaryFigure = plt.figure(figsize=size)
        self.primaryPlot = self.primaryFigure.add_subplot()
        formatPlot(self.primaryPlot, name + " Comparison", yaxis)
        self.primaryFigFilename = name.replace(" ","_") + "_Comparison"
        self.index = index
        self.name = name
        self.yaxis = yaxis

def formatPlot(plot, name, yaxis):
    plot.set_xscale('log')
    plot.set_yscale('log')
    plot.title.set_text(name)
    plot.set_xlabel('Refinement Iterations')
    plot.set_ylabel(yaxis)

def plotQuantity(dataList, name, yaxis, xdata, outpath, comparePlot):
    minVal = min(dataList)
    dataList = list(map(lambda p: p / minVal, dataList))
    outputFile = outputFormat.format(outpath, name.replace(" ","_"))
    fig = plt.figure(figsize=size)
    plot = fig.add_subplot()
    formatPlot(plot, name, yaxis)
    plot.plot(xdata, dataList)
    comparePlot.plot(xdata, dataList)
    fig.savefig(outputFile)
    plt.close(fig)

def processGraph(filepath, outpath, plotGroups):
    stats = {}
    stem = Path(filepath).stem
    graphName = parse(stemFormat,stem)[0]

    dataLines = []
    with open(filepath) as fp:
        for line in fp:
            parsed = parse(lineFormat, line)
            dataLines.append(parsed[1])

    stats["tolerance"] = list(map(float, dataLines[0].split()))
    stats["refineIter"] = list(map(float, dataLines[1].split()))
    for plotGroup in plotGroups:
        stat = list(map(float, dataLines[plotGroup.index].split()))
        zipped = zip(stats["refineIter"], stat)
        zipped = sorted(zipped, key=lambda x: x[0])
        stat = [x[1] for x in zipped]
        refine = [x[0] for x in zipped]
        plotQuantity(stat, graphName + " " + plotGroup.name, plotGroup.yaxis, refine, outpath, plotGroup.primaryPlot)

def main():

    dirpath = sys.argv[1]
    outpath = sys.argv[2]
    globMatch = "{}/*.txt".format(dirpath)

    plotGroups = []
    plotGroups.append(PlotGroup("Edge Cut Mean", "Normalized Edge Cut Mean", 7))
    plotGroups.append(PlotGroup("Edge Cut Min", "Normalized Edge Cut Min", 8))
    plotGroups.append(PlotGroup("Swaps Mean", "Normalized Swaps Mean", 10))
    plotGroups.append(PlotGroup("Swaps Min", "Normalized Swaps Min", 11))

    for file in glob(globMatch):
        filepath = file
        processGraph(filepath, outpath, plotGroups)

    for plotGroup in plotGroups:
        plotGroup.primaryFigure.savefig(outputFormat.format(outpath,plotGroup.primaryFigFilename))


if __name__ == "__main__":
    main()