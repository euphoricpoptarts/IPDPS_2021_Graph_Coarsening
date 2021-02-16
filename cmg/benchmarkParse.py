import sys
import os
from glob import glob
from parse import parse
from statistics import mean, stdev
import secrets
from pathlib import Path
from threading import Thread


tolerances = ["1e-7","1e-8","1e-9","1e-10","1e-11","1e-12","1e-13","1e-14","1e-15"]
sysCall = "./sgpar {} {} 0 0 0 10 {} {} > /dev/null"
form = "{} a {} {} {} b {}"

def concatStatWithSpace(stat):
    return ' '.join(map(str, stat))

class MultilevelData:
    def __init__(self):
        self.niters = []
        self.maxIters = []
        self.edgeCuts = []
        self.swaps = []

class MultilevelStats:
    def __init__(self):
        self.refineMean = []
        self.refineMin = []
        self.refineStdDev = []
        self.maxItersReached = []
        self.edgeCutMean = []
        self.edgeCutMin = []
        self.edgeCutStdDev = []
        self.swapsMean = []
        self.swapsMin = []
        self.swapsStdDev = []

def printStat(outFormat, fieldsByLevel, outfile):
    levelN = 0
    for field in fieldsByLevel:
        print(outFormat.format(levelN, concatStatWithSpace(field)), file=outfile)
        levelN += 1

def processGraph(filepath, bestPart, logFile):
    statsByTol = {}
    statsByTol["timeMean"] = []
    statsByTol["timeMin"] = []
    statsByTol["coarsenLevel"] = []
    for tolerance in tolerances:
        metricsPath = "metrics/group{}.txt".format(secrets.token_urlsafe(10))
        print("running sgpar on {} with tol {}, comparing to {}, data logged in {}".format(filepath, tolerance, bestPart, metricsPath))

        err = os.system(sysCall.format(filepath, metricsPath, bestPart, tolerance))

        if(err == 256):
            print("error code: {}".format(err))
            print("error produced by:")
            print(sysCall.format(filepath, metricsPath, bestPart, tolerance))
        else:
            totalTimes = []
            coarsenTimes = []
            refineTimes = []
            multilevel = []

            cnt = 0
            with open(metricsPath) as fp:
                for line in fp:
                    parsed = parse(form, line)
                    refineData = parsed[0].split()
                    niters = refineData[0::2]
                    niters.reverse()
                    maxIterReached = refineData[1::2]
                    maxIterReached.reverse()
                    totalTimes.append(float(parsed[1]))
                    coarsenTimes.append(float(parsed[2]))
                    refineTimes.append(float(parsed[3]))
                    multilevelResults = parsed[4].split()
                    multilevelEdgeCuts = multilevelResults[0::2]
                    multilevelEdgeCuts.reverse()
                    multilevelSwaps = multilevelResults[1::2]
                    multilevelSwaps.reverse()

                    data = zip(niters, maxIterReached, multilevelEdgeCuts, multilevelSwaps)

                    dataCount = 0
                    for datum in data:
                        dataCount += 1
                        if len(multilevel) < dataCount:
                            multilevel.append(MultilevelData())
                        multilevel[dataCount-1].niters.append(int(datum[0]))
                        multilevel[dataCount-1].maxIters.append(int(datum[1]))
                        multilevel[dataCount-1].edgeCuts.append(int(datum[2]))
                        multilevel[dataCount-1].swaps.append(int(datum[3]))

                    cnt += 1

            levelCount = 0
            for level in multilevel:
                levelCount += 1
                if len(statsByTol["coarsenLevel"]) < levelCount:
                    levelStats = MultilevelStats()
                    statsByTol["coarsenLevel"].append(levelStats)
                levelStats = statsByTol["coarsenLevel"][levelCount - 1]
                levelStats.refineMean.append(mean(level.niters))
                levelStats.refineMin.append(min(level.niters))
                if len(level.niters) > 1:
                    levelStats.refineStdDev.append(stdev(level.niters))
                else:
                    levelStats.refineStdDev.append(0)
                levelStats.edgeCutMean.append(mean(level.edgeCuts))
                levelStats.edgeCutMin.append(min(level.edgeCuts))
                if len(level.edgeCuts) > 1:
                    levelStats.edgeCutStdDev.append(stdev(level.edgeCuts))
                else:
                    levelStats.edgeCutStdDev.append(0)
                levelStats.swapsMean.append(mean(level.swaps))
                levelStats.swapsMin.append(min(level.swaps))
                if len(level.swaps) > 1:
                    levelStats.swapsStdDev.append(stdev(level.swaps))
                else:
                    levelStats.swapsStdDev.append(0)
                levelStats.maxItersReached.append(sum(level.maxIters))
            statsByTol["timeMean"].append(mean(totalTimes))
            statsByTol["timeMin"].append(min(totalTimes))

    output = open(logFile, "w")
    print("tolerances: {}".format(' '.join(tolerances)), file=output)
    print("mean total time: {}".format(concatStatWithSpace(statsByTol["timeMean"])), file=output)
    print("min total time: {}".format(concatStatWithSpace(statsByTol["timeMin"])), file=output)
    printStat("coarsen level {} mean refine iterations: {}", [level.refineMean for level in statsByTol["coarsenLevel"]], output)
    printStat("coarsen level {} min refine iterations: {}", [level.refineMin for level in statsByTol["coarsenLevel"]], output)
    printStat("coarsen level {} refine iterations std deviation: {}", [level.refineStdDev for level in statsByTol["coarsenLevel"]], output)
    printStat("coarsen level {} times max iter reached: {}", [level.maxItersReached for level in statsByTol["coarsenLevel"]], output)
    printStat("coarsen level {} mean edge cut: {}", [level.edgeCutMean for level in statsByTol["coarsenLevel"]], output)
    printStat("coarsen level {} min edge cut: {}", [level.edgeCutMin for level in statsByTol["coarsenLevel"]], output)
    printStat("coarsen level {} edge cut std deviation: {}", [level.edgeCutStdDev for level in statsByTol["coarsenLevel"]], output)
    printStat("coarsen level {} mean swaps to best partition: {}", [level.swapsMean for level in statsByTol["coarsenLevel"]], output)
    printStat("coarsen level {} min swaps to best partition: {}", [level.swapsMin for level in statsByTol["coarsenLevel"]], output)
    printStat("coarsen level {} swaps to best partition std deviation: {}", [level.swapsStdDev for level in statsByTol["coarsenLevel"]], output)
    print("end {} processing".format(filepath))

def main():

    dirpath = sys.argv[1]
    bestPartDir = sys.argv[2]
    logDir = sys.argv[3]
    globMatch = "{}/*.csr".format(dirpath)
    threads = []

    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        bestPart = "{}/{}.2.ptn".format(bestPartDir, stem)
        logFile = "{}/{}_Sampling_Data.txt".format(logDir, stem)
        #processGraph(filepath, bestPart, logFile)
        t = Thread(target=processGraph, args=(filepath, bestPart, logFile,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()