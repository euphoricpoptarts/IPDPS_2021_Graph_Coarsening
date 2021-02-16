import sys
import os
import re
from glob import glob
from parse import parse
from statistics import mean, stdev, median, geometric_mean
import secrets
import json
from pathlib import Path
from itertools import zip_longest
import math

stemParse = r"(.+)_(fm|spec)_(.*)_Sampling_Data"
lineParse = "{}mean={}, median={}, min={}, max={}, std-dev={}"
#fieldHeaders = "Graph, HEC, MIS, Match, MT"
fieldHeaders = "{Graph} & {hec} & {match} & {mtmetis} & {gosh} & {mis2}"
dnf = "{OOM}"

graphOrder = [("HV15R","HV15R"),
("rgg_n_2_24_s0","rgg24"),
("nlpkkt160","nlpkkt160"),
("europe_osm","europeOsm"),
("Cube_Coup_dt0","CubeCoup"),
("delaunay_n24","delaunay24"),
("Flan_1565","Flan1565"),
("ML_Geer","MLGeer"),
("cage15","cage15"),
("channel-500x100x100-b050","channel050"),
("indochina-2004","ic04"),
("com-Orkut","Orkut"),
("vas_stokes_4M","vasStokes4M"),
("kmer_U1a","kmerU1a"),
("kron_g500-logn21","kron21"),
("products","products"),
("hollywood-2009","hollywood09"),
("mycielskian17","mycielskian17"),
("citation","citation"),
("ppa","ppa")]

graphOrder2 = [("HV15R","HV15R"),
("rgg_n_2_24_s0","rgg24"),
("nlpkkt160","nlpkkt160"),
("europe_osm","europeOsm"),
("Cube_Coup_dt0","CubeCoup"),
("delaunay_n24","delaunay24"),
("Flan_1565","Flan1565"),
("ML_Geer","MLGeer"),
("cage15","cage15"),
("channel-500x100x100-b050","channel050"),
("gmean","GeoMean"),
("indochina-2004","ic04"),
("com-Orkut","Orkut"),
("vas_stokes_4M","vasStokes4M"),
("kmer_U1a","kmerU1a"),
("kron_g500-logn21","kron21"),
("products","products"),
("hollywood-2009","hollywood09"),
("mycielskian17","mycielskian17"),
("citation","citation"),
("ppa","ppa"),
("gmean","GeoMean")]

def getStats(filepath):
    with open(filepath,"r") as f:
       return json.load(f) 

def gpuBuildTable(graphs,data,outFile):
    with open(outFile,"w+") as f:
        print("GPU Build Comparison Table", file=f)
        for graph, graphSanitized in graphOrder:
            l = [graphSanitized]
            values = data[graph]
            exp = values[("spec","hec")]
            exp_gemm = values[("spec","hec_gemm")]
            exp_map = values[("spec","hec_hashmap")]
            l.append("{:.2f}".format(exp["coarsen-duration-seconds"]["median"]))
            l.append("{:.2f}".format(exp["coarsen-build-duration-seconds"]["median"] / exp["coarsen-duration-seconds"]["median"]))
            l.append("{:.2f}".format(exp_map["coarsen-build-duration-seconds"]["median"] / exp["coarsen-build-duration-seconds"]["median"]))
            l.append("{:.2f}".format(exp_gemm["coarsen-build-duration-seconds"]["median"] / exp["coarsen-build-duration-seconds"]["median"]))
            print(" & ".join(l) + " \\\\", file=f)

def hecLevelsTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print("CPU Coarse Levels Table", file=f)
        cpuvgpuratio = []
        for graph, graphSanitized in graphOrder:
            l = [graphSanitized]
            values = data[graph]
            cpu = values[("spec","hecCPU")]
            l.append("{:.0f}".format(cpu["number-coarse-levels"]["median"]))
            print(" & ".join(l) + " \\\\", file=f)

def hecSerialTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print("CPU HEC Map Parallel v Serial Table", file=f)
        cpuvgpuratio = []
        for graph, graphSanitized in graphOrder:
            l = [graphSanitized]
            values = data[graph]
            cpu = values[("spec","hecCPU")]
            gpu = values[("spec","hec")]
            serial = values[("spec","hecCPU_real_serial")]
            cpu_coarsen = cpu["coarsen-map-duration-seconds"]["median"] + cpu["coarsen-build-duration-seconds"]["median"]
            gpu_coarsen = gpu["coarsen-map-duration-seconds"]["median"] + gpu["coarsen-build-duration-seconds"]["median"]
            l.append("{:.2f}".format(serial["number-coarse-levels"]["median"] / cpu["number-coarse-levels"]["median"]))
            l.append("{:.2f}".format(gpu["number-coarse-levels"]["median"] / cpu["number-coarse-levels"]["median"]))
            l.append("{:.0f}".format(abs(serial["number-coarse-levels"]["median"] - cpu["number-coarse-levels"]["median"])))
            l.append("{:.2f}".format(serial["coarsen-map-duration-seconds"]["median"] / cpu["coarsen-map-duration-seconds"]["median"]))
            l.append("{:.2f}".format(cpu_coarsen / gpu_coarsen))
            cpuvgpuratio.append(cpu_coarsen / gpu_coarsen)
            print(" & ".join(l) + " \\\\", file=f)
        print(" & {:.2f}".format(geometric_mean(cpuvgpuratio)) + " \\\\", file=f)

def cpuBuildTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print("CPU Build Comparison Table", file=f)
        grco = []
        hashmap = []
        spgemm = []
        for graph, graphSanitized in graphOrder2:
            l = [graphSanitized]
            if graph == "gmean":
                l.append("0")
                l.append("{:.0f}".format(100*geometric_mean(grco)))
                l.append("{:.2f}".format(geometric_mean(hashmap)))
                l.append("{:.2f}".format(geometric_mean(spgemm)))
                grco = []
                hashmap = []
                spgemm = []
            else:
                values = data[graph]
                exp = values[("spec","hecCPU")]
                l.append("{:.2f}".format(exp["coarsen-duration-seconds"]["median"]))
                l.append("{:.0f}".format(100*exp["coarsen-build-duration-seconds"]["median"] / exp["coarsen-duration-seconds"]["median"]))
                grco.append(exp["coarsen-build-duration-seconds"]["median"] / exp["coarsen-duration-seconds"]["median"])
                try:
                    exp_map = values[("spec","hecCPU_hashmap")]
                    l.append("{:.2f}".format(exp_map["coarsen-build-duration-seconds"]["median"] / exp["coarsen-build-duration-seconds"]["median"]))
                    hashmap.append(exp_map["coarsen-build-duration-seconds"]["median"] / exp["coarsen-build-duration-seconds"]["median"])
                except KeyError:
                    l.append(dnf)
                try:
                    exp_gemm = values[("spec","hecCPU_gemm")]
                    l.append("{:.2f}".format(exp_gemm["coarsen-build-duration-seconds"]["median"] / exp["coarsen-build-duration-seconds"]["median"]))
                    spgemm.append(exp_gemm["coarsen-build-duration-seconds"]["median"] / exp["coarsen-build-duration-seconds"]["median"])
                except KeyError:
                    l.append(dnf)
            print(" & ".join(l) + " \\\\", file=f)

def gpuvcpuTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print("GPU vs CPU Coarsening table", file=f)
        for graph, graphSanitized in graphOrder:
            l = [graphSanitized]
            values = data[graph]
            gpu = values[("spec","hec")]
            cpu = values[("spec","hecCPU")]
            cpu_hash = values[("spec","hecCPU_hashmap")]
            gpu_hash = values[("spec","hec_hashmap")]
            l.append("{:.2f}".format(cpu["coarsen-duration-seconds"]["median"] / gpu["coarsen-duration-seconds"]["median"]))
            l.append("{:.2f}".format(cpu["coarsen-build-duration-seconds"]["median"] / gpu["coarsen-build-duration-seconds"]["median"]))
            l.append("{:.2f}".format(cpu_hash["coarsen-build-duration-seconds"]["median"] / gpu_hash["coarsen-build-duration-seconds"]["median"]))
            try:
                serial = values[("spec","hecCPUSerial")]
                l.append("{:.2f}".format(serial["coarsen-duration-seconds"]["median"] / cpu["coarsen-duration-seconds"]["median"]))
                l.append("{:.2f}".format(serial["coarsen-build-duration-seconds"]["median"] / cpu["coarsen-build-duration-seconds"]["median"]))
            except KeyError:
                l.append(dnf)
                l.append(dnf)
            print(" & ".join(l) + " \\\\", file=f)

def methodsCompTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print("Coarsening methods timing table", file=f)
        for graph, graphSanitized in graphOrder:
            l = [graphSanitized]
            values = data[graph]
            main = values[("spec","hec")]
            otherExps = ["hec2","match","mtmetis","gosh","gosh2","mis2"]
            for other in otherExps:
                try:
                    exp = values[("spec",other)]
                    l.append("{:.2f}".format(exp["coarsen-duration-seconds"]["median"] / main["coarsen-duration-seconds"]["median"]))
                except KeyError:
                    l.append(dnf)
            
            l.append("{:.0f}".format(main["number-coarse-levels"]["median"]))
            for other in otherExps:
                try:
                    exp = values[("spec",other)]
                    l.append("{:.0f}".format(exp["number-coarse-levels"]["median"]))
                except KeyError:
                    l.append(dnf)
            
            hecCoarseLevels = main["coarse-levels"]
            hecCoarsenRate = hecCoarseLevels[-1]["number-vertices"]["median"] / hecCoarseLevels[0]["number-vertices"]["median"]
            hecCoarsenRate = math.pow(hecCoarsenRate, 1/(main["number-coarse-levels"]["median"] - 1))
            l.append("{:.2f}".format(hecCoarsenRate))
           
            try:
                mtmetis = values[("spec","mtmetis")]
                mtCL = mtmetis["coarse-levels"]
                mtCR = mtCL[-1]["number-vertices"]["median"] / mtCL[0]["number-vertices"]["median"]
                mtCR = math.pow(mtCR, 1/(mtmetis["number-coarse-levels"]["median"] - 1))
                l.append("{:.2f}".format(mtCR))
            except KeyError:
                l.append(dnf)

            print(" & ".join(l) + " \\\\", file=f)

def methodsECCompTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print("Coarsening methods EC Table", file=f)
        for graph, graphSanitized in graphOrder:
            l = [graphSanitized]
            values = data[graph]
            main = values[("spec","hec")]
            otherExps = ["match","mtmetis","gosh","mis2"]
            l.append("{:.2f}".format(main["total-duration-seconds"]["median"]))
            l.append("{:.0f}".format(100 * main["coarsen-duration-seconds"]["median"] / main["total-duration-seconds"]["median"]))
            l.append("{:.0f}".format(main["edge-cut"]["median"]))
            for other in otherExps:
                try:
                    exp = values[("spec",other)]
                    l.append("{:.2f}".format(exp["edge-cut"]["median"] / main["edge-cut"]["median"]))
                except KeyError:
                    l.append(dnf)
            

            for other in ["hec"] + otherExps:
                try:
                    exp = values[("spec",other)]
                    l.append("{:.2f}".format(exp["edge-cut"]["std-dev"] / exp["edge-cut"]["mean"]))
                except KeyError:
                    l.append(dnf)

            print(" & ".join(l) + " \\\\", file=f)

def methodsECFMCompTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print("Coarsening methods EC FM Table", file=f)
        for graph, graphSanitized in graphOrder:
            l = [graphSanitized]
            values = data[graph]
            main = values[("fm","hec")]
            otherExps = ["hec2","match","mtmetis","gosh","gosh2","mis2"]
            l.append("{:.0f}".format(main["edge-cut"]["median"]))
            for other in otherExps:
                try:
                    exp = values[("fm",other)]
                    l.append("{:.2f}".format(exp["edge-cut"]["median"] / main["edge-cut"]["median"]))
                except KeyError:
                    l.append(dnf)
            

            for other in ["hec"] + otherExps:
                try:
                    exp = values[("fm",other)]
                    l.append("{:.2f}".format(exp["edge-cut"]["std-dev"] / exp["edge-cut"]["mean"]))
                except KeyError:
                    l.append(dnf)

            print(" & ".join(l) + " \\\\", file=f)

def fmTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print("FM EC Table", file=f)
        for graph in graphs:
            graphSanitized = graph.replace("_","\\textunderscore ")
            l = [graphSanitized]
            values = data[graph]
            cpu = values[("fm","hecCPU")]
            gpu = values[("fm","hec")]
            found = True
            try:
                serial = values[("fm","hecCPUSerial")]
            except KeyError:
                found = False
            l.append("{:.2f}".format(cpu["total-duration-seconds"]["median"]))
            l.append("{:.0f}".format(100 * cpu["coarsen-duration-seconds"]["median"] / cpu["total-duration-seconds"]["median"]))
            l.append("{:.2f}".format(cpu["total-duration-seconds"]["median"] / gpu["total-duration-seconds"]["median"]))
            l.append(" ")
            l.append(" ")
            if found:
                l.append("{:.0f}".format(serial["edge-cut"]["median"]))
                l.append("{:.2f}".format(gpu["edge-cut"]["median"] / serial["edge-cut"]["median"]))
                l.append("{:.2f}".format(cpu["edge-cut"]["median"] / serial["edge-cut"]["median"]))
            else:
                l.append(dnf)
                l.append(dnf)
                l.append(dnf) 

            print(" & ".join(l) + " \\\\", file=f)

def specTable(graphs,data,outFile):
    with open(outFile,"a+") as f:
        print("FM EC Table", file=f)
        for graph, graphSanitized in graphOrder:
            l = [graphSanitized]
            values = data[graph]
            gpu = values[("fm","hec")]
            found = True
            try:
                serial = values[("fm","hecCPUSerial")]
            except KeyError:
                found = False
            l.append("{:.0f}".format(gpu["edge-cut"]["median"]))
            try:
                cpu = values[("fm","hecCPU")]
                l.append("{:.0f}".format(cpu["edge-cut"]["median"]))
            except KeyError:
                l.append(dnf)
            if found:
                l.append("{:.0f}".format(serial["edge-cut"]["median"]))
            else:
                l.append(dnf)

            print(" , ".join(l), file=f)

def main():

    logDir = sys.argv[1]
    outFile = sys.argv[2]

    globMatch = "{}/*.json".format(logDir)

    data = {}
    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        stemMatch = re.match(stemParse, stem)
        if stemMatch is not None:
            graph = stemMatch.groups()[0]
            experiment = (stemMatch.groups()[1], stemMatch.groups()[2])
            if graph not in data:
                data[graph] = {}
            data[graph][experiment] = getStats(filepath)

    graphsSorted = [key for key in data]
    graphsSorted = sorted(graphsSorted, key = str.casefold)
    gpuBuildTable(graphsSorted, data, outFile)
    cpuBuildTable(graphsSorted, data, outFile)
    hecSerialTable(graphsSorted, data, outFile)
    hecLevelsTable(graphsSorted, data, outFile)
    gpuvcpuTable(graphsSorted, data, outFile)
    methodsCompTable(graphsSorted, data, outFile)
    methodsECCompTable(graphsSorted, data, outFile)
    methodsECFMCompTable(graphsSorted, data, outFile)
    specTable(graphsSorted, data, outFile)

if __name__ == "__main__":
    main()
