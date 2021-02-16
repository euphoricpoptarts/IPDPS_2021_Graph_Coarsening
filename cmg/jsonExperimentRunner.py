import sys
import os
import subprocess
from glob import glob
from parse import parse
from statistics import mean, stdev, median
import secrets
import json
from pathlib import Path
from threading import Thread, BoundedSemaphore
from itertools import zip_longest

cudaCall = "./sgpar.cuda"
gpuCalls = [("./build/hec_spec","spec_hec"),
("./build/hec2_spec","spec_hec2"),
("./build/hec_spec_spgemm","spec_hec_gemm"),
("./build/hec_spec_hashmap","spec_hec_hashmap"),
("./build/mtmetis_spec","spec_mtmetis"),
("./build/match_spec","spec_match"),
("./build/mis_spec","spec_mis2"),
("./build/gosh_spec","spec_gosh"),
("./build/gosh2_spec","spec_gosh2"),
("./build/hec_fm","fm_hec"),
("./build/hec2_fm","fm_hec2"),
("./build/mtmetis_fm","fm_mtmetis"),
("./build/match_fm","fm_match"),
("./build/mis_fm","fm_mis2"),
("./build/gosh2_fm","fm_gosh2"),
("./build/gosh_fm","fm_gosh")]

cpuCalls = [("./mp_build/hec_spec","spec_hecCPU"),
("./mp_build/hec_spec_hashmap","spec_hecCPU_hashmap"),
("./mp_build/hec_spec_serial","spec_hecCPU_real_serial"),
("./mp_build/hec_fm","fm_hecCPU"),
("./mp_build/hec_spec_spgemm","spec_hecCPU_gemm")]

#first one used for timing
#second one used only for cutsize
serialCalls = [("./mp_build/hec_spec","spec_hecCPUSerial"),
("./mp_build/hec_fm","fm_hecCPUSerial")]

rateLimit = BoundedSemaphore(value = 1)
waitLimit = 3600

def printStat(fieldTitle, statList, outfile):
    min_s = min(statList)
    max_s = max(statList)
    avg = mean(statList)
    sdev = "only one data-point"
    if len(statList) > 1:
        sdev = stdev(statList)
    med = median(statList)
    print("{}: mean={}, median={}, min={}, max={}, std-dev={}".format(fieldTitle, avg, med, min_s, max_s, sdev), file=outfile)

def listToStats(statList):
    stats = {}
    stats["min"] = min(statList)
    stats["max"] = max(statList)
    stats["mean"] = mean(statList)
    stats["std-dev"] = "only one data-point"
    if len(statList) > 1:
        stats["std-dev"] = stdev(statList)
    stats["median"] = median(statList)
    return stats

def dictToStats(data):
    output = {}
    for key, value in data.items():
        if len(value) > 0 and isinstance(value[0], dict):
            d = []
            for datum in value:
                d.append(dictToStats(datum))
            output[key] = d
        elif len(value) > 0:
            output[key] = listToStats(value)
    return output


def printDict(data, outfile):
    for key, value in data.items():
        if len(value) > 0 and isinstance(value[0], dict):
            for idx, datum in enumerate(value):
                print("{} Level {}:".format(key, idx), file=outfile)
                printDict(datum, outfile)
        elif len(value) > 0:
            printStat(key, value, outfile)

def transposeListOfDicts(data):
    data = [x for x in data if x is not None]
    transposed = {}
    if len(data) == 0:
        return transposed
    #all entries should have same fields
    fields = [x for x in data[0]]
    for field in fields:
        transposed[field] = [datum[field] for datum in data]

    fieldsToTranpose = []
    for key, value in transposed.items():
        if len(value) > 0 and isinstance(value[0], list):
            fieldsToTranpose.append(key)

    for field in fieldsToTranpose:
        #value is a list of lists, transform it into list of dicts
        aligned_lists = zip_longest(*transposed[field])
        dict_list = []
        for l in aligned_lists:
            d = transposeListOfDicts(l)
            dict_list.append(d)
        transposed[field] = dict_list
    return transposed

def analyzeMetrics(metricsPath, logFile):
    with open(metricsPath, "r") as fp:
        data = json.load(fp)

    data = transposeListOfDicts(data)

    with open(logFile, "w") as output:
        printDict(data, output)

    statsDict = dictToStats(data)
    jsonFile = os.path.splitext(logFile)[0] + ".json"
    with open(jsonFile, "w") as output:
        json.dump(statsDict, output)

def analyzeMetricsSpecial(metrics, logFile):
    data = []
    for metric in metrics:
        with open(metric, "r") as fp:
            data2 = json.load(fp)
        if data2 is not None and isinstance(data2, list):
            data.append(data2[0])

    data = transposeListOfDicts(data)

    with open(logFile, "w") as output:
        printDict(data, output)

    statsDict = dictToStats(data)
    jsonFile = os.path.splitext(logFile)[0] + ".json"
    with open(jsonFile, "w") as output:
        json.dump(statsDict, output)

def runExperiment(executable, filepath, metricDir, logFile, t_count):

    if(os.path.exists(logFile)):
        return
    myenv = os.environ.copy()
    myenv['OMP_NUM_THREADS'] = str(t_count)

    giveup = False
    while giveup is not True:
        metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
        call = [executable, filepath, metricsPath, "base_config.txt"]
        call_str = " ".join(call)
        with rateLimit:
            print("running {}".format(call_str), flush=True)
            stdout_f = "/var/tmp/sgpar_log.txt"
            with open(stdout_f, 'w') as fp:
                process = subprocess.Popen(call, stdout=fp, stderr=subprocess.DEVNULL, env=myenv)
            try:
                returncode = process.wait(timeout = waitLimit)
            except subprocess.TimeoutExpired:
                process.kill()
                print("Timeout reached by {}".format(call_str), flush=True)
                return

        if(returncode != 0):
            if returncode != -11:
                giveup = True
            print("error code: {}".format(returncode))
            print("error produced by:")
            print(call_str, flush=True)
        else:
            analyzeMetrics(metricsPath, logFile)
            return

def runExperimentSpecial(executable, filepath, metricDir, logFile, t_count):

    if(os.path.exists(logFile)):
        return
    myenv = os.environ.copy()
    myenv['OMP_NUM_THREADS'] = str(t_count)

    metrics = []
    for x in range(10):
        metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
        metrics.append(metricsPath)
        call = [executable, filepath, metricsPath, "one_config.txt"]
        call_str = " ".join(call)
        with rateLimit:
            print("running {}".format(call_str), flush=True)
            stdout_f = "/var/tmp/sgpar_log.txt"
            with open(stdout_f, 'w') as fp:
                process = subprocess.Popen(call, stdout=fp, stderr=subprocess.DEVNULL, env=myenv)
            try:
                returncode = process.wait(timeout = waitLimit)
            except subprocess.TimeoutExpired:
                process.kill()
                print("Timeout reached by {}".format(call_str), flush=True)
                return

        if(returncode != 0):
            print("error code: {}".format(returncode))
            print("error produced by:")
            print(call_str, flush=True)
    
    analyzeMetricsSpecial(metrics, logFile)

def processGraph(filepath, metricDir, logFileTemplate):
    
    #for t in [1,2,4,8,16,32]:
    """
    for call, name in gpuCalls:
        logFile = logFileTemplate.format(name)
        runExperiment(call, filepath, metricDir, logFile, 64)
    """
    for call, name in cpuCalls:
        logFile = logFileTemplate.format(name)
        runExperiment(call, filepath, metricDir, logFile, 64)
    for call, name in serialCalls:
        logFile = logFileTemplate.format(name)
        runExperiment(call, filepath, metricDir, logFile, 1)

    print("end {} processing".format(filepath), flush=True)

def reprocessMetricsFromLogFile(f_path):
    form = "running {} sgpar on csr/{}.csr, data logged in {}"
    reprocessList = []
    with open(f_path) as fp:
        for line in fp:
            r = parse(form, line)
            if r != None:
                reprocess = {}
                reprocess['metrics'] = r[2]
                reprocess['log'] = "redo_stats/" + r[1] + "_" + r[0].replace(" ","_") + ".txt"
                reprocessList.append(reprocess)

    for reprocess in reprocessList:
        print(reprocess)
        try:
            analyzeMetrics(reprocess['metrics'], reprocess['log'])
        except:
            print("Couldn't process last")

def main():

    dirpath = sys.argv[1]
    metricDir = sys.argv[2]
    logDir = sys.argv[3]
    globMatch = "{}/*.csr".format(dirpath)

    threads = []
    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        #will fill in the third argument later
        logFile = "{}/{}_{}_Sampling_Data.txt".format(logDir, stem,"{}")
        t = Thread(target=processGraph, args=(filepath, metricDir, logFile))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
