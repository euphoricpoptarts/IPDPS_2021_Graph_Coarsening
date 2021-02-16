#pragma once
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>

class ExperimentLoggerUtil {

public:
	enum class Measurement : int {
		Map,
		Build,
		Count,
		Prefix,
		Bucket,
		Dedupe,
		RadixSort,
		RadixDedupe,
        WriteGraph,
		Permute,
		MapConstruct,
		FMRefine,
		FMRecoarsen,
		Heavy,
		END
	};
	std::vector<std::string> measurementNames{
		"coarsen-map",
		"coarsen-build",
		"coarsen-count",
		"coarsen-prefix-sum",
		"coarsen-bucket",
		"coarsen-dedupe",
		"coarsen-radix-sort",
		"coarsen-radix-dedupe",
        "coarsen-write-graph",
		"coarsen-permute",
		"coarsen-map-construct",
		"fm-refine",
		"fm-recoarsen",
		"heavy"
	};
	std::vector<double> measurements;

private:
	class CoarseLevel {
	public:
		int refineIterations = 0;
		bool iterationMaxReached = false;
		uint64_t unrefinedEdgeCut = 0;
		uint64_t numVertices = 0;

		CoarseLevel(int refineIterations, bool iterationMaxReached, uint64_t numVertices) :
			refineIterations(refineIterations),
			iterationMaxReached(iterationMaxReached),
			numVertices(numVertices) {}

		void log(std::ofstream& f) {
			f << "{";
			f << "\"refine-iterations\":" << refineIterations << ',';
			f << "\"iteration-max-reached\":" << iterationMaxReached << ',';
			f << "\"number-vertices\":" << numVertices << ',';
			f << "\"unrefined-edge-cut\":" << unrefinedEdgeCut;
			f << "}";
		}
	};

	int numCoarseLevels = 0;
	std::vector<CoarseLevel> coarseLevels;
	uint64_t finestEdgeCut = 0;
	uint64_t edgeCut4Way = 0;
	uint64_t partitionDiff = 0;
	double totalDurationSeconds = 0;
	double coarsenDurationSeconds = 0;
	double refineDurationSeconds = 0;
	double coarsenDedupeDurationSeconds = 0;
	double coarsenCountDurationSeconds = 0;
	double coarsenBucketDurationSeconds = 0;
	double coarsenPrefixSumDurationSeconds = 0;
	double coarsenMapDurationSeconds = 0;
	double coarsenBuildDurationSeconds = 0;
    double mapPasses[2] = {0,0};
    double mapFirstPassRate[2] = {0,0};
    int callCount = 0;

public:
	ExperimentLoggerUtil() :
		measurements(static_cast<int>(Measurement::END), 0.0)
	{}

    void addMapPassData(double assignRate, double passes){
        if(callCount > 1) return;
        mapPasses[callCount] = passes;
        mapFirstPassRate[callCount] = assignRate;
        callCount++;
    }

	void addCoarseLevel(int refineIterations, bool iterationMaxReached, uint64_t unrefinedEdgeCut) {
		coarseLevels.emplace_back(refineIterations, iterationMaxReached, unrefinedEdgeCut);
		numCoarseLevels++;
	}

	void modifyCoarseLevelEC(int level, uint64_t unrefinedEdgeCut) {
		if (coarseLevels.size() > 0) {
			//coarse levels are backwards
			int top = coarseLevels.size() - 1;
			coarseLevels[top - level].unrefinedEdgeCut = unrefinedEdgeCut;
		}
	}

	void setFinestEdgeCut(uint64_t finestEdgeCut) {
		this->finestEdgeCut = finestEdgeCut;
	}

	void setEdgeCut4Way(uint64_t edgeCut4Way) {
		this->edgeCut4Way = edgeCut4Way;
	}

	void setPartitionDiff(uint64_t partitionDiff) {
		this->partitionDiff = partitionDiff;
	}

	void setTotalDurationSeconds(double totalDurationSeconds) {
		this->totalDurationSeconds = totalDurationSeconds;
	}

	void setCoarsenDurationSeconds(double coarsenDurationSeconds) {
		this->coarsenDurationSeconds = coarsenDurationSeconds;
	}

	void setRefineDurationSeconds(double refineDurationSeconds) {
		this->refineDurationSeconds = refineDurationSeconds;
	}

	void addMeasurement(Measurement m, double val) {
		measurements[static_cast<int>(m)] += val;
	}

	double getMeasurement(Measurement m) {
		return measurements[static_cast<int>(m)];
	}

	void log(char* filename, bool first, bool last) {
		std::ofstream f;
		f.open(filename, std::ios::app);

		if (f.is_open()) {
			if (first) {
				f << "[";
			}
			f << "{";
			f << "\"edge-cut\":" << finestEdgeCut << ',';
			f << "\"edge-cut-four-way\":" << edgeCut4Way << ",";
			f << "\"partition-diff\":" << partitionDiff << ',';
			f << "\"total-duration-seconds\":" << totalDurationSeconds << ',';
			f << "\"coarsen-duration-seconds\":" << coarsenDurationSeconds << ',';
			f << "\"refine-duration-seconds\":" << refineDurationSeconds << ',';
			f << "\"number-coarse-levels\":" << numCoarseLevels << ',';
			f << "\"level-1-first-two-passes-assign-rate\":" << mapFirstPassRate[0] << ',';
			f << "\"level-1-total-passes\":" << mapPasses[0] << ',';
			f << "\"level-2-first-two-passes-assign-rate\":" << mapFirstPassRate[1] << ',';
			f << "\"level-2-total-passes\":" << mapPasses[1] << ',';
			for (int i = 0; i < static_cast<int>(Measurement::END); i++) {
				f << "\"" << measurementNames[i] << "-duration-seconds\":" << measurements[i] << ",";
			}
			f << "\"coarse-levels\":[";

			bool firstLog = true;
			for (CoarseLevel l : coarseLevels) {
				if (!firstLog) {
					f << ',';
				}
				l.log(f);
				firstLog = false;
			}

			f << "]";
			f << "}";
			if (!last) {
				f << ",";
			}
			else {
				f << "]";
			}
			f.close();
		}
		else {
			std::cerr << "Could not open " << filename << std::endl;
		}
	}
};
