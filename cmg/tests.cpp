#include "sgpar.h"
#include <stdio.h>
#include <iostream>

using namespace sgpar;

bool hashmap_deduplication_test() {
	sgp_eid_t offsets[5] = { 0, 10, 15, 25, 31 };
	sgp_vid_t edges_per_source[5] = { 0, 0, 0, 0, 0 };
	sgp_vid_t dests[31];
	sgp_wgt_t wgts[31];
	for (sgp_vid_t i = 0; i < 4; i++) {
		for (sgp_eid_t j = offsets[i]; j < offsets[i + 1]; j++) {
			dests[j] = j - offsets[i];
			wgts[j] = j;
		}
	}

	dests[16] = 2;
	dests[17] = 4;
	dests[18] = 3;
	dests[19] = 2;
	dests[20] = 4;
	dests[21] = 2;
	dests[22] = 0;
	dests[23] = 3;
	dests[24] = 1;

	sgp_wgt_t expected_wgts[5] = { 15 + 22, 24, 16 + 19 + 21, 18 + 23, 17 + 20 };

	atom_eid_t total_edges(0);

	hashmap_deduplicate(offsets + 2, dests, wgts, edges_per_source + 2, &total_edges);

	bool success = true;

	success = success && (total_edges == 5);
	success = success && (edges_per_source[2] = 5);
	
	std::unordered_map<sgp_vid_t, sgp_wgt_t> actual_weights;
	for (sgp_eid_t i = offsets[2]; i < offsets[2] + edges_per_source[2]; i++) {
		actual_weights.insert({ dests[i], wgts[i] });
		std::cout << "Dest: " << dests[i] << "; Weight: " << wgts[i] << std::endl;
	}

	success = success && (actual_weights.size() == 5);

	for (sgp_vid_t i = 0; i < 5; i++) {
		success = success && (actual_weights.count(i) == 1);
		if (success) {
			success = success && (actual_weights.at(i) == expected_wgts[i]);
		}
	}

	return success;
}

bool heap_deduplication_test() {
	sgp_eid_t offsets[5] = { 0, 10, 15, 25, 31 };
	sgp_vid_t edges_per_source[5] = { 0, 0, 0, 0, 0 };
	sgp_vid_t dests[31];
	sgp_wgt_t wgts[31];
	for (sgp_vid_t i = 0; i < 4; i++) {
		for (sgp_eid_t j = offsets[i]; j < offsets[i + 1]; j++) {
			dests[j] = j - offsets[i];
			wgts[j] = j;
		}
	}

	dests[16] = 2;
	dests[17] = 4;
	dests[18] = 3;
	dests[19] = 2;
	dests[20] = 4;
	dests[21] = 2;
	dests[22] = 0;
	dests[23] = 3;
	dests[24] = 1;

	sgp_wgt_t expected_wgts[5] = { 15 + 22, 24, 16 + 19 + 21, 18 + 23, 17 + 20 };

	atom_eid_t total_edges(0);

	heap_deduplicate(offsets + 2, dests, wgts, edges_per_source + 2, &total_edges);

	bool success = true;

	success = success && (total_edges == 5);
	success = success && (edges_per_source[2] = 5);

	std::unordered_map<sgp_vid_t, sgp_wgt_t> actual_weights;
	for (sgp_eid_t i = offsets[2]; i < offsets[2] + edges_per_source[2]; i++) {
		actual_weights.insert({ dests[i], wgts[i] });
		std::cout << "Dest: " << dests[i] << "; Weight: " << wgts[i] << std::endl;
	}

	success = success && (actual_weights.size() == 5);

	for (sgp_vid_t i = 0; i < 5; i++) {
		success = success && (actual_weights.count(i) == 1);
		if (success) {
			success = success && (actual_weights.at(i) == expected_wgts[i]);
		}
	}

	return success;
}

int main() {
	if (hashmap_deduplication_test()) {
		printf("Hashmap deduplication test success\n");
	}
	else {
		printf("Hashmap deduplication test failure\n");
	}
	if (heap_deduplication_test()) {
		printf("Heap deduplication test success\n");
	}
	else {
		printf("Heap deduplication test failure\n");
	}
}