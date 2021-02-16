#include <stdio.h>
#include <stdlib.h>

#define SGPAR_IMPLEMENTATION
#include "sgpar.h"

#ifdef __cplusplus
using namespace sgpar;
#endif

int main(int argc, char **argv) {

    if (argc < 4) {
        printf("You input %d args\n", argc);
        fprintf(stderr, "Usage: %s csr_filename metrics_filename config_file\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *filename = argv[1];
    char *metrics = argv[2];

    config_t config;
    CHECK_SGPAR(sgp_load_config(argv[3], &config));

    sgp_graph_t g;
    CHECK_SGPAR( sgp_load_graph(&g, filename) );
    printf("n: %ld, m: %ld\n", g.nvertices, g.nedges);
    printf("coarsening_alg: %d, refine_alg: %d, local_alg %d, num_iter %d\n", 
                    config.coarsening_alg, config.refine_alg, config.local_search_alg, config.num_iter);

    sgp_vid_t *part;
    part = (sgp_vid_t *) malloc(g.nvertices * sizeof(sgp_vid_t));
    SGPAR_ASSERT(part != NULL);

    sgp_vid_t* best_part = NULL;
    int compare_part = 0;
    if(argc > 4){
        best_part = (sgp_vid_t*)malloc(g.nvertices * sizeof(sgp_vid_t));
        SGPAR_ASSERT(best_part != NULL);
        CHECK_SGPAR( sgp_load_partition(best_part, g.nvertices, argv[4]));
        compare_part = 1;
    }

    long edgecut_min = 1<<30;
    sgp_pcg32_random_t rng;
    rng.state = time(NULL);
    rng.inc   = 1;
#ifdef _KOKKOS
    Kokkos::initialize();
#endif
    for (int i=0; i < config.num_iter; i++) {
        sgp_eid_t edgecut = 0;
#ifdef EXPERIMENT
        ExperimentLoggerUtil experiment;
#endif
        CHECK_SGPAR( sgp_partition_graph(part, &edgecut, &config, 0, g,
#ifdef EXPERIMENT
            experiment,
#endif
                                        &rng) );

        sgp_vid_t part_diff = 0;
        if (compare_part) {
            CHECK_SGPAR(compute_partition_edit_distance(part, best_part, g.nvertices, &part_diff));
        }

        if (config.num_partitions == 4) {
            sgp_graph_t g1, g2;
            CHECK_SGPAR(sgp_use_partition(part, g, &g1, &g2));

            sgp_vid_t* part1;
            part1 = (sgp_vid_t*)malloc(g1.nvertices * sizeof(sgp_vid_t));
            SGPAR_ASSERT(part1 != NULL);

            sgp_vid_t* part2;
            part2 = (sgp_vid_t*)malloc(g2.nvertices * sizeof(sgp_vid_t));
            SGPAR_ASSERT(part2 != NULL);

            sgp_eid_t ec1 = 0, ec2 = 0;

#ifdef EXPERIMENT
            ExperimentLoggerUtil experiment1;
#endif
            CHECK_SGPAR(sgp_partition_graph(part1, &ec1, &config, 0, g1,
#ifdef EXPERIMENT
                experiment1,
#endif
                &rng));

#ifdef EXPERIMENT
            ExperimentLoggerUtil experiment2;
#endif
            CHECK_SGPAR(sgp_partition_graph(part2, &ec2, &config, 0, g2,
#ifdef EXPERIMENT
                experiment2,
#endif
                &rng));

#ifdef EXPERIMENT
            experiment.setEdgeCut4Way(edgecut + ec1 + ec2);
#endif
            sgp_free_graph(&g1);
            sgp_free_graph(&g2);
        }

#ifdef EXPERIMENT
        experiment.setPartitionDiff(part_diff);
#endif
        
        if (edgecut < edgecut_min) {
            edgecut_min = edgecut;
        }
#ifdef EXPERIMENT
        bool first = true, last = true;
        if (i > 0) {
            first = false;
        }
        if (i + 1 < config.num_iter) {
            last = false;
        }
        experiment.log(metrics, first, last);
#endif
    }
#ifdef _KOKKOS
    Kokkos::finalize();
#endif
    printf("graph %s, min edgecut found is %ld\n", 
                    filename, edgecut_min);

    /*
    FILE *outfp = fopen("parts.txt", "w");
    for (sgp_vid_t i=0;  i<g.nvertices; i++) {
        fprintf(outfp, "%d\n", part[i]); 
    }
    fclose(outfp);
    */

    CHECK_SGPAR( sgp_free_graph(&g) );
    free(part);
    if (compare_part) {
        free(best_part);
    }

    return EXIT_SUCCESS;
}

