/** \file    sgpar.h
 *  \brief   Multilevel spectral graph partitioning
 *  \authors Kamesh Madduri, Shad Kirmani, and Michael Gilbert
 *  \date    September 2019
 *  \license MIT License 
 */

#ifndef SGPAR_H_
#define SGPAR_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#else
#include <sys/time.h>
#endif

#ifdef _KOKKOS
#include "coarsen_kokkos.h"
#include "eigensolve_kokkos.h"
#endif

#ifdef __cplusplus
#include <atomic>
#include <unordered_map>
// #define USE_GNU_PARALLELMODE
#ifdef USE_GNU_PARALLELMODE
#include <parallel/algorithm> // for parallel sort
#else 
#include <algorithm>          // for STL sort
#endif

#ifdef EXPERIMENT
#include "ExperimentLoggerUtil.cpp"
#endif
#else
#include <stdatomic.h>
#endif

#ifdef __cplusplus
namespace sgpar {
//extern "C" {
#endif

//prevent conflicts with definitions_kokkos.h (included from coarsen_kokkos.h)
#ifndef _KOKKOS
#ifdef __cplusplus
#define SGPAR_API 
#endif // __cplusplus

#ifndef __cplusplus
#ifdef SGPAR_STATIC
#define SGPAR_API static
#else
#define SGPAR_API extern
#endif // __SGPAR_STATIC
#endif

#define SGPAR_USE_ASSERT
#ifdef SGPAR_USE_ASSERT
#ifndef SGPAR_ASSERT
#include <assert.h>
#define SGPAR_ASSERT(expr) assert(expr)
#endif
#else
#define SGPAR_ASSERT(expr) 
#endif

/**********************************************************
 *  PCG Random Number Generator
 **********************************************************
 */

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

typedef struct { uint64_t state;  uint64_t inc; } sgp_pcg32_random_t;

uint32_t sgp_pcg32_random_r(sgp_pcg32_random_t* rng) {
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**********************************************************
 * Internal
 **********************************************************
 */
#if defined(SGPAR_HUGEGRAPHS)
typedef uint64_t sgp_vid_t;
#define SGP_INFTY UINT64_MAX
typedef uint64_t sgp_eid_t;
#elif defined(SGPAR_LARGEGRAPHS)
typedef uint32_t sgp_vid_t;
#define SGP_INFTY UINT32_MAX
typedef uint64_t sgp_eid_t;
#else
typedef uint32_t sgp_vid_t;
#define SGP_INFTY UINT32_MAX
typedef uint32_t sgp_eid_t;
#endif
typedef double sgp_real_t;
#ifdef _KOKKOS
typedef sgp_real_t sgp_wgt_t;
#else
typedef sgp_vid_t sgp_wgt_t;
#endif

#ifndef SGPAR_COARSENING_VTX_CUTOFF
#define SGPAR_COARSENING_VTX_CUTOFF 50
#endif

#ifndef SGPAR_COARSENING_MAXLEVELS
#define SGPAR_COARSENING_MAXLEVELS 100
#endif

#ifdef _OPENMP
#ifdef __cplusplus
typedef std::atomic<sgp_vid_t> atom_vid_t;
typedef std::atomic<sgp_eid_t> atom_eid_t;
#else
typedef _Atomic sgp_vid_t atom_vid_t;
typedef _Atomic sgp_eid_t atom_eid_t;
#endif
#else
typedef sgp_vid_t atom_vid_t;
typedef sgp_eid_t atom_eid_t;
#endif

static sgp_real_t SGPAR_POWERITER_TOL = 1e-10;
static sgp_real_t MAX_COARSEN_RATIO = 0.9;

//100 trillion
#define SGPAR_POWERITER_ITER 100000000000000

typedef struct {
    sgp_vid_t   nvertices;
    sgp_eid_t   nedges;
    sgp_eid_t* source_offsets;
    sgp_vid_t* edges_per_source;
    sgp_vid_t* destination_indices;
    sgp_wgt_t* weighted_degree;
    sgp_wgt_t* eweights;
} sgp_graph_t;

#define CHECK_SGPAR(func)                                                      \
{                                                                              \
    int status = (func);                                                       \
    if (status != 0) {                                                         \
        printf("sgpar Error: return value %d at line %d. Exiting ... \n",      \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_RETSTAT(func)                                                    \
{                                                                              \
    int status = (func);                                                       \
    if (status != 0) {                                                         \
        printf("Error: return value %d at line %d. Exiting ...\n",             \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

SGPAR_API double sgp_timer() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + ((1e-6) * tp.tv_usec));
#endif
}
#endif

typedef struct configuration {
    int coarsening_alg;
    int refine_alg;
    int local_search_alg;
    int num_iter;
    double tol;
    int num_partitions;
} config_t;

SGPAR_API int change_tol(sgp_real_t new_tol){
    SGPAR_POWERITER_TOL = new_tol;

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_coarsen_ACE(sgp_graph_t* interp,
    sgp_vid_t* nvertices_coarse_ptr,
    const sgp_graph_t g,
    const int coarsening_level,
    sgp_pcg32_random_t* rng) {

    sgp_vid_t n = g.nvertices;

    sgp_vid_t* vperm = (sgp_vid_t*)malloc(n * sizeof(sgp_vid_t));
    SGPAR_ASSERT(vperm != NULL);

    sgp_vid_t* vcmap = (sgp_vid_t*)malloc(n * sizeof(sgp_vid_t));

    for (sgp_vid_t i = 0; i < n; i++) {
        vcmap[i] = SGP_INFTY;
        vperm[i] = i;
    }


    for (sgp_vid_t i = n - 1; i > 0; i--) {
        sgp_vid_t v_i = vperm[i];
#ifndef SGPAR_HUGEGRAPHS
        uint32_t j = (sgp_pcg32_random_r(rng)) % (i + 1);
#else
        uint64_t j1 = (sgp_pcg32_random_r(rng)) % (i + 1);
        uint64_t j2 = (sgp_pcg32_random_r(rng)) % (i + 1);
        uint64_t j = ((j1 << 32) + j2) % (i + 1);
#endif 
        sgp_vid_t v_j = vperm[j];
        vperm[i] = v_j;
        vperm[j] = v_i;
    }

    sgp_vid_t nvertices_coarse = 0;

    sgp_real_t threshold = 0.3;

    //add some vertices to the representative set
    if (coarsening_level == 1) {
        for (sgp_vid_t i = 0; i < n; i++) {
            sgp_vid_t u = vperm[i];

            sgp_real_t degree_total = 0.0;
            sgp_real_t degree_representative = 0.0;
            for (sgp_eid_t j = g.source_offsets[u];
                j < g.source_offsets[u + 1]; j++) {
                sgp_vid_t v = g.destination_indices[j];
                degree_total += 1.0;
                if (vcmap[v] != SGP_INFTY) {
                    degree_representative += 1.0;
                }

            }
            if (degree_representative / degree_total < threshold) {
                vcmap[u] = nvertices_coarse++;
            }
        }
    }
    else {
        for (sgp_vid_t i = 0; i < n; i++) {
            sgp_vid_t u = vperm[i];

            sgp_real_t degree_total = 0.0;
            sgp_real_t degree_representative = 0.0;
            for (sgp_eid_t j = g.source_offsets[u];
                j < g.source_offsets[u + 1]; j++) {
                sgp_vid_t v = g.destination_indices[j];
                degree_total += g.eweights[j];
                if (vcmap[v] != SGP_INFTY) {
                    degree_representative += g.eweights[j];
                }

            }
            if (degree_representative / degree_total < threshold) {
                vcmap[u] = nvertices_coarse++;
            }
        }
    }
    free(vperm);

    interp->source_offsets = (sgp_eid_t*) malloc((n + 1) * sizeof(sgp_eid_t));

    interp->source_offsets[0] = 0;
    for (sgp_vid_t u = 0; u < n; u++) {
        sgp_eid_t counter = 0;
        if (vcmap[u] != SGP_INFTY) {
            counter = 1;
        }
        else {
            for (sgp_eid_t j = g.source_offsets[u]; j < g.source_offsets[u + 1]; j++) {
                sgp_vid_t v = g.destination_indices[j];
                if (vcmap[v] != SGP_INFTY) {
                    counter++;
                }
            }
        }
        interp->source_offsets[u + 1] = interp->source_offsets[u] + counter;
    }
    interp->destination_indices = (sgp_vid_t*)malloc(interp->source_offsets[n] * sizeof(sgp_vid_t));
    interp->eweights = (sgp_wgt_t*) malloc(interp->source_offsets[n] * sizeof(sgp_wgt_t));
    //compute the interpolation weights
    if (coarsening_level == 1) {
        for (sgp_vid_t u = 0; u < n; u++) {
            sgp_eid_t offset = interp->source_offsets[u];
            if (vcmap[u] != SGP_INFTY) {
                interp->destination_indices[offset] = vcmap[u];
                interp->eweights[offset] = 1.0;
            }
            else {
                sgp_real_t degree_representative = 0.0;
                //count sum weights to representative vertices
                for (sgp_eid_t j = g.source_offsets[u]; j < g.source_offsets[u + 1]; j++) {
                    sgp_vid_t v = g.destination_indices[j];
                    if (vcmap[v] != SGP_INFTY) {
                        degree_representative += 1.0;
                    }
                }
                for (sgp_eid_t j = g.source_offsets[u]; j < g.source_offsets[u + 1]; j++) {
                    sgp_vid_t v = g.destination_indices[j];
                    if (vcmap[v] != SGP_INFTY) {
                        interp->destination_indices[offset] = vcmap[v];
                        interp->eweights[offset] = 1.0 / degree_representative;
                        offset++;
                    }
                }
            }
        }
    }
    else {
        for (sgp_vid_t u = 0; u < n; u++) {
            sgp_eid_t offset = interp->source_offsets[u];
            if (vcmap[u] != SGP_INFTY) {
                interp->destination_indices[offset] = vcmap[u];
                interp->eweights[offset] = 1.0;
            }
            else {
                sgp_real_t degree_representative = 0.0;
                //count sum weights to representative vertices
                for (sgp_eid_t j = g.source_offsets[u]; j < g.source_offsets[u + 1]; j++) {
                    sgp_vid_t v = g.destination_indices[j];
                    if (vcmap[v] != SGP_INFTY) {
                        degree_representative += g.eweights[j];
                    }
                }
                for (sgp_eid_t j = g.source_offsets[u]; j < g.source_offsets[u + 1]; j++) {
                    sgp_vid_t v = g.destination_indices[j];
                    if (vcmap[v] != SGP_INFTY) {
                        interp->destination_indices[offset] = vcmap[v];
                        interp->eweights[offset] = g.eweights[j] / degree_representative;
                        offset++;
                    }
                }
            }
        }
    }
    free(vcmap);

    *nvertices_coarse_ptr = nvertices_coarse;

    return EXIT_SUCCESS;

}

SGPAR_API int sgp_coarsen_heavy_edge_matching(sgp_vid_t* vcmap,
                                              sgp_vid_t* nvertices_coarse_ptr,
                                              const sgp_graph_t g,
                                              const int coarsening_level,
                                              sgp_pcg32_random_t* rng) {

	sgp_vid_t n = g.nvertices;

	sgp_vid_t* vperm = (sgp_vid_t*)malloc(n * sizeof(sgp_vid_t));
	SGPAR_ASSERT(vperm != NULL);

	for (sgp_vid_t i = 0; i < n; i++) {
		vcmap[i] = SGP_INFTY;
		vperm[i] = i;
	}


	for (sgp_vid_t i = n - 1; i > 0; i--) {
		sgp_vid_t v_i = vperm[i];
#ifndef SGPAR_HUGEGRAPHS
		uint32_t j = (sgp_pcg32_random_r(rng)) % (i + 1);
#else
		uint64_t j1 = (sgp_pcg32_random_r(rng)) % (i + 1);
		uint64_t j2 = (sgp_pcg32_random_r(rng)) % (i + 1);
		uint64_t j = ((j1 << 32) + j2) % (i + 1);
#endif 
		sgp_vid_t v_j = vperm[j];
		vperm[i] = v_j;
		vperm[j] = v_i;
	}

	sgp_vid_t nvertices_coarse = 0;
	
    if(coarsening_level == 1){
        //match each vertex with its first unmatched neighbor
        for (sgp_vid_t i = 0; i < n; i++) {
            sgp_vid_t u = vperm[i];
            if (vcmap[u] == SGP_INFTY) {
                sgp_vid_t match = u;

                for (sgp_eid_t j = g.source_offsets[u] + 1;
                    j < g.source_offsets[u + 1]; j++) {
                    sgp_vid_t v = g.destination_indices[j];
                    //v must be unmatched to be considered
                    if (vcmap[v] == SGP_INFTY) {
                        j = g.source_offsets[u + 1];//break the loop
                        match = v;
                    }

                }
                sgp_vid_t coarse_vtx = nvertices_coarse++;
                vcmap[u] = coarse_vtx;
                vcmap[match] = coarse_vtx;//u and match are the same when matching with self
            }
        }
    }
    else {        
        //match the vertices, in random order, with the vertex on their heaviest adjacent edge
        //if no unmatched adjacent vertex exists, match with self
        for (sgp_vid_t i = 0; i < n; i++) {
            sgp_vid_t u = vperm[i];
            if (vcmap[u] == SGP_INFTY) {
                sgp_vid_t match = u;
                sgp_wgt_t max_ewt = 0;

                for (sgp_eid_t j = g.source_offsets[u] + 1;
                    j < g.source_offsets[u + 1]; j++) {
                    sgp_vid_t v = g.destination_indices[j];
                    //v must be unmatched to be considered
                    if (max_ewt < g.eweights[j] && vcmap[v] == SGP_INFTY) {
                        max_ewt = g.eweights[j];
                        match = v;
                    }

                }
                sgp_vid_t coarse_vtx = nvertices_coarse++;
                vcmap[u] = coarse_vtx;
                vcmap[match] = coarse_vtx;//u and match are the same when matching with self
            }
        }
    }

	free(vperm);

	*nvertices_coarse_ptr = nvertices_coarse;

	return EXIT_SUCCESS;

}
#ifdef _OPENMP
SGPAR_API int sgp_coarsen_HEC(sgp_vid_t *vcmap, 
                              sgp_vid_t *nvertices_coarse_ptr, 
                              const sgp_graph_t g, 
                              const int coarsening_level,
                              sgp_pcg32_random_t *rng) {
    sgp_vid_t n = g.nvertices;

    sgp_vid_t *vperm = (sgp_vid_t *) malloc(n * sizeof(sgp_vid_t));
    SGPAR_ASSERT(vperm != NULL);

    for (sgp_vid_t i=0; i<n; i++) {
        vcmap[i] = SGP_INFTY;
        vperm[i] = i;
    }


    for (sgp_vid_t i=n-1; i>0; i--) {
        sgp_vid_t v_i = vperm[i];
#ifndef SGPAR_HUGEGRAPHS
        uint32_t j = (sgp_pcg32_random_r(rng)) % (i+1);
#else
        uint64_t j1 = (sgp_pcg32_random_r(rng)) % (i+1);
        uint64_t j2 = (sgp_pcg32_random_r(rng)) % (i+1);
        uint64_t j  = ((j1<<32) + j2) % (i+1);
#endif 
        sgp_vid_t v_j = vperm[j];
        vperm[i] = v_j;
        vperm[j] = v_i;
    }

    sgp_vid_t *hn = (sgp_vid_t *) malloc(n * sizeof(sgp_vid_t));
    SGPAR_ASSERT(hn != NULL);

	omp_lock_t * v_locks = (omp_lock_t*) malloc(n * sizeof(omp_lock_t));
	SGPAR_ASSERT(v_locks != NULL);

#ifdef __cplusplus
	std::atomic<sgp_vid_t> nvertices_coarse(0);
#else
    _Atomic sgp_vid_t nvertices_coarse = 0;
#endif

#pragma omp parallel
{

	int tid = omp_get_thread_num();
	sgp_pcg32_random_t t_rng;
	t_rng.state = rng->state + tid;
	t_rng.inc = rng->inc;

	for (sgp_vid_t i = 0; i < n; i++) {
		hn[i] = SGP_INFTY;
	}

#pragma omp barrier

	if (coarsening_level == 1) {
#pragma omp for
		for (sgp_vid_t i = 0; i < n; i++) {
			sgp_vid_t adj_size = g.source_offsets[i + 1] - g.source_offsets[i];
            if (adj_size == 0) {
                //no edges, so pair this vertex to a random vertex
                sgp_vid_t hn_i = (sgp_pcg32_random_r(&t_rng)) % (n - 1);
                //ensure that this vertex is not paired to itself
                if (hn_i == i) {
                    hn_i++;
                }
                hn[i] = hn_i;
            }
            else {
                sgp_vid_t offset = (sgp_pcg32_random_r(&t_rng)) % adj_size;
                // sgp_vid_t offset = 0;
                hn[i] = g.destination_indices[g.source_offsets[i] + offset];
            }
		}
	}
	else {
#pragma omp for
		for (sgp_vid_t i = 0; i < n; i++) {
            if (g.edges_per_source[i] == 0) {
                //no edges, so pair this vertex to a random vertex
                sgp_vid_t hn_i = (sgp_pcg32_random_r(&t_rng)) % (n - 1);
                //ensure that this vertex is not paired to itself
                if (hn_i == i) {
                    hn_i++;
                }
                hn[i] = hn_i;
            }
            else {
                sgp_vid_t hn_i = g.destination_indices[g.source_offsets[i]];
                sgp_wgt_t max_ewt = g.eweights[g.source_offsets[i]];

                for (sgp_eid_t j = g.source_offsets[i] + 1; j < g.source_offsets[i] + g.edges_per_source[i]; j++) {
                    if (max_ewt < g.eweights[j]) {
                        max_ewt = g.eweights[j];
                        hn_i = g.destination_indices[j];
                    }

                }
                hn[i] = hn_i;
            }
		}
	}

#pragma omp for
    for(sgp_vid_t i = 0; i < n; i++){
        omp_init_lock(v_locks + i);
    }

#pragma omp for
	for (sgp_vid_t i = 0; i < n; i++) {
		sgp_vid_t u = vperm[i];
		sgp_vid_t v = hn[u];

		sgp_vid_t less = u, more = v;
		if (v < u) {
			less = v;
			more = u;
		}

		omp_set_lock(v_locks + less);
		omp_set_lock(v_locks + more);
		if (vcmap[u] == SGP_INFTY) {
			if (vcmap[v] == SGP_INFTY) {
				vcmap[v] = nvertices_coarse++;
			}
			vcmap[u] = vcmap[v];
		}
		omp_unset_lock(v_locks + more);
		omp_unset_lock(v_locks + less);
	}

#pragma omp for
    for(sgp_vid_t i = 0; i < n; i++){
        omp_destroy_lock(v_locks + i);
    }
}
    
    free(hn);
    free(vperm);
	free(v_locks);

    *nvertices_coarse_ptr = nvertices_coarse;
    
    return EXIT_SUCCESS;
}
#else
SGPAR_API int sgp_coarsen_HEC(sgp_vid_t* vcmap,
    sgp_vid_t* nvertices_coarse_ptr,
    const sgp_graph_t g,
    const int coarsening_level,
    sgp_pcg32_random_t* rng) {
    sgp_vid_t n = g.nvertices;

    sgp_vid_t* vperm = (sgp_vid_t*)malloc(n * sizeof(sgp_vid_t));
    SGPAR_ASSERT(vperm != NULL);

    for (sgp_vid_t i = 0; i < n; i++) {
        vcmap[i] = SGP_INFTY;
        vperm[i] = i;
    }


    for (sgp_vid_t i = n - 1; i > 0; i--) {
        sgp_vid_t v_i = vperm[i];
#ifndef SGPAR_HUGEGRAPHS
        uint32_t j = (sgp_pcg32_random_r(rng)) % (i + 1);
#else
        uint64_t j1 = (sgp_pcg32_random_r(rng)) % (i + 1);
        uint64_t j2 = (sgp_pcg32_random_r(rng)) % (i + 1);
        uint64_t j = ((j1 << 32) + j2) % (i + 1);
#endif 
        sgp_vid_t v_j = vperm[j];
        vperm[i] = v_j;
        vperm[j] = v_i;
    }

    sgp_vid_t* hn = (sgp_vid_t*)malloc(n * sizeof(sgp_vid_t));
    SGPAR_ASSERT(hn != NULL);

    for (sgp_vid_t i = 0; i < n; i++) {
        hn[i] = SGP_INFTY;
    }

    if (coarsening_level == 1) {
        for (sgp_vid_t i = 0; i < n; i++) {
            sgp_vid_t adj_size = g.source_offsets[i + 1] - g.source_offsets[i];
            if (adj_size == 0) {
                //no edges, so pair this vertex to a random vertex
                sgp_vid_t hn_i = (sgp_pcg32_random_r(rng)) % (n - 1);
                //ensure that this vertex is not paired to itself
                if (hn_i == i) {
                    hn_i++;
                }
                hn[i] = hn_i;
            }
            else {
                sgp_vid_t offset = (sgp_pcg32_random_r(rng)) % adj_size;
                // sgp_vid_t offset = 0;
                hn[i] = g.destination_indices[g.source_offsets[i] + offset];
            }
        }
    }
    else {
        for (sgp_vid_t i = 0; i < n; i++) {
            if (g.edges_per_source[i] == 0) {
                //no edges, so pair this vertex to a random vertex
                sgp_vid_t hn_i = (sgp_pcg32_random_r(rng)) % (n - 1);
                //ensure that this vertex is not paired to itself
                if (hn_i == i) {
                    hn_i++;
                }
                hn[i] = hn_i;
            }
            else {
                sgp_vid_t hn_i = g.destination_indices[g.source_offsets[i]];
                sgp_wgt_t max_ewt = g.eweights[g.source_offsets[i]];

                for (sgp_eid_t j = g.source_offsets[i] + 1; j < g.source_offsets[i] + g.edges_per_source[i]; j++) {
                    if (max_ewt < g.eweights[j]) {
                        max_ewt = g.eweights[j];
                        hn_i = g.destination_indices[j];
                    }

                }
                hn[i] = hn_i;
            }
        }
    }

    sgp_vid_t nvertices_coarse = 0;

    for (sgp_vid_t i = 0; i < n; i++) {
        sgp_vid_t u = vperm[i];
        sgp_vid_t v = hn[u];
        if (vcmap[u] == SGP_INFTY) {
            if (vcmap[v] == SGP_INFTY) {
                vcmap[v] = nvertices_coarse++;
            }
            vcmap[u] = vcmap[v];
        }
    }

    free(hn);
    free(vperm);

    *nvertices_coarse_ptr = nvertices_coarse;

    return EXIT_SUCCESS;
}
#endif

typedef struct {
    sgp_vid_t u;
    sgp_vid_t v;
    sgp_vid_t w;
} edge_triple_t;
#ifdef __cplusplus

inline static bool uvw_cmpfn_inc(const edge_triple_t& a, 
                                 const edge_triple_t& b) {
    if (a.u != b.u) {
        return (a.u < b.u); // sort by u, increasing order
    } else {
        if (a.v != b.v) {
            return (a.v < b.v); // sort by v, increasing order
        } else {
            return (a.w > b.w); // sort by w, increasing order
        }
    }
}
#else
static int uvw_cmpfn_inc(const void *a, const void *b) {
    sgp_vid_t *av = ((sgp_vid_t *) a);
    sgp_vid_t *bv = ((sgp_vid_t *) b);
    if (av[0] > bv[0]) {
        return 1;
    }
    if (av[0] < bv[0]) {
        return -1;
    }
    if (*av == *bv) {
        if (av[1] > bv[1])
            return 1;
        if (av[1] < bv[1])
            return -1;
        if (av[1] == bv[1]) {
            if (av[2] < bv[2])
                return 1;
            if (av[2] > bv[2]) 
                return -1;
        }
    }
    return 0;
}
#endif

typedef struct {
    sgp_real_t ev;
    sgp_vid_t  u;
} sgp_vv_pair_t;

#ifdef __cplusplus
inline static bool vu_cmpfn_inc(const sgp_vv_pair_t& a, 
                                const sgp_vv_pair_t& b) {
    if (a.ev != b.ev) {
        return (a.ev < b.ev); // sort by ev
    } else {
        return (a.u < b.u); // sort by u, increasing order
    }
}
#else
static int vu_cmpfn_inc(const void *a, const void *b) {
    sgp_vv_pair_t *av = ((sgp_vv_pair_t *) a);
    sgp_vv_pair_t *bv = ((sgp_vv_pair_t *) b);
    if ((*av).ev > (*bv).ev) {
        return 1;
    }
    if ((*av).ev < (*bv).ev) {
        return -1;
    }
    if ((*av).ev == (*bv).ev) {
        if ((*av).u > (*bv).u)
            return 1;
        else
            return -1;
    }
    return 0;
}
#endif

//assumption: source_offsets[rangeBegin] <= target < source_offsets[rangeEnd] 
//
static sgp_vid_t binary_search_find_source_index(sgp_eid_t *source_offsets, sgp_vid_t rangeBegin, sgp_vid_t rangeEnd, sgp_eid_t target){
    if(rangeBegin + 1 == rangeEnd){
        return rangeBegin;
    }
    int rangeMiddle = (rangeBegin + rangeEnd) >> 1;
    if(source_offsets[rangeMiddle] <= target){
        return binary_search_find_source_index(source_offsets, rangeMiddle, rangeEnd, target);
    } else {
        return binary_search_find_source_index(source_offsets, rangeBegin, rangeMiddle, target);
    }
}

static sgp_eid_t binary_search_find_first_self_loop(edge_triple_t *edges, sgp_eid_t rangeBegin, sgp_eid_t rangeEnd){
    if(rangeBegin + 1 == rangeEnd){
        return rangeEnd;
    }
    int rangeMiddle = (rangeBegin + rangeEnd) >> 1;
    if(edges[rangeMiddle].u != SGP_INFTY){
        return binary_search_find_first_self_loop(edges, rangeMiddle, rangeEnd);
    } else {
        return binary_search_find_first_self_loop(edges, rangeBegin, rangeMiddle);
    }
}

void heap_deduplicate(sgp_eid_t* offset_bottom, sgp_vid_t* dest_by_source, sgp_wgt_t* wgt_by_source, sgp_vid_t* edges_per_source, sgp_eid_t* gc_nedges) {

    sgp_eid_t bottom = *offset_bottom;
    sgp_eid_t top = *(offset_bottom + 1);
    sgp_vid_t size = top - bottom;
    sgp_eid_t offset = bottom;
    sgp_eid_t last_offset = offset;
    //max heapify (root at source_bucket_offset[u+1] - 1)
    for (sgp_vid_t i = size / 2; i > 0; i--) {
        sgp_eid_t heap_node = top - i, leftC = top - 2 * i, rightC = top - 1 - 2 * i;
        sgp_vid_t j = i;
        //heapify heap_node
        while ((2 * j <= size && dest_by_source[heap_node] < dest_by_source[leftC]) || (2 * j + 1 <= size && dest_by_source[heap_node] < dest_by_source[rightC])) {
            if (2 * j + 1 > size || dest_by_source[leftC] > dest_by_source[rightC]) {
                sgp_vid_t swap = dest_by_source[leftC];
                dest_by_source[leftC] = dest_by_source[heap_node];
                dest_by_source[heap_node] = swap;

                sgp_wgt_t w_swap = wgt_by_source[leftC];
                wgt_by_source[leftC] = wgt_by_source[heap_node];
                wgt_by_source[heap_node] = w_swap;
                j = 2 * j;
            }
            else {
                sgp_vid_t swap = dest_by_source[rightC];
                dest_by_source[rightC] = dest_by_source[heap_node];
                dest_by_source[heap_node] = swap;

                sgp_wgt_t w_swap = wgt_by_source[rightC];
                wgt_by_source[rightC] = wgt_by_source[heap_node];
                wgt_by_source[heap_node] = w_swap;
                j = 2 * j + 1;
            }
            heap_node = top - j, leftC = top - 2 * j, rightC = top - 1 - 2 * j;
        }
    }

    //heap sort
    for (sgp_eid_t i = bottom; i < top; i++) {

        sgp_vid_t top_swap = dest_by_source[top - 1];
        dest_by_source[top - 1] = dest_by_source[i];
        dest_by_source[i] = top_swap;

        sgp_wgt_t top_w_swap = wgt_by_source[top - 1];
        wgt_by_source[top - 1] = wgt_by_source[i];
        wgt_by_source[i] = top_w_swap;

        size--;

        sgp_vid_t j = 1;
        sgp_eid_t heap_node = top - j, leftC = top - 2 * j, rightC = top - 1 - 2 * j;
        //re-heapify root node
        while ((2 * j <= size && dest_by_source[heap_node] < dest_by_source[leftC]) || (2 * j + 1 <= size && dest_by_source[heap_node] < dest_by_source[rightC])) {
            if (2 * j + 1 > size || dest_by_source[leftC] > dest_by_source[rightC]) {
                sgp_vid_t swap = dest_by_source[leftC];
                dest_by_source[leftC] = dest_by_source[heap_node];
                dest_by_source[heap_node] = swap;

                sgp_wgt_t w_swap = wgt_by_source[leftC];
                wgt_by_source[leftC] = wgt_by_source[heap_node];
                wgt_by_source[heap_node] = w_swap;
                j = 2 * j;
            }
            else {
                sgp_vid_t swap = dest_by_source[rightC];
                dest_by_source[rightC] = dest_by_source[heap_node];
                dest_by_source[heap_node] = swap;

                sgp_wgt_t w_swap = wgt_by_source[rightC];
                wgt_by_source[rightC] = wgt_by_source[heap_node];
                wgt_by_source[heap_node] = w_swap;
                j = 2 * j + 1;
            }
            heap_node = top - j, leftC = top - 2 * j, rightC = top - 1 - 2 * j;
        }

        //sub-array is now sorted from bottom to i

        if (last_offset < offset) {
            if (dest_by_source[last_offset] == dest_by_source[i]) {
                wgt_by_source[last_offset] += wgt_by_source[i];
            }
            else {
                dest_by_source[offset] = dest_by_source[i];
                wgt_by_source[offset] = wgt_by_source[i];
                last_offset = offset;
                offset++;
                (*gc_nedges)++;
            }
        }
        else {
            offset++;
            (*gc_nedges)++;
        }
    }
    *edges_per_source = offset - *offset_bottom;
}

#ifdef __cplusplus
void hashmap_deduplicate(sgp_eid_t* offset_bottom, sgp_vid_t* dest_by_source, sgp_wgt_t* wgt_by_source, sgp_vid_t* edges_per_source, sgp_eid_t* gc_nedges) {

    sgp_eid_t bottom = *offset_bottom;
    sgp_eid_t top = *(offset_bottom + 1);
    sgp_eid_t next_offset = bottom;
    std::unordered_map<sgp_vid_t, sgp_eid_t> map;
    map.reserve(top - bottom);
    //hashing sort
    for (sgp_eid_t i = bottom; i < top; i++) {

        sgp_vid_t v = dest_by_source[i];

        if (map.count(v) > 0) {
            sgp_eid_t idx = map.at(v);

            wgt_by_source[idx] += wgt_by_source[i];
        }
        else {
            map.insert({ v, next_offset });
            dest_by_source[next_offset] = dest_by_source[i];
            wgt_by_source[next_offset] = wgt_by_source[i];
            next_offset++;
            (*gc_nedges)++;
        }
    }

    *edges_per_source = next_offset - *offset_bottom;
}
#endif

#ifdef _OPENMP

void parallel_prefix_sum_tree(sgp_eid_t* gc_source_offsets, sgp_vid_t nc, int t_id, int total_threads) {

	//tree-reduction upwards first (largest index contains sum of whole array)
	sgp_vid_t multiplier = 1, prev_multiplier = 1;
	while (multiplier < nc) {
		multiplier <<= 1;
		sgp_vid_t pos = 0;
		//prevent unsigned rollover
		if (nc >= t_id * multiplier) {
			//standard reduction would have sum of whole array in lowest index
			//this makes it easier to compute the indices we need to add
			pos = nc - t_id * multiplier;
		}
#pragma omp barrier
		//strictly greater because gc_source_offsets[0] is always zero
		while (pos > prev_multiplier) {
			gc_source_offsets[pos] = gc_source_offsets[pos] + gc_source_offsets[pos - prev_multiplier];
			//prevent unsigned rollover
			if (pos >= multiplier * total_threads) {
				pos -= multiplier * total_threads;
			}
			else {
				pos = 0;
			}
		}
		prev_multiplier = multiplier;
	}

	//compute left-sums from the root of the tree downwards
	multiplier >>= 1;
	sgp_vid_t next_multiplier = multiplier >> 1;
	while (next_multiplier > 0) {
		sgp_vid_t pos = 0;
		if (nc > (next_multiplier + t_id * multiplier)) {
			pos = nc - (next_multiplier + t_id * multiplier);
		}
		//strictly greater because gc_source_offsets[0] is always zero
#pragma omp barrier
		while (pos > next_multiplier) {
			gc_source_offsets[pos] = gc_source_offsets[pos] + gc_source_offsets[pos - next_multiplier];
			//prevent unsigned rollover
			if (pos >= multiplier * total_threads) {
				pos -= multiplier * total_threads;
			}
			else {
				pos = 0;
			}
		}
		multiplier = next_multiplier;
		next_multiplier >>= 1;
	}
#pragma omp barrier
}

void parallel_prefix_sum(sgp_eid_t* gc_source_offsets, sgp_vid_t nc, int t_id, int total_threads) {

    sgp_vid_t idx_per_thread = (nc / total_threads) + 1;

    sgp_vid_t start = t_id * idx_per_thread;

    sgp_vid_t end = start + idx_per_thread;

    if (end > nc) {
        end = nc;
    }

    for (sgp_vid_t i = start + 1; i < end; i++) {
        gc_source_offsets[i + 1] += gc_source_offsets[i];
    }

#pragma omp barrier

#pragma omp single
{
    for (sgp_vid_t i = 2 * idx_per_thread; i <= nc; i += idx_per_thread) {
        gc_source_offsets[i] += gc_source_offsets[i - idx_per_thread];
    }
}

    end = start + idx_per_thread - 1;
    if (end > nc) {
        end = nc;
    }
    if (end < start) {
        start = end;
    }
    sgp_eid_t add = gc_source_offsets[start];

    for (sgp_vid_t i = start; i < end; i++) {
        gc_source_offsets[i + 1] += add;
    }

#pragma omp barrier
}

SGPAR_API int sgp_build_coarse_graph_msd(sgp_graph_t* gc,
    sgp_vid_t* vcmap,
    const sgp_graph_t g,
    const int coarsening_level,
    double* time_ptrs,
    const int coarsening_alg) {
    sgp_vid_t n = g.nvertices;
    sgp_vid_t nc = gc->nvertices;

    double start_dedupe = 0;
    double start_count = 0;
    double start_prefix = 0;
    double start_bucket = 0;

    //radix sort source vertices, then sort edges

#ifdef __cplusplus
    atom_vid_t * edges_per_source_atomic = new atom_vid_t[nc];
    SGPAR_ASSERT(edges_per_source_atomic != NULL);
#else
    atom_vid_t* edges_per_source_atomic = (atom_vid_t*) calloc(nc, sizeof(atom_vid_t));
    SGPAR_ASSERT(edges_per_source_atomic != NULL);
#endif

    sgp_vid_t* mapped_edges = (sgp_vid_t*)malloc(g.source_offsets[n] * sizeof(sgp_vid_t));
    SGPAR_ASSERT(mapped_edges != NULL);

    sgp_eid_t* source_bucket_offset = (sgp_eid_t*)calloc(nc + 1, sizeof(sgp_eid_t));
    SGPAR_ASSERT(source_bucket_offset != NULL);
    source_bucket_offset[0] = 0;

    sgp_vid_t* dest_by_source;
    sgp_wgt_t* wgt_by_source;

    sgp_eid_t gc_count[256];

    sgp_vid_t * edges_per_source = (sgp_vid_t *) malloc(nc * sizeof(sgp_vid_t));
    SGPAR_ASSERT(edges_per_source != NULL);
    start_count = sgp_timer();

#pragma omp parallel
{
    sgp_vid_t total_threads = omp_get_num_threads();
    sgp_vid_t t_id = omp_get_thread_num();
    gc_count[t_id] = 0;

#pragma omp for
    for (sgp_vid_t i = 0; i < nc; i++) {
        edges_per_source_atomic[i] = 0;
    }

    //count edges per vertex
#pragma omp for
    for (sgp_vid_t i = 0; i < n; i++) {
        sgp_vid_t u = vcmap[i];
        sgp_eid_t end_offset = g.source_offsets[i + 1];
        if (coarsening_level != 1) {
            end_offset = g.source_offsets[i] + g.edges_per_source[i];
        }
        for (sgp_eid_t j = g.source_offsets[i]; j < end_offset; j++) {
            sgp_vid_t v = vcmap[g.destination_indices[j]];
            mapped_edges[j] = v;
            if (u != v) {
                edges_per_source_atomic[u]++;
            }
        }
    }

#pragma omp single
{
    time_ptrs[2] += (sgp_timer() - start_count);
    start_prefix = sgp_timer();
}

#pragma omp for
    for (sgp_vid_t i = 0; i < nc; i++) {
        source_bucket_offset[i + 1] = edges_per_source_atomic[i];
        edges_per_source_atomic[i] = 0; // will use as counter again
    }

    //prefix sums to compute bucket offsets
    parallel_prefix_sum(source_bucket_offset, nc, t_id, total_threads);

#pragma omp single
{
    time_ptrs[3] += (sgp_timer() - start_prefix);
    start_bucket = sgp_timer();
    dest_by_source = (sgp_vid_t*)malloc(source_bucket_offset[nc] * sizeof(sgp_vid_t));
    SGPAR_ASSERT(dest_by_source != NULL);
    wgt_by_source = (sgp_wgt_t*)malloc(source_bucket_offset[nc] * sizeof(sgp_wgt_t));
    SGPAR_ASSERT(wgt_by_source != NULL);
}

    //sort by source first
#pragma omp for
    for (sgp_vid_t i = 0; i < n; i++) {
        sgp_vid_t u = vcmap[i];
        sgp_eid_t end_offset = g.source_offsets[i + 1];
        if (coarsening_level != 1) {
            end_offset = g.source_offsets[i] + g.edges_per_source[i];
        }
        for (sgp_eid_t j = g.source_offsets[i]; j < end_offset; j++) {
            sgp_vid_t v = mapped_edges[j];
            if (u != v) {
                //edges_per_source[u]++ is atomic_fetch_add
                sgp_eid_t offset = source_bucket_offset[u] + edges_per_source_atomic[u]++;

                dest_by_source[offset] = v;
                if (coarsening_level != 1) {
                    wgt_by_source[offset] = g.eweights[j];
                }
                else {
                    wgt_by_source[offset] = 1;
                }
            }
        }
    }
#pragma omp single
{
    free(mapped_edges);
#ifdef __cplusplus
    delete[] edges_per_source_atomic;
#else
    free(edges_per_source_atomic);
#endif
    time_ptrs[4] += (sgp_timer() - start_bucket);
    start_dedupe = sgp_timer();
}

    //sort by dest and deduplicate
#pragma omp for schedule(dynamic, 16)
    for (sgp_vid_t u = 0; u < nc; u++) {

        sgp_vid_t size = source_bucket_offset[u + 1] - source_bucket_offset[u];

#ifdef __cplusplus
        //heapsort
        if ((coarsening_alg & 6) == 2 && size < 10) {
            heap_deduplicate(source_bucket_offset + u, dest_by_source, wgt_by_source, edges_per_source + u, gc_count + t_id);
        }
        //hashmap sort
        else {
            hashmap_deduplicate(source_bucket_offset + u, dest_by_source, wgt_by_source, edges_per_source + u, gc_count + t_id);
        }
#else
        heap_deduplicate(source_bucket_offset + u, dest_by_source, wgt_by_source, edges_per_source + u, gc_count + t_id);
#endif
    }

#pragma omp single
{
    time_ptrs[5] += (sgp_timer() - start_dedupe);

    sgp_eid_t gc_nedges = 0;
    for (sgp_vid_t i = 0; i < total_threads; i++) {
        gc_nedges += gc_count[i];
    }

    gc_nedges = gc_nedges / 2;

    gc->nedges = gc_nedges;
    gc->destination_indices = dest_by_source;
    gc->source_offsets = source_bucket_offset;
    gc->eweights = wgt_by_source;
    gc->edges_per_source = edges_per_source;

    gc->weighted_degree = (sgp_wgt_t*)malloc(nc * sizeof(sgp_wgt_t));
    assert(gc->weighted_degree != NULL);
}

#pragma omp for
    for (sgp_vid_t i = 0; i < nc; i++) {
        sgp_wgt_t degree_wt_i = 0;
        sgp_eid_t end_offset = gc->source_offsets[i] + gc->edges_per_source[i];
        for (sgp_eid_t j = gc->source_offsets[i]; j < end_offset; j++) {
            degree_wt_i += gc->eweights[j];
        }
        gc->weighted_degree[i] = degree_wt_i;
    }
}

    return EXIT_SUCCESS;
}
#else

SGPAR_API int sgp_build_coarse_graph_msd(sgp_graph_t* gc,
    sgp_vid_t* vcmap,
    const sgp_graph_t g,
    const int coarsening_level,
    double* time_ptrs,
    const int coarsening_alg) {
    sgp_vid_t n = g.nvertices;
    sgp_vid_t nc = gc->nvertices;

    sgp_eid_t ec = 0;

    //radix sort source vertices, then sort edges

    double start_count = sgp_timer();

    //count edges per vertex
    sgp_vid_t* edges_per_source = (sgp_vid_t*)calloc(nc, sizeof(sgp_vid_t));
    SGPAR_ASSERT(edges_per_source != NULL);
    sgp_vid_t* mapped_edges = (sgp_vid_t*)malloc(g.source_offsets[n] * sizeof(sgp_vid_t));
    SGPAR_ASSERT(mapped_edges != NULL);

    for (sgp_vid_t i = 0; i < n; i++) {
        sgp_vid_t u = vcmap[i];
        sgp_eid_t end_offset = g.source_offsets[i + 1];
        if (coarsening_level != 1) {
            end_offset = g.source_offsets[i] + g.edges_per_source[i];
        }
        for (sgp_eid_t j = g.source_offsets[i]; j < end_offset; j++) {
            sgp_vid_t v = vcmap[g.destination_indices[j]];
            mapped_edges[j] = v;
            if (u != v) {
                edges_per_source[u]++;
                ec++;
            }
        }
    }

    time_ptrs[2] += (sgp_timer() - start_count);
    double start_prefix = sgp_timer();

    //prefix sums to compute bucket offsets
    sgp_eid_t* source_bucket_offset = (sgp_eid_t*)calloc(nc + 1, sizeof(sgp_eid_t));
    SGPAR_ASSERT(source_bucket_offset != NULL);
    for (sgp_vid_t i = 0; i < nc; i++) {
        source_bucket_offset[i + 1] = source_bucket_offset[i] + edges_per_source[i];
        edges_per_source[i] = 0;//reset to be used as a counter
    }

    time_ptrs[3] += (sgp_timer() - start_prefix);
    double start_bucket = sgp_timer();

    //sort by source first
    sgp_vid_t* dest_by_source = (sgp_vid_t*)malloc(ec * sizeof(sgp_vid_t));
    SGPAR_ASSERT(dest_by_source != NULL);
    sgp_wgt_t* wgt_by_source = (sgp_wgt_t*)malloc(ec * sizeof(sgp_wgt_t));
    SGPAR_ASSERT(wgt_by_source != NULL);
    for (sgp_vid_t i = 0; i < n; i++) {
        sgp_vid_t u = vcmap[i];
        sgp_eid_t end_offset = g.source_offsets[i + 1];
        if (coarsening_level != 1) {
            end_offset = g.source_offsets[i] + g.edges_per_source[i];
        }
        for (sgp_eid_t j = g.source_offsets[i]; j < end_offset; j++) {
            sgp_vid_t v = mapped_edges[j];
            if (u != v) {
                sgp_eid_t offset = source_bucket_offset[u] + edges_per_source[u];
                edges_per_source[u]++;

                dest_by_source[offset] = v;
                if (coarsening_level != 1) {
                    wgt_by_source[offset] = g.eweights[j];
                }
                else {
                    wgt_by_source[offset] = 1;
                }
            }
        }
    }
    free(mapped_edges);

    time_ptrs[4] += (sgp_timer() - start_bucket);
    double start_dedupe = sgp_timer();

    //sort by dest and deduplicate
    sgp_eid_t gc_nedges = 0;
    for (sgp_vid_t u = 0; u < nc; u++) {

        sgp_vid_t size = source_bucket_offset[u + 1] - source_bucket_offset[u];
        //heapsort
        if ((coarsening_alg & 6) == 2 && size < 10) {
            heap_deduplicate(source_bucket_offset + u, dest_by_source, wgt_by_source, edges_per_source + u, &gc_nedges);
        }
        //hashmap sort
        else {
            hashmap_deduplicate(source_bucket_offset + u, dest_by_source, wgt_by_source, edges_per_source + u, &gc_nedges);
        }

    }

    time_ptrs[5] += (sgp_timer() - start_dedupe);

    gc_nedges /= 2;

    gc->nedges = gc_nedges;
    gc->destination_indices = dest_by_source;
    gc->source_offsets = source_bucket_offset;
    gc->eweights = wgt_by_source;
    gc->edges_per_source = edges_per_source;

    gc->weighted_degree = (sgp_wgt_t*)malloc(nc * sizeof(sgp_wgt_t));
    assert(gc->weighted_degree != NULL);

    for (sgp_vid_t i = 0; i < nc; i++) {
        sgp_wgt_t degree_wt_i = 0;
        sgp_eid_t end_offset = gc->source_offsets[i] + gc->edges_per_source[i];
        for (sgp_eid_t j = gc->source_offsets[i]; j < end_offset; j++) {
            degree_wt_i += gc->eweights[j];
        }
        gc->weighted_degree[i] = degree_wt_i;
    }

    return EXIT_SUCCESS;
}
#endif 

SGPAR_API int sgp_coarsen_one_level(sgp_graph_t* gc, sgp_vid_t* vcmap,
    const sgp_graph_t g,
    const int coarsening_level,
    const int coarsening_alg,
    sgp_pcg32_random_t* rng,
    double* time_ptrs) {

    double start_map = sgp_timer();
    if ((coarsening_alg & 1) == 0) {
        sgp_vid_t nvertices_coarse;
        sgp_coarsen_HEC(vcmap, &nvertices_coarse, g, coarsening_level, rng);
        gc->nvertices = nvertices_coarse;
    }
    else if ((coarsening_alg & 1) == 1) {
        sgp_vid_t nvertices_coarse;
        sgp_coarsen_heavy_edge_matching(vcmap, &nvertices_coarse, g, coarsening_level, rng);
        gc->nvertices = nvertices_coarse;
    }
    time_ptrs[0] += (sgp_timer() - start_map);

    double start_build = sgp_timer();

    sgp_build_coarse_graph_msd(gc, vcmap, g, coarsening_level, time_ptrs, coarsening_alg);
    time_ptrs[1] += (sgp_timer() - start_build);

    return EXIT_SUCCESS;
}


SGPAR_API int sgp_vec_normalize(sgp_real_t *u, int64_t n) {

    assert(u != NULL);
    sgp_real_t squared_sum = 0;

    for (int64_t i=0; i<n; i++) {
        squared_sum += u[i]*u[i];
    }
    sgp_real_t sum_inv = 1/sqrt(squared_sum);

    for (int64_t i=0; i<n; i++) {
        u[i] = u[i]*sum_inv;
    }
    return EXIT_SUCCESS;
}

#ifdef _OPENMP
SGPAR_API int sgp_vec_normalize_omp(sgp_real_t *u, int64_t n) {

    assert(u != NULL);
    static sgp_real_t squared_sum = 0;

#pragma omp single
    squared_sum = 0;

#pragma omp for reduction(+:squared_sum)
    for (int64_t i=0; i<n; i++) {
        squared_sum += u[i]*u[i];
    }
    sgp_real_t sum_inv = 1/sqrt(squared_sum);

    #pragma omp single 
    {
        //printf("squared_sum %3.3f\n", squared_sum);
    }

#pragma omp for
    for (int64_t i=0; i<n; i++) {
        u[i] = u[i]*sum_inv;
    }
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_dotproduct_omp(sgp_real_t *dot_prod_ptr, 
                                 sgp_real_t *u1, sgp_real_t *u2, int64_t n) {

    static sgp_real_t dot_prod = 0;

#pragma omp single
    dot_prod = 0;
    
#pragma omp for reduction(+:dot_prod)
    for (int64_t i=0; i<n; i++) {
        dot_prod += u1[i]*u2[i];
    }
    *dot_prod_ptr = dot_prod;

    return EXIT_SUCCESS;
}


SGPAR_API int sgp_vec_orthogonalize_omp(sgp_real_t *u1, sgp_real_t *u2, int64_t n) {

    sgp_real_t mult1;
    sgp_vec_dotproduct_omp(&mult1, u1, u2, n);

#pragma omp for
    for (int64_t i=0; i<n; i++) {
        u1[i] -= mult1*u2[i];
    }
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_D_orthogonalize_omp(sgp_real_t *u1, sgp_real_t *u2, 
                        sgp_wgt_t *D,  int64_t n) {

    //u1[i] = u1[i] - (dot(u1, D*u2)/dot(u2, D*u2)) * u2[i]

    sgp_real_t mult1;
    sgp_vec_dotproduct_omp(&mult1, u1, u2, n);

    static sgp_real_t mult_numer = 0;
    static sgp_real_t mult_denom = 0;

#pragma omp single
{
    mult_numer = 0;
    mult_denom = 0;
}

#pragma omp for reduction(+:mult_numer, mult_denom)
    for (int64_t i=0; i<n; i++) {
        mult_numer += u1[i]*D[i]*u2[i];
        mult_denom += u2[i]*D[i]*u2[i];
    }

#pragma omp for
    for (int64_t i=0; i<n; i++) {
        u1[i] -= mult_numer*u2[i]/mult_denom;
    }

    return EXIT_SUCCESS;
}

SGPAR_API void sgp_power_iter_eigenvalue_log(sgp_real_t *u, sgp_graph_t g){
    sgp_real_t eigenval = 0;
    sgp_real_t eigenval_max = 0;
    sgp_real_t eigenval_min = 2;
    for (sgp_vid_t i=0; i<(g.nvertices); i++) {
        sgp_vid_t weighted_degree = g.source_offsets[i+1]-g.source_offsets[i];
        sgp_real_t u_i = weighted_degree*u[i];
        sgp_real_t matvec_i = 0;
        for (sgp_eid_t j=g.source_offsets[i]; 
                       j<g.source_offsets[i+1]; j++) {
            matvec_i += u[g.destination_indices[j]];
        }
        u_i -= matvec_i;
        sgp_real_t eigenval_est = u_i/u[i];
        if (eigenval_est < eigenval_min) {
            eigenval_min = eigenval_est;
        }
        if (eigenval_est > eigenval_max) {
            eigenval_max = eigenval_est;
        }
        eigenval += (u_i*u_i)*1e9;
    }

    printf("eigenvalue = %1.9lf (%1.9lf %1.9lf), "
                    "edge cut lb %5.0lf "
                    "gap ratio %.0lf\n", 
                    eigenval*1e-9, 
                    eigenval_min, eigenval_max,
                    eigenval*1e-9*(g.nvertices)/4,
                    ceil(1.0/(1.0-eigenval*1e-9)));
}

SGPAR_API int sgp_power_iter(sgp_real_t *u, sgp_graph_t g, const int normLap, const int final
#ifdef EXPERIMENT
    , ExperimentLoggerUtil& experiment
#endif
) {

    sgp_vid_t n = g.nvertices;

    sgp_real_t *vec1 = (sgp_real_t *) malloc(n*sizeof(sgp_real_t));
    SGPAR_ASSERT(vec1 != NULL);

    sgp_wgt_t *weighted_degree = NULL;
    
    if(normLap && final){
        weighted_degree = (sgp_wgt_t *) malloc(n*sizeof(sgp_wgt_t));
        assert(weighted_degree != NULL);
        for (sgp_vid_t i=0; i<n; i++) {
            weighted_degree[i] = g.source_offsets[i+1] - g.source_offsets[i];
        }
    }

    sgp_wgt_t gb = 2.0;
    if(!normLap){
        if(!final){
            gb = 2*g.weighted_degree[0];
            for (sgp_vid_t i=1; i<n; i++) {
                if(gb < 2*g.weighted_degree[i]) {
                    gb = 2*g.weighted_degree[i];
                }
            }
        } else {
            gb = 2*(g.source_offsets[1]-g.source_offsets[0]);
            for (sgp_vid_t i=1; i<n; i++) {
                if (gb < 2*(g.source_offsets[i+1]-g.source_offsets[i])) {
                    gb = 2*(g.source_offsets[i+1]-g.source_offsets[i]);
                }
            }
        }
    }


    uint64_t g_niter = 0;
    uint64_t iter_max = (uint64_t)SGPAR_POWERITER_ITER / (uint64_t)n;

    sgp_real_t* v = (sgp_real_t*)malloc(n * sizeof(sgp_real_t));
    SGPAR_ASSERT(v != NULL);
#pragma omp parallel shared(u)
{

    uint64_t niter = 0;
#pragma omp for
    for (sgp_vid_t i=0; i<n; i++) {
        vec1[i] = 1.0;
    }

#if 0
    sgp_real_t mult = 0;
    sgp_vec_dotproduct_omp(&mult, vec1, u, n);

#pragma omp for
    for (sgp_vid_t i=0; i<n; i++) {
        u[i] -= mult*u[i]/n;
    }
    sgp_vec_normalize_omp(u, n);
#endif

    sgp_vec_normalize_omp(vec1, n); 
    if(!normLap){
        sgp_vec_orthogonalize_omp(u, vec1, n);
    } else {
        sgp_vec_D_orthogonalize_omp(u, vec1, g.weighted_degree, n);
    }
    sgp_vec_normalize_omp(u, n);

#pragma omp for
    for (sgp_vid_t i=0; i<n; i++) {
        v[i] = u[i];
    }

    sgp_real_t tol = SGPAR_POWERITER_TOL;
    sgp_real_t dotprod = 0, lastDotprod = 1;
    while (fabs(dotprod - lastDotprod) > tol && (niter < iter_max)) {

        // u = v
#pragma omp for
        for (sgp_vid_t i=0; i<n; i++) {
            u[i] = v[i];
        }

        // sparse matrix multiplication
#pragma omp for
        for (sgp_vid_t i=0; i<n; i++) {
            // sgp_real_t v_i = g.weighted_degree[i]*u[i];
            sgp_real_t weighted_degree_inv = 1.0;
            sgp_real_t v_i;
            if (normLap) {
                if (final) {
                    weighted_degree_inv = 1.0 / weighted_degree[i];
                    v_i = 0.5 * u[i];
                }
                else {
                    weighted_degree_inv = 1.0 / g.weighted_degree[i];
                    v_i = 0.5 * u[i];
                }
            }
            else {
                if (final) {
                    sgp_vid_t weighted_degree = g.source_offsets[i + 1] - g.source_offsets[i];
                    v_i = (gb - weighted_degree) * u[i];
                }
                else {
                    v_i = (gb - g.weighted_degree[i]) * u[i];
                }
            }
            sgp_real_t matvec_i = 0;

            sgp_eid_t end_offset = g.source_offsets[i + 1];
            if (!final) {
                end_offset = g.source_offsets[i] + g.edges_per_source[i];
            }

            for (sgp_eid_t j = g.source_offsets[i]; j < end_offset; j++) {
                if (final) {
                    matvec_i += u[g.destination_indices[j]];
                }
                else {
                    matvec_i += u[g.destination_indices[j]] * g.eweights[j];
                }
            }
            if (normLap) {
                v_i += 0.5 * matvec_i * weighted_degree_inv;
            }
            else {
                v_i += matvec_i;
            }
            v[i] = v_i;
        }

        if(!normLap){
            sgp_vec_orthogonalize_omp(v, vec1, n);
        }
        sgp_vec_normalize_omp(v, n);
		lastDotprod = dotprod;
        sgp_vec_dotproduct_omp(&dotprod, u, v, n);
        niter++;
    }

#pragma omp single
    {
        g_niter = niter;
    }
}
    free(v);

    int max_iter_reached = 0;
    if (g_niter >= iter_max) {
        printf("exceeded max iter count, ");
        max_iter_reached = 1;
    }
    printf("number of iterations: %lu\n", g_niter);
#ifdef EXPERIMENT
    experiment.addCoarseLevel(g_niter, max_iter_reached, n);
#endif
    if(!normLap && final){
        sgp_power_iter_eigenvalue_log(u, g);
    }

    free(vec1);
    if(normLap && final){
        free(weighted_degree);
    }
    return EXIT_SUCCESS;
}
#else
SGPAR_API int sgp_vec_normalize_serial(sgp_real_t* u, int64_t n) {

    assert(u != NULL);
    static sgp_real_t squared_sum = 0;

    squared_sum = 0;

    for (int64_t i = 0; i < n; i++) {
        squared_sum += u[i] * u[i];
    }
    sgp_real_t sum_inv = 1 / sqrt(squared_sum);

    for (int64_t i = 0; i < n; i++) {
        u[i] = u[i] * sum_inv;
    }
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_dotproduct_serial(sgp_real_t* dot_prod_ptr,
    sgp_real_t* u1, sgp_real_t* u2, int64_t n) {

    static sgp_real_t dot_prod = 0;

    dot_prod = 0;

    for (int64_t i = 0; i < n; i++) {
        dot_prod += u1[i] * u2[i];
    }
    *dot_prod_ptr = dot_prod;

    return EXIT_SUCCESS;
}


SGPAR_API int sgp_vec_orthogonalize_serial(sgp_real_t* u1, sgp_real_t* u2, int64_t n) {

    sgp_real_t mult1;
    sgp_vec_dotproduct_serial(&mult1, u1, u2, n);

    for (int64_t i = 0; i < n; i++) {
        u1[i] -= mult1 * u2[i];
    }
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_D_orthogonalize_serial(sgp_real_t* u1, sgp_real_t* u2,
    sgp_wgt_t* D, int64_t n) {

    //u1[i] = u1[i] - (dot(u1, D*u2)/dot(u2, D*u2)) * u2[i]

    sgp_real_t mult1;
    sgp_vec_dotproduct_serial(&mult1, u1, u2, n);

    static sgp_real_t mult_numer = 0;
    static sgp_real_t mult_denom = 0;

    mult_numer = 0;
    mult_denom = 0;

    for (int64_t i = 0; i < n; i++) {
        mult_numer += u1[i] * D[i] * u2[i];
        mult_denom += u2[i] * D[i] * u2[i];
    }

    for (int64_t i = 0; i < n; i++) {
        u1[i] -= mult_numer * u2[i] / mult_denom;
    }

    return EXIT_SUCCESS;
}

SGPAR_API void sgp_power_iter_eigenvalue_log(sgp_real_t* u, sgp_graph_t g) {
    sgp_real_t eigenval = 0;
    sgp_real_t eigenval_max = 0;
    sgp_real_t eigenval_min = 2;
    for (sgp_vid_t i = 0; i < (g.nvertices); i++) {
        sgp_vid_t weighted_degree = g.source_offsets[i + 1] - g.source_offsets[i];
        sgp_real_t u_i = weighted_degree * u[i];
        sgp_real_t matvec_i = 0;
        for (sgp_eid_t j = g.source_offsets[i];
            j < g.source_offsets[i + 1]; j++) {
            matvec_i += u[g.destination_indices[j]];
        }
        u_i -= matvec_i;
        sgp_real_t eigenval_est = u_i / u[i];
        if (eigenval_est < eigenval_min) {
            eigenval_min = eigenval_est;
        }
        if (eigenval_est > eigenval_max) {
            eigenval_max = eigenval_est;
        }
        eigenval += (u_i * u_i) * 1e9;
    }

    printf("eigenvalue = %1.9lf (%1.9lf %1.9lf), "
        "edge cut lb %5.0lf "
        "gap ratio %.0lf\n",
        eigenval * 1e-9,
        eigenval_min, eigenval_max,
        eigenval * 1e-9 * (g.nvertices) / 4,
        ceil(1.0 / (1.0 - eigenval * 1e-9)));
}

SGPAR_API int sgp_power_iter(sgp_real_t* u, sgp_graph_t g, int normLap, int final
#ifdef EXPERIMENT
    , ExperimentLoggerUtil& experiment
#endif
                            ) {

    sgp_vid_t n = g.nvertices;

    sgp_real_t* vec1 = (sgp_real_t*)malloc(n * sizeof(sgp_real_t));
    SGPAR_ASSERT(vec1 != NULL);

    sgp_wgt_t* weighted_degree = NULL;

    if (normLap && final) {
        weighted_degree = (sgp_wgt_t*)malloc(n * sizeof(sgp_wgt_t));
        assert(weighted_degree != NULL);
        for (sgp_vid_t i = 0; i < n; i++) {
            weighted_degree[i] = g.source_offsets[i + 1] - g.source_offsets[i];
        }
    }

    sgp_wgt_t gb = 2.0;
    if (!normLap) {
        if (!final) {
            gb = 2 * g.weighted_degree[0];
            for (sgp_vid_t i = 1; i < n; i++) {
                if (gb < 2 * g.weighted_degree[i]) {
                    gb = 2 * g.weighted_degree[i];
                }
            }
        }
        else {
            gb = 2 * (g.source_offsets[1] - g.source_offsets[0]);
            for (sgp_vid_t i = 1; i < n; i++) {
                if (gb < 2 * (g.source_offsets[i + 1] - g.source_offsets[i])) {
                    gb = 2 * (g.source_offsets[i + 1] - g.source_offsets[i]);
                }
            }
        }
    }

    uint64_t niter = 0;
    uint64_t iter_max = (uint64_t)SGPAR_POWERITER_ITER / (uint64_t)n;

        for (sgp_vid_t i = 0; i < n; i++) {
            vec1[i] = 1.0;
        }

#if 0
        sgp_real_t mult = 0;
        sgp_vec_dotproduct_serial(&mult, vec1, u, n);

#pragma omp for
        for (sgp_vid_t i = 0; i < n; i++) {
            u[i] -= mult * u[i] / n;
        }
        sgp_vec_normalize_serial(u, n);
#endif

        sgp_vec_normalize_serial(vec1, n);
        if (!normLap) {
            sgp_vec_orthogonalize_serial(u, vec1, n);
        }
        else {
            sgp_vec_D_orthogonalize_serial(u, vec1, g.weighted_degree, n);
        }
        sgp_vec_normalize_serial(u, n);

        sgp_real_t* v = (sgp_real_t*)malloc(n * sizeof(sgp_real_t));
        SGPAR_ASSERT(v != NULL);

        for (sgp_vid_t i = 0; i < n; i++) {
            v[i] = u[i];
        }

        sgp_real_t tol = SGPAR_POWERITER_TOL;
        sgp_real_t dotprod = 0, lastDotprod = 1;
        while (fabs(dotprod - lastDotprod) > tol && (niter < iter_max)) {

            // u = v
            for (sgp_vid_t i = 0; i < n; i++) {
                u[i] = v[i];
            }

            // v = Lu
            for (sgp_vid_t i = 0; i < n; i++) {
                // sgp_real_t v_i = g.weighted_degree[i]*u[i];
                sgp_real_t weighted_degree_inv = 1.0;
                sgp_real_t v_i;
                if (!normLap) {
                    if (!final) {
                        v_i = (gb - g.weighted_degree[i]) * u[i];
                    }
                    else {
                        sgp_vid_t weighted_degree = g.source_offsets[i + 1] - g.source_offsets[i];
                        v_i = (gb - weighted_degree) * u[i];
                    }
                }
                else {
                    weighted_degree_inv = 1.0 / g.weighted_degree[i];
                    v_i = 0.5 * u[i];
                }
                sgp_real_t matvec_i = 0;
                sgp_eid_t end_offset = g.source_offsets[i + 1];
                if (!final) {
                    end_offset = g.source_offsets[i] + g.edges_per_source[i];
                }
                for (sgp_eid_t j = g.source_offsets[i]; j < end_offset; j++) {
                    if (!final) {
                        matvec_i += u[g.destination_indices[j]] * g.eweights[j];
                    }
                    else {
                        matvec_i += u[g.destination_indices[j]];
                    }
                }
                // v_i -= matvec_i;
                if (!normLap) {
                    v_i += matvec_i;
                }
                else {
                    v_i += 0.5 * matvec_i * weighted_degree_inv;
                }
                v[i] = v_i;
            }

            if (!normLap) {
                sgp_vec_orthogonalize_serial(v, vec1, n);
            }
            sgp_vec_normalize_serial(v, n);
            lastDotprod = dotprod;
            sgp_vec_dotproduct_serial(&dotprod, u, v, n);
            niter++;
        }

        if (niter == iter_max) {
            printf("exceeded max iter count, ");
        }
        printf("number of iterations: %d\n", niter);
        free(v);

    int max_iter_reached = 0;
    if (niter >= iter_max) {
        printf("exceeded max iter count, ");
        max_iter_reached = 1;
    }
    printf("number of iterations: %lu\n", niter);
#ifdef EXPERIMENT
        experiment.addCoarseLevel(niter, max_iter_reached, n);
#endif
    if (!normLap && final) {
        sgp_power_iter_eigenvalue_log(u, g);
    }

    free(vec1);
    if (normLap && final) {
        free(weighted_degree);
    }
    return EXIT_SUCCESS;
}
#endif

SGPAR_API int write_sorted_eigenvec(sgp_vv_pair_t* vu_pair, sgp_vid_t n) {
    FILE* infp = fopen("eigenvec_debug.txt", "w");
    if (infp == NULL) {
        printf("Error: Could not open config file %s. Exiting ...\n", "eigenvec_debug.txt");
        return EXIT_FAILURE;
    }
    for (sgp_vid_t i = 0; i < n; i++) {
#ifdef SGPAR_HUGEGRAPHS
        fprintf(infp, "%llu %.20f\n", vu_pair[i].u, vu_pair[i].ev);
#else
        fprintf(infp, "%u %.20f\n", vu_pair[i].u, vu_pair[i].ev);
#endif
    }
    CHECK_RETSTAT(fclose(infp));

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_compute_partition(sgp_vid_t *part, sgp_vid_t num_partitions, 
                                    sgp_eid_t *edgecut, int perc_imbalance_allowed, 
                                    int local_search_alg,
                                    sgp_real_t *evec,
                                    sgp_graph_t g) {

    sgp_vid_t n = g.nvertices;
    sgp_vv_pair_t *vu_pair;
    vu_pair = (sgp_vv_pair_t *) malloc(n * sizeof(sgp_vv_pair_t));
    assert(vu_pair != NULL);

    //sort based on value of eigenvector corresponding to vertex
    for (sgp_vid_t i = 0; i<n; i++) {
        vu_pair[i].ev = evec[i];
        vu_pair[i].u  = i;
    }   
#ifdef __cplusplus
#if defined(USE_GNU_PARALLELMODE) && defined(_OPENMP)
    __gnu_parallel::sort(((sgp_vv_pair_t *) vu_pair), 
                         ((sgp_vv_pair_t *) vu_pair)+n,
                         vu_cmpfn_inc,
                        __gnu_parallel::quicksort_tag());
#else
    std::sort(((sgp_vv_pair_t *) vu_pair), 
              ((sgp_vv_pair_t *) vu_pair)+n,
              vu_cmpfn_inc);
#endif
#else
    qsort(vu_pair, n, sizeof(sgp_vv_pair_t), vu_cmpfn_inc);
#endif

#ifdef EIGENVEC_DEBUG
    write_sorted_eigenvec(vu_pair, n);
#endif

    // I'll just leave this here for now
    assert((num_partitions == 2) || (num_partitions == 4));

    if (num_partitions == 4) {
        num_partitions = 2;
    }

    long max_part_size = ceil(n/((double) num_partitions));

    // allow some imbalance
    sgp_vid_t imbr = floor(max_part_size*(1.0 + perc_imbalance_allowed/100.0));
    sgp_vid_t imbl = n - imbr;
    for (sgp_vid_t i=0; i<imbl; i++) {
        if(vu_pair[i].u >= n){
            return EXIT_FAILURE;
        }
        part[vu_pair[i].u] = 0;
    }
    for (sgp_vid_t i=imbl; i<n; i++) {
        if(vu_pair[i].u >= n){
            return EXIT_FAILURE;
        }
        part[vu_pair[i].u] = 1;
    }

    //compute edgecut based on current partition
    long edgecut_curr = 0;
    for (sgp_vid_t i=0; i<n; i++) {
        sgp_vid_t part_i = part[i];
        for (sgp_eid_t j=g.source_offsets[i]; j<g.source_offsets[i+1]; j++) {
            sgp_vid_t v = g.destination_indices[j];
            if (part[v] != part_i) {
                edgecut_curr++;
            }
        }
    }

    long edgecut_min = edgecut_curr;
    long curr_split = imbl;

    //checks each vertex between imbl and imbr (which are in partition 1)
    //adds them to partition 0 if doing so would reduce the edge cut
    for (sgp_vid_t i=imbl; i<imbr; i++) {
        /* add vert at position i to comm 0 */
        sgp_vid_t u = vu_pair[i].u;
        long ec_update = 0;
        for (sgp_eid_t j=g.source_offsets[u]; j<g.source_offsets[u+1]; j++) {
            sgp_vid_t v = g.destination_indices[j];
            if (part[v] == 1) {
                ec_update++;
            } else {
                ec_update--;
            }
        }
        edgecut_curr = edgecut_curr + 2*ec_update;

        if (edgecut_curr <= edgecut_min) {
            part[u] = 0;
            edgecut_min = edgecut_curr;
            curr_split = i+1;
            /*
            curr_split = n - i - 1;
            if ((n - i - 1) < (i+1)) {
                curr_split = i + 1;
            }
            */
        } 
    }

    free(vu_pair);

    printf("After bipartitioning, the partitions have %ld and %ld vertices, "
           "and the edge cut is %ld.\n", 
           curr_split, n-curr_split, edgecut_min/2);
    *edgecut = edgecut_min/2;

    if (local_search_alg == 0) 
        return 0;

    int64_t n_left = curr_split-n/3;
    if (n_left < 0) {
        n_left = 0;
    }
    int64_t n_right = curr_split+n/3;
    if (n_right > ((int64_t) n)) {
        n_right = n;
    }
    int num_swaps = 0;
    int64_t ec_change = 0;
    int max_swaps = n/3;
    while (num_swaps < max_swaps) {
        int64_t ec_dec_max = 0;
        sgp_vid_t move_right_vert = SGP_INFTY;
        for (long i=n_left; i<curr_split; i++) {
            sgp_vid_t part_i = part[i];
            
            int64_t ec_dec_i = 0;
            for (sgp_eid_t j=g.source_offsets[i]; 
                           j<g.source_offsets[i+1]; j++) {
                sgp_vid_t v = g.destination_indices[j];
                if (part[v] != part_i) {
                    ec_dec_i++;
                } else {
                    ec_dec_i--;
                }
            }
            if (ec_dec_i > ec_dec_max) {
                ec_dec_max = ec_dec_i;
                move_right_vert = i; 
            }
        }
        if (move_right_vert == SGP_INFTY) {
            // printf("Exiting before swap\n");
            break;
        }
        // printf("ec dec is %ld, moving vert %lu to the right\n",
        //                ec_dec_max, (uint64_t) move_right_vert);
        if (part[move_right_vert] == 0) {
            part[move_right_vert] = 1;
        } else {
            part[move_right_vert] = 0;
        }
        ec_change += ec_dec_max;
        int64_t ec_dec_max_prev = ec_dec_max;

        ec_dec_max = 0;
        sgp_vid_t move_left_vert = SGP_INFTY;
        for (long i=curr_split; i<n_right; i++) {
            sgp_vid_t part_i = part[i];
            int64_t ec_dec_i = 0;
            for (sgp_eid_t j=g.source_offsets[i]; 
                           j<g.source_offsets[i+1]; j++) {
                sgp_vid_t v = g.destination_indices[j];
                if (part[v] != part_i) {
                    ec_dec_i++;
                } else {
                    ec_dec_i--;
                }
            }
            if (ec_dec_i > ec_dec_max) {
                ec_dec_max = ec_dec_i;
                move_left_vert = i; 
            }
        }
        if (move_left_vert == SGP_INFTY) {
            /* Roll back prev swap and exit */
            if (part[move_right_vert] == 0) {
                part[move_right_vert] = 1;
            } else {
                part[move_right_vert] = 0;
            }
            ec_change -= ec_dec_max_prev;
            // printf("Incomplete swap, exiting\n");
            break;
        }
        // printf("ec dec is %ld, moving vert %lu to the left\n",
        //                ec_dec_max, (uint64_t) move_left_vert);
        if (part[move_left_vert] == 0) {
            part[move_left_vert] = 1;
        } else {
            part[move_left_vert] = 0;
        }
        ec_change += ec_dec_max;

        num_swaps++;
    }
    printf("Total change: %ld, swaps %d, new edgecut %ld\n", 
                    ec_change, num_swaps, edgecut_min/2-ec_change);

    edgecut_curr = 0;
    for (sgp_vid_t i=0; i<n; i++) {
        sgp_vid_t part_i = part[i];
        for (sgp_eid_t j=g.source_offsets[i]; j<g.source_offsets[i+1]; j++) {
            sgp_vid_t v = g.destination_indices[j];
            if (part[v] != part_i) {
                edgecut_curr++;
            }
        }
    }
    // fprintf(stderr, "computed %ld, est %ld\n", edgecut_curr/2,
    //                    edgecut_min/2-ec_change);
    assert(edgecut_curr/2 == (edgecut_min/2-ec_change));
    *edgecut = edgecut_curr/2;

#if 0
    sgp_real_t bin_width = 0.005;
    int64_t num_bins = ((int64_t) (2.0/bin_width) + 1);
    int64_t *bin_counts = (int64_t *) malloc(num_bins * sizeof(int64_t));
    for (int64_t i=0; i<num_bins; i++) {
        bin_counts[i] = 0; 
    }
    for (int64_t i=0; i<((int64_t)n); i++) {
        int64_t bin_num = ((int64_t) floor((1+evec[i])/bin_width));
        bin_counts[bin_num]++;
    }

    int64_t cumulative_bin_perc = 0;
    for (int64_t i=0; i<num_bins; i++) {
        int64_t bin_perc = (100*bin_counts[i])/n;
        if (bin_perc > 0) {
            cumulative_bin_perc += bin_perc;
            printf("bin %ld, perc %ld\n", i, bin_perc);
        }
    }
    printf("cumulative bin percentage: %ld\n", cumulative_bin_perc);

    free(bin_counts);
#endif

    return EXIT_SUCCESS;
}    

SGPAR_API int sgp_improve_partition(sgp_vid_t *part, sgp_vid_t num_partitions, 
                                    sgp_eid_t *edgecut, int perc_imbalance_allowed, 
                                    sgp_real_t *evec,
                                    sgp_graph_t g) {

    return EXIT_SUCCESS;
}    


/**********************************************************
 * API
 **********************************************************
 */

SGPAR_API int sgp_load_graph(sgp_graph_t *g, char *csr_filename);
SGPAR_API int sgp_free_graph(sgp_graph_t *g);
SGPAR_API int sgp_load_partition(sgp_vid_t *part, sgp_vid_t size, char *part_filename);
SGPAR_API int sgp_use_partition(sgp_vid_t* part, const sgp_graph_t g, sgp_graph_t* g1, sgp_graph_t* g2);
SGPAR_API int sgp_load_config(const char* config_f, config_t * config);
SGPAR_API int compute_partition_edit_distance(const sgp_vid_t* part1, const sgp_vid_t* part2, sgp_vid_t size, sgp_vid_t *diff);
SGPAR_API int sgp_partition_graph(sgp_vid_t *part,
                                  sgp_eid_t *edge_cut,
                                  config_t *config,
                                  const int perc_imbalance_allowed,
                                  const sgp_graph_t g,
#ifdef EXPERIMENT
    ExperimentLoggerUtil& experiment,
#endif
                                  sgp_pcg32_random_t* rng);


#ifdef __cplusplus
}
#endif
#endif // SGPAR_H_

/**********************************************************
 * Implementation 
 **********************************************************
 */
#ifdef SGPAR_IMPLEMENTATION

#ifdef __cplusplus
namespace sgpar {
#endif

SGPAR_API int sgp_load_graph(sgp_graph_t *g, char *csr_filename) {

    FILE *infp = fopen(csr_filename, "rb");
    if (infp == NULL) {
        printf("Error: Could not open input file. Exiting ...\n");
        return EXIT_FAILURE;
    }
    long n, m;
    long unused_vals[4];
    SGPAR_ASSERT(fread(&n, sizeof(long), 1, infp) != 0);
    SGPAR_ASSERT(fread(&m, sizeof(long), 1, infp) != 0);
    SGPAR_ASSERT(fread(unused_vals, sizeof(long), 4, infp) != 0);
    g->nvertices = n;
    g->nedges = m/2;
    g->source_offsets = (sgp_eid_t *) malloc((g->nvertices+1)*sizeof(sgp_eid_t));
    SGPAR_ASSERT(g->source_offsets != NULL);
    g->destination_indices = (sgp_vid_t *) malloc(2*g->nedges*sizeof(sgp_vid_t));
    SGPAR_ASSERT(g->destination_indices != NULL);
    size_t nitems_read = fread(g->source_offsets, sizeof(sgp_eid_t), g->nvertices+1, infp);
    SGPAR_ASSERT(nitems_read == ((size_t) g->nvertices+1));
    nitems_read = fread(g->destination_indices, sizeof(sgp_vid_t), 2*g->nedges, infp);
    SGPAR_ASSERT(nitems_read == ((size_t) 2*g->nedges));
    CHECK_RETSTAT( fclose(infp) );
    g->eweights = NULL;
    g->weighted_degree = NULL;
    g->edges_per_source = NULL;
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_free_graph(sgp_graph_t *g) {

    if (g->source_offsets != NULL) {
        free(g->source_offsets);
        g->source_offsets = NULL;
    }
    
    if (g->destination_indices != NULL) {
        free(g->destination_indices);
        g->destination_indices = NULL;
    }

    if (g->eweights != NULL) {
        free(g->eweights);
        g->eweights = NULL;
    }

    if (g->weighted_degree != NULL) {
        free(g->weighted_degree);
        g->weighted_degree = NULL;
    }

    if (g->edges_per_source != NULL) {
        free(g->edges_per_source);
        g->edges_per_source = NULL;
    }

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_load_partition(sgp_vid_t *part, sgp_vid_t size, char *part_filename){
    FILE *infp = fopen(part_filename, "r");
    if (infp == NULL) {
        printf("Error: Could not open partition file %s. Exiting ...\n", part_filename);
        return EXIT_FAILURE;
    }

    for(sgp_vid_t i = 0; i < size; i++){
#if SGPAR_HUGEGRAPHS
        if(fscanf(infp, "%li", part + i) == 0){
#else
        if(fscanf(infp, "%i", part + i) == 0){
#endif
            return EXIT_FAILURE;
        }
    }
    CHECK_RETSTAT(fclose(infp));

    return EXIT_SUCCESS;
}

//partitions assumed to same vertex labellings
SGPAR_API int compute_partition_edit_distance(const sgp_vid_t* part1, const sgp_vid_t* part2, sgp_vid_t size, sgp_vid_t *diff){

    sgp_vid_t d = 0; //difference if partition labelling is same
    sgp_vid_t d2 = 0; //difference if partition labelling is swapped
    for(sgp_vid_t i = 0; i < size; i++){
        if(part1[i] != part2[i]){
            d++;
        } else {
            d2++;
        }
    }

    if(d < d2){
        *diff = d/2;
    } else {
        *diff = d2/2;
    }
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_partition_graph(sgp_vid_t *part,
                                  sgp_eid_t *edge_cut,
                                  config_t * config,
                                  const int perc_imbalance_allowed,
                                  const sgp_graph_t g,
#ifdef EXPERIMENT
                                  ExperimentLoggerUtil& experiment,
#endif
                                  sgp_pcg32_random_t* rng) {

    printf("sgpar settings: %d %lu %.16f\n", 
                    SGPAR_COARSENING_VTX_CUTOFF, 
                    (uint64_t) SGPAR_POWERITER_ITER,
                    SGPAR_POWERITER_TOL);

#ifdef _KOKKOS
    double start_time = sgp_timer();

    std::list<sgpar_kokkos::matrix_type> coarse_graphs, interp_mtxs;
    std::list<sgpar_kokkos::vtx_view_t> vtx_weights;
    CHECK_SGPAR(sgpar_kokkos::sgp_generate_coarse_graphs(&g, coarse_graphs, interp_mtxs, vtx_weights, rng, experiment));

    double fin_coarsening_time = sgp_timer();
    sgp_real_t* eigenvec = (sgp_real_t*)malloc(g.nvertices * sizeof(sgp_real_t));

    CHECK_SGPAR(sgpar_kokkos::sgp_eigensolve(eigenvec, coarse_graphs, interp_mtxs, vtx_weights, rng, config->refine_alg
        , experiment
        ));

    coarse_graphs.clear();
    interp_mtxs.clear();
    double fin_final_level_time = sgp_timer();
    //I don't feel like redoing the timing stuff rn
    double fin_refine_time = sgp_timer();

    //retry in case of spurious memory errors
    while(sgp_compute_partition(part, config->num_partitions, edge_cut,
        perc_imbalance_allowed,
        config->local_search_alg,
        eigenvec, g) != EXIT_SUCCESS);

    free(eigenvec);
    experiment.setTotalDurationSeconds(fin_final_level_time - start_time);
    experiment.setCoarsenDurationSeconds(fin_coarsening_time - start_time);
    experiment.setRefineDurationSeconds(fin_final_level_time - fin_coarsening_time);
    experiment.setFinestEdgeCut(*edge_cut);
    experiment.modifyCoarseLevelEC(0, *edge_cut);

    printf("Coarsening permutation time: %.8f\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Permute));
    printf("Coarsening map construction time: %.8f\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::MapConstruct));
    printf("Coarsening map total time: %.8f\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Map));
    printf("Coarsening heavy find total time: %.8f\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Heavy));
    printf("FM coarsen time: %.8f\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::FMRecoarsen));
    printf("Coarsen Dedupe time: %.8f\n", experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Dedupe));
    printf("Total: %3.3lf s, coarsening %3.3lf %3.0lf%% "
        "(sort %3.3lf %3.0lf%%), "
        "refine %3.3lf s (%3.3lf s, %3.0lf%% + %3.3lf, %3.0lf%%)\n",
        fin_final_level_time - start_time,
        fin_coarsening_time - start_time,
        (fin_coarsening_time - start_time) * 100 / (fin_final_level_time - start_time),
        experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Build),
        experiment.getMeasurement(ExperimentLoggerUtil::Measurement::Build) * 100 / (fin_final_level_time - start_time),
        fin_final_level_time - fin_coarsening_time,
        fin_final_level_time - fin_refine_time,
        100 * (fin_final_level_time - fin_refine_time) / (fin_final_level_time - start_time),
        fin_refine_time - fin_coarsening_time,
        100 * (fin_refine_time - fin_coarsening_time) /
    (fin_final_level_time - start_time));
#else

    int coarsening_level = 0;
    sgp_graph_t g_all[SGPAR_COARSENING_MAXLEVELS];
    sgp_vid_t *vcmap[SGPAR_COARSENING_MAXLEVELS];
    
    for (int i=0; i<SGPAR_COARSENING_MAXLEVELS; i++) {
        g_all[i].nvertices = 0;
        g_all[i].source_offsets = NULL;
        g_all[i].destination_indices = NULL;
        g_all[i].eweights = NULL;
        g_all[i].weighted_degree = NULL;
        g_all[i].edges_per_source = NULL;
    }
    g_all[0].nvertices = g.nvertices; g_all[0].nedges = g.nedges;
    g_all[0].source_offsets = g.source_offsets;
    g_all[0].destination_indices = g.destination_indices;

    double start_time = sgp_timer();
    double time_counters[6] = { 0, 0, 0, 0, 0, 0 };

    int coarsen_ratio_exceeded = 0;
    //generate all coarse graphs
    while ((coarsening_level < (SGPAR_COARSENING_MAXLEVELS-1)) && 
           (coarsen_ratio_exceeded == 0) && 
           (g_all[coarsening_level].nvertices > SGPAR_COARSENING_VTX_CUTOFF) &&
           ((config->coarsening_alg & 16) == 0)) {
        coarsening_level++;
        printf("Calculating coarse graph %d\n", coarsening_level);
        vcmap[coarsening_level-1] = (sgp_vid_t *) 
                                    malloc(g_all[coarsening_level-1].nvertices
                                                 * sizeof(sgp_vid_t));
        SGPAR_ASSERT(vcmap[coarsening_level-1] != NULL);
        CHECK_SGPAR( sgp_coarsen_one_level(&g_all[coarsening_level],
                                            vcmap[coarsening_level-1],
                                            g_all[coarsening_level-1], 
                                            coarsening_level, config->coarsening_alg, 
                                            rng, time_counters) );

        if (config->coarsening_alg & 1) {
            sgp_real_t coarsen_ratio = (sgp_real_t)g_all[coarsening_level].nvertices / (sgp_real_t)g_all[coarsening_level - 1].nvertices;
            if (coarsen_ratio > MAX_COARSEN_RATIO) {
                coarsen_ratio_exceeded = 1;
            }
        }
    }

    //don't use the coarsest level if it has too few vertices
    if (g_all[coarsening_level].nvertices < 30) {
        sgp_free_graph(g_all + coarsening_level);
        coarsening_level--;
    }

    printf("Coarsest level: %d\n", coarsening_level);

    int num_coarsening_levels = coarsening_level+1;

    double fin_coarsening_time = sgp_timer();

    sgp_vid_t gc_nvertices = g_all[num_coarsening_levels-1].nvertices;
    sgp_real_t *eigenvec[SGPAR_COARSENING_MAXLEVELS];
    eigenvec[num_coarsening_levels-1] = (sgp_real_t *) 
                                        malloc(gc_nvertices*sizeof(sgp_real_t));
    SGPAR_ASSERT(eigenvec[num_coarsening_levels-1] != NULL);
    //randomly initialize guess eigenvector for coarsest graph
    for (sgp_vid_t i=0; i<gc_nvertices; i++) {
        eigenvec[num_coarsening_levels-1][i] = 
                            ((double) sgp_pcg32_random_r(rng))/UINT32_MAX;
    }

    sgp_vec_normalize(eigenvec[num_coarsening_levels-1], gc_nvertices);
    if ((config->coarsening_alg & 16) == 0) { /* bit 4 (0-indexed) of coarsening_alg indicates no coarsening if set */
        printf("Coarsening level %d, ", num_coarsening_levels-1);        
        if (config->refine_alg == 0) {
            CHECK_SGPAR( sgp_power_iter(eigenvec[num_coarsening_levels-1], 
                g_all[num_coarsening_levels-1], 0, 0
#ifdef EXPERIMENT
                , experiment 
#endif
            ) );
        } else {
            CHECK_SGPAR( sgp_power_iter(eigenvec[num_coarsening_levels-1], 
                   g_all[num_coarsening_levels-1], 1, 0
#ifdef EXPERIMENT
                , experiment
#endif
            ) );
        }
    }

    for (int l = num_coarsening_levels - 2; l >= 0; l--) {
        sgp_vid_t gcl_n = g_all[l].nvertices;
        eigenvec[l] = (sgp_real_t*)malloc(gcl_n * sizeof(sgp_real_t));
        SGPAR_ASSERT(eigenvec[l] != NULL);

        //prolong eigenvector from coarser level to finer level
        for (sgp_vid_t i = 0; i < gcl_n; i++) {
            eigenvec[l][i] = eigenvec[l + 1][vcmap[l][i]];
        }
        
#ifndef COARSE_EIGEN_EC
        free(eigenvec[l + 1]);
#ifndef _KOKKOS
        free(vcmap[l]);
#endif
#endif

        sgp_vec_normalize(eigenvec[l], gcl_n);

        //don't do refinement for finest level here
        if (l > 0) {
            printf("Coarsening level %d, ", l);
            if (config->refine_alg == 0) {
                sgp_power_iter(eigenvec[l], g_all[l], 0, 0
#ifdef EXPERIMENT
                    , experiment
#endif
                );
            }
            else {
                sgp_power_iter(eigenvec[l], g_all[l], 1, 0
#ifdef EXPERIMENT
                    , experiment
#endif
                );
            }
        }
    }

    double fin_refine_time = sgp_timer();

    printf("Coarsening level %d, ", 0);
    if (config->refine_alg == 0) {
        CHECK_SGPAR( sgp_power_iter(eigenvec[0], g_all[0], 0, 1
#ifdef EXPERIMENT
            , experiment
#endif
        ) );
    } else {
        CHECK_SGPAR( sgp_power_iter(eigenvec[0], g_all[0], 1, 1
#ifdef EXPERIMENT
            , experiment
#endif
        ));
    }
    double fin_final_level_time = sgp_timer();

    printf("Total: %3.3lf s, coarsening %3.3lf %3.0lf%% "
                    "(sort %3.3lf %3.0lf%%), "
                    "refine %3.3lf s (%3.3lf s, %3.0lf%% + %3.3lf, %3.0lf%%)\n", 
                    fin_final_level_time-start_time,
                    fin_coarsening_time-start_time,
                    (fin_coarsening_time-start_time)*100/(fin_final_level_time-start_time),
                    time_counters[1],
                    time_counters[1]*100/(fin_final_level_time-start_time),
                    fin_final_level_time-fin_coarsening_time,
                    fin_final_level_time-fin_refine_time,
                    100*(fin_final_level_time-fin_refine_time)/(fin_final_level_time-start_time),
                    fin_refine_time-fin_coarsening_time,
                    100*(fin_refine_time-fin_coarsening_time)/
                    (fin_final_level_time-start_time));

#ifdef EXPERIMENT
    experiment.setTotalDurationSeconds(fin_final_level_time - start_time);
    experiment.setCoarsenDurationSeconds(fin_coarsening_time - start_time);
    experiment.setRefineDurationSeconds(fin_final_level_time - fin_coarsening_time);
    experiment.addMeasurement(ExperimentLoggerUtil::Measurment::Map, time_counters[0]);
    experiment.addMeasurement(ExperimentLoggerUtil::Measurment::Build, time_counters[1]);
    experiment.addMeasurement(ExperimentLoggerUtil::Measurment::Count, time_counters[2]);
    experiment.addMeasurement(ExperimentLoggerUtil::Measurment::Prefix, time_counters[3]);
    experiment.addMeasurement(ExperimentLoggerUtil::Measurment::Bucket, time_counters[4]);
    experiment.addMeasurement(ExperimentLoggerUtil::Measurment::Dedupe, time_counters[5]);
#endif


    for (int i=1; i<num_coarsening_levels; i++) {
        sgp_free_graph(&g_all[i]);
    }

#ifdef COARSE_EIGEN_EC
        for (int l = num_coarsening_levels - 1; l >= 1; l--) {
            printf("Computing edge cut for eigenvector prolonged from coarse level %d\n", l);

            //prolong eigenvector to finest level
            for (int l2 = l - 1; l2 >= 0; l2--) {
                sgp_real_t* prev_prolonged = eigenvec[l];
                sgp_vid_t gcl2_n = g_all[l2].nvertices;
                sgp_real_t* prolonged_eigenvec = (sgp_real_t*)malloc(gcl2_n * sizeof(sgp_real_t));
                SGPAR_ASSERT(prolonged_eigenvec != NULL);
                for (sgp_vid_t i = 0; i < gcl2_n; i++) {
                    prolonged_eigenvec[i] = prev_prolonged[vcmap[l2][i]];
                }
                free(prev_prolonged);
                eigenvec[l] = prolonged_eigenvec;
            }
            sgp_compute_partition(part, config->num_partitions, edge_cut,
                perc_imbalance_allowed,
                config->local_search_alg,
                eigenvec[l], g);

            free(vcmap[l - 1]);
            free(eigenvec[l]);

            //unsigned int part_diff = 0;
            //if (config->compare_part) {
                //CHECK_SGPAR(compute_partition_edit_distance(part, best_part, g.nvertices, &part_diff));
            //}

#ifdef EXPERIMENT
            experiment.modifyCoarseLevelEC(l, *edge_cut);
#endif
        }
#endif

    

    sgp_compute_partition(part, config->num_partitions, edge_cut,
        perc_imbalance_allowed,
        config->local_search_alg,
        eigenvec[0], g);

#ifdef EXPERIMENT
        experiment.setFinestEdgeCut(*edge_cut);
        experiment.modifyCoarseLevelEC(0, *edge_cut);
#endif

    sgp_improve_partition(part, config->num_partitions, edge_cut,
                           perc_imbalance_allowed,
                           eigenvec[0], g);
    free(eigenvec[0]);
#endif

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_use_partition(sgp_vid_t* part, const sgp_graph_t g, sgp_graph_t* g1, sgp_graph_t* g2) {
    sgp_vid_t n = g.nvertices;

    sgp_vid_t* mapping = (sgp_vid_t*)malloc(n * sizeof(sgp_vid_t));
    sgp_vid_t g1_count = 0;
    sgp_vid_t g2_count = 0;
    //compute vertex mappings
    for (sgp_vid_t i = 0; i < n; i++) {
        if (part[i] == 0) {
            mapping[i] = g1_count++;
        }
        else {
            mapping[i] = g2_count++;
        }
    }

    sgp_eid_t* g1_source_offsets = (sgp_eid_t*)malloc((g1_count + 1) * sizeof(sgp_eid_t));
    sgp_eid_t* g2_source_offsets = (sgp_eid_t*)malloc((g2_count + 1) * sizeof(sgp_eid_t));
    g1_source_offsets[0] = 0;
    g2_source_offsets[0] = 0;
    g1_count = 0;
    g2_count = 0;

    //compute source offsets
    for (sgp_vid_t i = 0; i < n; i++) {
        if (part[i] == 0) {
            g1_source_offsets[g1_count + 1] = g1_source_offsets[g1_count];
        }
        else {
            g2_source_offsets[g2_count + 1] = g2_source_offsets[g2_count];
        }
        for (sgp_eid_t j = g.source_offsets[i]; j < g.source_offsets[i + 1]; j++) {
            sgp_vid_t dest = g.destination_indices[j];
            if (part[i] == part[dest]) {
                if (part[i] == 0) {
                    g1_source_offsets[g1_count + 1]++;
                }
                else {
                    g2_source_offsets[g2_count + 1]++;
                }
            }
        }
        if (part[i] == 0) {
            g1_count++;
        }
        else {
            g2_count++;
        }
    }

    sgp_vid_t* g1_dest_indices = (sgp_vid_t*)malloc(g1_source_offsets[g1_count] * sizeof(sgp_vid_t));
    sgp_vid_t* g2_dest_indices = (sgp_vid_t*)malloc(g2_source_offsets[g2_count] * sizeof(sgp_vid_t));
    sgp_eid_t g1_ec = 0;
    sgp_eid_t g2_ec = 0;
    //write re-labelled edges
    for (sgp_vid_t i = 0; i < n; i++) {
        for (sgp_eid_t j = g.source_offsets[i]; j < g.source_offsets[i + 1]; j++) {
            sgp_vid_t dest = g.destination_indices[j];
            if (part[i] == part[dest]) {
                if (part[i] == 0) {
                    g1_dest_indices[g1_ec++] = mapping[dest];
                }
                else {
                    g2_dest_indices[g2_ec++] = mapping[dest];
                }
            }
        }
    }

    g1->destination_indices = g1_dest_indices;
    g1->source_offsets = g1_source_offsets;
    g1->nedges = g1_ec;
    g1->nvertices = g1_count;
    g1->eweights = NULL;
    g1->edges_per_source = NULL;
    g1->weighted_degree = NULL;

    g2->destination_indices = g2_dest_indices;
    g2->source_offsets = g2_source_offsets;
    g2->nedges = g2_ec;
    g2->nvertices = g2_count;
    g2->eweights = NULL;
    g2->edges_per_source = NULL;
    g2->weighted_degree = NULL;

    return EXIT_SUCCESS;
}

SGPAR_API int sgp_load_config(const char* config_f, config_t* c) {

    FILE* infp = fopen(config_f, "r");
    if (infp == NULL) {
        printf("Error: Could not open config file %s. Exiting ...\n", config_f);
        return EXIT_FAILURE;
    }
    if (fscanf(infp, "%i", &c->coarsening_alg) == 0) {
        return EXIT_FAILURE;
    }
    if (fscanf(infp, "%i", &c->refine_alg) == 0) {
        return EXIT_FAILURE;
    }
    if (fscanf(infp, "%i", &c->local_search_alg) == 0) {
        return EXIT_FAILURE;
    }
    if (fscanf(infp, "%i", &c->num_partitions) == 0) {
        return EXIT_FAILURE;
    }
    if (fscanf(infp, "%i", &c->num_iter) == 0) {
        return EXIT_FAILURE;
    }
    if (fscanf(infp, "%lf", &c->tol) == 0) {
        return EXIT_FAILURE;
    }
    CHECK_RETSTAT(fclose(infp));

    if (SGPAR_POWERITER_TOL != c->tol) {
        printf("Using non-default tolerance: %lf\n", c->tol);
        CHECK_SGPAR(change_tol(c->tol));
    }

    return EXIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif // namespace sgpar

#endif // SGPAR_IMPLEMENTATION
