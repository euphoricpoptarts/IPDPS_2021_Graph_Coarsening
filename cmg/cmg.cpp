#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <vector>
#include <cmath>
#include <random>
#include <cfloat>
#include <limits>
#include <climits>
#include <omp.h>

#ifdef USE_GNU_PARALLELMODE
#include <parallel/algorithm> // for parallel sort
#else 
#include <algorithm>          // for STL sort
#endif

#define COARSE_VTX_CUTOFF 25
#define TOL 1e-12

typedef struct graph {
    long n;
    long m;
    unsigned int *rowOffsets;
    unsigned int *adj;
    unsigned int *degree;
    unsigned int *coarseID;
    unsigned int *eweights;
    unsigned int clevel; // Coarsening level
    double *u2;          // Fiedler vector
    double *u3;          // Third vector
    graph *cg;           // Pointer to coarse graph
    long vec2_itrs;      // iterations for Fiedler vec
    long vec3_itrs;      // iterations reqd for third vec
} graph_t;

typedef struct {
    unsigned int u;
    unsigned int v;
    unsigned int w;
} edge_triple_t;

inline static bool cmpfn(const edge_triple_t& a, const edge_triple_t& b) {
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

#if 0
static int vu_cmpfn_inc(const void *a, const void *b) {
    unsigned int *av = ((unsigned int *) a);
    unsigned int *bv = ((unsigned int *) b);
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

static int HEC(graph_t *g) {
    
    std::chrono::duration<double> elt;
    auto startTimerCoarseningStep = std::chrono::high_resolution_clock::now();  
  
    // Coarsen if number of vertices are greater than 25
    if (g->n < COARSE_VTX_CUTOFF) {
        g->coarseID = NULL;
        g->cg = NULL;
        return 0;
    }

    // q holds the fine-to-coarse vertex id mapping
    unsigned int *q;
    q = (unsigned int *) malloc (sizeof(unsigned int) * g->n);
    for (long i=0; i<g->n; i++) {
        q[i] = UINT_MAX;
    }

    // cvc is the coarse vertex count
    int cvc = 0;

    // p holds the shuffled vertex ids
    unsigned int *p = (unsigned int *) malloc (sizeof(unsigned int) * g->n);
    for (long i=0; i<g->n; i++) {
        p[i] = i;
    }
  
    // fixing seed for debugging
    // unsigned int seed = time(NULL);
    // unsigned int seed = 42;
    // srand (seed);

    // parallelize this step
    std::random_shuffle(p, p + g->n);

    // hn holds the heaviest adjacency of every vertex
    unsigned int *hn = (unsigned int *) malloc (sizeof(unsigned int) * g->n);
    for (long i=0; i<g->n; i++) {
        hn[i] = 0;
    }

    // we will first populate the hn array
    // can be parallelized, no dependencies
    if (g->clevel == 0) {
        // simpler case, all unit weights
        for (long u=0; u<g->n; u++) {
            hn[u] = g->adj[g->rowOffsets[u]]; 
        }
    } else {
        for (long u=0; u<g->n; u++) {
            unsigned int hn_u = UINT_MAX;
            unsigned int maxewgt = 0;
            for (unsigned int k=g->rowOffsets[u]; k < g->rowOffsets[u+1]; k++) {
                if (maxewgt < g->eweights[k]) {
                    maxewgt = g->eweights[k];
                    hn_u = g->adj[k];
                }
            }
            // this shouldn't happen
            if(maxewgt == 0) {
                std::cout << "u= " << u << " #ngbrs= " 
                    << g->rowOffsets[u+1] - g->rowOffsets[u] << std::endl;
                exit(1);
            }
            hn[u] = hn_u; 
        }
    }

    // we now look at the vertices in permuted order
    // and compute the coarse vertex count
  
#if 0
    // serial code
    for (long i=0; i<g->n; i++) {
        unsigned int u = p[i];
        unsigned int v = hn[u];
        if (q[u] == UINT_MAX) {
            if (q[v] == UINT_MAX) {
                q[v] = cvc++;
            }
            q[u] = q[v];
        }
    }
#endif

    // parallelization of above loop
    unsigned int *offsets_tid;
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    unsigned int chunk_size = g->n;
    unsigned int voff = tid*chunk_size;
    unsigned int thr_vcount = voff;

    if (tid == 0) {
        offsets_tid = (unsigned int *) malloc((nthreads+1)*sizeof(unsigned int));
        assert(offsets_tid != NULL);
        for (int i=0; i<=nthreads; i++) {
            offsets_tid[i] = 0;
        }
    }

#pragma omp barrier

#pragma omp for schedule(static)
    for (long i=0; i<g->n; i++) {
        unsigned int u = p[i];
        unsigned int v = hn[u];
        unsigned int qu = q[u];
        unsigned int qv = q[v];
        if (qu == UINT_MAX) {
            if (qv == UINT_MAX) {
                // unsigned int qc = __sync_fetch_and_add(&cvc, 1);
                unsigned int qc = thr_vcount++;
                unsigned int qoldu = __sync_val_compare_and_swap(&q[u], UINT_MAX, qc);  
                if (qoldu != UINT_MAX) {
                    // std::cerr << "u's community was set by another thread, it is now " << qoldu << std::endl;
                    thr_vcount--;
                    continue;
                }
                unsigned int qoldv = __sync_val_compare_and_swap(&q[v], UINT_MAX, qc);
                if (qoldv != UINT_MAX) { // reset u's value?
                    // std::cerr << "somebody already set u's community; we introduced an unnecessary vertex with id " << qc << std::endl;
                    // unsigned int quprev = __sync_val_compare_and_swap(&q[u], qoldu, qoldv);
                }
            } else {
                __sync_val_compare_and_swap(&q[u], UINT_MAX, qv);
            }
        }
    }

#pragma omp flush

    offsets_tid[tid+1] = thr_vcount - voff;
  
// #pragma omp critical
//  std::cout << "tid " << tid << " , visited count is " << thr_vcount - voff << std::endl;

#pragma omp barrier

    if (tid == 0) {
        for (int i=1; i<=nthreads; i++) {
            offsets_tid[i] += offsets_tid[i-1];
        }
        cvc = offsets_tid[nthreads];
        // std::cerr << "Total coarse vert count " << cvc << std::endl;
    }

#pragma omp barrier

#pragma omp for schedule(static)
    for (long i=0; i<g->n; i++) {
        unsigned int qi = q[i];
        unsigned int qi_owner_thread = qi/chunk_size;
        q[i] = offsets_tid[qi_owner_thread] + (qi - qi_owner_thread*chunk_size);
        assert(q[i] >= 0);
        assert(q[i] < g->n);
    }

}

    for (long i=0; i<g->n; i++) {
        if (q[i] == UINT_MAX) {
            std::cout << "Vertex " << i << " community not set" << std::endl;
        }
        assert(q[i] != UINT_MAX);
    }
 
    int n_coarse = cvc;
    std::cout << "Coarse vertex count: " << n_coarse << std::endl;
    free(hn);
    free(p);

    g->coarseID = (unsigned int *) q;
    long c = cvc;

    graph_t *cg = (graph_t *) malloc(sizeof(graph_t));
    cg->n = c;
    cg->clevel = g->clevel + 1;
    cg->cg = NULL;
    cg->u2 = NULL; // Feidler vector
    cg->u3 = NULL; // Third eigenvector
    cg->vec2_itrs = 0;
    cg->vec3_itrs = 0;

    auto endTimerCoarseningStep = std::chrono::high_resolution_clock::now();
    elt = endTimerCoarseningStep - startTimerCoarseningStep;
    // std::cout << "time 1: " << elt.count() << " s." << std::endl << std::endl;
  
    startTimerCoarseningStep = std::chrono::high_resolution_clock::now();  
    // Vertex neighbor counts
    unsigned int *vcounts = (unsigned int *) malloc(cg->n * sizeof(unsigned int));
#pragma omp parallel for
    for (long i=0; i<cg->n; i++) {
        vcounts[i] = 0;
    }

    // Coarse graph edges and edge weights
    unsigned int *coarse_edges = (unsigned int *) malloc(3 * g->m * sizeof(unsigned int));
    assert(coarse_edges != NULL);

    unsigned int nonzero_edges = 0;
    for (long i=0; i<g->n; i++) {
        unsigned int u = g->coarseID[i];
        for (unsigned int j=g->rowOffsets[i]; j<g->rowOffsets[i+1]; j++) {
            unsigned int v = g->coarseID[g->adj[j]];
            coarse_edges[3*nonzero_edges] = u;
            coarse_edges[3*nonzero_edges+1] = v;
            if (u != v) {
                coarse_edges[3*nonzero_edges+2] = g->eweights[j];
                nonzero_edges++;
            }
        }
    }
    endTimerCoarseningStep = std::chrono::high_resolution_clock::now();
    elt = endTimerCoarseningStep - startTimerCoarseningStep;
    // std::cout << "time 2: " << elt.count() << " s." << std::endl << std::endl;
    startTimerCoarseningStep = std::chrono::high_resolution_clock::now();  
    // qsort(coarse_edges, nonzero_edges, 3*sizeof(int), vu_cmpfn_inc);
  
#ifdef USE_GNU_PARALLELMODE
    __gnu_parallel::sort(((edge_triple_t *) coarse_edges), ((edge_triple_t *) coarse_edges)+nonzero_edges, cmpfn,
      __gnu_parallel::quicksort_tag()); // quicksort_tag() is optional
#else
    std::sort(((edge_triple_t *) coarse_edges), ((edge_triple_t *) coarse_edges)+nonzero_edges, cmpfn);
#endif

    endTimerCoarseningStep = std::chrono::high_resolution_clock::now();
    elt = endTimerCoarseningStep - startTimerCoarseningStep;
    // std::cout << "time 3: " << elt.count() << " s." << std::endl << std::endl;
    
    startTimerCoarseningStep = std::chrono::high_resolution_clock::now();  
  
    /* count the number of coarse edges */
    unsigned int start_idx = 0;
    while(coarse_edges[3*start_idx+2] == 0) {
        start_idx++;
    }

    vcounts[0]++;
    for (unsigned int i=start_idx+1; i<nonzero_edges; i++) {
        unsigned int prev_u = coarse_edges[3*(i-1)];
        unsigned int prev_v = coarse_edges[3*(i-1)+1];
        unsigned int curr_u = coarse_edges[3*i];
        unsigned int curr_v = coarse_edges[3*i+1];
        if (curr_u == curr_v) {
            continue;
        }
        if ((curr_u != prev_u) || (curr_v != prev_v)) {
            vcounts[curr_u] ++;
        }
    }
  
    endTimerCoarseningStep = std::chrono::high_resolution_clock::now();
    elt = endTimerCoarseningStep - startTimerCoarseningStep;
    
    // std::cout << "time 4: " << elt.count() << " s." << std::endl << std::endl;
    startTimerCoarseningStep = std::chrono::high_resolution_clock::now();  
    /*
    for (int i=0; i<cg->n; i++) {
        assert(vcounts[i] != 0);
    }
    */
    unsigned int *rowOffsetsCoarse;
    rowOffsetsCoarse = (unsigned int *) malloc((cg->n+1)*sizeof(unsigned int));
    assert(rowOffsetsCoarse != NULL);
    rowOffsetsCoarse[0] = 0;
    for (int i=0; i<cg->n; i++) {
        rowOffsetsCoarse[i+1] = rowOffsetsCoarse[i] + vcounts[i];
    }

    cg->rowOffsets = rowOffsetsCoarse;

    int m_coarse = rowOffsetsCoarse[cg->n];
    cg->m = rowOffsetsCoarse[cg->n];

    for (int i=0; i<cg->n; i++) {
        vcounts[i] = 0;
    }

    /* Allocate coarse edge weights array */
    unsigned int *eweights;
    eweights = (unsigned int *) malloc(m_coarse * sizeof(unsigned int));
    assert(eweights != NULL);

    unsigned int *adjCoarse;
    adjCoarse = (unsigned int *) malloc(m_coarse*sizeof(unsigned int));
    assert(adjCoarse != NULL);

    endTimerCoarseningStep = std::chrono::high_resolution_clock::now();
    elt = endTimerCoarseningStep - startTimerCoarseningStep;
    // std::cout << "time 5: " << elt.count() << " s." << std::endl << std::endl;
    startTimerCoarseningStep = std::chrono::high_resolution_clock::now();  
    /* update coarse edge weights */
    adjCoarse[0] = coarse_edges[3*start_idx+1];
    eweights[0] = coarse_edges[3*start_idx+2];
    vcounts[0] ++;

    for (unsigned int i=start_idx+1; i<nonzero_edges; i++) { 
        unsigned int curr_u = coarse_edges[3*i];
        unsigned int curr_v = coarse_edges[3*i+1];
        if (curr_u == curr_v) {
            continue;
        }
        unsigned int prev_u = coarse_edges[3*(i-1)];
        unsigned int prev_v = coarse_edges[3*(i-1)+1];
        unsigned int idx = rowOffsetsCoarse[curr_u] + vcounts[curr_u];
        if ((curr_u != prev_u) || (curr_v != prev_v)) {
            adjCoarse[idx] = curr_v;
            eweights[idx] = coarse_edges[3*i+2];
            vcounts[curr_u] ++;
        } else {
            eweights[idx-1] += coarse_edges[3*i+2];
        }
    }

    cg->adj = adjCoarse;
    cg->eweights = eweights;

    cg->degree = (unsigned int *) malloc(sizeof(unsigned int) * cg->n);
    assert( cg->degree != NULL);
    for (unsigned int i=0; i < cg->n; i ++) {
        unsigned int sum_wgts = 0;
        for (unsigned int j=cg->rowOffsets[i]; j < cg->rowOffsets[i+1]; j++) {
            sum_wgts += cg->eweights[j];
        }
        cg->degree[i] = sum_wgts;
    }

    g->cg = cg;

    free(coarse_edges);
    free(vcounts);

    endTimerCoarseningStep = std::chrono::high_resolution_clock::now();
    elt = endTimerCoarseningStep - startTimerCoarseningStep;
    // std::cout << "time 6: " << elt.count() << " s." << std::endl << std::endl;

    HEC(cg);

    return 0;
}

void freeGraph(graph_t *g) {
    if (g->cg != NULL) {
        freeGraph(g->cg);
    }
    free(g->rowOffsets);
    free(g->adj);
    free(g->eweights);
    free(g->degree);
    free(g->coarseID);
    if (g->u2 != NULL)
        free(g->u2);
    if (g->u3 != NULL)
        free(g->u3);
    free(g);
}

int print_adj(const char *lbl, graph_t *g) {
    std::cout << lbl << "--- ";
    for (long i=0; i<g->n; i++) {
        for (unsigned int j=g->rowOffsets[i]; j<g->rowOffsets[i+1]; j++) {
            std::cout << "(" << i << "," << g->adj[j] << ", " << g->eweights[j] << ")  " << std::endl;
        }
    }
    std::cout << std::endl;
    return 0;
}

int get_coarsening_levels(graph_t *g) {
    if (g->cg == NULL)
        return g->clevel;
    return get_coarsening_levels(g->cg);
}

int print_coarsening_info(graph_t *g) {
    if (g == NULL) return 0;
    std::cout << "Coarsening level " << g->clevel 
    << ", vertices=" << g->n 
    << ", edges=" << g->m/2 
    << ", iter_count=" << g->vec2_itrs << std::endl;
    print_coarsening_info(g->cg);
    return 0;
}

int print_vec(const char *lbl, double *u, long n) {
    assert(u != NULL);

    std::cout << lbl << ": ";
    for (long i=0; i<n; i++) {
        std::cout << u[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}

double sum_vec(double *u, long n) {
    assert(u != NULL);
  
    double sum = 0.0;
    for (long i=0; i<n; i++) {
        sum += u[i];
    }
    return sum;
}

double diff_vec(double *u1, double *u2, long n) {
    double sum = 0.0;

    for (long i=0; i<n; i++) {
        double diff = u1[i] - u2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double get_2_norm(double *u, long n) {
    assert(u != NULL);
  
    double sum = 0.0;
    for (long i=0; i<n; i++) {
        sum += u[i]*u[i];
    }
    return sqrt(sum);
}

void normalize(double *u, long n) {
    assert(u != NULL);
  
    double norm2 = get_2_norm(u, n);
    for (long i=0; i<n; i++) {
        u[i] = u[i]/norm2;
    }
}

void copy_vec(double *u, double *v, long n) {
    assert(u != NULL);
    assert(v != NULL);
    for (long i=0; i<n; i++) {
        v[i] = u[i];
    }
}

void make_ones_vec(double *u, long n) {
    assert(u != NULL);
    for (long i=0; i<n; i++) {
        u[i] = 1.0;
    }
}

void make_zeros_vec(double *u, long n) {
    assert(u != NULL);
    for (long i=0; i<n; i++) {
        u[i] = 0.0;
    }
}

void scalar_mult(double *u, long n, double scalar) {
    assert(u != NULL);
    for (long i=0; i<n; i++) {
        u[i] *= scalar;
    }
}

double dot_prod_vecs(double *u1, double *u2, long n) {
    assert(u1 != NULL);
    assert(u2 != NULL);

    double sum = 0.0;
    for (long i=0; i<n; i++) {
        sum += u1[i]*u2[i];
    }
    return sum;
}

void D_orthogonalize(double *u1, double *u2, graph_t *g) {
  
    //u1[i] = u1[i] - (dot(u1, D*u2)/dot(u2, D*u2)) * u2[i]
    long n = g->n;
    assert(u1 != NULL);
    assert(u2 != NULL);

    double mult1 = 0;
    double mult2 = 0;
    for (long i=0; i<n; i++) {
        double tmp = g->degree[i] * u2[i];
        mult1 += u1[i] * tmp;
        mult2 += u2[i] * tmp;
    }

    double mult = mult1/mult2;
    for (unsigned int i=0; i<n; i++) {
        u1[i] -= mult * u2[i];
    }

}

void orthogonalize(double *u1, double *u2, long n) {
    assert(u1 != NULL);
    assert(u2 != NULL);

    double mult1 = dot_prod_vecs(u1, u2, n);
    for (unsigned int i=0; i<n; i++) {
        u1[i] -= mult1*u2[i];
    }
}

void koren(graph_t *g, int vecnum) {
    double *v1 = (double *) malloc(sizeof(double) * g->n);
    make_ones_vec(v1, g->n);
    normalize(v1, g->n);

    double *u2 = (double *) malloc(sizeof(double) * g->n);
    if (vecnum == 3) {
        copy_vec(g->u3, u2, g->n);
    } else {
        copy_vec(g->u2, u2, g->n);
    }
  
    D_orthogonalize(u2, v1, g);
    if (vecnum == 3) {
        D_orthogonalize(u2, g->u2, g);
    }
    normalize(u2, g->n);

    double *v2 = (double *) malloc(sizeof(double) * g->n);
    make_zeros_vec(v2, g->n);

    long k = 0;

    double dp_v2_u2 = dot_prod_vecs(v2, u2, g->n);
    // double diff = diff_vec(v2, u2, g->n);

    double tol = TOL;

    while(dp_v2_u2 < 1.0 - tol) {
        if (g->clevel == 0 && k > 100) {
            printf("Max iterations reached\n");
            break;
        }
        k ++;

        copy_vec(u2, v2, g->n);
    
        // u2 = (I + D^-1 * A)/2
        for (unsigned int i=0; i<g->n; i++) {
            double sum = v2[i]/2.0;
            for (unsigned int j=g->rowOffsets[i]; j<g->rowOffsets[i+1]; j++) {
                sum += g->eweights[j] * v2[ g->adj[j] ]/(g->degree[i]*2.0);
            }
            u2[i] =  sum;
        }

        if (sum_vec(u2, g->n) > 0.01) {
            D_orthogonalize(u2, v1, g);
            if (vecnum == 3) {
                D_orthogonalize(u2, g->u2, g);
                break;
            }
        }

        normalize(u2, g->n);

        // diff = diff_vec(v2, u2, g->n);
        dp_v2_u2 = dot_prod_vecs(v2, u2, g->n);
    }

    if (vecnum == 3) {
        copy_vec(u2, g->u3, g->n);
        g->vec3_itrs = k;
    } else {
        copy_vec(u2, g->u2, g->n);
        g->vec2_itrs = k;
    }

    free(v2);
    free(v1);
    free(u2);

}

void power_it(graph_t *g, int vecnum) {

  /*
   * MATLAB code from Urschel
   * 
   * function [u2, k]=graphpowerit(u2,L)
   * n=length(u2);
   * v1=ones(n,1)/sqrt(n);
   * %Defining Bg
   * g=max(sum(abs(L)));
   * Bg=g*speye(n,n)-L;
   * u2=u2-(u2'*v1)*v1;
   * u2=u2/norm(u2);
   * v2=zeros(n,1);
   *
   * k = 0;
   *
   * while v2'*u2 < 1-10^(-6)
   *  k = k+1;
   *  v2=u2;
   *  if sum(v2)>.01
   *    v2=v2-(v2'*v1)*v1;
   *  end
   *
   *  u2=Bg*v2;
   *  u2=u2/norm(u2);
   *
   *  end
   *
   */

    double gr = -1.0;
    for (long i=0; i<g->n; i++) {
        double sum = 0;
        for (unsigned int j=g->rowOffsets[i]; j<g->rowOffsets[i+1]; j++) {
            sum += g->eweights[j];
        }
        sum *= 2; //Adding diagonal element
        if (gr < sum) {
            gr = sum;
        }
    }
  
    double *v1 = (double *) malloc(sizeof(double) * g->n);
    make_ones_vec(v1, g->n);
    normalize(v1, g->n);

    double *u2 = (double *) malloc(sizeof(double) * g->n);
    if (vecnum == 3) {
        copy_vec(g->u3, u2, g->n);
    } else {
        copy_vec(g->u2, u2, g->n);
    }
  
    orthogonalize(u2, v1, g->n);
    if (vecnum == 3) {
        orthogonalize(u2, g->u2, g->n);
    }
    normalize(u2, g->n);

    double *v2 = (double *) malloc(sizeof(double) * g->n);
    make_zeros_vec(v2, g->n);

    long k = 0;

    double dp_v2_u2 = dot_prod_vecs(v2, u2, g->n);
    // double diff = diff_vec(v2, u2, g->n);

    double tol = TOL;
    while(dp_v2_u2 < 1.0 - tol) {
    // while(diff > tol) {
        k ++;
        if (k > 1000000) {
            fprintf(stderr, "Million iterations, breaking\n");
            break;
        }
        copy_vec(u2, v2, g->n);
    
        // u2 = Bg * v2;
        for (unsigned int i=0; i<g->n; i++) {
            double sum = (gr - g->degree[i]) * v2[i];
            for (unsigned int j=g->rowOffsets[i]; j<g->rowOffsets[i+1]; j++) {
                sum += g->eweights[j] * v2[ g->adj[j] ];
            }
            u2[i] =  sum;
        }

        if (sum_vec(u2, g->n) > 0.01) {
            orthogonalize(u2, v1, g->n);
            if (vecnum == 3) {
                orthogonalize(u2, g->u2, g->n);
            }
        }

        normalize(u2, g->n);

        // diff = diff_vec(v2, u2, g->n);
        dp_v2_u2 = dot_prod_vecs(v2, u2, g->n);
    }

    if (vecnum == 3) {
        copy_vec(u2, g->u3, g->n);
        g->vec3_itrs = k;
    } else {
        copy_vec(u2, g->u2, g->n);
        g->vec2_itrs = k;
    }
    printf("Number of iterations %d\n", k);
    free(v2);
    free(v1);
    free(u2);
}

void compute_vector(graph_t *g, int refine_type, int vecnum) {
    if (vecnum == 3) {
        g->u3 = (double *) malloc(sizeof(double) * g->n);
    } else {
        g->u2 = (double *) malloc(sizeof(double) * g->n);
    }

    if (g->cg == NULL) {
    
        // For coarsest graph
        // initialize with random numbers
        std::default_random_engine generator(time(NULL));
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (unsigned int i=0; i < g->n; i++) {
            if(vecnum == 3) {
                g->u3[i] = distribution(generator);
            } else {
                g->u2[i] = distribution(generator);
            }
        }
    } else {
        compute_vector(g->cg, refine_type, vecnum);
    
        // Project vector from coarse graph
        for (unsigned int i=0; i < g->n; i++) {
            unsigned int coarse_index = g->coarseID[i];
            if (vecnum == 3) {
                g->u3[i] = g->cg->u3[coarse_index];
            } else {
                g->u2[i] = g->cg->u2[coarse_index];
            }
        }
    }
    if (refine_type == 1) {
        koren(g, vecnum);
    } else {
        // refine vector using power iteration
        power_it(g, vecnum);
    }


}

void partition(graph_t *g) {

    std::vector<std::pair<double, unsigned int> > part_ids;

    for (long i = 0; i<g->n; i++) {
        part_ids.push_back(std::make_pair(g->u2[i], i));
    }   
    std::sort(part_ids.begin(), part_ids.end());


    long split = ceil(g->n/2.0);
    fprintf(stderr, "split is %ld\n", split);
    // allow x% imbalance
    long imbr = floor(split*1.01);
    long imbl = g->n - imbr;
    std::vector<unsigned int> CompMapping(g->n, 0);
    for (long i=0; i<imbl; i++) {
        CompMapping[part_ids[i].second] = 0;
    }
    for (long i=imbl; i<g->n; i++) {
        CompMapping[part_ids[i].second] = 1;
    }

    int edgecut_curr = 0;
    for (long i=0; i<g->n; i++) {
        unsigned int part_i = CompMapping[i];
        for (unsigned int j=g->rowOffsets[i]; j<g->rowOffsets[i+1]; j++) {
            unsigned int v = g->adj[j];
            if (CompMapping[v] != part_i) {
                edgecut_curr++;
            }
        }
    }
    // fprintf(stderr, "Vert frac %lf, edge cut is %u\n", ((double) split-imb)/g->n, edgecut_curr/2);

    int edgecut_min = edgecut_curr;
    unsigned int curr_split = g->n - imbl + 1;

    for (long i=imbl; i<imbr; i++) {
        /* add vert at position i to comm 0 */
        unsigned int u = part_ids[i].second;
        CompMapping[u] = 0;
        int ec_update = 0;
        for (unsigned int j=g->rowOffsets[u]; j<g->rowOffsets[u+1]; j++) {
            unsigned int v = g->adj[j];
            if (CompMapping[v] == 1) {
                ec_update++;
            } else {
                ec_update--;
            }
        }
        edgecut_curr = edgecut_curr + 2*ec_update;
        // std::cout << "Vert frac " << ((double) i)/g->n << ", edge cut is " << edgecut_curr/2 << std::endl;
        if (edgecut_curr <= edgecut_min) {
            edgecut_min = edgecut_curr;
            curr_split = g->n - i - 1;
            if ((g->n - i - 1) < (i+1)) {
                curr_split = i + 1;
            }
        } 
    }
    std::cout << "Large part with imbalance is " 
              << curr_split 
              << ", edge cut is " << edgecut_min/2 << std::endl;

    double *vec = (double *) malloc (sizeof(double) * g->n);

    for (unsigned int i = 0; i<g->n; i++) {
        vec[i] = g->u2[i];
    }
    std::sort(vec, vec + g->n);
  
    double median = 0.0;
    if (g->n % 2 == 0) {
        median = (vec[g->n / 2 - 1] + vec[g->n / 2])/2.0;
    } else {
        median = vec[g->n / 2];
    }
    free(vec);

    long ones = 0;
    long zeros = 0;
    unsigned int *cutVec = (unsigned int *) malloc (sizeof(unsigned int) * g->n);
    for (unsigned int i=0; i<g->n; i++) {
        if (g->u2[i] < median) {
            cutVec[i] = 0;
            zeros ++;
        } else {
            cutVec[i] = 1;
            ones ++;
        }
    }

    double balance = (ones*1.0) / (zeros*1.0);

    unsigned int cutsize = 0;
    for (unsigned int i=0; i<g->n; i++) {
        for (unsigned int j=g->rowOffsets[i]; j<g->rowOffsets[i+1]; j++) {
            if (i < g->adj[j] && cutVec[i] != cutVec[g->adj[j]]) {
                cutsize ++;
            } 
        }
    }

    // std::cout << "Cut size is " << cutsize << " with balance " << balance << std::endl;
    printf("Using median, cut size is %d with balance %0.2f\n", cutsize, balance);
    free(cutVec);
}

void report_quality(graph_t *g) {
    // Measure quality of eigenvector
    double r = 0;
    double sqerror = 0;
    for (unsigned int i=0; i<g->n; i++) {
        double sum = (g->rowOffsets[i+1] - g->rowOffsets[i]) * g->u2[i];
        for (unsigned int j=g->rowOffsets[i]; j<g->rowOffsets[i+1]; j++) {
            sum -= g->u2[ g->adj[j] ];
        }

        r += sum * g->u2[i];
        sqerror += (sum - r * g->u2[i]) * (sum - r * g->u2[i]);
    }

    // std::cout << "Error estimate ||(L-rI)y|| = " << sqrt(sqerror) << std::endl;
    printf("Error estimate ||(L-rI)y|| = %.2e\n", sqrt(sqerror));
}

void print_coordinates(graph_t *g, char *csrfilename) {
    char xyzfilename[2048];
    sprintf(xyzfilename, "%s.xyz", csrfilename);
    FILE *xyzfile = fopen(xyzfilename, "w");
    for (unsigned int i=0; i<g->n; i++) {
        fprintf(xyzfile, "%lf,%lf\n", g->u2[i], g->u3[i]);
    }
    fclose(xyzfile);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Usage: "<< argv[0] 
                  << " <csr filename> <power(0)/koren(1)>"
                  << std::endl;
        return 1;
    }

    char *inputFilename = argv[1];
    std::chrono::duration<double> total_elt; 
    
    // Read CSR file
    auto startTimerPart = std::chrono::high_resolution_clock::now();  
    FILE *infp = fopen(inputFilename, "rb");
    if (infp == NULL) {
        std::cout << "Error: Could not open input file. Exiting ..." <<
        std::endl; 
        return 1;   
    }
  
    int refine_type = atoi(argv[2]);

    long n, m;
    long rest[4];
    unsigned int *rowOffsets, *adj, *eweights, *degree;
    fread(&n, 1, sizeof(long), infp);
    fread(&m, 1, sizeof(long), infp);
    fread(rest, 4, sizeof(long), infp);
    rowOffsets = (unsigned int *) malloc (sizeof(unsigned int) * (n+1));
    degree = (unsigned int *) malloc (sizeof(unsigned int) * n);
    adj = (unsigned int *) malloc (sizeof(unsigned int) * m);
    eweights = (unsigned int *) malloc (sizeof(unsigned int) * m);
    fread(rowOffsets, n+1, sizeof(unsigned int), infp);
    fread(adj, m, sizeof(unsigned int), infp);
    for (unsigned int i=0; i<m; i++) {
        eweights[i] = 1;
    }
    for (unsigned int i=0; i<n; i++) {
        degree[i] = rowOffsets[i+1] - rowOffsets[i];
    }

    fclose(infp);
    auto endTimerPart = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elt = endTimerPart - startTimerPart;
    std::cout << "CSR read time: " << elt.count() << " s." << std::endl;
    std::cout << "Num edges: " << m/2 << " , vertices: " << n << std::endl;

    auto startTimerCoarsen = std::chrono::high_resolution_clock::now();  
    graph_t *g = (graph_t *) malloc(sizeof(graph_t));;
    g->n = n; g->m = m;
    g->rowOffsets = rowOffsets;
    g->adj = adj;  
    g->eweights = eweights;
    g->degree = degree;
    g->clevel = 0;
    g->u2 = NULL; //Fiedler vector
    g->u3 = NULL; //Third vector
    g->vec2_itrs = 0;
    g->cg = NULL;

    // Perform coarsening
    HEC(g);

    auto endTimerCoarsen = std::chrono::high_resolution_clock::now();
    elt = endTimerCoarsen - startTimerCoarsen;
    total_elt = elt;
    // std::cout << "Coarsening time: " << elt.count() << " s." << std::endl;
    printf("Coarsening time: %.3f s.\n", elt.count());

    auto startTimerVectorCompute = std::chrono::high_resolution_clock::now();  

    // Compute Fiedler vector
    compute_vector(g, refine_type, 2);

    auto endTimerVectorCompute = std::chrono::high_resolution_clock::now();
    elt = endTimerVectorCompute - startTimerVectorCompute;
    total_elt += elt;
    // std::cout << "Fiedler Vector Compute time: " << elt.count() << " s." << std::endl;
    printf("Fiedler Vector Compute time: %.3f s.\n", elt.count());

    auto startTimerThirdVectorCompute = std::chrono::high_resolution_clock::now();  

    // Compute third vector
    compute_vector(g, refine_type, 3);

    auto endTimerThirdVectorCompute = std::chrono::high_resolution_clock::now();
    elt = endTimerThirdVectorCompute - startTimerThirdVectorCompute;
    std::cout << "Third Vector Compute time: " << elt.count() << " s." << std::endl << std::endl;

    print_coarsening_info(g);

    // Report quality of eigenvector
    report_quality(g);

    // Print coordinates
    print_coordinates(g, inputFilename);

    // Compute partition quality
    partition(g);

    // std::cout << "Total time (coarsen+fiedler): " << total_elt.count() << " s." << std::endl << std::endl;
    printf("Total time (coarsen+fiedler): %.3f s\n", total_elt.count());

    freeGraph(g);
    return 0;
}
