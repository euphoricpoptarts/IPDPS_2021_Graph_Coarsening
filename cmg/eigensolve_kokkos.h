#pragma once

#include "definitions_kokkos.h"
#include "KokkosBlas1_dot.hpp"
#include "coarsen_kokkos.h"
#include "heuristics.h"
#include <limits>
#include <cstdlib>
#include <cmath>

namespace sgpar {
namespace sgpar_kokkos {

SGPAR_API int sgp_vec_normalize_kokkos(eigenview_t& u, sgp_vid_t n) {
    sgp_real_t squared_sum = 0;

    squared_sum = KokkosBlas::dot(u, u);
    sgp_real_t sum_inv = 1 / sqrt(squared_sum);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(int64_t i) {
        u(i) = u(i) * sum_inv;
    });
    return EXIT_SUCCESS;
}


SGPAR_API int sgp_vec_orthogonalize_kokkos(eigenview_t& u1, eigenview_t& u2, sgp_vid_t n) {

    sgp_real_t mult1 = KokkosBlas::dot(u1,u2);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        u1(i) -= mult1 * u2(i);
    });
    return EXIT_SUCCESS;
}

SGPAR_API int sgp_vec_D_orthogonalize_kokkos(eigenview_t& u1, eigenview_t& u2,
    Kokkos::View<sgp_wgt_t*>& D, sgp_vid_t n) {

    //u1[i] = u1[i] - (dot(u1, D*u2)/dot(u2, D*u2)) * u2[i]

    sgp_real_t mult1 = KokkosBlas::dot(u1, u2);

    sgp_real_t mult_numer = 0.0;
    sgp_real_t mult_denom = 0.0;

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const sgp_vid_t & i, sgp_real_t & thread_mult_numer) {
        thread_mult_numer += u1(i) * D(i) * u2(i);
    }, mult_numer);
    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const sgp_vid_t & i, sgp_real_t & thread_mult_denom) {
        thread_mult_denom += u2(i) * D(i) * u2(i);
    }, mult_denom);

    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        u1(i) -= mult_numer * u2(i) / mult_denom;
    });
    return EXIT_SUCCESS;
}

SGPAR_API void sgp_power_iter_eigenvalue_log(eigenview_t& u, const matrix_type& g) {
    sgp_real_t eigenval = 0;
    sgp_real_t eigenval_max = 0;
    sgp_real_t eigenval_min = 2;

    sgp_vid_t n = g.numRows();

    eigenview_t v("v", n);
    KokkosSparse::spmv("N", 1.0, g, u, 0.0, v);

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(sgp_vid_t i, sgp_real_t& local_sum) {
        sgp_vid_t weighted_degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        sgp_real_t u_i = weighted_degree * u(i);
        sgp_real_t matvec_i = v(i);
        u_i -= matvec_i;
        v(i) = u_i;
        local_sum += (u_i * u_i) * 1e9;
    }, eigenval);

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(sgp_vid_t i, sgp_real_t& local_max) {
        sgp_real_t eigenval_est = v(i) / u(i);
        if (local_max < eigenval_est) {
            local_max = eigenval_est;
        }
    }, Kokkos::Max<sgp_real_t, Kokkos::HostSpace>(eigenval_max));

    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(sgp_vid_t i, sgp_real_t& local_min) {
        sgp_real_t eigenval_est = v(i) / u(i);
        if (local_min > eigenval_est) {
            local_min = eigenval_est;
        }
    }, Kokkos::Min<sgp_real_t, Kokkos::HostSpace>(eigenval_min));

    printf("eigenvalue = %1.9lf (%1.9lf %1.9lf), "
        "edge cut lb %5.0lf "
        "gap ratio %.0lf\n",
        eigenval * 1e-9,
        eigenval_min, eigenval_max,
        eigenval * 1e-9 * n / 4,
        ceil(1.0 / (1.0 - eigenval * 1e-9)));
}

SGPAR_API int sgp_power_iter(eigenview_t& u, const matrix_type& g, int normLap, int final
#ifdef EXPERIMENT
    , ExperimentLoggerUtil& experiment
#endif
    ) {

    sgp_vid_t n = g.numRows();

    eigenview_t vec1("vec1", n);
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
        vec1(i) = 1.0;
    });

    wgt_view_t weighted_degree("weighted degree",n);
    Kokkos::parallel_for("", n, KOKKOS_LAMBDA(sgp_vid_t i) {
        sgp_wgt_t degree_wt_i = 0;
        sgp_eid_t end_offset = g.graph.row_map(i + 1);
        for (sgp_eid_t j = g.graph.row_map(i); j < end_offset; j++) {
            degree_wt_i += g.values(j);
        }
        weighted_degree(i) = degree_wt_i;
    });

    sgp_wgt_t gb = 2.0;
    if (!normLap) {
        Kokkos::parallel_reduce("max weighted degree", n, KOKKOS_LAMBDA(sgp_vid_t i, sgp_wgt_t& local_max){
            sgp_wgt_t swap = 2.0 * weighted_degree(i);
            if (local_max < swap) {
                local_max = swap;
            }
        }, Kokkos::Max<sgp_wgt_t, Kokkos::HostSpace>(gb));
        if (gb < 2.0) {
            gb = 2.0;
        }
    }

    sgp_vec_normalize_kokkos(vec1, n);
    if (!normLap) {
        sgp_vec_orthogonalize_kokkos(u, vec1, n);
    }
    else {
        sgp_vec_D_orthogonalize_kokkos(u, vec1, weighted_degree, n);
    }
    sgp_vec_normalize_kokkos(u, n);

    eigenview_t v("v", n);
    //is this necessary?
    Kokkos::deep_copy(v, u);

    sgp_real_t tol = SGPAR_POWERITER_TOL;
    uint64_t niter = 0;
    uint64_t iter_max = (uint64_t)SGPAR_POWERITER_ITER / (uint64_t)n;
    sgp_real_t dotprod = 0, lastDotprod = 1;
    while (fabs(dotprod - lastDotprod) > tol && (niter < iter_max)) {

        Kokkos::deep_copy(u, v);

        KokkosSparse::spmv("N", 1.0, g, u, 0.0, v);

        // v = Lu
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(sgp_vid_t i) {
            // sgp_real_t v_i = g.weighted_degree[i]*u[i];
            sgp_real_t weighted_degree_inv, v_i;
            if (!normLap) {
                v_i = (gb - weighted_degree(i)) * u(i);
            }
            else {
                weighted_degree_inv = 1.0 / weighted_degree(i);
                v_i = 0.5 * u(i);
            }
            // v_i -= matvec_i;
            if (!normLap) {
                v_i += v(i);
            }
            else {
                v_i += 0.5 * v(i) * weighted_degree_inv;
            }
            v(i) = v_i;
        });

        if (!normLap) {
            sgp_vec_orthogonalize_kokkos(v, vec1, n);
        }
        sgp_vec_normalize_kokkos(v, n);
        lastDotprod = dotprod;
        dotprod = KokkosBlas::dot(u, v);
        niter++;
    }
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

    return EXIT_SUCCESS;
}

void fm_create_ds(const eigenview_t& partition_device, const matrix_type& g_device, const vtx_view_t& vtx_w_device, const sgp_eid_t maxE,
    sgp_eid_t& cutsize, vtx_mirror_t& bucketsA, vtx_mirror_t& bucketsB, vtx_mirror_t& ll_next_out, vtx_mirror_t& ll_prev_out, vtx_mirror_t& free_vtx_out, Kokkos::View<int64_t*>::HostMirror& gains_out) {

    sgp_vid_t n = g_device.numRows();

    sgp_eid_t totalBuckets = 2 * maxE + 1;
    vtx_view_t bucketsA_cap("buckets cap 0", totalBuckets), bucketsB_cap("buckets cap 1", totalBuckets);

    //extra space for each bucket for simpler logic
    Kokkos::parallel_for(totalBuckets, KOKKOS_LAMBDA(sgp_eid_t i){
        bucketsA_cap(i) = 1;
        bucketsB_cap(i) = 1;
    });

    vtx_view_t bucketsA_device("buckets part 0", totalBuckets), bucketsB_device("buckets part 1", totalBuckets);
    edge_view_t bucketsA_offset("buckets part 0", totalBuckets + 1), bucketsB_offset("buckets part 1", totalBuckets + 1);
    vtx_view_t ll_next("linked list nexts", n), ll_prev("linked list prevs", n);
    Kokkos::View<int64_t*> gains("gains", n);

    cutsize = 0;
    //calculate gains and cutsize
    Kokkos::parallel_reduce("find gains", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread, sgp_eid_t & thread_sum){
        sgp_real_t gain = 0;
        sgp_vid_t i = thread.league_rank();
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, g_device.graph.row_map(i), g_device.graph.row_map(i + 1)), [=](const sgp_eid_t j, sgp_real_t& local_sum) {
            sgp_vid_t v = g_device.graph.entries(j);
            if (partition_device(i) != partition_device(v)) {
                local_sum += g_device.values(j);
            }
            else {
                local_sum -= g_device.values(j);
            }
            }, gain);
        sgp_eid_t local_cut = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, g_device.graph.row_map(i), g_device.graph.row_map(i + 1)), [=](const sgp_eid_t j, sgp_eid_t& local_sum) {
            sgp_vid_t v = g_device.graph.entries(j);
            if (partition_device(i) != partition_device(v)) {
                local_sum += g_device.values(j);
            }
            }, local_cut);
        Kokkos::single(Kokkos::PerTeam(thread), [&]() {
            gains(i) = gain;
            thread_sum += local_cut;
            sgp_eid_t bucket = static_cast<sgp_eid_t>(gain + maxE);
            if (partition_device(i) == 0.0) {
                Kokkos::atomic_increment(&bucketsA_cap(bucket));
            }
            else {
                Kokkos::atomic_increment(&bucketsB_cap(bucket));
            }
            });
    }, cutsize);

    //calculate where to put vertices for each bucket
    Kokkos::parallel_scan("align linked lists", totalBuckets, KOKKOS_LAMBDA(const sgp_eid_t i, sgp_eid_t & update, const bool final){
        update += bucketsA_cap(i);

        if (final) {
            bucketsA_offset(i + 1) = update;
        }
    });
    Kokkos::View<sgp_eid_t> ba_size_sub = Kokkos::subview(bucketsA_offset, totalBuckets);
    sgp_eid_t ba_size = 0;
    Kokkos::deep_copy(ba_size, ba_size_sub);
    Kokkos::parallel_scan("align linked lists", totalBuckets, KOKKOS_LAMBDA(const sgp_eid_t i, sgp_eid_t & update, const bool final){
        update += bucketsB_cap(i);

        if (final) {
            bucketsB_offset(i + 1) = update;
        }
    });
    Kokkos::View<sgp_eid_t> bb_size_sub = Kokkos::subview(bucketsB_offset, totalBuckets);
    sgp_eid_t bb_size = 0;
    Kokkos::deep_copy(bb_size, bb_size_sub);

    Kokkos::parallel_for(totalBuckets, KOKKOS_LAMBDA(sgp_eid_t i){
        bucketsA_cap(i) = 0;
        bucketsB_cap(i) = 0;
        bucketsA_device(i) = SGP_INFTY;
        bucketsB_device(i) = SGP_INFTY;
    });

    vtx_view_t bucketsA_vtx("bucketsA vertices", ba_size), bucketsB_vtx("bucketsB vertices", bb_size);
    Kokkos::parallel_for(ba_size, KOKKOS_LAMBDA(sgp_eid_t i){
        bucketsA_vtx(i) = SGP_INFTY;
    });
    Kokkos::parallel_for(bb_size, KOKKOS_LAMBDA(sgp_eid_t i){
        bucketsB_vtx(i) = SGP_INFTY;
    });

    vtx_view_t free_vtx("free vertices", n);

    Kokkos::parallel_for("move vertices to buckets", n, KOKKOS_LAMBDA(sgp_vid_t i){
        sgp_eid_t bucket = static_cast<sgp_eid_t>(gains(i) + maxE);
        if (partition_device(i) == 0.0) {
            sgp_eid_t offset = Kokkos::atomic_fetch_add(&bucketsA_cap(bucket), 1);
            offset += bucketsA_offset(bucket);
            bucketsA_vtx(offset) = i;
        }
        else {
            sgp_eid_t offset = Kokkos::atomic_fetch_add(&bucketsB_cap(bucket), 1);
            offset += bucketsB_offset(bucket);
            bucketsB_vtx(offset) = i;
        }
        ll_next(i) = SGP_INFTY;
        ll_prev(i) = SGP_INFTY;
        free_vtx(i) = 1;
    });

    Kokkos::parallel_for("create buckets for part 0", ba_size, KOKKOS_LAMBDA(sgp_eid_t i){
        sgp_vid_t u = bucketsA_vtx(i);
        if (u != SGP_INFTY) {
            if (i != 0 && bucketsA_vtx(i - 1) != SGP_INFTY) {
                ll_prev(u) = bucketsA_vtx(i - 1);
            }
            else {
                sgp_eid_t bucket = static_cast<sgp_eid_t>(gains(u) + maxE);
                bucketsA_device(bucket) = u;
            }
            //last element of bucketsA_vtx must be SGP_INFTY so this can't be out of bounds
            //also if the next element is SGP_INFTY, then this is effectively a no-op
            ll_next(u) = bucketsA_vtx(i + 1);
        }
    });

    Kokkos::parallel_for("create buckets for part 1", bb_size, KOKKOS_LAMBDA(sgp_eid_t i){
        sgp_vid_t u = bucketsB_vtx(i);
        if (u != SGP_INFTY) {
            if (i != 0 && bucketsB_vtx(i - 1) != SGP_INFTY) {
                ll_prev(u) = bucketsB_vtx(i - 1);
            }
            else {
                sgp_eid_t bucket = static_cast<sgp_eid_t>(gains(u) + maxE);
                bucketsB_device(bucket) = u;
            }
            //last element of bucketsB_vtx must be SGP_INFTY so this can't be out of bounds
            //also if the next element is SGP_INFTY, then this is effectively a no-op
            ll_next(u) = bucketsB_vtx(i + 1);
        }
    });

    bucketsA = Kokkos::create_mirror(bucketsA_device);
    bucketsB = Kokkos::create_mirror(bucketsB_device);
    ll_next_out = Kokkos::create_mirror(ll_next);
    ll_prev_out = Kokkos::create_mirror(ll_prev);
    free_vtx_out = Kokkos::create_mirror(free_vtx);
    gains_out = Kokkos::create_mirror(gains);

    Kokkos::deep_copy(bucketsA, bucketsA_device);
    Kokkos::deep_copy(bucketsB, bucketsB_device);
    Kokkos::deep_copy(ll_next_out, ll_next);
    Kokkos::deep_copy(ll_prev_out, ll_prev);
    Kokkos::deep_copy(free_vtx_out, free_vtx);
    Kokkos::deep_copy(gains_out, gains);
}

struct imbCut {
    int64_t cut = 0;
    int64_t imb = 0;
    sgp_vid_t idx = 0;
};

struct bestImbCut
{

	Kokkos::View<int64_t*> scan_cuts;
    Kokkos::View<int64_t*> scan_imbs;
    int64_t start_cut;
    int64_t start_imb, target_imb;
	Kokkos::View<imbCut> result;
	sgp_vid_t end_of_range;

    bestImbCut(Kokkos::View<int64_t*> scan_cuts,
        Kokkos::View<int64_t*> scan_imbs,
        int64_t start_cut,
        int64_t start_imb,
		int64_t target_imb,
		Kokkos::View<imbCut> result,
		sgp_vid_t end_of_range)
        : scan_cuts(scan_cuts)
        , scan_imbs(scan_imbs)
        , start_cut(start_cut)
        , start_imb(start_imb)
		, target_imb(target_imb)
   		, result(result)
   		, end_of_range(end_of_range) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const sgp_vid_t i, imbCut& update, const bool final) const
    {
        bool swap = false;
        if (abs(scan_imbs(i)) <= target_imb && update.cut >= scan_cuts(i)) {
            swap = true;
        } 
        else if (abs(update.imb) > abs(scan_imbs(i)) && 1.2 * update.cut > scan_cuts(i)) {
            swap = true;
        }
        if (swap) {
            update.imb = scan_imbs(i);
            update.cut = scan_cuts(i);
            //idx is exclusive of last vtx to be swapped
            update.idx = i + 1;
        }

		if(final && i == end_of_range){
			result() = update;
			//result().imb = update.imb;
			//result().cut = update.cut;
			//result().idx = update.idx;
		}
    }

    KOKKOS_INLINE_FUNCTION
        void join(volatile imbCut& update, const volatile imbCut& compare) const
    {
        bool swap = false;
		if(abs(compare.imb) <= target_imb && update.cut >= compare.cut){
			swap = true;
		}
		else if (abs(update.imb) >= abs(compare.imb) && update.cut > compare.cut) {
            swap = true;
        }
        else if (abs(update.imb) > abs(compare.imb) && 1.2 * update.cut > compare.cut) {
            swap = true;
        }
        if (swap) {
            update.imb = compare.imb;
            update.cut = compare.cut;
            update.idx = compare.idx;
        }
    }

    KOKKOS_INLINE_FUNCTION
        void init(imbCut& update) const
    {
        update.imb = start_imb;
        update.cut = start_cut;
        update.idx = 0;
    }
};

struct scanVtxWgt
{
    vtx_view_t v_wgt, in_perm;
    Kokkos::View<int64_t*> imbScan;

    scanVtxWgt(vtx_view_t v_wgt,
        vtx_view_t in_perm,
        Kokkos::View<int64_t*> imbScan)
        : v_wgt(v_wgt)
        , in_perm(in_perm)
        , imbScan(imbScan) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const sgp_vid_t i, int64_t& update, const bool final) const
    {
        sgp_vid_t u = in_perm(i);
        if (final) {
            imbScan(i) = update;
        }
        update += 2*v_wgt(u);
    }
};

struct findInterleaveIdx
{

    int64_t target_imb;
    Kokkos::View<int64_t*> imbScan;

    findInterleaveIdx(int64_t target_imb,
        Kokkos::View<int64_t*> imbScan)
        : target_imb(target_imb)
        , imbScan(imbScan) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const sgp_vid_t i, sgp_vid_t& update) const
    {
        if (imbScan(i) >= target_imb) {
            if (i < update) {
                update = i;
            }
        }
    }

    //min reduction
    KOKKOS_INLINE_FUNCTION
        void join(volatile sgp_vid_t& update, volatile const sgp_vid_t& input) const {
        if (input < update) update = input;
    }

    KOKKOS_INLINE_FUNCTION
        void join(sgp_vid_t& update, const sgp_vid_t& input) const {
        if (input < update) update = input;
    }

    KOKKOS_INLINE_FUNCTION
        void init(sgp_vid_t& update) const {
        update = SGP_INFTY;
    }
};

struct interleaveParts
{

    vtx_view_t first_part, last_part, out_perm;

    sgp_vid_t interleave_loc;

    interleaveParts(vtx_view_t first_part,
        vtx_view_t last_part,
        vtx_view_t out_perm,
        sgp_vid_t interleave_loc)
        : first_part(first_part)
        , last_part(last_part)
        , out_perm(out_perm)
        , interleave_loc(interleave_loc) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const sgp_vid_t i, sgp_vid_t& update, const bool final) const
    {
        //both middle conditions can't be simultaneously true
        if (i < interleave_loc) {
            if (final) {
                out_perm(update) = first_part(i);
            }
            update += 1;
        }
        else if(i >= first_part.extent(0)) {
            if (final) {
                out_perm(update) = last_part(i - interleave_loc);
            }
            update += 1;
        }
        else if (i >= interleave_loc + last_part.extent(0)) {
            if (final) {
                out_perm(update) = first_part(i);
            }
            update += 1;
        }
        else {
            if (final) {
                out_perm(update) = last_part(i - interleave_loc);
                out_perm(update + 1) = first_part(i);
            }
            update += 2;
        }
    }
};

struct countPart
{
    vtx_view_t in_perm;

    eigenview_t part;
    double correct_part;

    countPart(vtx_view_t in_perm,
        eigenview_t part,
        double correct_part)
        : in_perm(in_perm)
        , part(part)
        , correct_part(correct_part) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const sgp_vid_t i, sgp_vid_t& update) const
    {
        sgp_vid_t u = in_perm(i);

        if (part(u) == correct_part) {
            update += 1;
        }
    }
};

struct sortPart
{
    vtx_view_t in_perm, out_perm;

    eigenview_t part;
    double correct_part;

    sortPart(vtx_view_t in_perm,
        vtx_view_t out_perm,
        eigenview_t part,
        double correct_part)
        : in_perm(in_perm)
        , out_perm(out_perm)
        , part(part)
        , correct_part(correct_part) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const sgp_vid_t i, sgp_vid_t& update, const bool final) const
    {
        sgp_vid_t u = in_perm(i);

        if (part(u) == correct_part) {
            if (final) {
                out_perm(update) = u;
            }
            update += 1;
        }
    }
};

vtx_view_t interleave_for_balance(const eigenview_t& partition, const vtx_view_t& rand_perm, const vtx_view_t& vtx_w, const int64_t start_imb, const sgp_vid_t n) {
    //move permutation into parts but respect ordering
    vtx_view_t part0, part1;
    sgp_vid_t part0_size = 0, part1_size = 0;
    countPart part0_count(rand_perm, partition, 0.0);
    Kokkos::parallel_reduce("find part 0 size", n, part0_count, part0_size);
    part0 = vtx_view_t("part 0", part0_size);
    sortPart part0_sort(rand_perm, part0, partition, 0.0);
    Kokkos::parallel_scan("move part 0", n, part0_sort);
    countPart part1_count(rand_perm, partition, 1.0);
    Kokkos::parallel_reduce("find part 0 size", n, part1_count, part1_size);
    part1 = vtx_view_t("part 1", part1_size);
    sortPart part1_sort(rand_perm, part1, partition, 1.0);
    Kokkos::parallel_scan("move part 1", n, part1_sort);
    vtx_view_t perm("swap order", n);

	/*sgp_vid_t x = 0, y = 0;
	int64_t imb = start_imb;
	while(x < part0_size || y < part1_size){
		if(x < part0_size && y < part1_size){
			if(imb > 0){
				perm(x + y) = part0(x);
				imb -= 2*vtx_w(part0(x));
				x++;
			} else {
				perm(x + y) = part1(y);
				imb += 2*vtx_w(part1(y));
				y++;
			}
		} else if(x < part0_size){
			perm(x + y) = part0(x);
			x++;
		} else {
			perm(x + y) = part1(y);
			y++;
		}
	}
	return perm;*/

    //compute location of interleave
    int64_t target_imb = abs(start_imb) / 10;
    if (target_imb < 20) {
        target_imb = 20;
    }
    target_imb = abs(start_imb) - target_imb;
    sgp_vid_t interleave = 0;
    if (target_imb > 0 && start_imb > 0) {
        Kokkos::View<int64_t*> imbalance_part("scan for imbalance reducer", part0_size);
        scanVtxWgt scan_imb(vtx_w, part0, imbalance_part);
        Kokkos::parallel_scan("scan imbalance", part0_size, scan_imb);
        findInterleaveIdx find_interleave(target_imb, imbalance_part);
        Kokkos::parallel_reduce("find interleave", part0_size, find_interleave, interleave);
    }
    else if (target_imb > 0) {
        Kokkos::View<int64_t*> imbalance_part("scan for imbalance reducer", part1_size);
        scanVtxWgt scan_imb(vtx_w, part1, imbalance_part);
        Kokkos::parallel_scan("scan imbalance", part1_size, scan_imb);
        findInterleaveIdx find_interleave(target_imb, imbalance_part);
        Kokkos::parallel_reduce("find interleave", part1_size, find_interleave, interleave);
    }

    sgp_vid_t interleave_size = 0;
    if (start_imb > 0) {
        interleaveParts interleave_parts = interleaveParts(part0, part1, perm, interleave);
        interleave_size = part1_size + interleave;
        if (part0_size > interleave_size) {
            interleave_size = part0_size;
        }
        Kokkos::parallel_scan("interleave parts", interleave_size, interleave_parts);
    }
    else {
        interleaveParts interleave_parts = interleaveParts(part1, part0, perm, interleave);
        interleave_size = part0_size + interleave;
        if (part1_size > interleave_size) {
            interleave_size = part1_size;
        }
        Kokkos::parallel_scan("interleave parts", interleave_size, interleave_parts);
    }

    return perm;
}

sgp_eid_t fm_refine_par(eigenview_t& partition, const matrix_type& g, const vtx_view_t& vtx_w, ExperimentLoggerUtil& experiment, const int level) {
    sgp_vid_t n = g.numRows();

    pool_t rand_pool(std::time(nullptr));
    //perm contains the order in which we will swap the vertices
    vtx_view_t start_perm("starting permutation", n);
    vtx_view_t chosen_at("reverse permutation", n);
    Kokkos::parallel_for("init reverse", n, KOKKOS_LAMBDA(const sgp_vid_t i){
        chosen_at(i) = SGP_INFTY;
    });

    int64_t start_imb = 0;
    Kokkos::parallel_reduce("calculate starting imb", n, KOKKOS_LAMBDA(const sgp_vid_t i, int64_t & local_imb){
        if (partition(i) == 0.0) {
            local_imb += vtx_w(i);
        }
        else {
            local_imb -= vtx_w(i);
        }
    }, start_imb);
    Kokkos::View<sgp_vid_t> chosen_total("chosen total");
    sgp_vid_t chosen = 0;
    do {
        vtx_view_t pre_commit_chosen("pre commit chosen", n);
        chosen = 0;
        //choose some vertices that have a good gain based on the currently chosen vtx
        Kokkos::parallel_reduce("choose some vtx", n, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t& local_sum){
            if (chosen_at(i) == SGP_INFTY) {
                int64_t gain = 0;
                for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                    sgp_vid_t v = g.graph.entries(j);
                    sgp_wgt_t wgt = g.values(j);
                    //has v been chosen
                    bool vPartSwapped = (chosen_at(v) < n);
                    if (partition(i) != partition(v)) {
                        if (vPartSwapped) {
                            gain -= wgt;
                        }
                        else {
                            gain += wgt;
                        }
                    }
                    else {
                        if (vPartSwapped) {
                            gain += wgt;
                        }
                        else {
                            gain -= wgt;
                        }
                    }
                }
                if (gain > 0) {
                    sgp_vid_t write_loc = Kokkos::atomic_fetch_add(&chosen_total(), 1);
                    start_perm(write_loc) = i;
                    pre_commit_chosen(i) = write_loc;
                    local_sum++;
                }
                else {
                    pre_commit_chosen(i) = SGP_INFTY;
                }
            } else {
                pre_commit_chosen(i) = SGP_INFTY;
			}
        }, chosen);

        //this can be optimized to be only on the chosen range
        Kokkos::parallel_for("commit chosen", n, KOKKOS_LAMBDA(const sgp_vid_t i){
            if (pre_commit_chosen(i) < n) {
                chosen_at(i) = pre_commit_chosen(i);
            }
        });
    } while (chosen > 10);

    //this might not be necessary if we only interleave on the chosen range
    Kokkos::parallel_scan("move rest of vtx", n, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t & update, const bool final){
        //i was not chosen
        if (chosen_at(i) >= n) {
            if (final) {
                start_perm(chosen_total() + update) = i;
            }
            update += 1;
        }
    });

    vtx_view_t mid_perm = interleave_for_balance(partition, start_perm, vtx_w, start_imb, n);

    sgp_vid_t chosen_host = 0;
    Kokkos::deep_copy(chosen_host, chosen_total);

    vtx_view_t end_perm("final permutation", n);
    Kokkos::deep_copy(chosen_total, 0);
    Kokkos::View<sgp_vid_t> end_write("end write");

    vtx_view_t reverse_map("reversed", n);
    Kokkos::parallel_for("construct reverse map", n, KOKKOS_LAMBDA(sgp_vid_t i) {
        reverse_map(mid_perm(i)) = i;
    });

    Kokkos::parallel_for("remove bad gains", n, KOKKOS_LAMBDA(const sgp_vid_t i){
		int64_t gain = 0, gainLost = 0;
        sgp_vid_t u = mid_perm(i);
		if(chosen_at(u) < n){
        for (sgp_eid_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
            sgp_vid_t v = g.graph.entries(j);
            sgp_wgt_t wgt = g.values(j);
            //has v been chosen
            bool vPartSwappedBefore = (reverse_map(v) < reverse_map(u));
            bool vPartSwappedAfter = !vPartSwappedBefore && (reverse_map(v) < chosen_host);
            if (partition(u) != partition(v)) {
                if (vPartSwappedBefore) {
                    gain -= wgt;
                }
                else {
                    gain += wgt;
                }
                if (vPartSwappedAfter) {
                    gainLost += wgt;
                }
            }
            else {
                if (vPartSwappedBefore) {
                    gain += wgt;
                }
                else {
                    gain -= wgt;
                }
                if (vPartSwappedAfter) {
                    gainLost -= wgt;
                }
            }
        }
        //will decide later what to do with gainLost
        if (gain > 0) {
            sgp_vid_t write_loc = Kokkos::atomic_fetch_add(&chosen_total(), 1);
            end_perm(write_loc) = u;
        }
        else {
            sgp_vid_t write_loc = n - 1 - Kokkos::atomic_fetch_add(&end_write(), 1);
            end_perm(write_loc) = u;
        }
		}
    });

    //this might not be necessary if we only interleave on the chosen range
    Kokkos::parallel_scan("move rest of vtx", n, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t& update, const bool final){
        //i was not chosen
        sgp_vid_t u = mid_perm(i);
        if (chosen_at(u) >= n) {
            if (final) {
                end_perm(chosen_total() + update) = u;
            }
            update += 1;
        }
    });

    vtx_view_t perm = interleave_for_balance(partition, end_perm, vtx_w, start_imb, n);

    Kokkos::parallel_for("construct reverse map", n, KOKKOS_LAMBDA(sgp_vid_t i) {
        reverse_map(perm(i)) = i;
    });

    Kokkos::View<int64_t*> gains("gains", n);
	Kokkos::View<int64_t*> scan_cuts("scan cuts", n);
    Kokkos::View<int64_t*> scan_imbs("scan imbs", n);
    int64_t start_cut = 0;

    Kokkos::parallel_reduce("calculate gains and starting cut", n, KOKKOS_LAMBDA(const sgp_vid_t i, int64_t & local_cut){
        int64_t gain = 0;
        sgp_eid_t cut = 0;
        for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
            sgp_vid_t v = g.graph.entries(j);
            sgp_wgt_t wgt = g.values(j);
            //will v be swapped before i is swapped?
            bool vPartSwapped = (reverse_map(v) < reverse_map(i));
            if (partition(i) != partition(v)) {
                cut += wgt;
                if (vPartSwapped) {
                    gain -= wgt;
                }
                else {
                    gain += wgt;
                }
            }
            else {
                if (vPartSwapped) {
                    gain += wgt;
                }
                else {
                    gain -= wgt;
                }
            }
        }
        gains(i) = 2*gain;
        local_cut += cut;
    }, start_cut);
    
	printf("Level %i; Unrefined balance: %li, cutsize: %li\n", level, start_imb, start_cut);

    Kokkos::parallel_scan("cut scan", n, KOKKOS_LAMBDA(const sgp_vid_t i, int64_t & update, const bool final){
        sgp_vid_t u = perm(i);
        //gain is the quantity the edge cut will decrease by
        update -= gains(u);
        if (final) {
            scan_cuts(i) = start_cut + update;
        }
    });

    Kokkos::parallel_scan("imb scan", n, KOKKOS_LAMBDA(const sgp_vid_t i, int64_t & update, const bool final){
        sgp_vid_t u = perm(i);
        //vtx is swapping partitions so the sign is swapped from start imb calculation and value is doubled
        if (partition(u) == 0.0) {
            update -= 2*vtx_w(u);
        }
        else {
            update += 2*vtx_w(u);
        }
        if (final) {
            scan_imbs(i) = start_imb + update;
        }
    });
/*	for(sgp_vid_t i = 0; i < n; i++){
		printf("perm(%u): %u, imb: %li, cut: %li\n", i, perm(i), scan_imbs(i), scan_cuts(i));
		printf("part(%u): %.1f\n", perm(i), partition(perm(i)));
	}
*/
    imbCut result;
	result.imb = abs(start_imb);
	result.cut = start_cut;
	result.idx = 0;
	Kokkos::View<imbCut> dev_result("dev result");
	Kokkos::deep_copy(dev_result, result);

	int64_t target_imb = abs(start_imb)/10;
	if(target_imb < 20) target_imb = 20;
    bestImbCut findBest(scan_cuts, scan_imbs, start_cut, start_imb, target_imb, dev_result, n - 1);

	/*for(sgp_vid_t i = 0; i < n; i++){
		bool choose = false;
		if(abs(scan_imbs(i)) <= target_imb && scan_cuts(i) < result.cut){
			choose = true;
		}
		else if(abs(scan_imbs(i)) > target_imb)
		{
			if (abs(scan_imbs(i)) < result.imb && scan_cuts(i) < 1.05 * result.cut) {
        		choose = true;
			}
        	else if (abs(scan_imbs(i)) <= result.imb && scan_cuts(i) <= result.cut) {
        		choose = true;
			}
		}
		if(choose){
			result.imb = abs(scan_imbs(i));
			result.cut = scan_cuts(i);
			result.idx = i + 1;
		}
	}*/
    Kokkos::parallel_scan("find best partition", n, findBest);
	Kokkos::deep_copy(result, dev_result);

	if(result.idx > 0){
    Kokkos::parallel_for("perform swaps", result.idx, KOKKOS_LAMBDA(const sgp_vid_t i){
        sgp_vid_t u = perm(i);
        if (partition(u) == 0.0) {
            partition(u) = 1.0;
        }
        else {
            partition(u) = 0.0;
        }
    });
	}

    printf("Level %i; Refined balance: %li, cutsize: %li, swaps: %u\n", level, result.imb, result.cut, result.idx);
    return result.cut;
}

//edge cuts are bounded by |E| of finest graph (assuming unweighted edges for finest graph)
//also we assume ALL coarse edge weights are integral
//this code is DISGUSTING, you've been warned
sgp_eid_t fm_refine(eigenview_t& partition_device, const matrix_type& g_device, const vtx_view_t& vtx_w_device, ExperimentLoggerUtil& experiment, const int level) {
    //return fm_refine_par(partition_device, g_device, vtx_w_device, experiment, level);
	
	sgp_vid_t n = g_device.numRows();

    sgp_eid_t maxE = 0;

    Kokkos::Timer timer;

    eigen_mirror_t partition = Kokkos::create_mirror(partition_device);
    Kokkos::deep_copy(partition, partition_device);

    vtx_mirror_t vtx_w = Kokkos::create_mirror(vtx_w_device);
    Kokkos::deep_copy(vtx_w, vtx_w_device);

    edge_mirror_t row_map = Kokkos::create_mirror(g_device.graph.row_map);
    vtx_mirror_t entries = Kokkos::create_mirror(g_device.graph.entries);
    wgt_mirror_t values = Kokkos::create_mirror(g_device.values);

    Kokkos::deep_copy(row_map, g_device.graph.row_map);
    Kokkos::deep_copy(entries, g_device.graph.entries);
    Kokkos::deep_copy(values, g_device.values);

    host_graph_t graph(entries, row_map);
    host_matrix_t g("interpolate", n, values, graph);

    Kokkos::parallel_reduce("max e find", policy(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member& thread, sgp_eid_t & t_max){
        sgp_vid_t i = thread.league_rank();
        sgp_eid_t weighted_degree = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, g_device.graph.row_map(i), g_device.graph.row_map(i + 1)), [=](const sgp_eid_t j, sgp_eid_t& local_sum) {
            local_sum += g_device.values(j);
        }, weighted_degree);
        Kokkos::single(Kokkos::PerTeam(thread), [&]() {
            if (t_max < weighted_degree) {
                t_max = weighted_degree;
            }
        });
    }, Kokkos::Max<sgp_eid_t, Kokkos::HostSpace>(maxE));

#ifdef _DEBUG
    printf("maxE: %u\n", maxE);
#endif

    sgp_eid_t cutsize = 0;
    vtx_mirror_t bucketsA, bucketsB, ll_next, ll_prev, free_vtx;
    Kokkos::View<int64_t*>::HostMirror gains;
    fm_create_ds(partition_device, g_device, vtx_w_device, maxE, cutsize, bucketsA, bucketsB, ll_next, ll_prev, free_vtx, gains);

    int64_t balance = 0;
    for (sgp_vid_t i = 0; i < n; i++) {
        if (partition(i) == 0) {
            balance += vtx_w(i);
        }
        else {
            balance -= vtx_w(i);
        }
    }
    Kokkos::View<int64_t*>::HostMirror balances("balances", n);
    vtx_mirror_t swap_order("swap order", n);
    edge_mirror_t cutsizes("cutsizes", n);

    //printf("Level: %i; Unrefined balance: %li, cutsize: %lu\n", level, balance, cutsize);

    int64_t start_balance = abs(balance);
    int64_t start_cut = cutsize;
    int64_t bucket_offsetA = 2 * maxE;
    int64_t bucket_offsetB = bucket_offsetA;
    sgp_vid_t total_swaps = 0;
    sgp_eid_t min_cut = start_cut;
    sgp_vid_t argmin = SGP_INFTY;
    int64_t min_imb = start_balance;
	int64_t target_imb = start_balance / 10;
	if(target_imb < 20){
		target_imb = 20;
	}
    bool start_counter = true;
    int counter = 0;
    while (bucket_offsetA >= 0 || bucket_offsetB >= 0) {
        sgp_vid_t swap_a = SGP_INFTY;
        if (bucket_offsetA >= 0) swap_a = bucketsA(bucket_offsetA);
        sgp_vid_t swap_b = SGP_INFTY; 
        if (bucket_offsetB >= 0) swap_b = bucketsB(bucket_offsetB);
        sgp_vid_t swap = SGP_INFTY;
        
        bool choose_a = false, choose_b = false;

        //select a vertex to swap and remove from datastructure
        if (swap_a != SGP_INFTY && swap_b == SGP_INFTY) {
            if (balance > 0 || bucket_offsetB < 0) {
                choose_a = true;
            }
            bucket_offsetB--;
        }
        else if (swap_a == SGP_INFTY && swap_b != SGP_INFTY) {
            if (balance < 0 || bucket_offsetA < 0) {
                choose_b = true;
            }
            bucket_offsetA--;
        }
        else if (swap_a != SGP_INFTY && swap_b != SGP_INFTY) {
            if (balance > 0) {
                choose_a = true;
            }
            else if (balance < 0) {
                choose_b = true;
            }
            else {
                if (bucket_offsetA > bucket_offsetB) {
                    choose_a = true;
                }
                else {
                    choose_b = true;
                }
            }
        }
        else {
            bucket_offsetA--;
            bucket_offsetB--;
        }

        if (choose_a) {
            swap = swap_a;
            sgp_vid_t next = ll_next(swap);
            bucketsA(bucket_offsetA) = next;
            if (next != SGP_INFTY) {
                ll_prev(next) = SGP_INFTY;
            }
        }
        else if (choose_b) {
            swap = swap_b;
            sgp_vid_t next = ll_next(swap);
            bucketsB(bucket_offsetB) = next;
            if (next != SGP_INFTY) {
                ll_prev(next) = SGP_INFTY;
            }
        }

        //swap and modify datastructure
        if (swap != SGP_INFTY) {
            //the gain only counts the outward edges, but in an undirected graphs these edges are duplicated from other vertices
            cutsize -= 2*gains(swap);
            if (partition(swap) == 0.0) {
                partition(swap) = 1.0;
                balance -= 2*vtx_w(swap);
            }
            else {
                partition(swap) = 0.0;
                balance += 2*vtx_w(swap);
            }
            free_vtx(swap) = 0;
            swap_order(total_swaps) = swap;
            cutsizes(total_swaps) = cutsize;
            balances(total_swaps) = balance;
            if(start_counter) counter++;

            //we ideally want both the cut and imbalance to improve
            //however we prioritize the balance, and bound how much the cut can decay by
			if(abs(balance) <= target_imb && cutsize <= min_cut){
                min_imb = abs(balance);
                min_cut = cutsize;
                argmin = total_swaps;
                start_counter = true;
                counter = 0;
			}
			else if(abs(balance) > target_imb)
			{
				if (abs(balance) < min_imb && cutsize < 1.05 * min_cut) {
            	    min_imb = abs(balance);
                	min_cut = cutsize;
                	argmin = total_swaps;
                	start_counter = true;
                	counter = 0;
            	}
            	else if (abs(balance) <= min_imb && cutsize <= min_cut) {
                	min_imb = abs(balance);
                	min_cut = cutsize;
                	argmin = total_swaps;
                	start_counter = true;
                	counter = 0;
            	}
			}

            total_swaps++;
            for (sgp_eid_t j = g.graph.row_map(swap); j < g.graph.row_map(swap + 1); j++) {
                sgp_vid_t v = g.graph.entries(j);
                if (free_vtx(v) == 1) {
                    //multiply by 2 because this edge already counted towards v's gain, now it must count in the opposite direction
                    int64_t gain_change = 2*g.values(j);
                    if (partition(swap) == partition(v)) {
                        gain_change = -gain_change;
                    }
                    //connect previous and last nodes of ll together
                    sgp_vid_t old_ll_next = ll_next(v);
                    sgp_vid_t old_ll_prev = ll_prev(v);
                    if (old_ll_next != SGP_INFTY) {
                        ll_prev(old_ll_next) = old_ll_prev;
                    }
                    if (old_ll_prev != SGP_INFTY) {
                        ll_next(old_ll_prev) = old_ll_next;
                    }

                    int64_t old_gain = gains(v);
                    int64_t next_gain = old_gain + gain_change;
                    gains(v) = next_gain;
                    //old_ll_next becomes new head of ll if v was old head
                    if (old_ll_prev == SGP_INFTY) {
                        if (partition(v) == 0.0) {
                            bucketsA(old_gain + maxE) = old_ll_next;
                        }
                        else {
                            bucketsB(old_gain + maxE) = old_ll_next;
                        }
                    }

                    //old head of ll that v will be inserted into
                    sgp_vid_t new_ll_head = bucketsA(next_gain + maxE);
                    if (partition(v) == 0.0) {
                        bucketsA(next_gain + maxE) = v;
                        //need to move the seek head back if some entries are being moved up
                        if (next_gain + maxE > bucket_offsetA) {
                            bucket_offsetA = next_gain + maxE;
                        }
                    }
                    else {
                        new_ll_head = bucketsB(next_gain + maxE);
                        bucketsB(next_gain + maxE) = v;
                        if (next_gain + maxE > bucket_offsetB) {
                            bucket_offsetB = next_gain + maxE;
                        }
                    }

                    if (new_ll_head != SGP_INFTY) {
                        ll_prev(new_ll_head) = v;
                    }
                    ll_next(v) = new_ll_head;
                    ll_prev(v) = SGP_INFTY;
                }
            }
            if (counter > 500) {
                bucket_offsetA = -1;
                bucket_offsetB = -1;
            }
        }
    }

    sgp_vid_t undo_from = 0;
    if (argmin != SGP_INFTY) {
        undo_from = argmin + 1;
    }

    for (sgp_vid_t i = undo_from; i < total_swaps; i++) {
        sgp_vid_t undo = swap_order(i);
        if (partition(undo) == 0.0) {
            partition(undo) = 1.0;
        }
        else {
            partition(undo) = 0.0;
        }
    }

    printf("Level %i; Unrefined balance: %li, cutsize: %li\n", level, balance, start_cut);
    printf("Level %i; Refined balance: %li, cutsize: %u\n", level, min_imb, min_cut);
    Kokkos::deep_copy(partition_device, partition);

    experiment.addMeasurement(ExperimentLoggerUtil::Measurement::FMRefine, timer.seconds());
    timer.reset();
    return min_cut;
}

eigenview_t init_gggp(const matrix_type& cg,
                      const vtx_view_t& c_vtx_w_device){
    printf("Doing GGGP\n");

    sgp_vid_t gc_n = cg.numRows();

    sgp_vid_t vtx_w_total = 0;

    vtx_mirror_t c_vtx_w = Kokkos::create_mirror(c_vtx_w_device);
    Kokkos::deep_copy(c_vtx_w, c_vtx_w_device);

    printf("coarse vertex count: %u\n", gc_n);
    for (sgp_vid_t i = 0; i < c_vtx_w.extent(0); i++) {
        vtx_w_total += c_vtx_w(i);
    }

    edge_mirror_t row_map = Kokkos::create_mirror(cg.graph.row_map);
    vtx_mirror_t entries = Kokkos::create_mirror(cg.graph.entries);
    wgt_mirror_t values = Kokkos::create_mirror(cg.values);

    Kokkos::deep_copy(row_map, cg.graph.row_map);
    Kokkos::deep_copy(entries, cg.graph.entries);
    Kokkos::deep_copy(values, cg.values);

    eigenview_t best_cg_part("best cg part", gc_n);
    eigen_mirror_t cg_m = Kokkos::create_mirror(best_cg_part);
    Kokkos::deep_copy(cg_m, best_cg_part);
    sgp_real_t cutmin = SGP_INFTY;
	
	for (sgp_vid_t i = 0; i < gc_n; i++) {
        //reset coarse partition
        for (sgp_vid_t j = 0; j < gc_n; j++) {
            cg_m(j) = 0;
        }
        cg_m(i) = 1;
        sgp_vid_t count = c_vtx_w(i);
        //incrementally grow partition 1
        while (count < vtx_w_total / 2) {
            sgp_vid_t argmin = i;
            sgp_real_t min = SGP_INFTY;
            //find minimum increase to cutsize for moving a vertex from partition 0 to partition 1
            for (sgp_vid_t u = 0; u < gc_n; u++) {
                if (cg_m(u) == 0) {
                    sgp_real_t cutLoss = 0;
                    for (sgp_eid_t j = row_map(u); j < row_map(u + 1); j++) {
                        sgp_vid_t v = entries(j);
                        if (cg_m(v) == 0) {
                            cutLoss += values(j);
                        }
                        else {
                            cutLoss -= values(j);
                        }
                    }
                    if (cutLoss < min) {
                        min = cutLoss;
                        argmin = u;
                    }
                }
            }
            cg_m(argmin) = 1;
            count += c_vtx_w(argmin);
        }
        sgp_real_t edge_cut = 0;
        //find total cutsize
        for (sgp_vid_t u = 0; u < gc_n; u++) {
            for (sgp_eid_t j = row_map(u); j < row_map(u + 1); j++) {
                sgp_vid_t v = entries(j);
                if (cg_m(v) != cg_m(u)) {
                    edge_cut += values(j);
                }
            }
        }
        //if cutsize less than best, replace best with current
        if (edge_cut < cutmin) {
            cutmin = edge_cut;
            Kokkos::deep_copy(best_cg_part, cg_m);
        }
    }
    return best_cg_part;
}

eigenview_t init_spectral(const matrix_type& cg,
                      const vtx_view_t& c_vtx_w){
    printf("Doing spectral\n");
    sgp_vid_t gc_n = cg.numRows();

    sgp_vid_t vtx_w_total = 0;

    printf("coarse vertex count: %u\n", gc_n);
    Kokkos::parallel_reduce(c_vtx_w.extent(0), KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t & thread_sum) {
        thread_sum += c_vtx_w(i);
    }, vtx_w_total);
    pool_t rand_pool(std::time(nullptr));
    eigenview_t coarse_guess("coarse_guess", gc_n);
    double max = (double)std::numeric_limits<uint64_t>::max();
    Kokkos::parallel_for(gc_n, KOKKOS_LAMBDA(sgp_vid_t i){
        gen_t generator = rand_pool.get_state();
        coarse_guess(i) = ((double)generator.urand64()) / max;
        coarse_guess(i) = 2.0 * coarse_guess(i) - 1.0;
        rand_pool.free_state(generator);
    });
    sgp_vec_normalize_kokkos(coarse_guess, gc_n);
    ExperimentLoggerUtil throwaway;
    sgp_power_iter(coarse_guess, cg, 0, 1
        , throwaway);

    vtx_view_t perm_d = sort_order<sgp_real_t, sgp_vid_t>(coarse_guess, 1.0, -1.0);
    vtx_mirror_t perm = Kokkos::create_mirror(perm_d);
    Kokkos::deep_copy(perm, perm_d);

    vtx_mirror_t c_vtx_m = Kokkos::create_mirror(c_vtx_w);
    Kokkos::deep_copy(c_vtx_m, c_vtx_w);

    sgp_vid_t sum = 0;
    eigen_mirror_t cg_m("coarse guess discretized mirror", gc_n);
    for(sgp_vid_t i = 0; i < gc_n; i++){
        sgp_vid_t u = perm(i);
        if(sum < vtx_w_total / 2){
            sum += c_vtx_m(u);
            cg_m(u) = 1.0;
        } else {
            cg_m(u) = 0.0;
        }
    }
    Kokkos::deep_copy(coarse_guess, cg_m);
    return coarse_guess;
}


eigenview_t sgp_recoarsen_one_level(const matrix_type& g,
    const vtx_view_t& f_vtx_w,
    const eigenview_t partition,
    sgp_eid_t min_cut, sgp_eid_t& out_cut, int refine_layer, 
    ExperimentLoggerUtil& experiment, bool mt, bool auto_replace = false, bool top = true) {

    sgp_eid_t last_cut = min_cut;
    eigenview_t fine_part = partition;
    printf("recoarsening level %d\n", refine_layer);
    bool was_top = top;
	while(true){
        matrix_type gc;
        matrix_type interpolation_graph;
        sgp_vid_t nvertices_coarse;
        vtx_view_t c_vtx_w;
        Kokkos::Timer timer;
        ExperimentLoggerUtil throwaway;
		if(mt){
        	sgp_recoarsen_match(interpolation_graph, &nvertices_coarse, g, fine_part, throwaway);
		} else {
        	sgp_recoarsen_HEC(interpolation_graph, &nvertices_coarse, g, fine_part);
		}

        sgp_build_coarse_graph(gc, c_vtx_w, interpolation_graph, g, f_vtx_w, 2, throwaway);
        experiment.addMeasurement(ExperimentLoggerUtil::Measurement::FMRecoarsen, timer.seconds());
        timer.reset();

        eigenview_t coarse_part("coarser partition", nvertices_coarse);
        if(gc.numRows() < 10){
           printf("stop recoarsening level %d\n", refine_layer);
           return fine_part;
        }
        if(gc.numRows() < 30 && auto_replace){
            coarse_part = init_gggp(gc, c_vtx_w);
            sgp_eid_t last_cut2 = min_cut;
            do{
                last_cut2 = min_cut;
                min_cut = fm_refine(coarse_part, gc, c_vtx_w, experiment, refine_layer + 1);
            } while(last_cut2 != min_cut);
                eigenview_t fine_recoarsened("new fine partition", g.numRows());
                KokkosSparse::spmv("N", 1.0, interpolation_graph, coarse_part, 0.0, fine_recoarsened);
                fine_part = fine_recoarsened;
            out_cut = min_cut;
            printf("stop recoarsening level %d\n", refine_layer);
            return fine_part;
        } else if (gc.numRows() < 30 || gc.numRows() == g.numRows()) {
            printf("stop recoarsening level %d\n", refine_layer);
            return fine_part;
        }
        Kokkos::parallel_for("create coarse partition", g.numRows(), KOKKOS_LAMBDA(sgp_vid_t i) {
            sgp_vid_t coarse_vtx = interpolation_graph.graph.entries(i);
            double part_wgt = static_cast<double>(f_vtx_w(i)) / static_cast<double>(c_vtx_w(coarse_vtx));
            Kokkos::atomic_add(&coarse_part(coarse_vtx), part_wgt * fine_part(i));
        });
        //not strictly necessary but you never know with floating point rounding errors
        Kokkos::parallel_for("discretize partition", nvertices_coarse, KOKKOS_LAMBDA(sgp_vid_t i) {
            coarse_part(i) = round(coarse_part(i));
        });

        sgp_eid_t last_cut2 = min_cut;
        do{
            last_cut2 = min_cut;
            min_cut = fm_refine(coarse_part, gc, c_vtx_w, experiment, refine_layer + 1);
        } while(last_cut2 != min_cut);
        if(top){
            //explore an entirely new partition
            coarse_part = sgp_recoarsen_one_level(gc, c_vtx_w, coarse_part, min_cut, min_cut, refine_layer + 1, experiment, mt, true, false);
            do{
                last_cut2 = min_cut;
                min_cut = fm_refine(coarse_part, gc, c_vtx_w, experiment, refine_layer + 1);
            } while(last_cut2 != min_cut);
            //refine the new partition
            coarse_part = sgp_recoarsen_one_level(gc, c_vtx_w, coarse_part, min_cut, min_cut, refine_layer + 1, experiment, mt, false, false);
            do{
                last_cut2 = min_cut;
                min_cut = fm_refine(coarse_part, gc, c_vtx_w, experiment, refine_layer + 1);
            } while(last_cut2 != min_cut);
        } else {
            //continue doing whatever we're doing
            coarse_part = sgp_recoarsen_one_level(gc, c_vtx_w, coarse_part, min_cut, min_cut, refine_layer + 1, experiment, mt, auto_replace, false);
            do{
                last_cut2 = min_cut;
                min_cut = fm_refine(coarse_part, gc, c_vtx_w, experiment, refine_layer + 1);
            } while(last_cut2 != min_cut);
        }
        eigenview_t fine_recoarsened("new fine partition", g.numRows());
        KokkosSparse::spmv("N", 1.0, interpolation_graph, coarse_part, 0.0, fine_recoarsened);
        do{
            last_cut2 = min_cut;
            min_cut = fm_refine(fine_recoarsened, g, f_vtx_w, experiment, refine_layer);
        } while(last_cut2 != min_cut);
        //partition can coarsen to become incredibly unbalanced, this is an easy way to detect max imbalance
        if(min_cut == 0){
            return fine_part;
        }
        if(auto_replace || min_cut < last_cut){
            fine_part = fine_recoarsened;
            out_cut = min_cut;
        }
		//the closer tol is to 1, the smaller that the cut decrease must be before quitting
		double tol = 0.99;
		if(refine_layer >= 1 && refine_layer <= 1){
			tol = 0.999;
		} else if(refine_layer > 1){
			tol = 0.9999;
		}
        if(!top && min_cut > tol*last_cut) {
            printf("stop recoarsening level %d\n", refine_layer);
            return fine_part;
        }
        //only explore on first run
        if(was_top && min_cut < tol*last_cut){
			top = true;
		} else {
			top = false;
		}
        if(auto_replace || min_cut < last_cut){
			last_cut = min_cut;
        }
    }
}

SGPAR_API int sgp_eigensolve(sgp_real_t* eigenvec, std::list<matrix_type>& graphs, std::list<matrix_type>& interpolates, std::list<vtx_view_t>& vtx_weights, sgp_pcg32_random_t* rng, int refine_alg
    , ExperimentLoggerUtil& experiment) {

    sgp_vid_t gc_n = graphs.rbegin()->numRows();
    eigenview_t coarse_guess("coarse_guess", gc_n);
    eigenview_t::HostMirror cg_m = Kokkos::create_mirror(coarse_guess);
    //randomly initialize guess eigenvector for coarsest graph
#ifndef FM
    for (sgp_vid_t i = 0; i < gc_n; i++) {
        cg_m(i) = ((double)sgp_pcg32_random_r(rng)) / UINT32_MAX;
    }
    Kokkos::deep_copy(coarse_guess, cg_m);
    sgp_vec_normalize_kokkos(coarse_guess, gc_n);
#else
    matrix_type cg = *graphs.rbegin();
    vtx_view_t c_vtx_w = *vtx_weights.rbegin();

	if(gc_n > 200){
    	coarse_guess = init_spectral(cg, c_vtx_w);
	} else {
		coarse_guess = init_gggp(cg, c_vtx_w);
	}
#endif


    auto graph_iter = graphs.rbegin(), interp_iter = interpolates.rbegin();
    auto vtx_w_iter = vtx_weights.rbegin();
    auto end = --graphs.rend();

    int refine_layer = graphs.size();
    //there is always one more refinement than interpolation
#ifdef FM
	sgp_eid_t cutsize = SGP_INFTY;
#endif
    while (graph_iter != end) {
        //refine
#ifdef FM
        printf("Refining layer %i\n", refine_layer);
        refine_layer--;
        sgp_eid_t old_cutsize = cutsize;
        do {
            old_cutsize = cutsize;
            cutsize = fm_refine(coarse_guess, *graph_iter, *vtx_w_iter, experiment, refine_layer);
        } while (cutsize != old_cutsize); //could be larger if the balance improved
#ifdef RECOARSEN
        coarse_guess = sgp_recoarsen_one_level(*graph_iter, *vtx_w_iter, coarse_guess, cutsize, cutsize, refine_layer, experiment, false);
        coarse_guess = sgp_recoarsen_one_level(*graph_iter, *vtx_w_iter, coarse_guess, cutsize, cutsize, refine_layer, experiment, true);
		do {
            old_cutsize = cutsize;
            cutsize = fm_refine(coarse_guess, *graph_iter, *vtx_w_iter, experiment, refine_layer);
        } while (cutsize != old_cutsize); //could be larger if the balance improved
#endif
#else
        CHECK_SGPAR(sgp_power_iter(coarse_guess, *graph_iter, refine_alg, 0
#ifdef EXPERIMENT
            , experiment
#endif
            ));
#endif
        graph_iter++;
        vtx_w_iter++;

        //interpolate
        eigenview_t fine_vec("fine vec", graph_iter->numRows());
        KokkosSparse::spmv("N", 1.0, *interp_iter, coarse_guess, 0.0, fine_vec);
        coarse_guess = fine_vec;
        interp_iter++;
    }

#ifdef FM
    printf("Refining layer %i\n", refine_layer);
    refine_layer--;
    sgp_eid_t old_cutsize = cutsize;
    do {
        old_cutsize = cutsize;
        cutsize = fm_refine(coarse_guess, *graph_iter, *vtx_w_iter, experiment, 0);
    } while (cutsize != old_cutsize); //could be larger if the balance improved
#ifdef RECOARSEN
    coarse_guess = sgp_recoarsen_one_level(*graph_iter, *vtx_w_iter, coarse_guess, cutsize, cutsize, refine_layer, experiment, false);
    coarse_guess = sgp_recoarsen_one_level(*graph_iter, *vtx_w_iter, coarse_guess, cutsize, cutsize, refine_layer, experiment, true);
	do {
        old_cutsize = cutsize;
        cutsize = fm_refine(coarse_guess, *graph_iter, *vtx_w_iter, experiment, 0);
    } while (cutsize != old_cutsize); //could be larger if the balance improved
#endif
#else
    //last refine
    CHECK_SGPAR(sgp_power_iter(coarse_guess, *graph_iter, refine_alg, 1
#ifdef EXPERIMENT
        , experiment
#endif
        ));
#endif

    eigenview_t::HostMirror eigenmirror = Kokkos::create_mirror(coarse_guess);
    Kokkos::deep_copy(eigenmirror, coarse_guess);

    Kokkos::parallel_for(host_policy(0, graph_iter->numRows()), KOKKOS_LAMBDA(sgp_vid_t i) {
        eigenvec[i] = eigenmirror(i);
    });

    return EXIT_SUCCESS;
}

}
}
