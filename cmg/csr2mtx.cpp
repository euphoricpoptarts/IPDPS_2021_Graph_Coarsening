#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

using sgp_vid_t = uint32_t;
using sgp_eid_t = uint32_t;
using sgp_wgt_t = sgp_vid_t;
typedef struct {
    sgp_vid_t   nvertices;
    sgp_eid_t   nedges;
    sgp_eid_t* source_offsets;
    sgp_vid_t* edges_per_source;
    sgp_vid_t* destination_indices;
    sgp_wgt_t* weighted_degree;
    sgp_wgt_t* eweights;
} sgp_graph_t;

int sgp_load_graph(sgp_graph_t *g, char *csr_filename) {

    FILE *infp = fopen(csr_filename, "rb");
    if (infp == NULL) {
        printf("Error: Could not open input file. Exiting ...\n");
        return 1;
    }
    long n, m;
    long unused_vals[4];
    fread(&n, sizeof(long), 1, infp);
    fread(&m, sizeof(long), 1, infp);
    fread(unused_vals, sizeof(long), 4, infp);
    g->nvertices = n;
    g->nedges = m/2;
    g->source_offsets = (sgp_eid_t *) malloc((g->nvertices+1)*sizeof(sgp_eid_t));
    g->destination_indices = (sgp_vid_t *) malloc(2*g->nedges*sizeof(sgp_vid_t));
    size_t nitems_read = fread(g->source_offsets, sizeof(sgp_eid_t), g->nvertices+1, infp);
    nitems_read = fread(g->destination_indices, sizeof(sgp_vid_t), 2*g->nedges, infp);
    fclose(infp);
    g->eweights = NULL;
    g->weighted_degree = NULL;
    g->edges_per_source = NULL;
    return 0;
}

int write_g(sgp_graph_t g, char *out_f){
	std::ostringstream out_s;
	sgp_vid_t n = g.nvertices;
	for(sgp_vid_t u = 0; u < n; u++)
	{
		for(sgp_eid_t j = g.source_offsets[u]; j < g.source_offsets[u + 1]; j++){
			sgp_vid_t v = g.destination_indices[j];
			if(u > v) {
				out_s << (u + 1) << " " << (v + 1) << std::endl;
			}
		}
	}
	std::ofstream out(out_f);
	out << "%%MatrixMarket matrix coordinate pattern symmetric" << std::endl;
	out << n << " " << n << " " << (g.source_offsets[n]/2) << std::endl;
	out << out_s.str();
	out.close();
	return 0;
}

int main(int argc, char** argv){
	if(argc < 3){
		printf("Usage: ./exe in out\n");
	}
	char* in_f = argv[1];
	char* out_f = argv[2];

	sgp_graph_t g;
	if(sgp_load_graph(&g, in_f)) return 1;

	return write_g(g, out_f);
}
