#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <cassert>

using namespace std;

int main(int argc, char *argv[]) {

    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <MM filename> <binary CSR filename>\n";
        exit(1);
    }

    char* inFileName = argv[1];
    FILE* inFilePtr = fopen(inFileName, "r");
    if (inFilePtr == NULL) {
        cout << "Error: could not open input file " << argv[1] << "! Exiting" << endl;
        exit(1);
    }

    char line[2048];
    uint64_t lineCount = 0;

    uint64_t N = 0;
    uint64_t M = 0;
    uint64_t N1 = 0;
    uint64_t N2 = 0;

    // default is symmetric, otherwise create a bipartite graph
    int symmetricFormat  = 1;

    // default is pattern, otherwise read real weights and discard them
    int patternFormat    = 1;

    // Initially storing adjacencies in vectors
    vector< vector<uint64_t> > AdjVector;

    // Read the file
    while (fgets(line, sizeof(line), inFilePtr) != NULL) {
        if (line[0] == '%') {
            // header
            if (line[1] == '%') {
                string str(line);
                string headerWord("coordinate");
                size_t found;
                found = str.find(headerWord);
                if (found == string::npos) {
                    cout << "Error: File is not in coordinate format! Exiting." << endl;
                    exit(1);
                }

                headerWord = string("general");
                found = str.find(headerWord);
                if (found != string::npos) {
                    cout << "Non-zero pattern is not symmetric." << endl;
                    symmetricFormat = 0;
                }

                headerWord = string("pattern");
                found = str.find(headerWord);
                if (found == string::npos) {
                    headerWord = string("real");
                    found = str.find(headerWord);
                    if (found != string::npos) {
                        cout << "Matrix has real weights, ignoring them." << endl;
                        patternFormat = 0;
                    }
                }
            }
            // Skip comment lines
            continue;
        }

        lineCount++;

        if (lineCount == 1) {
            sscanf(line, "%ld %ld %ld\n", &N1, &N2, &M);
            cout << "Matrix size: " << N1 << " " << N2 << " " << M << endl;
            if (symmetricFormat) {
                assert(N1 == N2);
                N = N1;
            } else {
                if (N1 == N2) {
                    // read A+A'
                    symmetricFormat = 1;
                    cout << "Reading general matrix as an undirected graph." << endl;
                    N = N1;
                } else {
                    cout << "Reading general matrix as a bipartite graph." << endl;
                    N = N1+N2;
                }
            }
            AdjVector.resize(N);
        } else {
            uint64_t u;
            uint64_t v;
            if (patternFormat) {
                sscanf(line, "%lu %lu\n", &u, &v);
            } else {
                double w;
                sscanf(line, "%lu %lu %lf\n", &u, &v, &w);
            }
            // cout << u << " " << v << endl;
            assert(u > 0); assert(v > 0);
            // filter self loops
            if (symmetricFormat) {
                if (u != v)  {
                    // store 0-indexed vertex ids
                    AdjVector[u-1].push_back(v-1);
                    AdjVector[v-1].push_back(u-1);
                }
            } else {
                u += N2;
                AdjVector[u-1].push_back(v-1);
                AdjVector[v-1].push_back(u-1);
            }
        }
    }
    fclose(inFilePtr);
    cout << lineCount-1 << " lines read from file. Non-zero count is given to be " << M << "." << endl;

    // Sort the adjacencies of each vertex
    for (uint64_t i=0; i<N; i++) {
        sort(AdjVector[i].begin(), AdjVector[i].end());
    }

    // Remove parallel edges
    for (uint64_t i=0; i<N; i++)  {
        vector<uint64_t> AdjVectorNoDup;
        if (AdjVector[i].size() > 1) {
            uint64_t prevVtx = AdjVector[i][0];
            AdjVectorNoDup.push_back(prevVtx);
            for (uint64_t len = 1; len<AdjVector[i].size(); len++) {
                uint64_t currVtx = AdjVector[i][len];
                if (currVtx != prevVtx) {
                    prevVtx = currVtx;
                    AdjVectorNoDup.push_back(prevVtx);
                }
            }
            AdjVector[i] = AdjVectorNoDup;
        }
    }

    // Sort the adjacencies again
    for (uint64_t i=0; i<N; i++) {
        sort(AdjVector[i].begin(), AdjVector[i].end());
    }

    // Get edge count
    M = 0;
    for (uint64_t i=0; i<N; i++) {
        M += AdjVector[i].size();
    }
    cout << "After deduplication and self loop removal, n: " << N << ", m: " << M/2 << endl;

    // Identify largest connected component
    uint64_t numComps = 0;
    vector<uint64_t> CompID(N, 0);
    vector<uint64_t> S(N);
    // vector<unsigned int> compSizes(N+1, 0);
    // compSizes[0] = 0;
    uint64_t largestCompSize = 0;
    uint64_t largestCompID   = 0;
 
    for (uint64_t i=0; i<N; i++) {
        if (CompID[i] != 0) {
            continue;
        }

        // Do a BFS from vertex i
        numComps++;       
        S[0] = i;
        CompID[i] = numComps;
        uint64_t currentPosS = 0;
        uint64_t visitedCount = 1;

        while (currentPosS != visitedCount) {
            uint64_t currentVert = S[currentPosS];
            for (uint64_t j=0; j<AdjVector[currentVert].size(); j++) {
                uint64_t v = AdjVector[currentVert][j];
                if (CompID[v] == 0) {
                    S[visitedCount++] = v;
                    CompID[v] = numComps;
                }
            }
            currentPosS++;
        }
        // compSizes[numComps] = visitedCount;
        if (visitedCount > largestCompSize) {
            largestCompSize = visitedCount;
            largestCompID = numComps;
        }
    }

    if (numComps == 1) {
        cout << "There is just one connected component." << endl;
    } else {
        cout << "There are " << numComps << " components and the largest component size is " 
             << largestCompSize << endl;
    }

    uint64_t newID = 1;
    uint64_t newM = 0;
    for (uint64_t i=0; i<N; i++) {
        if (CompID[i] == largestCompID) {
            CompID[i] = newID++;
            newM += AdjVector[i].size();
        } else {
            CompID[i] = 0;
        }
    }
    assert((newID-1) == largestCompSize);
    
    // Store old-to-new vertex mapping to disk
    /*
    if (numComps > 1) {
        ofstream ofs((string(argv[1])+".vmapping").c_str(), std::ofstream::out);
        for (long i=0; i<N; i++) {
            ofs << CompID[i] << endl;
        }
        ofs.close();
    }
    */

    // Get final Adj vectors and offsets
    uint64_t Nold = N;
    N = largestCompSize;
    cout << "Old edge count: " << M/2 << ", new edge count: " << newM/2 << endl; 
    M = newM;
    uint64_t* num_edges = new uint64_t[N+1];
    uint64_t* adj = new uint64_t[M];

    // Update num_edges array
    num_edges[0] = 0;
    for (uint64_t i=0; i<Nold; i++) {
        if (CompID[i] != 0) {
            uint64_t u = CompID[i]-1;
            num_edges[u+1] = num_edges[u] + AdjVector[i].size();
        }
    }
    assert(num_edges[N] == newM);

    // Update adj array 
    for (uint64_t i=0; i<Nold; i++) {
        if (CompID[i] != 0) {
            uint64_t u = CompID[i]-1;
            for (size_t j=0; j<AdjVector[i].size(); j++) {
                uint64_t v = CompID[AdjVector[i][j]]-1;
                uint64_t adjPos = num_edges[u] + j;
                assert(adjPos < M);
                adj[adjPos] = v;
            }
        }
    }

    // Write to CSR file
    char* outFileName = argv[2];

    cout << "Writing binary file to disk." << endl;
    FILE* writeBinaryPtr = fopen(outFileName, "wb");
    if (writeBinaryPtr == NULL) {
        cout << "Error: could not open output CSR file " << argv[2] << "! Exiting" << endl;
        exit(1);
    }

    long undirected = 1;
    long verification_graph = 0;
    long graph_type = 0;
    long one_indexed = 1;

    fwrite(&N, sizeof(uint64_t), 1, writeBinaryPtr);
    fwrite(&M, sizeof(uint64_t), 1, writeBinaryPtr);
    fwrite(&undirected, sizeof(long), 1, writeBinaryPtr);
    fwrite(&graph_type, sizeof(long), 1, writeBinaryPtr);
    fwrite(&one_indexed, sizeof(long), 1, writeBinaryPtr);
    fwrite(&verification_graph, sizeof(long), 1, writeBinaryPtr);
    fwrite(&num_edges[0], sizeof(uint64_t), N+1, writeBinaryPtr);
    fwrite(&adj[0], sizeof(uint64_t), M, writeBinaryPtr);
    fclose(writeBinaryPtr);

    /*
    for (long i=0; i<N; i++) {
        for (unsigned int j=num_edges[i]; j<num_edges[i+1]; j++) {
            cout << i+1 << " " << adj[j]+1 << endl;
        }
    }
    */

    delete [] num_edges;
    delete [] adj;

    return 0;
}
