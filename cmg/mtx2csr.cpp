#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <cassert>
#include <sstream>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char *argv[]) {

    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <MM filename> <binary CSR filename>\n";
        exit(1);
    }

    char* inFileName = argv[1];
	ifstream inFile(inFileName);
    if (!inFile.is_open()) {
        cout << "Error: could not open input file " << argv[1] << "! Exiting" << endl;
        exit(1);
    }
	std::stringstream buffer;
	buffer << inFile.rdbuf();
	inFile.close();

	string lineS;
    const char * line;
    long lineCount = 0;

    long N = 0;
    long M = 0;
    long N1 = 0;
    long N2 = 0;

    // default is symmetric, otherwise create a bipartite graph
    int symmetricFormat  = 1;

    // default is pattern, otherwise read real weights and discard them
    int patternFormat    = 1;

    // Initially storing adjacencies in vectors
    vector< vector<unsigned int> > AdjVector;

    // Read the file
    while (getline(buffer, lineS)) {
		line = lineS.c_str();
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
            unsigned int u;
            unsigned int v;
            if (patternFormat) {
                sscanf(line, "%u %u\n", &u, &v);
            } else {
                double w;
                sscanf(line, "%u %u %lf\n", &u, &v, &w);
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
    cout << lineCount-1 << " lines read from file. Non-zero count is given to be " << M << "." << endl;

    // Sort the adjacencies of each vertex
    for (long i=0; i<N; i++) {
        sort(AdjVector[i].begin(), AdjVector[i].end());
    }

    // Remove parallel edges
    for (long i=0; i<N; i++)  {
        vector<unsigned int> AdjVectorNoDup;
        if (AdjVector[i].size() > 1) {
            unsigned int prevVtx = AdjVector[i][0];
            AdjVectorNoDup.push_back(prevVtx);
            for (unsigned int len = 1; len<AdjVector[i].size(); len++) {
                unsigned int currVtx = AdjVector[i][len];
                if (currVtx != prevVtx) {
                    prevVtx = currVtx;
                    AdjVectorNoDup.push_back(prevVtx);
                }
            }
            AdjVector[i] = AdjVectorNoDup;
        }
    }

    // Sort the adjacencies again
    for (long i=0; i<N; i++) {
        sort(AdjVector[i].begin(), AdjVector[i].end());
    }

    // Get edge count
    M = 0;
    for (int i=0; i<N; i++) {
        M += AdjVector[i].size();
    }
    cout << "After deduplication and self loop removal, n: " << N << ", m: " << M/2 << endl;

    // Identify largest connected component
    unsigned int numComps = 0;
    vector<unsigned int> CompID(N, 0);
    vector<unsigned int> S(N);
    // vector<unsigned int> compSizes(N+1, 0);
    // compSizes[0] = 0;
    unsigned int largestCompSize = 0;
    unsigned int largestCompID   = 0;
 
    for (long i=0; i<N; i++) {
        if (CompID[i] != 0) {
            continue;
        }

        // Do a BFS from vertex i
        numComps++;       
        S[0] = i;
        CompID[i] = numComps;
        unsigned int currentPosS = 0;
        unsigned int visitedCount = 1;

        while (currentPosS != visitedCount) {
            unsigned int currentVert = S[currentPosS];
            for (unsigned int j=0; j<AdjVector[currentVert].size(); j++) {
                unsigned int v = AdjVector[currentVert][j];
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

    unsigned newID = 1;
    unsigned newM = 0;
    for (long i=0; i<N; i++) {
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
    long Nold = N;
    N = largestCompSize;
    cout << "Old edge count: " << M/2 << ", new edge count: " << newM/2 << endl; 
    M = newM;
    unsigned int* num_edges = new unsigned int[N+1];
    unsigned int* adj = new unsigned int[M];

    // Update num_edges array
    num_edges[0] = 0;
    for (long i=0; i<Nold; i++) {
        if (CompID[i] != 0) {
            unsigned int u = CompID[i]-1;
            num_edges[u+1] = num_edges[u] + AdjVector[i].size();
        }
    }
    assert(num_edges[N] == newM);

    // Update adj array 
    for (long i=0; i<Nold; i++) {
        if (CompID[i] != 0) {
            unsigned int u = CompID[i]-1;
            for (size_t j=0; j<AdjVector[i].size(); j++) {
                unsigned int v = CompID[AdjVector[i][j]]-1;
                unsigned int adjPos = num_edges[u] + j;
                assert(((long) adjPos) < M);
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

    fwrite(&N, sizeof(long), 1, writeBinaryPtr);
    fwrite(&M, sizeof(long), 1, writeBinaryPtr);
    fwrite(&undirected, sizeof(long), 1, writeBinaryPtr);
    fwrite(&graph_type, sizeof(long), 1, writeBinaryPtr);
    fwrite(&one_indexed, sizeof(long), 1, writeBinaryPtr);
    fwrite(&verification_graph, sizeof(long), 1, writeBinaryPtr);
    fwrite(&num_edges[0], sizeof(unsigned int), N+1, writeBinaryPtr);
    fwrite(&adj[0], sizeof(unsigned int), M, writeBinaryPtr);
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
