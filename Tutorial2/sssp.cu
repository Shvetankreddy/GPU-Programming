#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

const  int INF = INT_MAX;

struct Graph {
    int V;
    int E;
    vector<int> rowPtr;
    vector<int> col;
    vector<unsigned int> weight;
};

void readGraph(Graph &G){
    ifstream fin("Graph.txt");
    if (!fin) {
        cout << "Error opening Graph.txt\n";
        return;
    }

    fin >> G.V >> G.E;

    vector<int> U(G.E); vector<int> V(G.E); vector<int> W(G.E);

    for (int i = 0; i < G.E; i++)
        fin >> U[i] >> V[i] >> W[i];

    G.rowPtr.assign(G.V + 1, 0);
    G.col.resize(G.E);
    G.weight.resize(G.E);

    for (int i = 0; i < G.E; i++)
        G.rowPtr[U[i] + 1]++;

    for (int i = 1; i <= G.V; i++)
        G.rowPtr[i] += G.rowPtr[i - 1];

    vector<int> pos = G.rowPtr;

    for (int i = 0; i < G.E; i++) {
        int idx = pos[U[i]]++;
        G.col[idx] = V[i];
        G.weight[idx] = W[i];
    }
}

__global__ void SSSPKernel(int V,int *rowPtr,int *col,unsigned int *weight,unsigned int *dist){
    
    int v = blockIdx.x * blockDim.x + threadIdx.x;
        if (v >= V) return;

        if (dist[v] == INF) return;

    for (int e = rowPtr[v]; e < rowPtr[v + 1]; e++) {
        int u = col[e];
        unsigned int newDist = dist[v] + weight[e];
        atomicMin(&dist[u], newDist);
    }
}

int main(){
    Graph G;

    readGraph(G);

    int *d_rowPtr;
    int *d_col;
    unsigned int *d_weight;
    unsigned int *d_dist;

    cudaMalloc(&d_rowPtr, (G.V + 1) * sizeof(int));
    cudaMalloc(&d_col, G.E * sizeof(int));
    cudaMalloc(&d_weight, G.E * sizeof(unsigned int));
    cudaMalloc(&d_dist, G.V * sizeof(unsigned int));

    cudaMemcpy(d_rowPtr,G.rowPtr.data(),(G.V + 1) * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_col,G.col.data(),G.E * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_weight,G.weight.data(),G.E * sizeof(unsigned int),cudaMemcpyHostToDevice);

    vector<unsigned int> dist(G.V, INF);
    dist[0] = 0;

    cudaMemcpy(d_dist,dist.data(),G.V * sizeof(unsigned int),cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (G.V + blockSize - 1) / blockSize;

    for (int i = 0; i < G.V - 1; i++) {
        SSSPKernel<<<gridSize, blockSize>>>(G.V,d_rowPtr,d_col,d_weight,d_dist);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(dist.data(),d_dist,G.V * sizeof(unsigned int),cudaMemcpyDeviceToHost);

    for (int i = 0; i < G.V; i++) {
        if (dist[i] == INF)
            cout << "INF\n";
        else
            cout << dist[i] << '\n';
    }

    cudaFree(d_rowPtr);
    cudaFree(d_col);
    cudaFree(d_weight);
    cudaFree(d_dist);

    return 0;
}