#include <bits/stdc++.h>
using namespace std;

const int INF = INT_MAX;

struct Edge {
    int u;
    int v;
    int weight;
};

void readGraph(int &V, int &E, vector<Edge> &edges){
    ifstream fin("Graph.txt");

    if (!fin) {
        cerr << "Error: Could not open Graph.txt\n";
        return ;
    }

    fin >> V >> E;

    edges.resize(E);

    for (int i = 0; i < E; i++) {
        fin >> edges[i].u>> edges[i].v>> edges[i].weight;
    }

    fin.close();
}

void ssspCPU(int V,vector<Edge> &edges,int source,vector<int> &dist){
    dist.assign(V, INF);
    dist[source] = 0;

    for (int i = 0; i < V - 1; i++) {
        bool changed = false;
        for (const Edge &edge : edges) {
            if (dist[edge.u] == INF)
                continue;
            int newDist =dist[edge.u] + edge.weight;

            if (newDist < dist[edge.v]) {
                dist[edge.v] = newDist;
                changed = true;
            }
        }

        if (!changed) break;
    }
}

int main(){
    int V, E;
    vector<Edge> edges;

    readGraph(V, E, edges);

    int source=0;
    if (source < 0 || source >= V) {
        cout << "Invalid source vertex\n";
        return 1;
    }
    vector<int> dist;

    ssspCPU(V, edges, source, dist);

    for (int v = 0; v < V; v++) {
        if (dist[v] == INF)
            cout << "INF\n";
        else
            cout << dist[v] << '\n';
    }

    return 0;
}