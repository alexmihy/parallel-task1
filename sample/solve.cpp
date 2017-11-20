#include "include/graph_library.h"

#include "boost_API.h"

#define ITERATIONS_COUNT 1

void calculate_bellman_ford(Graph<int, float> &graph)
{
    int vertices_count = 0;
    long long edges_count = 0;

    int threads = omp_get_max_threads();

    vertices_count = graph.get_vertices_count();
    edges_count = graph.get_edges_count();

    SingleSourceShortestPaths<int, float> operation;
    operation.set_omp_threads(threads);

    vector<float> boost_result;
    boost_result.resize(vertices_count);
    float *result = new float [vertices_count];
    double avg_time = 0, time_start, time_end, time_exec;
    for (int i = 0; i < ITERATIONS_COUNT; i++) {
        int source_vertex = rand() % vertices_count;

        //cout << "source_vertex : " << source_vertex << endl;

        //cout << "Calculating boost implementation" << endl;

        double boost_time;
        boost_time = BoostAPI<int, float>::bellman_ford_shortest_paths(graph, boost_result, source_vertex);

        //cout << "boost calculated" << endl;

        time_start = omp_get_wtime();
        operation.gpu_bellman_ford(graph, result, source_vertex);
        time_end = omp_get_wtime();

        time_exec = time_end - time_start;
        //cout << "ITERATION " << i + 1 << ": " << time_exec / 60.0 << " sec" << endl;
        //cout << "ITERATION " << i + 1 << ": " << edges_count / (time_exec * 1e6) << " MTEPS" << endl;
        cout << "BOOST-only performance: " << edges_count / (boost_time * 1e6) << " MTEPS" << endl;

        for (int v = 0; v < vertices_count; v++) {
            if (((v == source_vertex) && (boost_result[v] != 0 || result[v] != 0))
                ||
                ((v != source_vertex) && (fabs(boost_result[v] - result[v]) > 1e-7))) {
                cout << "v : " << v << endl;
                cout << "boost : " << boost_result[v] << endl;
                cout << "our   : " << result[v] << endl;
                cout << "fabs  : " << fabs(boost_result[v] - result[v]) << endl;
                cout << "EPS   : " << 1e-7 << endl;
                throw "boost result difers with hand-made implementation";
            }
        }

        avg_time += time_exec;
    }
    avg_time /= ITERATIONS_COUNT;
    //cout << "Tests done!" << endl;
    //cout << "average execution time : " << avg_time / 60.0 << endl;

    delete[] result;
}

int main(int argc, char **argv) 
{
    if (argc < 2) {
        cout << "Usage: ./fb <path to input" << endl;
        return 1;
    }

    try {
        const char *path = argv[1];
        
        int vertices_count = 0;
        long long edges_count = 0;

        Graph<int, float> graph(0, 0, EDGES_LIST, true);

        GraphStorageAPI<int, float>::load_from_edges_list_bin_file(graph, path);

        
        vertices_count = graph.get_vertices_count();
        edges_count = graph.get_edges_count();

        cout << "Graph loadad:" << endl;
        cout << "vertices: " << vertices_count << endl;
        cout << "edges:    " << edges_count << endl;

        cout << "Graph without optimization" << endl;
        calculate_bellman_ford(graph);

        graph.convert_to_edges_list();
        GraphOptimizationAPI<int, float>::optimize_graph_for_GPU(graph, sizeof(int), PASCAL);

        cout << "Graph with optimization" << endl;
        calculate_bellman_ford(graph);
    }

    catch (const char *error) {
        cout << error << endl;
    }
    catch (...) {
        cout << "unknown error" << endl;
    }

    cout << "execution done!" << endl;

    return 0;
}