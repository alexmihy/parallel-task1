#include "include/graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void check(bool *cpu_parallel_result, bool *knl_result, int vertices_count)
{
    bool knl_correct = true;
    for(int i = 0; i < vertices_count; i++)
    {
        #ifdef __USE_KNL__
        if(cpu_parallel_result[i] != knl_result[i])
        {
            knl_correct = false;
        }
        #endif
    }

    #ifdef __USE_KNL__
    cout << "KNL check: " << ((knl_correct) ? " correct " : " error") << endl;
    #endif
    
    delete []cpu_parallel_result;
    
    #ifdef __USE_KNL__
    delete []knl_result;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void test_cpu_sequential(Graph<_TVertexValue, _TEdgeWeight> &_graph, BFS<_TVertexValue, _TEdgeWeight> _operation,
                         bool *_result, int _source_vertex)
{
    cout << "Doing BFS in CPU sequential mode:" << endl;
    double t1 = omp_get_wtime();
    _operation.cpu_sequential_bfs(_graph, _result, _source_vertex);
    double t2 = omp_get_wtime();
    cout << "sequential CPU perfomance: " << _graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void test_cpu_parallel(Graph<_TVertexValue, _TEdgeWeight> &_graph, BFS<_TVertexValue, _TEdgeWeight> _operation,
                       bool *_result, int _source_vertex)
{
    cout << "Doing BFS in CPU parallel mode:" << endl;
    double t1 = omp_get_wtime();
    _operation.cpu_parallel_bfs(_graph, _result, _source_vertex);
    double t2 = omp_get_wtime();
    cout << "parallel CPU perfomance: " << _graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void test_knl(Graph<_TVertexValue, _TEdgeWeight> &_graph, BFS<_TVertexValue, _TEdgeWeight> _operation,
                       bool *_result, int _source_vertex)
{
    cout << "Doing BFS in KNL mode:" << endl;
    double t1 = omp_get_wtime();
    _operation.knl_bfs(_graph, _result, _source_vertex);
    double t2 = omp_get_wtime();
    cout << "KNL perfomance: " << _graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void test_gpu(Graph<_TVertexValue, _TEdgeWeight> &_graph, BFS<_TVertexValue, _TEdgeWeight> _operation,
              bool *_result, int _source_vertex)
{
    cout << "Doing BFS in GPU mode:" << endl;
    double t1 = omp_get_wtime();
    _operation.gpu_bfs(_graph, _result, _source_vertex);
    double t2 = omp_get_wtime();
    cout << "GPU perfomance: " << _graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void read_data_from_file(Graph<_TVertexValue, _TEdgeWeight> &_graph, string _file_name, int vertices_pow)
{
    // open file
    ifstream graph_file(_file_name, ios::binary);
    if (!graph_file.is_open())
        return;
    
    long long file_size = get_filesize(_file_name);
    long long edges_count = file_size / (2 * sizeof(int) + sizeof(_TEdgeWeight));
    int vertices_count = pow(2.0, vertices_pow);
    cout << vertices_count / (1024 * 1024) << endl;
    
    // add edges from file
    for (long long i = 0; i < edges_count; i++)
    {
        int src_id = 0, dst_id = 0;
        
        graph_file.read(reinterpret_cast<char*>(&src_id), sizeof(int));
        graph_file.read(reinterpret_cast<char*>(&dst_id), sizeof(int));
        
        _graph.add_edge(src_id, dst_id, 1.0);
    }
    
    // add vertices from file
    for (int i = 0; i < vertices_count; i++)
    {
        _graph.add_vertex(i, 0);
    }
    
    graph_file.close();
}

#include <fstream>
using std::fstream;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        int scale = atoi(argv[1]);
        
        int threads = omp_get_max_threads();
        
        // load graph
        Graph<int, float> graph(0, 0, EDGES_LIST, true);
        
        /*if (!GraphStorageAPI<int, float>::load_from_edges_list_bin_file(graph, input_file_name))
            throw "Error: no such graph file";
        cout << "Input graph has been loaded from file " << input_file_name << "." << endl << endl;*/
        
        // generate graph
        cout << "Generating graph..." << endl;
        GraphGenerationAPI<int, float>::R_MAT_parallel(graph, pow(2.0, scale), 16, 45, 20, 20, 15, threads, true);
        cout << "done!" << endl << endl;
        
        int vertices_count = graph.get_vertices_count();

        BFS<int, float> operation;
        operation.set_omp_threads(threads);
        
        graph.convert_to_edges_list();
        
        bool *cpu_parallel_result = new bool[vertices_count];
        bool *gpu_parallel_result = new bool[vertices_count];
        
        
        operation.cpu_parallel_bfs(graph, cpu_parallel_result, 0);
        
        cout << " done parallel " << endl;
        
        operation.cpu_sequential_bfs(graph, gpu_parallel_result, 0);
        
        cout << " done seq " << endl;
        
        /*unsigned int seed = int(time(NULL));
        for(int i = 0; i < 10; i++)
        {
            int source_vertex = rand_r(&seed) % vertices_count;
            cout << "source vertex " << source_vertex << endl;
            
            operation.cpu_parallel_bfs(graph, cpu_parallel_result, source_vertex);
            operation.gpu_bfs(graph, gpu_parallel_result, source_vertex);
            
            for(int j = 0; j < vertices_count; j++)
            {
                if(cpu_parallel_result[j] != gpu_parallel_result[j])
                {
                    cout << "error in " << i << "-th test" << endl;
                    break;
                }
            }
        }*/
        
        delete []cpu_parallel_result;
        delete []gpu_parallel_result;
    }
    catch (const char *error)
    {
        cout << error << endl;
        getchar();
        return 1;
    }
    catch (...)
    {
        cout << "unknown error" << endl;
    }
    
    cout << "press any key to exit..." << endl;
    getchar();
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
