#include "include/graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
#ifdef __USE_GPU__
#include "nvgraph.h"
template <typename _TVertexValue, typename _TEdgeWeight>
void nvgraph_check(Graph<_TVertexValue, _TEdgeWeight> &_graph, vector<int> &_source_vertices)
{
    _graph.convert_to_adjacency_list();
    _graph.transpose();
    _graph.convert_to_compressed_adjacency_list();
    
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    long long *vertices_to_edges_ptrs = graph_data.vertices_to_edges_ptrs;
    int *dst_ids = graph_data.edges_dst_ids;
    _TEdgeWeight *weights = graph_data.edges_weights;
    
    _TEdgeWeight *sssp = (_TEdgeWeight*) malloc(vertices_count * sizeof(_TEdgeWeight));
    int *destination_offsets = (int*) malloc((vertices_count + 1) * sizeof(int));
    for(int i = 0; i < (vertices_count + 1); i++)
        destination_offsets[i] = (int)vertices_to_edges_ptrs[i];
    
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t vertex_dimT = CUDA_R_32F;
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    
    nvgraphCreate(&handle);
    nvgraphCreateGraphDescr(handle, &graph);
    CSC_input->nvertices = vertices_count;
    CSC_input->nedges = edges_count;
    CSC_input->destination_offsets = destination_offsets;
    CSC_input->source_indices = dst_ids;
    
    nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32);
    nvgraphAllocateVertexData(handle, graph, 1, &vertex_dimT);
    nvgraphAllocateEdgeData(handle, graph, 1, &edge_dimT);
    nvgraphSetEdgeData(handle, graph, (void*)weights, 0);
    
    double avg_performance = 0;
    for(int i = 0; i < _source_vertices.size(); i++)
    {
        double t1 = omp_get_wtime();
        nvgraphSssp(handle, graph, 0, &_source_vertices[i], 0);
        nvgraphGetVertexData(handle, graph, (void*)sssp, 0);
        double t2 = omp_get_wtime();
        cout << "NVGRaph perfomance: " << _graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl;
        avg_performance += _graph.get_edges_count() / ((t2 - t1) * 1e6);
    }
    
    cout << "NVGRaph average perfomance: " << (avg_performance / _source_vertices.size()) << " MTEPS" << endl << endl;
    
    delete [] sssp;
    delete [] destination_offsets;
}
#endif
*/

#define RUN_TIME 55

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    try
    {
        // init graph library
        NodeData node_data;
        init_library(node_data);
        node_data.print_node_properties();
        
        //int scale = atoi(argv[1]);
        //string optimization_required = argv[2];
        string file_name = argv[1];
    
        int threads = omp_get_max_threads();
        
        // create empty graph
        Graph<int, float> graph(0, 0, EDGES_LIST, true);
        
        // generate graph
        /*cout << "Generating graph..." << endl;
        GraphGenerationAPI<int, float>::R_MAT_parallel(graph, (int)pow(2.0, scale), 32, 45, 20, 20, 15, threads, true);
        cout << "done!" << endl << endl;*/
        
        // load graph
        GraphStorageAPI<int, float>::load_from_edges_list_bin_file(graph, file_name);
        
        cout << "graph loaded!" << endl;
        
        cout << "graph size: " << (double)(graph.get_edges_count() * (2*sizeof(int) + sizeof(float))) / (1024.0 * 1024.0 * 1024.0) << " GB" << endl << endl;
        
        int vertices_count = graph.get_vertices_count();
        
        // source vertices
        vector<int> source_vertices(10);
        for(int i = 0; i < 5; i++)
        source_vertices[i] = rand() % 100;

        // convert graph to edges list format
        graph.convert_to_edges_list();
        
        // init operation
        SingleSourceShortestPaths<int, float> operation;
        operation.set_omp_threads(threads);
        
        float *cpu_result = new float[vertices_count];
        double avg_performance = 0;
        double time_start = omp_get_wtime();
        int min = 1;
        for(int i = 0; true; i++)
        {
            int source_vertex = rand() % vertices_count;
            operation.cpu_bellman_ford(graph, cpu_result, source_vertex);
            double time_current = omp_get_wtime();
            double exec_time = time_current - time_start;
            
            if (exec_time / 60.0 >= min)
            {
                cout << i << "-th iteration on min " << exec_time / 60.0 << endl << endl;
                min++;
            }
            
            if (exec_time / 60.0 >= RUN_TIME)
            {
                cout << "done " << i << " iterations" << endl << endl;
                break;
            }
        }
        
        /*
        // now we optimize graph
        if(optimization_required == "true")
        {
            cout << "Graph optimization started... " << endl;
            GraphOptimizationAPI<int, float>::optimize_graph_for_GPU(graph, sizeof(float), PASCAL);
            cout << "done!" << endl << endl;
        }
        
        // run library code
        float *gpu_result = new float[vertices_count];
        double avg_performance = 0;
        for(int i = 0; i < source_vertices.size(); i++)
            avg_performance += operation.gpu_bellman_ford(graph, gpu_result, source_vertices[i]);
        cout << "Opt avg perfomance: " << avg_performance / source_vertices.size() << " MTEPS" << endl << endl;
        
        // run nvgraph code
        cout << "NVgraph test" << endl;
        nvgraph_check(graph, source_vertices);
        
        delete []gpu_result;
        cout << "done!" << endl;*/
    }
    catch (const char *error)
    {
        cout << error << endl;
    }
    catch (...)
    {
        cout << "unknown error" << endl;
    }
    
    cout << "press any key to exit..." << endl;
    
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
