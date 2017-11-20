#include "include/graph_library.h"
#include <cuda_profiler_api.h>

#include <queue>
#include <map>

#include "boost_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void parse_cmd_params(int _argc, char **_argv, string &_input_file_name, string &_template_file_name, string &_mode, bool &_boost_check)
{
    _boost_check = false;
    _input_file_name  = "input.graph";
    _template_file_name = "template";
    
    for (int i = 1; i < _argc; i++)
    {
        string option(_argv[i]);
        
        if (option.compare("-infile") == 0)
        {
            _input_file_name = _argv[++i];
        }
        
        if (option.compare("-template") == 0)
        {
            _template_file_name = _argv[++i];
        }
        
        if (option.compare("-mode") == 0)
        {
            _mode = _argv[++i];
        }
        
        if (option.compare("-boost_check") == 0)
        {
            _boost_check = true;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void save_subgraph_isomorphism_log_file(string _input_file_name, string _template_file_name, Graph<_TVertexValue, _TEdgeWeight> &_big_graph,
                                        Graph<_TVertexValue, _TEdgeWeight> &_small_graph, double _total_time, double _loading_input_time,
                                        double _conversion_time, double _processing_time, time_t &_time_start, time_t &_time_finish)
{
    ofstream log_file("perf.log");
    
    log_file << "*****************************************" << endl;
    log_file << "1. Subgraph isomorphism algorithm" << endl;
    log_file << "*****************************************" << endl << endl;
    log_file << "2. Input" << endl;
    
    // get path
    char path[PATH_MAX + 1];
#if defined(_WIN32)
    cout << "Creating log file not supported in windows" << endl;
    return;
#elif defined(__unix__) || defined(__unix) || defined(unix)
    readlink("/proc/self/exe", path, PATH_MAX);
#elif defined(__APPLE__) && defined(__MACH__)
    int bufsize = PATH_MAX + 1;
    _NSGetExecutablePath(path, &bufsize);
#endif
    
    log_file << "2.1 " << _input_file_name << endl;
    log_file << "Path " << path << endl;
    log_file << "Size " << get_filesize(_input_file_name) / 1024 << " Kb" << endl << endl;
    
    log_file << "2.2 Vertexes " << _big_graph.get_vertices_count() << endl;
    log_file << "Edges " << _big_graph.get_edges_count() << endl;
    log_file << "properties: unknown" << endl << endl;
    
    log_file << "2.3 " << _template_file_name << endl;
    log_file << "Path " << path << endl;
    log_file << "Size " << get_filesize(_template_file_name) / 1024 << " Kb" << endl << endl;
    
    log_file << "2.2 Vertexes " << _small_graph.get_vertices_count() << endl;
    log_file << "Edges " << _small_graph.get_edges_count() << endl;
    log_file << "properties: unknown" << endl << endl;
    
    log_file << "*****************************************" << endl;
    log_file << "3. Output " << endl << endl;
    
    log_file << "No output required." << endl << endl;
    
    log_file << "*****************************************" << endl;
    log_file << "4. Performance" << endl << endl;
    
    log_file << "4.1 Hardware " << endl;
    log_file << "CPU: Intel Xeon 5570" << endl;
    log_file << "Accelerator: GPU Nvidia K40" << endl;
    log_file << "RAM size: 64 GB" << endl << endl;
    
    log_file << "4.2 Execution time " << endl;
    log_file << "Time start " << ctime(&_time_start);
    log_file << "Time finish " << ctime(&_time_finish);
    log_file << "Total time, seconds " << _total_time << endl << endl;
    
    log_file << "Loading input data " << _loading_input_time << endl;
    log_file << "Graph conversion " << _conversion_time << endl;
    log_file << "Graph processing " << _processing_time << endl;
    log_file << "Output conversion " << "not required" << endl;
    log_file << "Saving output data " << "included in processing time" << endl << endl;
    
    log_file << "4.3 Performance metrics" << endl;
    log_file << "Subgraph Isomorphic time " << _processing_time << " seconds" << endl;
    
    log_file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        // required variables
        bool seq_check, boost_check;
        string input_file_name, template_graph_file_name, mode;
        double t1, t2;
        double total_time, loading_input_time, graph_conversion_time, processing_time, saving_output_time;
        
        // parse cmd comand line
        parse_cmd_params(argc, argv, input_file_name, template_graph_file_name, mode, boost_check);
        
        // get start time
        double time_start_omp = omp_get_wtime();
        time_t time_start = time(0);
        localtime(&time_start);
        
        // init library
        NodeData node_data;
        init_library(node_data);
        node_data.print_node_properties();
        
        t1 = omp_get_wtime();
        // load graph G1
        Graph<int, float> loaded_graph(0, 0, EDGES_LIST, false);
        if (!GraphStorageAPI<int, float>::load_from_edges_list_bin_file(loaded_graph, input_file_name))
            throw "Error: no such graph file";
        cout << "Input graph has been loaded from file " << input_file_name << "." << endl << endl;
        
        // load graph G2
        ifstream template_graph_file(template_graph_file_name);
        string line;
        getline(template_graph_file, line);
        int v_count = 0;
        istringstream ss(line);
        ss >> v_count;
        cout << "teplate graph: " << v_count << " vertices" << endl;
        Graph<int, float> small_graph(v_count, 0, ADJACENCY_LIST, true);
        for(int i = 0; i < v_count; i++)
            small_graph.add_vertex(i, 0);
        while (getline(template_graph_file, line))
        {
            istringstream ss(line);
            int src_id = 0, dst_id = 0;
            ss >> src_id >> dst_id;
            cout << "(" << src_id << " " << dst_id << ")" << endl;
            small_graph.add_edge(src_id, dst_id, 1);
            small_graph.add_edge(dst_id, src_id, 1);
        }
        t2 = omp_get_wtime();
        loading_input_time = t2 - t1;
        
        // convert graphs
        t1 = omp_get_wtime();
        Graph<int, float> big_graph(0, 0, ADJACENCY_LIST, true);
        for(int i = 0; i < loaded_graph.get_vertices_count(); i++)
        {
            big_graph.add_vertex(i, 0);
        }
        for(long long i = 0; i < loaded_graph.get_edges_count(); i++)
        {
            Edge<float> cur_edge = loaded_graph.iterate_edges(i);
            big_graph.add_edge(cur_edge.src_id, cur_edge.dst_id, cur_edge.weight);
            big_graph.add_edge(cur_edge.dst_id, cur_edge.src_id, cur_edge.weight);
        }
        big_graph.convert_to_compressed_adjacency_list();
        small_graph.convert_to_compressed_adjacency_list();
        t2 = omp_get_wtime();
        graph_conversion_time = t2 - t1;
        
        // perform computations
        t1 = omp_get_wtime();
        SubgraphIsomorphism<int, float> operation;
        cout << "Computing subgraph isomorphism in ";
        if (mode == "sequential")
        {
            cout << "SEQUENTIAL mode: " << endl;
            operation.ullman_algorithm(big_graph, small_graph, USE_SEQUENTIAL_MODE);
        }
        else if (mode == "cpu")
        {
            cout << "CPU mode: " << endl;
            operation.ullman_algorithm(big_graph, small_graph, USE_CPU_MODE);
        }
        else
        {
            throw "Error: unknown computational mode in MST test";
        }
        t2 = omp_get_wtime();
        processing_time = t2 - t1;
        
        cout << "time: " << t2 - t1 << " sec" << endl;
        
        // get finish time
        time_t time_finish = time(0);
        localtime(&time_finish);
        double time_finish_omp = omp_get_wtime();
        total_time = time_finish_omp - time_start_omp;
        
        // print isomorphisms using boost
        if(boost_check)
            BoostAPI<int, float>::VF2(big_graph, small_graph);
        
        // save log file
        save_subgraph_isomorphism_log_file(input_file_name, template_graph_file_name, big_graph, small_graph, total_time,
                                           loading_input_time, graph_conversion_time, processing_time, time_start, time_finish);
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
    getchar();
    
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
