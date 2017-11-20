#include "include/graph_library.h"
#include <cuda_profiler_api.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void parse_cmd_params(int _argc, char **_argv, string &_input_file_name, string &_mode, bool &_seq_check)
{
    _input_file_name  = "input.graph";
    _mode = "sequential";
    _seq_check = false;
    
    for (int i = 1; i < _argc; i++)
    {
        string option(_argv[i]);
        
        if (option.compare("-infile") == 0)
        {
            _input_file_name = _argv[++i];
        }
        
        if (option.compare("-mode") == 0)
        {
            _mode = _argv[++i];
        }
        
        if (option.compare("-seq_check") == 0)
        {
            _seq_check = true;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        // required variables
        string input_file_name, mode;
        double t1, t2;
        double perfomance, processing_time;
        bool seq_check;
        
        // get cmd params
        parse_cmd_params(argc, argv, input_file_name, mode, seq_check);
        
        // init graph library
        NodeData node_data;
        init_library(node_data);
        node_data.print_node_properties();
        
        // load graph
        Graph<int, float> graph(0, 0, EDGES_LIST, false);
        if (!GraphStorageAPI<int, float>::load_from_edges_list_bin_file(graph, input_file_name))
            throw "Error: no such graph file";
        cout << "Input graph has been loaded from file " << input_file_name << "." << endl << endl;
        
        // convert graph
        graph.convert_to_directed();
        graph.convert_to_compressed_adjacency_list();
        
        // perform computations
        cout << "Computing bridges in ";
        BridgesDetection<int, float> operation;
        bool *bridges_result = new bool[graph.get_edges_count()];
        
        t1 = omp_get_wtime();
        if(mode == "cpu")
        {
            cout << "CPU mode" << endl;
            operation.parallel_cpu_tarjan(graph, bridges_result, omp_get_max_threads());
        }
        else if(mode == "gpu")
        {
            cout << "GPU mode" << endl;
            operation.parallel_gpu_tarjan(graph, bridges_result);
        }
        else if(mode == "sequential")
        {
            cout << "SEQUENTIAL mode" << endl;
            operation.parallel_cpu_tarjan(graph, bridges_result, 1);
        }
        else
        {
            throw "Unknown computational mode";
        }
        t2 = omp_get_wtime();
        
        processing_time = t2 - t1;
        perfomance = graph.get_edges_count() / ((processing_time) * 1e6);
		cout << "time: " << processing_time << " sec" << endl;
        cout << "perfomance: " << perfomance << " MTEPS" << endl << endl;
        
        // perform check
        if(seq_check)
        {
            int error_count = 0;
            cout << "Sequential check:" << endl;
            bool *check_result = new bool[graph.get_edges_count()];
            operation.parallel_cpu_tarjan(graph, check_result, 1);
            for(long long i = 0; i < graph.get_edges_count(); i++)
                if(check_result[i] != bridges_result[i])
                    error_count++;
            delete[] check_result;
            cout << "ERROR COUNT: " << error_count << endl;
        }
        
        // free output memory
        delete []bridges_result;
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
