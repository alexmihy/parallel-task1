#include "include/graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    try
    {
        string input_file_name = argv[1];
        
        // init library
        NodeData node_data;
        init_library(node_data);
        node_data.print_node_properties();
        
        // load graph
        Graph<int, float> graph(0, 0, EDGES_LIST, true);
        if (!GraphStorageAPI<int, float>::load_from_edges_list_bin_file(graph, input_file_name))
            throw "Error: no such graph file";
        cout << "Input graph has been loaded from file " << input_file_name << "." << endl << endl;
        
        graph.convert_to_edges_list();
        
        long long before_sorting = GraphOptimizationAPI<int, float>::check_gpu_memory_transactions_count(graph, sizeof(float));
        
        GraphOptimizationAPI<int, float>::optimize_graph_for_CPU(graph, sizeof(float));

        long long after_sorting = GraphOptimizationAPI<int, float>::check_gpu_memory_transactions_count(graph, sizeof(float));
        
        cout << "before sorting: " << before_sorting << endl;
        cout << "after sorting: " << after_sorting << endl;
        
        cout << "rate: " << (double)(before_sorting)/after_sorting << endl;
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
