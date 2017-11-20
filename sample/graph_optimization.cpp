#include "include/graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void parse_cmd_params(int _argc, char **_argv, string &_input_file_name, string &_output_file_name, bool &_directed,
                      string &_optimization_mode, int &_elem_size)
{
    // set deafualt params
    bool input_name_selected = false;
    bool output_name_selected = false;
    _directed = true;
    _optimization_mode = "cpu";
    _elem_size = sizeof(float);
    
    // get params from cmd line
    for (int i = 1; i < _argc; i++)
    {
        string option(_argv[i]);
        
        if (option.compare("-infile") == 0)
        {
            _input_file_name = _argv[++i];
            input_name_selected = true;
        }
        
        if (option.compare("-outfile") == 0)
        {
            _output_file_name = _argv[++i];
            output_name_selected = true;
        }
        
        if (option.compare("-undirected") == 0)
        {
            _directed = false;
        }
        
        if (option.compare("-directed") == 0)
        {
            _directed = true;
        }
        
        if (option.compare("-cpu") == 0)
        {
            _optimization_mode = "cpu";
        }
        
        if (option.compare("-gpu") == 0)
        {
            _optimization_mode = "gpu";
        }
        
        if (option.compare("-knl") == 0)
        {
            _optimization_mode = "knl";
        }
        
        if (option.compare("-float") == 0)
        {
            _elem_size = sizeof(float);
        }
        
        if (option.compare("-bool") == 0)
        {
            _elem_size = sizeof(bool);
        }
    }
    
    if(!input_name_selected)
        throw "Error: no input graph file name selected";
    if(!output_name_selected)
        _output_file_name = "opt_" + _input_file_name;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        string input_file_name, output_file_name, optimization_mode;
        bool directed;
        int elem_size = 0;
        parse_cmd_params(argc, argv, input_file_name, output_file_name, directed, optimization_mode, elem_size);
        
        // init library
        NodeData node_data;
        init_library(node_data);
        node_data.print_node_properties();
        
        // load graph
        Graph<int, float> graph(0, 0, EDGES_LIST, directed);
        if (!GraphStorageAPI<int, float>::load_from_edges_list_bin_file(graph, input_file_name))
            throw "Error: no such graph file";
        cout << "Input graph has been loaded from file " << input_file_name << "." << endl << endl;
        
        cout << "optimizing for " << optimization_mode << " mode" << endl;
        
        // optimize graph for CPU
        if(optimization_mode == "cpu")
            GraphOptimizationAPI<int, float>::optimize_graph_for_CPU(graph, elem_size);
        
        // optimize graph for GPU
        if(optimization_mode == "gpu")
            GraphOptimizationAPI<int, float>::optimize_graph_for_GPU(graph, elem_size);
        
        // optimize graph for GPU
        if(optimization_mode == "knl")
            GraphOptimizationAPI<int, float>::optimize_graph_for_KNL(graph, elem_size);
        
        // save graph
        GraphStorageAPI<int, float>::save_to_edges_list_bin_file(graph, output_file_name);
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
