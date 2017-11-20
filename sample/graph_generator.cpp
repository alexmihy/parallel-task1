#include "include/graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void parse_cmd_params(int _argc, char **_argv, string &_graph_type, int &_scale, int &_avg_degree, string &_output_file_name,
                      bool &_directed, ContainerType &_graph_file_format, bool &_optimized, bool &_help, int &_threads)
{
    // set deafualt params
	bool name_set = false;
	_directed = true;
    _graph_file_format = EDGES_LIST;
    _graph_type = "RMAT";
    _scale = 3;
    _avg_degree = 1;
    _optimized = false;
    _help = false;

    // get params from cmd line
	for (int i = 1; i < _argc; i++)
	{
		string option(_argv[i]);
        
        if (option.compare("-threads") == 0)
        {
            _threads = atoi(_argv[++i]);
        }

		if (option.compare("-type") == 0)
		{
			_graph_type = _argv[++i];
		}

		if (option.compare("-scale") == 0) 
		{
			_scale = atoi(_argv[++i]);
		}

		if (option.compare("-avg_degree") == 0) 
		{
			_avg_degree = atoi(_argv[++i]);
		}

		if (option.compare("-file") == 0)
		{
			_output_file_name = _argv[++i];
			name_set = true;
		}

		if (option.compare("-undirected") == 0)
		{
			_directed = false;
		}

		if (option.compare("-directed") == 0)
		{
			_directed = true;
		}
        
        if (option.compare("-optimized") == 0)
        {
            _optimized = true;
        }
        
        if (option.compare("-file_format") == 0)
        {
            // check if file format value is correct
            if(string(_argv[++i]) == "edges_list")
            {
                _graph_file_format = EDGES_LIST;
            }
            else if(string(_argv[++i]) == "edges_list")
            {
                _graph_file_format = ADJACENCY_LIST;
            }
            else
            {
                throw "Error: unknown graph file format";
            }
        }
        
        if((option.compare("-help") == 0) || (option.compare("--help") == 0))
        {
            _help = true;
        }
	}

	if (!name_set)
		_output_file_name = "test_graph_" + to_string((long long)(_scale));
}
           
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
void print_help()
{
    cout << "-type " << "(RMAT, SCCA2, random-uniform)" << endl;
    cout << "-scale " << "vertices count = 2^scale" << endl;
    cout << "-avg_degree " << "average graph connections count" << endl;
    cout << "-file " << "output file name to save" << endl;
    cout << "-directed/-undirected " << "graph direction mode, by default directed" << endl;
    cout << "-optimized " << "eliminate graph loops dublicate edges on generation stage" << endl;
    cout << "-file_format " << "storage format in output file (edges_list,edges_list)" << endl;
}
           
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	try
	{
		bool directed, optimized, help;
		int scale = 0, avg_degree = 0, threads;
		string graph_type, output_file_name;
        ContainerType graph_file_format;
		parse_cmd_params(argc, argv, graph_type, scale, avg_degree, output_file_name, directed, graph_file_format,
                         optimized, help, threads);
        
        if(help)
        {
            print_help();
            return 0;
        }
        
		NodeData node_data;
		init_library(node_data);
		node_data.print_node_properties();
        
        // print input params
        cout << "Input parameters: " << endl;
        cout << "Type: " << graph_type << endl;
        cout << "Scale: 2^" << scale << endl;
        cout << "Average degree: " << avg_degree << endl;
        cout << "Directed: " << directed << endl;
        cout << "Optimized: " << optimized << endl;

        cout << "Graph generation started..." << endl;
		double t1 = omp_get_wtime();
        
		Graph<int, float> graph(0, 0, EDGES_LIST, directed);
        srand(time(NULL));
		if (graph_type == "RMAT")
			GraphGenerationAPI<int, float>::R_MAT_parallel(graph, pow(2.0, scale), avg_degree, 45, 20, 20, 15, threads, directed);
		else if (graph_type == "SCCA2")
			GraphGenerationAPI<int, float>::SSCA2(graph, pow(2.0, scale), avg_degree, directed);
        else if (graph_type == "random-uniform")
            GraphGenerationAPI<int, float>::random_uniform(graph, pow(2.0, scale), avg_degree, directed, optimized);
		else
			throw "Error: unsupported graph type...";
		double t2 = omp_get_wtime();
		cout << "Generation time: " << t2 - t1 << " sec" << endl << endl;
        
        graph.convert_to_edges_list();
        
		cout << "Saving graph to " << output_file_name << " ..." << endl;
        t1 = omp_get_wtime();
        if(graph_file_format == EDGES_LIST)
        {
            GraphStorageAPI<int, float>::save_to_edges_list_bin_file(graph, output_file_name);
        }
        else if(graph_file_format == ADJACENCY_LIST)
        {
            GraphStorageAPI<int, float>::save_to_adjacency_list_bin_file(graph, output_file_name);
        }
        t2 = omp_get_wtime();
        cout << "Saving to binary file time: " << t2 - t1 << " sec" << endl << endl;
        
        if(optimized)
        {
            t1 = omp_get_wtime();
            cout << "Graph optimization started... " << endl;
            GraphOptimizationAPI<int, float>::optimize_graph_for_CPU(graph, sizeof(float));
            t2 = omp_get_wtime();
            cout << "done in " << t2 - t1 << " sec" << endl << endl;
            
            string opt_file_name = (string("opt_") + output_file_name);
            cout << "Saving optimized graph to " << opt_file_name << " ..." << endl;
            t1 = omp_get_wtime();
            if(graph_file_format == EDGES_LIST)
            {
                GraphStorageAPI<int, float>::save_to_edges_list_bin_file(graph, opt_file_name);
            }
            else if(graph_file_format == ADJACENCY_LIST)
            {
                GraphStorageAPI<int, float>::save_to_adjacency_list_bin_file(graph, opt_file_name);
            }
            t2 = omp_get_wtime();
            cout << "Saving to binary file time: " << t2 - t1 << " sec" << endl << endl;
        }
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
