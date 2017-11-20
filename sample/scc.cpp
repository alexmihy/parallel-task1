#include "include/graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool equal_components(int *_first_components, int *_second_components, int _vertices_count)
{
	map<int, int> f_s_equality;
	map<int, int> s_f_equality;

	for (int i = 0; i < _vertices_count; i++)
	{
		f_s_equality[_first_components[i]] = _second_components[i];
		s_f_equality[_second_components[i]] = _first_components[i];
	}

	bool result = true;
	for (int i = 0; i < _vertices_count; i++)
	{
		if (f_s_equality[_first_components[i]] != _second_components[i])
		{
			result = false;
		}
		if (s_f_equality[_second_components[i]] != _first_components[i])
		{
			result = false;
		}
	}

	return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void check_components_size(int *_components, int _vertices_count)
{
	map<int, int> frequency;
	for (int i = 0; i < _vertices_count; i++)
	{
		frequency[_components[i]]++;
	}

	map<int, int> sizes;

	for (auto i : frequency)
	{
		sizes[i.second]++;
	}

	cout << "sizes: " << endl;

	for (auto i : sizes)
		cout << i.second << " components of size: " << i.first << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	try
	{
        // input variables
        string input_file_name = argv[1];
        int threads_count = atoi(argv[2]);
        string check = argv[3];
        int source_vertex = 0;
        double t1, t2;
        
        // init library
        NodeData node_data;
        init_library(node_data);
        node_data.print_node_properties();
        
        // load graph
        Graph<int, float> graph(0, 0, EDGES_LIST, true);
        if (!GraphStorageAPI<int, float>::load_from_edges_list_bin_file(graph, input_file_name))
            throw "Error: no such graph file";
        cout << "Input graph has been loaded from file " << input_file_name << "." << endl << endl;
        graph.convert_to_compressed_adjacency_list();
        
        // perform sequential computations
        StronglyConnectedComponents<int, float> operation(threads_count);
        
        int *tarjan_result = new int[graph.get_vertices_count()];
        if(check == "-check")
        {
            cout << "Tarjan algorithm:" << endl;
            t1 = omp_get_wtime();
            operation.cpu_tarjan(graph, tarjan_result);
            t2 = omp_get_wtime();
            cout << "Tarjan time: " << t2 - t1 << " sec" << endl << endl;
            cout << "perfomance: " << graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
        }
        
        int *cpu_result = new int[graph.get_vertices_count()];
        int *gpu_result = new int[graph.get_vertices_count()];
        
        #ifdef __USE_KNL__
        cout << "KNL forward-backward algorithm:" << endl;
        GraphOptimizationAPI<int, float>::optimize_graph_for_KNL(graph, sizeof(int));
        operation.cpu_forward_backward(graph, cpu_result);
        
        operation.cpu_forward_backward(graph, cpu_result);
        #endif
        
        #ifdef __USE_GPU__
        cout << "GPU forward-backward algorithm:" << endl;
        
        operation.gpu_forward_backward(graph, gpu_result);
        
        GraphOptimizationAPI<int, float>::optimize_graph_for_GPU(graph, sizeof(int));
        
        operation.gpu_forward_backward(graph, gpu_result);
        
        operation.gpu_forward_backward(graph, gpu_result);
        #endif
        
        if(check == "-check")
        {
            check_components_size(cpu_result, graph.get_vertices_count());
            cout << "correct: " << equal_components(tarjan_result, cpu_result, graph.get_vertices_count()) << endl << endl;
            
            #ifdef __USE_GPU__
            check_components_size(gpu_result, graph.get_vertices_count());
            cout << "correct: " << equal_components(tarjan_result, gpu_result, graph.get_vertices_count()) << endl << endl;
            #endif
        }
        
        delete [] tarjan_result;
        delete[] cpu_result;
        delete[] gpu_result;
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
