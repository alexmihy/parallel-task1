#include "include/graph_library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	try
	{
        // required variables
		int scale = atoi(argv[1]);
        int degree = atoi(argv[2]);
        int omp_threads = atoi(argv[3]);
        double t1, t2;
        
        // init graph library
		NodeData node_data;
		init_library(node_data);
		node_data.print_node_properties();
        
        Graph<int, float> graph(0, 0, EDGES_LIST, true);
        
        cout << "Generating graph..." << endl;
        GraphGenerationAPI<int, float>::R_MAT_parallel(graph, pow(2.0, scale), degree, 45, 20, 20, 15, omp_threads, false);
        cout << "done!" << endl << endl;
    
        // perform computations
		MinimumSpanningTree<int, float> mst_algorithm(omp_threads);
        Graph<int, float> cpu_mst(0, 0, EDGES_LIST, false);
        Graph<int, float> gpu_mst(0, 0, EDGES_LIST, false);
        
        cout << "algorithm launched" << endl;
        
        t1 = omp_get_wtime();
        mst_algorithm.cpu_boruvka(graph, cpu_mst);
        t2 = omp_get_wtime();
        cout << "CPU Performance : " << 2*graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
        
        t1 = omp_get_wtime();
        mst_algorithm.gpu_boruvka(graph, gpu_mst);
        t2 = omp_get_wtime();
        cout << "GPU Performance : " << 2*graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
        
        //GraphVisualizationAPI<int, float>::create_graphviz_file(cpu_mst, "mst.txt", VISUALIZE_AS_DIRECTED);
        
        cout << "done" << endl;
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
