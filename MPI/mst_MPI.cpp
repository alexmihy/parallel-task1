#include "include/graph_library.h"

#include <mpi.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int proc_id = 0, proc_num = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

	try
	{
		NodeData node_data;
		init_library(node_data);

		// load data from cmd
		int scale = atoi(argv[1]);
		int avg_degree = atoi(argv[2]);
		int vertices_count = pow(2.0, scale);
		long long edges_count = (long long)vertices_count * avg_degree;
		long long edges_on_proc = edges_count / proc_num;

		// local node graphs
		if (proc_id == 0)
			cout << "total first graph size: " << (edges_count/2) * (2 * sizeof(int) + sizeof(float)) / (1024.0 * 1024.0 * 1024.0) << " GB " << endl;
		cout << proc_id << ") allocating in the beginig: " << (edges_on_proc/2) * (2*sizeof(int) + sizeof(float)) / (1024.0 * 1024.0 * 1024.0) << " GB " << endl;
		Graph<int, float> node_graph(vertices_count, edges_on_proc, EDGES_LIST, false, false);

		// init RMAT graph
		srand((unsigned int)time(NULL) + proc_id);
		GraphGenerationAPI<int, float>::R_MAT(node_graph, pow(2.0, scale), avg_degree / proc_num, 60, 10, 10, 20, false);

		// load pointers
		GraphContainerData<int, float> node_graph_data = node_graph.get_graph_data();
		int *node_src_ids = node_graph_data.edges_src_ids;
		int *node_dst_ids = node_graph_data.edges_dst_ids;
		float *node_weights = node_graph_data.edges_weights;

		MPI_Barrier(MPI_COMM_WORLD);
		double t1 = MPI_Wtime();

		// main parallel computations
		Graph<int, float> node_MST(0, 0, EDGES_LIST, false, true);
		MinimumSpanningTree<int, float> operation;
		operation.single_node_boruvka(node_graph, node_MST, node_data, USE_GPU_MODE);
		node_graph.empty();

		// compute edges count for gather on root
		int result_edges_count = node_MST.get_edges_count();
		int total_edges_in_mst = 0;
		int *recv_counts = new int[proc_num];
		int *displacements = new int[proc_num];
		MPI_Reduce(&result_edges_count, &total_edges_in_mst, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Gather(&result_edges_count, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

		cout << proc_id << ") mst tmp result: " << result_edges_count * (2 * sizeof(int) + sizeof(float)) / (1024.0 * 1024.0 * 1024.0) << " GB " << endl;

		if (proc_id == 0)
		{
			for (int i = 0; i < proc_num; i++)
			{
				displacements[i] = 0;
			}
			for (int i = 0; i < proc_num; i++)
			{
				for (int j = 0; j < i; j++)
					displacements[i] += recv_counts[j];
			}
		}

		if (proc_id == 0)
			cout << "total new graph size: " << total_edges_in_mst * (2 * sizeof(int) + sizeof(float)) / (1024.0 * 1024.0 * 1024.0) << " GB " << endl;

		// get pointers to data
		GraphContainerData<int, float> recv_graph_data;
		GraphContainerData<int, float> send_graph_data = node_MST.get_graph_data();
		Graph<int, float> *root_graph;
		if (proc_id == 0)
		{
			root_graph = new Graph<int, float>(vertices_count, total_edges_in_mst, EDGES_LIST, false, false);
			recv_graph_data = root_graph->get_graph_data();
		}
		 
		// merge fragments
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gatherv(send_graph_data.edges_src_ids, result_edges_count, MPI_INT, recv_graph_data.edges_src_ids, recv_counts, displacements, MPI_INT, 
			       0, MPI_COMM_WORLD);
		MPI_Gatherv(send_graph_data.edges_dst_ids, result_edges_count, MPI_INT, recv_graph_data.edges_dst_ids, recv_counts, displacements, MPI_INT,
				   0, MPI_COMM_WORLD);
		MPI_Gatherv(send_graph_data.edges_weights, result_edges_count, MPI_FLOAT, recv_graph_data.edges_weights, recv_counts, displacements, MPI_FLOAT,
			       0, MPI_COMM_WORLD);
		node_MST.empty();

		// compute last portion
		if (proc_id == 0)
		{
			Graph<int, float> root_MST;
			operation.single_node_boruvka(*root_graph, root_MST, node_data, USE_GPU_MODE);

			cout << "final result size: " << root_MST.get_edges_count() * (2 * sizeof(int) + sizeof(float)) / (1024.0 * 1024.0 * 1024.0) << " GB " << endl;

			cout << root_MST.get_vertices_count() << " " << root_MST.get_edges_count() << " " << operation.compute_MST_weight(root_MST) << endl;

			delete root_graph;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		double t2 = MPI_Wtime();
		if (proc_id == 0)
			cout << "time: " << t2 - t1 << " sec " << endl;
	}
	catch (const char *error)
	{
		cout << error << endl;
	}
	catch (...)
	{
		cout << "unknown error" << endl;
	}

	MPI_Finalize();
	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////