/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// main computation using both CPU and GPU
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void MinimumSpanningTree<_TVertexValue, _TEdgeWeight>::cpu_boruvka(Graph<_TVertexValue, _TEdgeWeight> &_graph,
	                                                               Graph<_TVertexValue, _TEdgeWeight> &_mst)
{

	// convert input graph into edges list format if required
	_graph.convert_to_edges_list();

    // result array
    bool *mst_edges = new bool[_graph.get_edges_count()];

    // compute MST
    cpu_boruvka_kernel(_graph, mst_edges);

    // save result
	update_mst(_graph, _mst, mst_edges);
    
    // free array
    delete[] mst_edges;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void MinimumSpanningTree<_TVertexValue, _TEdgeWeight>::gpu_boruvka(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                   Graph<_TVertexValue, _TEdgeWeight> &_mst)
{
    
    // convert input graph into edges list format if required
    _graph.convert_to_edges_list();
    
    // result array
    bool *mst_edges = new bool[_graph.get_edges_count()];
    
    // compute MST on GPU
    gpu_boruvka_kernel(_graph, mst_edges);
    
    // save result
    update_mst(_graph, _mst, mst_edges);
    
    // free array
    delete[] mst_edges;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void MinimumSpanningTree<_TVertexValue, _TEdgeWeight>::update_mst(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                  Graph<_TVertexValue, _TEdgeWeight> &_mst,
                                                                  bool *mst_edges)
{
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *src_ids = graph_data.edges_src_ids;
    int *dst_ids = graph_data.edges_dst_ids;
    _TEdgeWeight *weights = graph_data.edges_weights;
    
    // add vertices
    _mst.empty();
    for (int vertex_pos = 0; vertex_pos < vertices_count; vertex_pos++)
    {
        _mst.add_vertex(_graph.iterate_vertices(vertex_pos));
    }
    
	// add edges
	for (long long edge_pos = 0; edge_pos < edges_count; edge_pos++)
	{
		if (mst_edges[edge_pos])
		{
			_mst.add_edge(src_ids[edge_pos], dst_ids[edge_pos], weights[edge_pos]);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef __NO_CPP_11_SUPPORT__
template <typename _TVertexValue, typename _TEdgeWeight>
void MinimumSpanningTree<_TVertexValue, _TEdgeWeight>::cpu_boruvka_kernel(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                          bool *_mst_edges)
{
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *src_ids = graph_data.edges_src_ids;
    int *dst_ids = graph_data.edges_dst_ids;
    _TEdgeWeight *weights = graph_data.edges_weights;
    
	// additional data for computations
	UnionFind<int> components(vertices_count);
	vector<long long>cheapest(vertices_count);
	int num_trees = vertices_count, prev_num_trees = 0;

    // perform computations
	#pragma omp parallel num_threads(omp_threads)
	{
		// initialize result mst and components
		#pragma omp for schedule(static) 
		for (int i = 0; i < vertices_count; i++)
			components.clear(i);

		#pragma omp for schedule(static) 
		for (long long i = 0; i < edges_count; i++)
			_mst_edges[i] = false;

		#pragma omp barrier

		// run algorithm until components count couldn't be reduced
		while (num_trees != prev_num_trees)
		{
			// init cheapest and flatten components
			#pragma omp for schedule(static) 
			for (int i = 0; i < vertices_count; i++)
			{
				cheapest[i] = -1;
				components.flatten(i);
			}
			#pragma omp barrier
			 
			// find minimal edge from each component
			#pragma omp for schedule(guided, 1024)
			for (long long i = 0; i < edges_count; i++)
			{
				int set1 = components.get_parent(src_ids[i]);
				int set2 = components.get_parent(dst_ids[i]);

				if (set1 == set2)
					continue;

				long long cheapest_val = cheapest[set1];
				while (cheapest_val == -1 || weights[i] < weights[cheapest_val])
				{
					if (atomic_compare_exchange_weak((atomic<long long>*)&cheapest[set1], &cheapest_val, i)) {
						break;
					}
				}

				cheapest_val = cheapest[set2];
				while (cheapest_val == -1 || weights[i] < weights[cheapest_val])
				{
					if (atomic_compare_exchange_weak((atomic<long long>*)&cheapest[set2], &cheapest_val, i)) {
						break;
					}
				}
			}

			// remember previous components count
			#pragma omp master
			{
				prev_num_trees = num_trees;
			}

			#pragma omp barrier

			// merge components
			#pragma omp for schedule(guided, 1024)
			for (int i = 0; i < vertices_count; i++)
			{
				if (cheapest[i] != -1)
				{
					int set1 = components.get_parent(src_ids[cheapest[i]]);
					int set2 = components.get_parent(dst_ids[cheapest[i]]);

					if (set1 == set2)
						continue;

					if (components.merge(set1, set2))
					{
						_mst_edges[cheapest[i]] = true;
					}
					else
					{
						#pragma omp atomic
						num_trees++;
					}
					#pragma omp atomic
				    num_trees--;
				}
			}
			#pragma omp barrier
		}
	}
}
#else
template <typename _TVertexValue, typename _TEdgeWeight>
void MinimumSpanningTree<_TVertexValue, _TEdgeWeight>::cpu_boruvka_kernel(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                          bool *_mst_edges)
{
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *src_ids = graph_data.edges_src_ids;
    int *dst_ids = graph_data.edges_dst_ids;
    _TEdgeWeight *weights = graph_data.edges_weights;
    
    // additional data for computations
    UnionFind<int> components(vertices_count);
    vector<long long>cheapest(vertices_count);
    int num_trees = vertices_count, prev_num_trees = 0;
    
    // perform computations
    #pragma omp parallel num_threads(omp_threads)
    {
        int tmpPos = 1, newPos = 3, pos = 2;
        #pragma omp atomic capture
        {
            tmpPos = pos;
            pos =  newPos;
        }
        
        cout << pos << endl;
        
        /// initialize result mst and components
        #pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; i++)
            components.clear(i);
        
        #pragma omp for schedule(static)
        for (long long i = 0; i < edges_count; i++)
            _mst_edges[i] = false;
        
        #pragma omp barrier
        
        // run algorithm until components count couldn't be reduced
        while (num_trees != prev_num_trees)
        {
            // init cheapest and flatten components
            #pragma omp for schedule(static)
            for (int i = 0; i < vertices_count; i++)
            {
                cheapest[i] = -1;
                components.flatten(i);
            }
            #pragma omp barrier
            
            // find minimal edge from each component
            #pragma omp for schedule(guided, 1024)
            for (long long i = 0; i < edges_count; i++)
            {
                int set1 = components.get_parent(src_ids[i]);
                int set2 = components.get_parent(dst_ids[i]);
                
                if (set1 == set2)
                    continue;
                
                #pragma omp critical
                {
                    long long cheapest_val = cheapest[set1];
                    if(cheapest_val == -1 || weights[i] < weights[cheapest_val])
                    {
                        cheapest[set1] = i;
                    }
                }
                
                #pragma omp critical
                {
                    long long cheapest_val = cheapest[set2];
                    if(cheapest_val == -1 || weights[i] < weights[cheapest_val])
                    {
                        cheapest[set2] = i;
                    }
                }
            }
            
            // remember previous components count
            #pragma omp master
            {
                prev_num_trees = num_trees;
            }
            
            #pragma omp barrier
            
            // merge components
            #pragma omp for schedule(guided, 1024)
            for (int i = 0; i < vertices_count; i++)
            {
                if (cheapest[i] != -1)
                {
                    int set1 = components.get_parent(src_ids[cheapest[i]]);
                    int set2 = components.get_parent(dst_ids[cheapest[i]]);
                    
                    if (set1 == set2)
                        continue;
                    
                    if (components.merge(set1, set2))
                    {
                        _mst_edges[cheapest[i]] = true;
                    }
                    else
                    {
                        #pragma omp atomic
                        num_trees++;
                    }
                    #pragma omp atomic
                    num_trees--;
                }
            }
            #pragma omp barrier
        }
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void MinimumSpanningTree<_TVertexValue, _TEdgeWeight>::gpu_boruvka_kernel(Graph<_TVertexValue, _TEdgeWeight> &_input_graph,
                                                                          bool *_host_mst_edges)
{
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _input_graph.get_graph_data();
    int vertices_count = _input_graph.get_vertices_count();
    long long edges_count = _input_graph.get_edges_count();
    int *host_src_ids = graph_data.edges_src_ids;
    int *host_dst_ids = graph_data.edges_dst_ids;
    _TEdgeWeight *host_weights = graph_data.edges_weights;
    
	// pointers for device edges storage
	int *device_src_ids;
	int *device_dst_ids;
	_TEdgeWeight *device_weights;

	// pointers for results storage
	bool *device_mst_edges;

	// allocate memory for results
	SAFE_CALL(cudaMalloc((void**)&device_mst_edges, edges_count * sizeof(bool)));

	// memory for edges storage on device
	SAFE_CALL(cudaMalloc((void**)&device_src_ids, edges_count * sizeof(int)));
	SAFE_CALL(cudaMalloc((void**)&device_dst_ids, edges_count * sizeof(int)));
	SAFE_CALL(cudaMalloc((void**)&device_weights, edges_count * sizeof(_TEdgeWeight)));

	// set result data to 0
	SAFE_CALL(cudaMemset(device_mst_edges, 0, edges_count * sizeof(bool)));

	// copy edges data from host to device
	SAFE_CALL(cudaMemcpy(device_src_ids, host_src_ids, edges_count * sizeof(int), cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(device_dst_ids, host_dst_ids, edges_count * sizeof(int), cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(device_weights, host_weights, edges_count * sizeof(_TEdgeWeight), cudaMemcpyHostToDevice));

	// compute MST on current edges
    double t1 = omp_get_wtime();
	boruvka_wrapper(device_src_ids, device_dst_ids, device_weights, device_mst_edges, vertices_count, edges_count);
    double t2 = omp_get_wtime();
    cout << "Inner CPU perfomance: " << 2 * _input_graph.get_edges_count() / ((t2 - t1) * 1e6) << " GTEPS" << endl << endl;

	// copy results back to host
	SAFE_CALL(cudaMemcpy(_host_mst_edges, device_mst_edges, edges_count * sizeof(bool), cudaMemcpyDeviceToHost));

	// free all memory
	SAFE_CALL(cudaFree(device_src_ids));
	SAFE_CALL(cudaFree(device_dst_ids));
	SAFE_CALL(cudaFree(device_weights));
	SAFE_CALL(cudaFree(device_mst_edges));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// check functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
_TEdgeWeight MinimumSpanningTree<_TVertexValue, _TEdgeWeight>::compute_MST_weight(Graph<_TVertexValue, _TEdgeWeight> &_mst)
{
	_TEdgeWeight result = 0.0;
	_TEdgeWeight c = 0.0;

	for (int vertex_pos = 0; vertex_pos < _mst.get_vertices_count(); vertex_pos++)
	{
		Vertex<_TVertexValue> vertex = _mst.iterate_vertices(vertex_pos);

		for (int edge_pos = 0; edge_pos < _mst.get_vertex_connections_count(vertex_pos); edge_pos++)
		{
			Edge<_TEdgeWeight> edge = _mst.iterate_adjacent_edges(vertex_pos, edge_pos);

			_TEdgeWeight y = edge.weight - c;
			_TEdgeWeight t = result + y;
			c = (t - result) - y;
			result = t;
		}
	}

	return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool MinimumSpanningTree<_TVertexValue, _TEdgeWeight>::check_mst_results(Graph<_TVertexValue, _TEdgeWeight> &_first_mst,
	                                                                     Graph<_TVertexValue, _TEdgeWeight> &_second_mst)
{
	_first_mst.convert_to_adjacency_list();
	_second_mst.convert_to_adjacency_list();
	
	cout << endl;
	cout << "vertices_count: " << _first_mst.get_vertices_count() << " " << _second_mst.get_vertices_count() << endl;
	cout << "edges_count: " << _first_mst.get_edges_count() << " " << _second_mst.get_edges_count() << endl;
	
	bool result = true;

	if (_first_mst.get_vertices_count() != _second_mst.get_vertices_count())
		result = false;
	if (_first_mst.get_edges_count() != _second_mst.get_edges_count())
		result = false;

	_TEdgeWeight first_mst_sum = compute_MST_weight(_first_mst);
	_TEdgeWeight second_mst_sum = compute_MST_weight(_second_mst);

	cout << "MST weights: " << first_mst_sum << " " << second_mst_sum << endl;
	cout << "Difference: " << fabs(first_mst_sum - second_mst_sum) << endl;
	if (fabs(first_mst_sum - second_mst_sum) > numeric_limits<_TEdgeWeight>::epsilon())
		result = false;

	return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

