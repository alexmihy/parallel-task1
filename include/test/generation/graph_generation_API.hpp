/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphGenerationAPI<_TVertexValue, _TEdgeWeight>::random_uniform(Graph<_TVertexValue, _TEdgeWeight> &_graph, int _vertices_count,
															         int _average_degree, bool _directed, bool _optimized)
{
    if(_optimized)
        _graph.convert_to_adjacency_list();
    
	// check input parameters correctness
	if (!_directed)
		_average_degree /= 2;

	if (_average_degree > _vertices_count)
		throw "average connections in graph is greater than number of vertices";

	if ((_directed == true) && (_graph.check_if_directed() == false))
		throw "creating directed graph in non-directed container";

	// add vertices to graph
	_graph.empty();
	for (int cur_vertex = 0; cur_vertex < _vertices_count; cur_vertex++)
	{
		_TVertexValue vertex_val = rand_uniform_val<_TVertexValue>(1000);
		_graph.add_vertex(cur_vertex, vertex_val);
	}

	long long edges_count = (long long)_vertices_count * _average_degree;
	for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
	{
		int from = rand() % _vertices_count;
		int to = rand() % _vertices_count;
		_TEdgeWeight weight = rand_uniform_val<_TEdgeWeight>(10000);

        // add edge
        if(_optimized)
        {
            // eliminate self loop
            if(to == from)
            {
                cur_edge--;
                continue;
            }
            
            // eleminate repetitive edges
            if (_graph.get_edge_by_id(from, to) == -1)
            {
                _graph.add_edge(from, to, weight);
            }
            else
            {
                cur_edge--;
                continue;
            }
            if ((_directed == false) && (_graph.check_if_directed() == true))
            {
                _graph.add_edge(to, from, weight);
            }
        }
        else
        {
            _graph.add_edge(from, to, weight);
            if ((_directed == false) && (_graph.check_if_directed() == true))
            {
                _graph.add_edge(to, from, weight);
            }
        }
	}
    
    _graph.convert_to_edges_list();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphGenerationAPI<_TVertexValue, _TEdgeWeight>::SSCA2(Graph<_TVertexValue, _TEdgeWeight> &_graph, int _vertices_count, 
	                                                        int _max_clique_size, bool _directed)
{
    uint32_t TotVertices;
	uint32_t* clusterSizes;
	uint32_t* firstVsInCluster;
	uint32_t estTotClusters, totClusters;
	
	uint32_t *startVertex, *endVertex;
	uint32_t numEdges;
    uint32_t numIntraClusterEdges, numInterClusterEdges;
	_TEdgeWeight* weights;
	_TEdgeWeight MinWeight, MaxWeight;
    uint32_t MaxCliqueSize;
    uint32_t MaxParallelEdges = 1;
    double ProbUnidirectional = 1.0;
    double ProbIntercliqueEdges = 0.1;
	uint32_t i_cluster, currCluster;
	uint32_t *startV, *endV, *d;
	uint32_t estNumEdges, edgeNum;

	uint32_t i, j, k, t, t1, t2, dsize;
	double p;
	uint32_t* permV;

    // initialize RNG 

    MinWeight = 0.0;
    MaxWeight = 1.0;
	TotVertices = _vertices_count;

	// generate clusters
	MaxCliqueSize = _max_clique_size;
	estTotClusters = 1.25 * TotVertices / (MaxCliqueSize/2);
	clusterSizes = (uint32_t *) malloc(estTotClusters*sizeof(uint32_t));

	for(i = 0; i < estTotClusters; i++) 
	{
		clusterSizes[i] = 1 + (rand_uniform_val<double>(10000.0) *MaxCliqueSize);
	}
	
	totClusters = 0;

	firstVsInCluster = (uint32_t *) malloc(estTotClusters*sizeof(uint32_t));

	firstVsInCluster[0] = 0;
	for (i=1; i<estTotClusters; i++) 
	{
		firstVsInCluster[i] = firstVsInCluster[i-1] + clusterSizes[i-1];
		if (firstVsInCluster[i] > TotVertices-1)
			break;
	}

	totClusters = i;

	clusterSizes[totClusters-1] = TotVertices - firstVsInCluster[totClusters-1];

	// generate intra-cluster edges
	estNumEdges = (uint32_t) ((TotVertices * (double) MaxCliqueSize * (2-ProbUnidirectional)/2) +
		      	        (TotVertices * (double) ProbIntercliqueEdges/(1-ProbIntercliqueEdges))) * (1+MaxParallelEdges/2);

	if ((estNumEdges > ((1<<30) - 1)) && (sizeof(uint32_t*) < 8)) 
	{
		fprintf(stderr, "ERROR: long* should be 8 bytes for this problem size\n");
		fprintf(stderr, "\tPlease recompile the code in 64-bit mode\n");
		exit(-1);
	}

	edgeNum = 0;
	p = ProbUnidirectional;

	fprintf (stderr, "[allocating %3.3f GB memory ... ", (double) 2*estNumEdges*8/(1<<30));

	startV = (uint32_t *) malloc(estNumEdges*sizeof(uint32_t));
	endV = (uint32_t *) malloc(estNumEdges*sizeof(uint32_t));

	fprintf(stderr, "done] ");  

	for (i_cluster=0; i_cluster < totClusters; i_cluster++)
	{
		for (i = 0; i < clusterSizes[i_cluster]; i++)
		{
			for (j = 0; j < i; j++) 
			{
				for (k = 0; k<1 + ((uint32_t)(MaxParallelEdges - 1) * rand_uniform_val<double>(10000.0)); k++)
				{
					startV[edgeNum] = j + \
					firstVsInCluster[i_cluster];	
					endV[edgeNum] = i + \
					firstVsInCluster[i_cluster];
					edgeNum++;
				}	
			}
			
		}
	}
	numIntraClusterEdges = edgeNum;
	
	//connect the clusters
	dsize = (uint32_t) (log((double)TotVertices)/log(2));
	d = (uint32_t *) malloc(dsize * sizeof(uint32_t));
	for (i = 0; i < dsize; i++) {
		d[i] = (uint32_t) pow(2, (double) i);
	}

	currCluster = 0;

	for (i = 0; i < TotVertices; i++) 
	{
		p = ProbIntercliqueEdges;	
		for (j = currCluster; j<totClusters; j++) 
		{
			if ((i >= firstVsInCluster[j]) && (i < firstVsInCluster[j] + clusterSizes[j]))
			{
				currCluster = j;
				break;
			}	
		}
		for (t = 1; t < dsize; t++)
		{
			j = (i + d[t] + (uint32_t)(rand_uniform_val<double>(10000.0) * (d[t] - d[t - 1]))) % TotVertices;
			if ((j<firstVsInCluster[currCluster]) || (j>=firstVsInCluster[currCluster] + clusterSizes[currCluster]))
			{
				for (k = 0; k<1 + ((uint32_t)(MaxParallelEdges - 1)* rand_uniform_val<double>(10000.0)); k++)
				{
					if (p >  rand_uniform_val<double>(10000.0)) 
					{
						startV[edgeNum] = i;
						endV[edgeNum] = j;
						edgeNum++;	
					}	
				}	
			}
			p = p/2;
		}
	}
	
	numEdges = edgeNum;
	numInterClusterEdges = numEdges - numIntraClusterEdges;	

	free(clusterSizes);  
	free(firstVsInCluster);
	free(d);

	fprintf(stderr, "done\n");
	fprintf(stderr, "\tNo. of inter-cluster edges - %d\n", numInterClusterEdges);
	fprintf(stderr, "\tTotal no. of edges - %d\n", numEdges);

	// shuffle vertices to remove locality	
	fprintf(stderr, "Shuffling vertices to remove locality ... ");
	fprintf(stderr, "[allocating %3.3f GB memory ... ", (double)(TotVertices + 2 * numEdges) * 8 / (1 << 30));

	permV = (uint32_t *)malloc(TotVertices*sizeof(uint32_t));
	startVertex = (uint32_t *)malloc(numEdges*sizeof(uint32_t));
	endVertex = (uint32_t *)malloc(numEdges*sizeof(uint32_t));

	for (i = 0; i<TotVertices; i++) 
	{
		permV[i] = i;
	}

	for (i = 0; i<TotVertices; i++) 
	{
		t1 = i + rand_uniform_val<double>(10000.0) * (TotVertices - i);
		if (t1 != i)
		{
			t2 = permV[t1];
			permV[t1] = permV[i];
			permV[i] = t2;
		}
	}

	for (i = 0; i<numEdges; i++) 
	{
		startVertex[i] = permV[startV[i]];
		endVertex[i] = permV[endV[i]];
	}

	free(startV);
	free(endV);
	free(permV);

	// generate edge weights

	fprintf(stderr, "Generating edge weights ... ");
	weights = (_TEdgeWeight *)malloc(numEdges*sizeof(_TEdgeWeight));
	for (i = 0; i<numEdges; i++) 
	{
		weights[i] = MinWeight + (_TEdgeWeight)(MaxWeight - MinWeight) * rand_uniform_val<double>(10000.0);
	}
  
	vector<vector<uint32_t>> dests(TotVertices);
	vector<vector<_TEdgeWeight>> weight_vect(TotVertices);

	// add data to vertices to graph
	_graph.empty();
	for (uint32_t i = 0; i < TotVertices; i++)
	{
		_graph.add_vertex(i, rand()%100);
	}
    
    // add edges to graph
	for (uint32_t i = 0; i < numEdges; i++) 
	{
		_graph.add_edge(startVertex[i], endVertex[i], weights[i]);
        _graph.add_edge(endVertex[i], startVertex[i], weights[i]);
        /*if ((_directed == false) && (_graph.check_if_directed() == true))
        {
            _graph.add_edge(endVertex[i], startVertex[i], weights[i]);
        }*/
	}
	fprintf(stderr, "done\n");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphGenerationAPI<_TVertexValue, _TEdgeWeight>::R_MAT(Graph<_TVertexValue, _TEdgeWeight> &_graph, int _vertices_count, 
	                                                        int _average_connections, int _a_prob, int _b_prob, int _c_prob, int _d_prob, 
															bool _directed, bool _optimized)
{
    if(_optimized)
        _graph.convert_to_adjacency_list();
        
    // check input parameters correctness
    if (!_directed)
        _average_connections /= 2;
    
    if (_average_connections > _vertices_count)
        throw "average connections in graph is greater than number of vertices";
    
    if ((_directed == true) && (_graph.check_if_directed() == false))
        throw "creating directed graph in non-directed container";
    
    int n = (int)log2(_vertices_count);
    long long max_edges_count = _vertices_count * (long long)_average_connections;
    
    // add vertices to graph
    _graph.empty();
    _graph.resize(_vertices_count, max_edges_count);
    _graph.set_directed(_directed);
    for (int cur_vertex = 0; cur_vertex < _vertices_count; cur_vertex++)
    {
        _TVertexValue vertex_val = rand_uniform_val<_TVertexValue>(3000);
        _graph.add_vertex(cur_vertex, vertex_val);
    }
    
    // generate and add edges to graph
    for (long long cur_edge = 0; cur_edge < max_edges_count; cur_edge++)
    {
        int x_middle = _vertices_count / 2, y_middle = _vertices_count / 2;
        for (long long i = 1; i < n; i++)
        {
            int a_beg = 0, a_end = _a_prob;
            int b_beg = _a_prob, b_end = b_beg + _b_prob;
            int c_beg = _a_prob + _b_prob, c_end = c_beg + _c_prob;
            int d_beg = _a_prob + _b_prob + _c_prob, d_end = d_beg + _d_prob;
            
            int step = (int)pow(2, n - (i + 1));
            
            int probability = rand() % 100;
            if (a_beg <= probability && probability < a_end)
            {
                x_middle -= step, y_middle -= step;
            }
            else if (b_beg <= probability && probability < b_end)
            {
                x_middle -= step, y_middle += step;
            }
            else if (c_beg <= probability && probability < c_end)
            {
                x_middle += step, y_middle -= step;
            }
            else if (d_beg <= probability && probability < d_end)
            {
                x_middle += step, y_middle += step;
            }
        }
        if (rand() % 2 == 0)
            x_middle--;
        if (rand() % 2 == 0)
            y_middle--;
        
        int from = x_middle;
        int to = y_middle;
        _TEdgeWeight edge_weight = rand_uniform_val<_TEdgeWeight>(100000);
        
        // add edge
        if(_optimized)
        {
            // eliminate self loop
            if(to == from)
            {
                cur_edge--;
                continue;
            }
            
            // eleminate repetitive edges
            if (_graph.get_edge_by_id(from, to) == -1)
            {
                _graph.add_edge(from, to, edge_weight);
            }
            else
            {
                cur_edge--;
                continue;
            }
            if ((_directed == false) && (_graph.check_if_directed() == true))
            {
                _graph.add_edge(to, from, edge_weight);
            }
        }
        else
        {
            _graph.add_edge(from, to, edge_weight);
            if ((_directed == false) && (_graph.check_if_directed() == true))
            {
                _graph.add_edge(to, from, edge_weight);
            }
        }
    }
    
    //_graph.convert_to_edges_list();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphGenerationAPI<_TVertexValue, _TEdgeWeight>::R_MAT_parallel(Graph<_TVertexValue, _TEdgeWeight> &_graph, int _vertices_count,
                                                                     int _average_connections, int _a_prob, int _b_prob, int _c_prob,
                                                                     int _d_prob, int _omp_threads, bool _directed)
{
    _graph.convert_to_edges_list();
    
    int n = (int)log2(_vertices_count);
    int vertices_count = _vertices_count;
    long long edges_count = _vertices_count * _average_connections;
    
    if(_directed)
        _graph.resize(_vertices_count, edges_count, false);
    else
        _graph.resize(_vertices_count, 2*edges_count, false);
    
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int *src_ids = graph_data.edges_src_ids;
    int *dst_ids = graph_data.edges_dst_ids;
    _TEdgeWeight *weights = graph_data.edges_weights;
    
    cout << "using " << _omp_threads << " threads" << endl;
    
    // generate and add edges to graph
    unsigned int seed = 0;
    #pragma omp parallel num_threads(_omp_threads) private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();
        
        #pragma omp for schedule(static)
        for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
        {
            int x_middle = _vertices_count / 2, y_middle = _vertices_count / 2;
            for (long long i = 1; i < n; i++)
            {
                int a_beg = 0, a_end = _a_prob;
                int b_beg = _a_prob, b_end = b_beg + _b_prob;
                int c_beg = _a_prob + _b_prob, c_end = c_beg + _c_prob;
                int d_beg = _a_prob + _b_prob + _c_prob, d_end = d_beg + _d_prob;
            
                int step = (int)pow(2, n - (i + 1));
            
                int probability = rand_r(&seed) % 100;
                if (a_beg <= probability && probability < a_end)
                {
                    x_middle -= step, y_middle -= step;
                }
                else if (b_beg <= probability && probability < b_end)
                {
                    x_middle -= step, y_middle += step;
                }
                else if (c_beg <= probability && probability < c_end)
                {
                    x_middle += step, y_middle -= step;
                }
                else if (d_beg <= probability && probability < d_end)
                {
                    x_middle += step, y_middle += step;
                }
            }
            if (rand_r(&seed) % 2 == 0)
                x_middle--;
            if (rand_r(&seed) % 2 == 0)
                y_middle--;
        
            int from = x_middle;
            int to = y_middle;
            _TEdgeWeight edge_weight = static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);
        
            src_ids[cur_edge] = from;
            dst_ids[cur_edge] = to;
            weights[cur_edge] = edge_weight;
            
            if(!_directed)
            {
                src_ids[cur_edge] = to;
                dst_ids[cur_edge] = from;
                weights[cur_edge] = edge_weight;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphGenerationAPI<_TVertexValue, _TEdgeWeight>::convert_into_single_SCC(Graph<_TVertexValue, _TEdgeWeight> &_graph, 
	                                                                          bool _directed, NodeData &_node_data)
{
	/*if (!_directed)
	{
		ShortestPaths<_TVertexValue, _TEdgeWeight> shortest_paths;

		vector<int> source_vertex(1);
		source_vertex[0] = 0;

		Check_reachability<_TEdgeWeight> root_reachability(_graph.get_vertices_count());
		shortest_paths.single_node_bellman_ford(_graph, source_vertex, _node_data, &root_reachability, USE_CPU_MODE);
		_graph.convert_to_adjacency_list();

		for (auto cur_vertex : root_reachability.not_reachable_vertices)
		{
			_TEdgeWeight edge_weight = rand_uniform_val<_TEdgeWeight>(1000);
			_graph.add_edge(Edge<_TEdgeWeight>(0, cur_vertex, edge_weight));
			_graph.add_edge(Edge<_TEdgeWeight>(cur_vertex, 0, edge_weight));
		}

		cout << root_reachability.not_reachable_vertices.size() << " edges have been added!" << endl;
	}
	else
	{
		throw "convert_into_single_CC not implemented yet for directed graph";
	}*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
