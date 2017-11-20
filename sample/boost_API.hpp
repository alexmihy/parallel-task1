/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BoostAPI<_TVertexValue, _TEdgeWeight>::convert_to_boost_graph(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, 
																   GraphBoostDirected **_output_graph)
{
	int vertices_count = _input_graph.get_vertices_count();
	long long edges_count = _input_graph.get_edges_count();

	vector<pair<int, int>> edges_boost(edges_count);
	vector<_TEdgeWeight> weights_boost(edges_count);

	int boost_edge_pos = 0;
	for (int vertex_pos = 0; vertex_pos < _input_graph.get_vertices_count(); vertex_pos++)
	{
		for (int edge_pos = 0; edge_pos < _input_graph.get_vertex_connections_count(vertex_pos); edge_pos++)
		{
			Edge<_TEdgeWeight> edge = _input_graph.iterate_adjacent_edges(vertex_pos, edge_pos);
			edges_boost[boost_edge_pos] = pair <int, int>(edge.src_id, edge.dst_id);
			weights_boost[boost_edge_pos] = edge.weight;
			boost_edge_pos++;
		}
	}

	*_output_graph = new GraphBoostDirected(edges_boost.begin(), edges_boost.end(), weights_boost.begin(), vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BoostAPI<_TVertexValue, _TEdgeWeight>::convert_to_boost_graph(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, 
																   GraphBoostUndirected **_output_graph)
{
	int vertices_count = _input_graph.get_vertices_count();
	long long edges_count = _input_graph.get_edges_count();

	vector<pair<int, int>> edges_boost(edges_count);
	vector<_TEdgeWeight> weights_boost(edges_count);

	int boost_edge_pos = 0;
	for (int vertex_pos = 0; vertex_pos < _input_graph.get_vertices_count(); vertex_pos++)
	{
		for (int edge_pos = 0; edge_pos < _input_graph.get_vertex_connections_count(vertex_pos); edge_pos++)
		{
			Edge<_TEdgeWeight> edge = _input_graph.iterate_adjacent_edges(vertex_pos, edge_pos);
			edges_boost[boost_edge_pos] = pair <int, int>(edge.src_id, edge.dst_id);
			weights_boost[boost_edge_pos] = edge.weight;
			boost_edge_pos++;
		}
	}

	*_output_graph = new GraphBoostUndirected(edges_boost.begin(), edges_boost.end(), weights_boost.begin(), vertices_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
double BoostAPI<_TVertexValue, _TEdgeWeight>::prim_minimum_spanning_tree(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, 
																         Graph<_TVertexValue, _TEdgeWeight> &_output_MST)
{
    _input_graph.convert_to_adjacency_list();
    
	typedef adjacency_list<vecS, vecS, undirectedS, property<vertex_distance_t, int>, property <edge_weight_t, float> > gl_boost;

	gl_boost *boost_graph;
	convert_to_boost_graph(_input_graph, &boost_graph);

	vector <graph_traits <gl_boost>::vertex_descriptor> p(num_vertices(*boost_graph));

	double t1 = omp_get_wtime();
	boost::prim_minimum_spanning_tree(*boost_graph, &p[0]);
	double t2 = omp_get_wtime();
	cout << "boost prim time without conversion: " << t2 - t1 << " sec " << endl;
	cout << "boost prim perfomance: " << 2*_input_graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl;

	_output_MST.empty();
	for (int vertex_pos = 0; vertex_pos < _input_graph.get_vertices_count(); vertex_pos++)
		_output_MST.add_vertex(_input_graph.iterate_vertices(vertex_pos));
	for (size_t i = 0; i != p.size(); ++i)
	{
		if (p[i] != i)
		{
			Edge<_TEdgeWeight> edge = _input_graph.get_edge_by_id(i, p[i]);
			_output_MST.add_edge(i, p[i], edge.weight);
		}
	}
	return t2 - t1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BoostAPI<_TVertexValue, _TEdgeWeight>::kruskal_minimum_spanning_tree(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, 
																            Graph<_TVertexValue, _TEdgeWeight> &_output_MST)
{
    // create new type
    using namespace boost;
	typedef adjacency_list < vecS, vecS, undirectedS, no_property, property < edge_weight_t, float > > gl_boost;
    
    // create boost graph
	gl_boost *boost_graph;
    int vertices_count = _input_graph.get_vertices_count();
    long long edges_count = _input_graph.get_edges_count();
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _input_graph.get_graph_data();
    vector<pair<int, int>> edges_boost(edges_count);
    vector<_TEdgeWeight> weights_boost(edges_count);
    for(long long i = 0; i < edges_count; i++)
    {
        edges_boost[i] = pair <int, int>(graph_data.edges_src_ids[i], graph_data.edges_dst_ids[i]);
        weights_boost[i] = graph_data.edges_weights[i];
    }
    boost_graph = new gl_boost(edges_boost.begin(), edges_boost.end(), weights_boost.begin(), vertices_count);
    property_map < gl_boost, edge_weight_t >::type weights = get(edge_weight, *boost_graph);
	
    // perform computations
	typedef graph_traits < gl_boost >::edge_descriptor boostEdge;
    
	vector <boostEdge> spanning_tree;

	double t1 = omp_get_wtime();
	boost::kruskal_minimum_spanning_tree(*boost_graph, back_inserter(spanning_tree));
	double t2 = omp_get_wtime();
	cout << "kruskal time without data conversion: " << t2 - t1 << " sec " << endl;
    cout << "boost kruskal perfomance: " << 2*_input_graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl;
    
    // save results
	_output_MST.empty();
	for (int vertex_pos = 0; vertex_pos < _input_graph.get_vertices_count(); vertex_pos++)
		_output_MST.add_vertex(_input_graph.iterate_vertices(vertex_pos));

	for (auto ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei)
	{
		int src = source(*ei, *boost_graph);
		int dst = target(*ei, *boost_graph);
        float weight = weights[*ei];
		if (src != dst)
		{
			_output_MST.add_edge(src, dst, weight);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BoostAPI<_TVertexValue, _TEdgeWeight>::dijkstra_shortest_paths(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, vector<int> _source_vertices,
	                                                                ShortestPathsSaveResult<_TEdgeWeight> *_save_obj)
{
	typedef adjacency_list<vecS, vecS, directedS, property<vertex_distance_t, int>, property <edge_weight_t, float> > gl_boost;

	typedef graph_traits <gl_boost>::vertex_descriptor vertex_descriptor;

	gl_boost *boost_graph;
	convert_to_boost_graph(_input_graph, &boost_graph);

	property_map<gl_boost, edge_weight_t>::type weightmap = get(edge_weight, *boost_graph);
	vector<vertex_descriptor> p(num_vertices(*boost_graph));

	vector<_TEdgeWeight> output_distances(_input_graph.get_vertices_count());

	double t1 = omp_get_wtime();
	_save_obj->initialize();
	for (int i = 0; i < _source_vertices.size(); i++)
	{
		vertex_descriptor s = vertex(_source_vertices[i], *boost_graph);
		boost::dijkstra_shortest_paths(*boost_graph, s, predecessor_map(make_iterator_property_map(p.begin(), get(vertex_index, *boost_graph))).
			                           distance_map(make_iterator_property_map(output_distances.begin(), get(vertex_index, *boost_graph))));
		_save_obj->save_data(_source_vertices[i], &output_distances.front());
	}
	_save_obj->finalize();
	double t2 = omp_get_wtime();
	cout << "boost dijkstra time: " << t2 - t1 << " sec" << endl;
	cout << "boost dikstra perfomance: " << _source_vertices.size() * _input_graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS";
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BoostAPI<_TVertexValue, _TEdgeWeight>::bellman_ford_shortest_paths(Graph<_TVertexValue, _TEdgeWeight> &_input_graph,
	                                                                    vector<_TEdgeWeight> &_output_distances)
{
	/*typedef graph_traits <GraphBoost>::vertex_descriptor vertex_descriptor;

	GraphBoost boost_graph = convert_to_boost_graph(_input_graph, true);

	property_map<GraphBoost, edge_weight_t>::type weightmap = get(edge_weight, boost_graph);
	vector<vertex_descriptor> p(num_vertices(boost_graph));

	vertex_descriptor s = vertex(0, boost_graph);

	double t1 = omp_get_wtime();
	boost::bellman_ford_shortest_paths(boost_graph, 0, predecessor_map(make_iterator_property_map(p.begin(), get(vertex_index, boost_graph))).
		                               distance_map(make_iterator_property_map(_output_distances.begin(), get(vertex_index, boost_graph))));
	double t2 = omp_get_wtime();
	return t2 - t1;*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BoostAPI<_TVertexValue, _TEdgeWeight>::tarjan_scc(Graph<_TVertexValue, _TEdgeWeight> &_graph, vector<int> &_scc_result)
{
    using namespace boost;

    _graph.convert_to_edges_list();
    
    // get vertices and edges count
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    
    // create boost graph
    adjacency_list<vecS, vecS, directedS> boost_graph;
    for(int i = 0; i < vertices_count; i++)
    {
        add_vertex(boost_graph);
    }
    for(long long i = 0; i < edges_count; i++)
    {
        Edge<_TEdgeWeight> edge = _graph.iterate_edges(i);
        add_edge(edge.src_id, edge.dst_id, boost_graph);
    }
    typedef graph_traits<adjacency_list<vecS, vecS, directedS> >::vertex_descriptor Vertex;
    
    double t1 = omp_get_wtime();
    _scc_result.resize(vertices_count);
    vector<int> discover_time(vertices_count);
    vector<default_color_type> color(vertices_count);
    vector<Vertex> root(num_vertices(boost_graph));
    strong_components(boost_graph, make_iterator_property_map(_scc_result.begin(), get(vertex_index, boost_graph)),
                      root_map(make_iterator_property_map(root.begin(), get(vertex_index, boost_graph))).
                      color_map(make_iterator_property_map(color.begin(), get(vertex_index, boost_graph))).
                      discover_time_map(make_iterator_property_map(discover_time.begin(), get(vertex_index, boost_graph))));
    double t2 = omp_get_wtime();
    
    // print main perfomance
    double processing_time = t2 - t1;
    double perfomance = edges_count / ((processing_time) * 1e6);
    cout << "boost tarjan time: " << processing_time << " sec" << endl;
    cout << "boost tarjan perfomance: " << perfomance << " MTEPS" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BoostAPI<_TVertexValue, _TEdgeWeight>::VF2(Graph<_TVertexValue, _TEdgeWeight> &_big_graph, Graph<_TVertexValue, _TEdgeWeight> &_small_graph)
{
    cout << "Boost check" << endl;
    
    typedef adjacency_list<setS, vecS, bidirectionalS> graph_type;
    
    // Build graph1
    int num_vertices1 = _big_graph.get_vertices_count();
    graph_type graph1(num_vertices1);
    for(int src_id = 0; src_id < num_vertices1; src_id++)
    {
        for(int edge_pos = 0; edge_pos < _big_graph.get_vertex_connections_count(src_id); edge_pos++)
        {
            Edge<_TEdgeWeight> edge = _big_graph.iterate_adjacent_edges(src_id, edge_pos);
            int dst_id = edge.dst_id;
            if(src_id <= dst_id)
                add_edge(src_id, dst_id, graph1);
        }
    }
    
    // Build graph2
    int num_vertices2 = _small_graph.get_vertices_count();
    graph_type graph2(num_vertices2);
    for(int src_id = 0; src_id < num_vertices2; src_id++)
    {
        for(int edge_pos = 0; edge_pos < _small_graph.get_vertex_connections_count(src_id); edge_pos++)
        {
            Edge<_TEdgeWeight> edge = _small_graph.iterate_adjacent_edges(src_id, edge_pos);
            int dst_id = edge.dst_id;
            if(src_id <= dst_id)
                add_edge(src_id, dst_id, graph2);
        }
    }
    
    // Create callback to print mappings
    vf2_print_callback<graph_type, graph_type> callback(graph2, graph1);
    
    // Print out all subgraph isomorphism mappings between graph1 and graph2.
    // Vertices and edges are assumed to be always equivalent.
    vf2_subgraph_iso(graph2, graph1, callback);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////