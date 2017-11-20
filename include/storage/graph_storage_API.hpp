/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool GraphStorageAPI<_TVertexValue, _TEdgeWeight>::save_to_adjacency_list_bin_file(Graph<_TVertexValue, _TEdgeWeight> &_graph, string _file_name)
{
    ofstream graph_file(_file_name.c_str(), ios::binary);
	if (!graph_file.is_open())
		return false;

	bool directed = _graph.check_if_directed();
	int vertices_count = _graph.get_vertices_count();
	long long edges_count = _graph.get_edges_count();
	graph_file.write(reinterpret_cast<const char*>(&directed), sizeof(bool));
	graph_file.write(reinterpret_cast<const char*>(&vertices_count), sizeof(int));
	graph_file.write(reinterpret_cast<const char*>(&edges_count), sizeof(long long));

	for (int vertex_pos = 0; vertex_pos < _graph.get_vertices_count(); vertex_pos++)
    {
        // save vertex data
		Vertex<_TVertexValue>tmp_vertex = _graph.iterate_vertices(vertex_pos);
		int connections_count = _graph.get_vertex_connections_count(vertex_pos);
        graph_file.write(reinterpret_cast<const char*>(&tmp_vertex), sizeof(Vertex<_TVertexValue>));
        graph_file.write(reinterpret_cast<const char*>(&connections_count), sizeof(int));

        // save edges data
		for (int edge_pos = 0; edge_pos < _graph.get_vertex_connections_count(vertex_pos); edge_pos++)
        {
			Edge<_TEdgeWeight> edge = _graph.iterate_adjacent_edges(vertex_pos, edge_pos);
			graph_file.write(reinterpret_cast<const char*>(&edge.dst_id), sizeof(int));
			graph_file.write(reinterpret_cast<const char*>(&edge.weight), sizeof(_TEdgeWeight));
        }
    }
    graph_file.close();
	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool GraphStorageAPI<_TVertexValue, _TEdgeWeight>::save_to_edges_list_bin_file(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                               string _file_name)
{
    ofstream graph_file(_file_name.c_str(), ios::binary);
    if (!graph_file.is_open())
        return false;
    
    bool directed = _graph.check_if_directed();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    
    graph_file.write(reinterpret_cast<const char*>(&directed), sizeof(bool));
    graph_file.write(reinterpret_cast<const char*>(&vertices_count), sizeof(int));
    graph_file.write(reinterpret_cast<const char*>(&edges_count), sizeof(long long));
    
    for (long long i = 0; i < edges_count; i++)
    {
        Edge<_TEdgeWeight> edge = _graph.iterate_edges(i);
        graph_file.write(reinterpret_cast<const char*>(&edge.src_id), sizeof(int));
        graph_file.write(reinterpret_cast<const char*>(&edge.dst_id), sizeof(int));
        graph_file.write(reinterpret_cast<const char*>(&edge.weight), sizeof(_TEdgeWeight));
    }
    
    graph_file.close();
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool GraphStorageAPI<_TVertexValue, _TEdgeWeight>::load_from_adjacency_list_bin_file(Graph<_TVertexValue, _TEdgeWeight> &_graph, string _file_name)
{
    ifstream graph_file(_file_name, ios::binary);
	if (!graph_file.is_open())
		return false;

	bool directed = true;
	int vertices_count = 0;
	long long edges_count = 0;
	graph_file.read(reinterpret_cast<char*>(&directed), sizeof(bool));
	graph_file.read(reinterpret_cast<char*>(&vertices_count), sizeof(int));
	graph_file.read(reinterpret_cast<char*>(&edges_count), sizeof(long long));

	_graph.resize(vertices_count, edges_count);

	if (directed != _graph.check_if_directed())
		throw "loading directed graph to non-directed container(or undirected to directed)";
	_graph.set_directed(directed);

	for (int vertex_pos = 0; vertex_pos < vertices_count; vertex_pos++)
    {
        // load vertex data
        Vertex<_TVertexValue> vertex;
        int connections_count = 0;
        graph_file.read(reinterpret_cast<char*>(&vertex), sizeof(Vertex<_TVertexValue>));
        graph_file.read(reinterpret_cast<char*>(&connections_count), sizeof(int));

        // load edges data
		vector <int> adg_ids(connections_count);
        vector <_TEdgeWeight> adg_weights(connections_count);
		for (int edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
			graph_file.read(reinterpret_cast<char*>(&adg_ids[edge_pos]), sizeof(int));
			graph_file.read(reinterpret_cast<char*>(&adg_weights[edge_pos]), sizeof(_TEdgeWeight));
        }

		_graph.add_vertex(vertex.id, vertex.value, adg_ids, adg_weights);
    }

    graph_file.close();
	return true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool GraphStorageAPI<_TVertexValue, _TEdgeWeight>::load_from_edges_list_bin_file(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                                 string _file_name)
{
    bool directed = true;
    int vertices_count = 1;
    long long edges_count = 0;

    // open file
    ifstream graph_file(_file_name, ios::binary);
    if (!graph_file.is_open())
        return false;
    
    // read header or get edges count
    graph_file.read(reinterpret_cast<char*>(&directed), sizeof(bool));
    graph_file.read(reinterpret_cast<char*>(&vertices_count), sizeof(int));
    graph_file.read(reinterpret_cast<char*>(&edges_count), sizeof(long long));
    if (directed != _graph.check_if_directed())
        throw "loading directed graph to non-directed container(or undirected to directed)";
    
    _graph.set_directed(directed);
    _graph.resize(vertices_count, edges_count);
    
    std::map<int, int>vertices_map;
    
    // add edges from file
    for (long long i = 0; i < edges_count; i++)
    {
        int src_id = 0, dst_id = 0;
        _TEdgeWeight weight = 0;
        
        graph_file.read(reinterpret_cast<char*>(&src_id), sizeof(int));
        graph_file.read(reinterpret_cast<char*>(&dst_id), sizeof(int));
        graph_file.read(reinterpret_cast<char*>(&weight), sizeof(_TEdgeWeight));
        
        _graph.add_edge(src_id, dst_id, weight);
    }
    
    // add vertices from file
    for (int i = 0; i < vertices_count; i++)
    {
        _graph.add_vertex(i, 0);
    }
    
    graph_file.close();
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
