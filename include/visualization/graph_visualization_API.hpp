/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphVisualizationAPI<_TVertexValue, _TEdgeWeight>::print_vertices_matrix(Graph<_TVertexValue, _TEdgeWeight> &_graph)
{
    cout << "graph profile: " << endl;
    for (int i = 0; i < _graph.get_vertices_count(); i++)
    {
        for (int j = 0; j < _graph.get_vertices_count(); j++)
        {
			Edge<_TEdgeWeight> result = _graph.get_edge_by_id(i, j);
            if (result.dst_id == -1)
                cout << "0 ";
            else
                cout << "1 ";
        }
        cout << endl;
    }
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphVisualizationAPI<_TVertexValue, _TEdgeWeight>::print_container_data(Graph<_TVertexValue, _TEdgeWeight> &_graph)
{
	cout << endl << "gpaph is: " << (_graph.check_if_directed() ? "directed" : "non-directed") << endl;

	cout << endl << "container data: " << endl;
	if (_graph.get_container_type() == ADJACENCY_LIST || _graph.get_container_type() == COMPRESSED_ADJACENCY_LIST)
	{
		
		for (int vertex_pos = 0; vertex_pos < _graph.get_vertices_count(); vertex_pos++)
		{
			Vertex<_TVertexValue> vertex = _graph.iterate_vertices(vertex_pos);
			cout << "vertex id: " << vertex.id << ", value: " << vertex.value << endl;
			cout << "connected to: ";

			for (int edge_pos = 0; edge_pos < _graph.get_vertex_connections_count(vertex_pos); edge_pos++)
			{
				Edge<_TEdgeWeight> edge = _graph.iterate_adjacent_edges(vertex_pos, edge_pos);
				cout << " [" << edge.dst_id << ",  " << edge.weight << "]" << " ";
			}
			cout << endl << endl;
		}
		cout << endl;
	}
	else if (_graph.get_container_type() == EDGES_LIST)
	{
		throw "EDGES_LIST in print_container_data is not suppored yet";
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphVisualizationAPI<_TVertexValue, _TEdgeWeight>::create_graphviz_file(Graph<_TVertexValue, _TEdgeWeight> &_graph, 
	                                                                          string _output_file_name,
                                                                              GraphvizMode _mode)
{
	ofstream dot_output(_output_file_name.c_str());

	bool directed = _graph.check_if_directed();

	string connection;
	if (!directed || (_mode == VISUALIZE_AS_UNDIRECTED))
	{
        dot_output << "graph G {" << endl;
        connection = " -- ";
	}
	else
	{
        dot_output << "digraph G {" << endl;
        connection = " -> ";
	}
    
    for(int vertex_pos = 0; vertex_pos < _graph.get_vertices_count(); vertex_pos++)
    {
        Vertex<_TVertexValue> vertex = _graph.iterate_vertices(vertex_pos);
        dot_output << vertex.id << " [label= \"id=" << vertex.id << ", value=" << vertex.value << "\"] "<< endl;
    }
    
	if (_graph.get_container_type() == ADJACENCY_LIST || _graph.get_container_type() == COMPRESSED_ADJACENCY_LIST)
	{
		for (int vertex_pos = 0; vertex_pos < _graph.get_vertices_count(); vertex_pos++)
		{
			for (int edge_pos = 0; edge_pos < _graph.get_vertex_connections_count(vertex_pos); edge_pos++)
			{
				Edge<_TEdgeWeight> edge = _graph.iterate_adjacent_edges(vertex_pos, edge_pos);
                
                if((_mode == VISUALIZE_AS_UNDIRECTED) && (edge.src_id > edge.dst_id))
                    continue;
                
				dot_output << edge.src_id << connection << edge.dst_id << " [label = \" " << edge.weight << " \"];" << endl;
			}
		}
	}
	else if (_graph.get_container_type() == EDGES_LIST)
	{
		for (int edge_pos = 0; edge_pos < _graph.get_edges_count(); edge_pos++)
		{
			Edge<_TEdgeWeight> edge = _graph.iterate_edges(edge_pos);
            
            if((_mode == VISUALIZE_AS_UNDIRECTED) && (edge.src_id > edge.dst_id))
                continue;
            
			dot_output << edge.src_id << connection << edge.dst_id << " [label = \" " << edge.weight << " \"];" << endl;
		}
	}
	else
	{
		throw "unsupported graph format in create_graphviz_file function";
	}

	dot_output << "}";

	dot_output.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
