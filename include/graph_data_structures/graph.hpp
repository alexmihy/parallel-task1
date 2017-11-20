/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Graph<_TVertexValue, _TEdgeWeight>::Graph()
{
	current_container_type = ADJACENCY_LIST;
	data = new AdjacencyList<_TVertexValue, _TEdgeWeight>(0, true);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Graph<_TVertexValue, _TEdgeWeight>::Graph(int _vertices_count, long long _edges_count, ContainerType _container_type, bool _directed, 
									      bool _empty)
{
	current_container_type = _container_type;
	data = create_new_container(current_container_type, _vertices_count, _edges_count, _directed, _empty);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Graph<_TVertexValue, _TEdgeWeight>::~Graph()
{
	if (data != NULL)
		delete data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::resize(int _new_vertices_count, long long _new_edges_count, bool _empty)
{
	bool was_directed = data->check_if_directed();
	if (data != NULL)
		delete data;
	data = create_new_container(current_container_type, _new_vertices_count, _new_edges_count, was_directed, _empty);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Vertex<_TVertexValue> Graph<_TVertexValue, _TEdgeWeight>::iterate_vertices(int _vertex_pos) const
{
	return data->iterate_vertices(_vertex_pos);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> Graph<_TVertexValue, _TEdgeWeight>::iterate_edges(long long _edge_pos) const
{
	return data->iterate_edges(_edge_pos);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> Graph<_TVertexValue, _TEdgeWeight>::iterate_adjacent_edges(int _src_vertex_id, int _edge_pos) const
{
	return data->iterate_adjacent_edges(_src_vertex_id, _edge_pos);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Vertex<_TVertexValue> Graph<_TVertexValue, _TEdgeWeight>::get_vertex_by_id(int _id) const
{
	return data->get_vertex_by_id(_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> Graph<_TVertexValue, _TEdgeWeight>::get_edge_by_id(int _src_id, int _dst_id) const
{
	return data->get_edge_by_id(_src_id, _dst_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int Graph<_TVertexValue, _TEdgeWeight>::get_vertices_count() const
{
	return data->get_vertices_count();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
long long Graph<_TVertexValue, _TEdgeWeight>::get_edges_count() const
{
	return data->get_edges_count();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int Graph<_TVertexValue, _TEdgeWeight>::get_vertex_connections_count(int _id) const
{
	return data->get_vertex_connections_count(_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::set_vertex(int _id, _TVertexValue _value)
{
    return data->set_vertex(_id, _value);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::add_vertex(int _id, _TVertexValue _value)
{
	data->add_vertex(_id, _value);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::add_vertex(Vertex<_TVertexValue> _vertex)
{
	data->add_vertex(_vertex.id, _vertex.value);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::add_vertex(int _id, 
											        _TVertexValue _value, 
													const vector<int> &_adj_ids, 
									                const vector<_TEdgeWeight> &_adj_weights)
{
	data->add_vertex(_id, _value, _adj_ids, _adj_weights);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::add_edge(int _src_id, int _dst_id, _TEdgeWeight _weight)
{
	data->add_edge(_src_id, _dst_id, _weight);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::add_edge(Edge<_TEdgeWeight> _edge)
{
	data->add_edge(_edge.src_id, _edge.dst_id, _edge.weight);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::empty()
{
	data->empty();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphContainerData<_TVertexValue, _TEdgeWeight> Graph<_TVertexValue, _TEdgeWeight>::get_graph_data()
{
	return data->get_container_data();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
long long Graph<_TVertexValue, _TEdgeWeight>::memory_usage()
{
	if (current_container_type == COMPRESSED_ADJACENCY_LIST || current_container_type == ADJACENCY_LIST)
	{
		long long vertices_memory = (sizeof(int) + sizeof(_TVertexValue))*data->get_vertices_count();
		long long edges_memory = (2 * sizeof(int) + sizeof(_TEdgeWeight))*data->get_edges_count();
		return (double) (vertices_memory + edges_memory) / (1000.0 * 1000.0);
	}
	return -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
bool Graph<_TVertexValue, _TEdgeWeight>::operator==(const Graph<_TVertexValue, _TEdgeWeight> &_other) const
{
	if (this->get_vertices_count() != _other.get_vertices_count())
		return false;
	if (this->get_edges_count() != _other.get_edges_count())
		return false;

	if (this->current_container_type == _other.current_container_type)
	{
		if (this->current_container_type == ADJACENCY_LIST)
		{
			for (int vertex_pos = 0; vertex_pos < this->get_vertices_count(); vertex_pos++)
			{
				Vertex<_TVertexValue> first_graph_vertex  = this->iterate_vertices(vertex_pos);
				Vertex<_TVertexValue> second_graph_vertex = _other.iterate_vertices(vertex_pos);

				if (first_graph_vertex != second_graph_vertex)
					return false;

				vector<Edge<_TEdgeWeight>> first_graph_edges;
				vector<Edge<_TEdgeWeight>> second_graph_edges;
				for (int edges_pos = 0; edges_pos < this->get_vertex_connections_count(vertex_pos); edges_pos++)
				{
					first_graph_edges.push_back(this->iterate_adjacent_edges(vertex_pos, edges_pos));
					second_graph_edges.push_back(_other.iterate_adjacent_edges(vertex_pos, edges_pos));
				}
				sort(first_graph_edges.begin(), first_graph_edges.end(), edge_dst_comparator<_TEdgeWeight>);
				sort(second_graph_edges.begin(), second_graph_edges.end(), edge_dst_comparator<_TEdgeWeight>);

				for (int edges_pos = 0; edges_pos < this->get_vertex_connections_count(vertex_pos); edges_pos++)
				{
					if (first_graph_edges[edges_pos] != second_graph_edges[edges_pos])
						return false;
				}
			}
			return true;
		}
		else
		{
			throw "not supported container type for == operator";
		}
	}
	else
	{
		throw "container types are not equal in == operator";
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::operator=(const Graph<_TVertexValue, _TEdgeWeight> &_other)
{
	delete data;
	data = _other.data->clone();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::convert_to_adjacency_list()
{
	if (current_container_type == EDGES_LIST)
	{
		// create new container
		BaseContainer<_TVertexValue, _TEdgeWeight> *new_data = new AdjacencyList <_TVertexValue, _TEdgeWeight>(0, data->check_if_directed());

		// add vertices
		for (int vertex_pos = 0; vertex_pos < data->get_vertices_count(); vertex_pos++)
		{
			Vertex<_TVertexValue> vertex = data->iterate_vertices(vertex_pos);
			new_data->add_vertex(vertex.id, vertex.value);
		}

		// add edges
		for (long long edge_pos = 0; edge_pos < data->get_edges_count(); edge_pos++)
		{
			Edge<_TEdgeWeight> cur_edge = data->iterate_edges(edge_pos);
			new_data->add_edge(cur_edge.src_id, cur_edge.dst_id, cur_edge.weight);
		}

		// swap containers
		delete data;
		data = new_data;
		current_container_type = ADJACENCY_LIST;
	}
	else if (current_container_type == COMPRESSED_ADJACENCY_LIST)
	{
		// create new container
		BaseContainer<_TVertexValue, _TEdgeWeight> *new_data = new AdjacencyList <_TVertexValue, _TEdgeWeight>(0, data->check_if_directed());

		// traverse all vertices and adjacent edges
		for (int vertex_pos = 0; vertex_pos < data->get_vertices_count(); vertex_pos++)
		{
			Vertex<_TVertexValue> vertex = data->iterate_vertices(vertex_pos);
			int vertex_connections_count = get_vertex_connections_count(vertex_pos);

			vector<int> adj_ids(vertex_connections_count);
			vector<_TEdgeWeight> adj_weights(vertex_connections_count);

			for (int edge_pos = 0; edge_pos < vertex_connections_count; edge_pos++)
			{
				Edge<_TEdgeWeight> edge = data->iterate_adjacent_edges(vertex.id, edge_pos);
				adj_ids[edge_pos] = edge.dst_id;
				adj_weights[edge_pos] = edge.weight;
			}
			new_data->add_vertex(vertex.id, vertex.value, adj_ids, adj_weights);
		}

		// swap containers
		delete data;
		data = new_data;
		current_container_type = ADJACENCY_LIST;
	}
	else if (current_container_type == ADJACENCY_LIST)
	{
		// already in required format
		return;
	}
	else
	{
		throw "error in source container type";
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::convert_to_compressed_adjacency_list()
{
	if (current_container_type == ADJACENCY_LIST)
	{
		// create new container
		BaseContainer<_TVertexValue, _TEdgeWeight> *new_data =
			new CompressedAdjacencyList <_TVertexValue, _TEdgeWeight>(data->get_vertices_count(), data->get_edges_count(), data->check_if_directed());

		// traverse all vertices and adjacent edges
		for (int vertex_pos = 0; vertex_pos < data->get_vertices_count(); vertex_pos++)
		{
			Vertex<_TVertexValue> vertex = data->iterate_vertices(vertex_pos);
			int vertex_connections_count = get_vertex_connections_count(vertex_pos);

			vector<int> adj_ids(vertex_connections_count);
			vector<_TEdgeWeight> adj_weights(vertex_connections_count);

			for (int edge_pos = 0; edge_pos < vertex_connections_count; edge_pos++)
			{
				Edge<_TEdgeWeight> edge = data->iterate_adjacent_edges(vertex.id, edge_pos);
				adj_ids[edge_pos] = edge.dst_id;
				adj_weights[edge_pos] = edge.weight;
			}
			new_data->add_vertex(vertex.id, vertex.value, adj_ids, adj_weights);
		}

		// swap containers
		delete data;
		data = new_data;
		current_container_type = COMPRESSED_ADJACENCY_LIST;
	}
	else if (current_container_type == COMPRESSED_ADJACENCY_LIST)
	{
		// already in required format
		return; 
	}
    else if (current_container_type == EDGES_LIST)
    {
        convert_to_adjacency_list();
        convert_to_compressed_adjacency_list();
    }
	else
	{
		throw "not supported format in convert_to_compressed_adjacency_list";
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::convert_to_edges_list()
{
	if (current_container_type == COMPRESSED_ADJACENCY_LIST || current_container_type == ADJACENCY_LIST)
	{
		// create new container
		BaseContainer<_TVertexValue, _TEdgeWeight> *new_data =
			new EdgesList <_TVertexValue, _TEdgeWeight>(data->get_vertices_count(), data->get_edges_count(), data->check_if_directed(), true);

		// traverse all vertices and adjacent edges
		for (int vertex_pos = 0; vertex_pos < data->get_vertices_count(); vertex_pos++)
		{
			Vertex<_TVertexValue> vertex = data->iterate_vertices(vertex_pos);
			new_data->add_vertex(vertex.id, vertex.value);

			for (int edge_pos = 0; edge_pos < data->get_vertex_connections_count(vertex_pos); edge_pos++)
			{
				Edge<_TEdgeWeight> edge = data->iterate_adjacent_edges(vertex.id, edge_pos);
				new_data->add_edge(vertex.id, edge.dst_id, edge.weight);
			}
		}

		// swap containers
		delete data;
		data = new_data;
		current_container_type = EDGES_LIST;
	}
	else if (current_container_type == EDGES_LIST)
	{
		// already in required format
		return; 
	}
	else
	{
		throw "error in source container type";
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::convert_to_directed()
{
    if (current_container_type == COMPRESSED_ADJACENCY_LIST || current_container_type == ADJACENCY_LIST)
    {
        int old_container_type = current_container_type;
        BaseContainer<_TVertexValue, _TEdgeWeight> *directed_data =
                    new AdjacencyList<_TVertexValue, _TEdgeWeight> (0, true);
        for (int vertex_pos = 0; vertex_pos < data->get_vertices_count(); vertex_pos++)
        {
            Vertex<_TVertexValue> vertex = data->iterate_vertices(vertex_pos);
            directed_data->add_vertex(vertex.id, vertex.value);
            for (int edge_pos = 0; edge_pos < data->get_vertex_connections_count(vertex_pos); edge_pos++)
            {
                Edge<_TEdgeWeight> edge = data->iterate_adjacent_edges(vertex.id, edge_pos);
                directed_data->add_edge(vertex.id, edge.dst_id, edge.weight);
                if(vertex.id != edge.dst_id)
                    directed_data->add_edge(edge.dst_id, vertex.id, edge.weight);
            }
        }
        
        // set container
        delete data;
        data = directed_data;
        current_container_type = ADJACENCY_LIST;
        
        // convert to old format if required
        if(old_container_type == COMPRESSED_ADJACENCY_LIST)
        {
            this->convert_to_compressed_adjacency_list();
        }

    }
    else if (current_container_type == EDGES_LIST)
    {
        this->set_directed(true);
        long long old_edges_count = data->get_edges_count();
        for(long long edge_pos = 0; edge_pos < old_edges_count; edge_pos++)
        {
            Edge<_TEdgeWeight> edge = data->iterate_edges(edge_pos);
            if(edge.src_id != edge.dst_id)
            {
                data->add_edge(edge.dst_id, edge.src_id, edge.weight);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
ContainerType Graph<_TVertexValue, _TEdgeWeight>::get_container_type()
{
	return current_container_type;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void Graph<_TVertexValue, _TEdgeWeight>::transpose()
{
    if((current_container_type == COMPRESSED_ADJACENCY_LIST) || (current_container_type == ADJACENCY_LIST))
    {
        int vertices_count = data->get_vertices_count();
        BaseContainer<_TVertexValue, _TEdgeWeight> *transposed_data =
            new AdjacencyList<_TVertexValue, _TEdgeWeight> (0, true);
        
        for (int vertex_pos = 0; vertex_pos < data->get_vertices_count(); vertex_pos++)
        {
            Vertex<_TVertexValue> vertex = data->iterate_vertices(vertex_pos);
            transposed_data->add_vertex(vertex.id, vertex.value);
        }
        
        for (int vertex_pos = 0; vertex_pos < data->get_vertices_count(); vertex_pos++)
        {
            for (int edge_pos = 0; edge_pos < data->get_vertex_connections_count(vertex_pos); edge_pos++)
            {
                Edge<_TEdgeWeight> edge = data->iterate_adjacent_edges(vertex_pos, edge_pos);
                transposed_data->add_edge(edge.dst_id, edge.src_id, edge.weight);
            }
        }
        
        delete data;
        data = transposed_data;
        
        if(current_container_type == COMPRESSED_ADJACENCY_LIST)
            this->convert_to_compressed_adjacency_list();
    }
    else
    {
        throw "unsupported format in graph transpose operation";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// private functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
BaseContainer<_TVertexValue, _TEdgeWeight>* Graph<_TVertexValue, _TEdgeWeight>::create_new_container(ContainerType _container_type,
	                                                                                                 int _vertices_count, 
																									 long long _edges_count,
	                                                                                                 bool _directed,
																									 bool _empty)
{
	BaseContainer<_TVertexValue, _TEdgeWeight> *new_data = NULL;
	
	switch (_container_type)
	{
	case COMPRESSED_ADJACENCY_LIST:
		new_data = new CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>(_vertices_count, _edges_count, _directed);
		break;
	case ADJACENCY_LIST:
		new_data = new AdjacencyList<_TVertexValue, _TEdgeWeight>(_vertices_count, _directed);
		break;
	case EDGES_LIST:
		new_data = new EdgesList<_TVertexValue, _TEdgeWeight>(_vertices_count, _edges_count, _directed, _empty);
		break;
	default:
		throw "unsupported container type in graph";
	}
	return new_data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
