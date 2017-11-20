/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
EdgesList<_TVertexValue, _TEdgeWeight>::EdgesList(int _vertices_count, long long _edges_count, bool _directed, bool _empty)
{
	BaseContainer<_TVertexValue, _TEdgeWeight>::directed = _directed;

	vertices.values = NULL;
	edges.src_ids   = NULL;
	edges.dst_ids   = NULL;
	edges.weights   = NULL;
	
	vertices.top   = 0;
	vertices.size  = _vertices_count + 1;
	edges.top      = 0;
	edges.size     = _edges_count + 1;
	
	universal_malloc<_TVertexValue>(&vertices.values, vertices.size);
	universal_malloc<int>(&edges.src_ids, edges.size);
	universal_malloc<int>(&edges.dst_ids, edges.size);
	universal_malloc<_TEdgeWeight>(&edges.weights, edges.size);

	if (!_empty)
	{
		vertices.top = _vertices_count;
		edges.top = _edges_count;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
EdgesList<_TVertexValue, _TEdgeWeight>::~EdgesList()
{
	universal_free<_TVertexValue>(&vertices.values);
	universal_free<int>(&edges.src_ids);
	universal_free<int>(&edges.dst_ids);
	universal_free<_TEdgeWeight>(&edges.weights);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Vertex<_TVertexValue> EdgesList<_TVertexValue, _TEdgeWeight>::iterate_vertices(int _vertex_pos)
{
	if (_vertex_pos >= vertices.top)
		throw "Out of bounds in ITERATE_VERTICES function";

	return Vertex<_TVertexValue>(_vertex_pos, vertices.values[_vertex_pos]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> EdgesList<_TVertexValue, _TEdgeWeight>::iterate_edges(long long _edge_pos)
{
	if (_edge_pos >= edges.top)
		throw "Out of bounds in ITERATE_VERTICES function";

	return Edge<_TEdgeWeight>(edges.src_ids[_edge_pos], edges.dst_ids[_edge_pos], edges.weights[_edge_pos]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> EdgesList<_TVertexValue, _TEdgeWeight>::iterate_adjacent_edges(int _src_vertex_id, int _edge_pos)
{
	throw "not implemented yet";
	return Edge<_TEdgeWeight>(0, 0, 0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Vertex<_TVertexValue> EdgesList<_TVertexValue, _TEdgeWeight>::get_vertex_by_id(int _id)
{
	return iterate_vertices(_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> EdgesList<_TVertexValue, _TEdgeWeight>::get_edge_by_id(int _src_id, int _dst_id)
{
	throw "GET_EDGE_BY_id not implemented yet in EDGES LIST";
	return Edge<_TEdgeWeight>(0, 0, 0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int EdgesList<_TVertexValue, _TEdgeWeight>::get_vertices_count() const
{
	return vertices.top;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
long long EdgesList<_TVertexValue, _TEdgeWeight>::get_edges_count() const
{
	return edges.top;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int EdgesList<_TVertexValue, _TEdgeWeight>::get_vertex_connections_count(int _id)
{
	throw "not implemented yet";
	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesList<_TVertexValue, _TEdgeWeight>::set_vertex(int _id, _TVertexValue _value)
{
    if(_id >= vertices.top)
        throw "out of range in SET_VERTEX function";
    
    vertices.values[_id] = _value;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesList<_TVertexValue, _TEdgeWeight>::add_vertex(int _id, _TVertexValue _value)
{
	// resize vertices data if needed
	if ((vertices.top + 1) == vertices.size)
	{
		vertices.values = resize_array<_TVertexValue>(vertices.values, vertices.size, vertices.size * 2);
		vertices.size = vertices.size * 2;
	}

	// add vertex to the top
	vertices.values[vertices.top] = _value;
	vertices.top++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesList<_TVertexValue, _TEdgeWeight>::add_vertex(int _id, _TVertexValue _value,
														const vector<int> &_adj_ids, 
														const vector<_TEdgeWeight> &_adj_weights)
{
	// add vertex without edges to data set
	add_vertex(_id, _value);

	// set all edges, outgoing from current vertex
	for (int i = 0; i < (int)_adj_ids.size(); i++)
	{
		add_edge(_id, _adj_ids[i], _adj_weights[i]);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesList<_TVertexValue, _TEdgeWeight>::add_edge(int _src_id, int _dst_id, _TEdgeWeight _weight)
{
	if (!this->check_if_directed())
	{
		if (_src_id > _dst_id)
			swap(_src_id, _dst_id);
	}

	// resize edges data if needed
	if (edges.top == edges.size)
	{
		edges.src_ids = resize_array<int>(edges.src_ids, edges.size, edges.size * 2);
		edges.dst_ids = resize_array<int>(edges.dst_ids, edges.size, edges.size * 2);
		edges.weights = resize_array<_TEdgeWeight>(edges.weights, edges.size, edges.size * 2);
		edges.size = edges.size * 2;
	}

	// add vertex to the top
	edges.src_ids[edges.top] = _src_id;
	edges.dst_ids[edges.top] = _dst_id;
	edges.weights[edges.top] = _weight;
	edges.top++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void EdgesList<_TVertexValue, _TEdgeWeight>::empty()
{
	vertices.top = 0;
	edges.top = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphContainerData<_TVertexValue, _TEdgeWeight> EdgesList<_TVertexValue, _TEdgeWeight>::get_container_data()
{
	GraphContainerData<_TVertexValue, _TEdgeWeight> device_data;

	device_data.vertices_ids    = NULL;
	device_data.vertices_values = NULL;
	device_data.edges_src_ids   = edges.src_ids;
	device_data.edges_dst_ids   = edges.dst_ids;
	device_data.edges_weights   = edges.weights;

	return device_data;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
EdgesList<_TVertexValue, _TEdgeWeight>* EdgesList<_TVertexValue, _TEdgeWeight>::clone() const
{
	throw "Edges list clone not implemented yet";
	return new EdgesList<_TVertexValue, _TEdgeWeight>(0, 0, false, true);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
