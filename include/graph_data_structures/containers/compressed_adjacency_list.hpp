/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::CompressedAdjacencyList(int _vertices_count, long long _edges_count, bool _directed)
{
	BaseContainer<_TVertexValue, _TEdgeWeight>::directed = _directed;

	vertices.ids = NULL;
	vertices.edge_ptrs = NULL;
	vertices.values = NULL;
	edges.adj_ids = NULL;
	edges.weights = NULL;

	vertices.top = 0;
	vertices.size = _vertices_count + 1;
	universal_malloc<int>(&vertices.ids, vertices.size);
	universal_malloc<long long>(&vertices.edge_ptrs, vertices.size);
	universal_malloc<_TVertexValue>(&vertices.values, vertices.size);

	edges.top = 0;
	edges.size = _edges_count + 1;
	universal_malloc<int>(&edges.adj_ids, edges.size);
	universal_malloc<_TEdgeWeight>(&edges.weights, edges.size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::~CompressedAdjacencyList()
{
	universal_free<int>(&vertices.ids);
	universal_free<long long>(&vertices.edge_ptrs);
	universal_free<_TVertexValue>(&vertices.values);

	universal_free<int>(&edges.adj_ids);
	universal_free<_TEdgeWeight>(&edges.weights);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Vertex<_TVertexValue> CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::iterate_vertices(int _vertex_pos)
{
	if (_vertex_pos > vertices.top)
		throw "Out of bounds in GET_VERTEX_BY_POS function";

	Vertex<_TVertexValue> result(vertices.ids[_vertex_pos], vertices.values[_vertex_pos]);
	return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::iterate_edges(long long _edge_pos)
{
	throw "not implemented yet";

	return Edge<_TEdgeWeight>(-1, -1, -1);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::iterate_adjacent_edges(int _src_vertex_id, int _edge_pos)
{
	long long edge_index = vertices.edge_ptrs[_src_vertex_id] + _edge_pos;
	Edge<_TEdgeWeight> result(_src_vertex_id, edges.adj_ids[edge_index], edges.weights[edge_index]);

	return result;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Vertex<_TVertexValue> CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::get_vertex_by_id(int _id)
{
	return iterate_vertices(_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::get_edge_by_id(int _src_id, int _dst_id)
{
	Edge< _TEdgeWeight> result;

	if (!this->check_if_directed())
	{
		if (_src_id > _dst_id)
			swap(_src_id, _dst_id);
	}

	for (int i = 0; i < get_vertex_connections_count(_src_id); i++)
	{
		result = iterate_adjacent_edges(_src_id, i);
		if (result.dst_id == _dst_id)
		{
			return result;
		}
	}

	return Edge< _TEdgeWeight>(-1, -1, -1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::get_vertices_count() const
{
	return vertices.top;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
long long CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::get_edges_count() const
{
	return edges.top;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::get_vertex_connections_count(int _id)
{
	return (int)(vertices.edge_ptrs[_id + 1] - vertices.edge_ptrs[_id]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::set_vertex(int _id, _TVertexValue _value)
{
    if(_id >= vertices.top)
        throw "out of range in SET_VERTEX function";
    
    vertices.values[_id] = _value;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::add_vertex(int _id, _TVertexValue _value)
{
	// resize vertices data if needed
	if ((vertices.top + 1) == vertices.size)
	{
		vertices.ids       = resize_array<int>(vertices.ids, vertices.size, vertices.size * 2);
		vertices.values    = resize_array<_TVertexValue>(vertices.values, vertices.size, vertices.size * 2);
		vertices.edge_ptrs = resize_array<long long>(vertices.edge_ptrs, vertices.size, vertices.size * 2);
		vertices.size      = vertices.size * 2;
	}

	// check if vertex id is correct
	if (vertices.top > 0 && (vertices.ids[vertices.top - 1] + 1) != _id)
		throw "invalid vertex ID in add_vertex function";

	// add vertex to the top
	vertices.ids[vertices.top]    = _id;
	vertices.values[vertices.top] = _value;
	if (vertices.top == 0)
		vertices.edge_ptrs[vertices.top] = 0;
	vertices.top++;

	// add last NULL vertex
	vertices.ids[vertices.top]       = 0;
	vertices.values[vertices.top]    = 0;
	vertices.edge_ptrs[vertices.top] = vertices.edge_ptrs[vertices.top - 1];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::add_vertex(int _id, _TVertexValue _value, 
																	  const vector<int> &_adj_ids, 
																	  const vector<_TEdgeWeight> &_adj_weights)
{
	// add vertex without edges to data set
	add_vertex(_id, _value);

	int adj_edges_count = (int)_adj_ids.size();

	// set all edges, outgoing from current vertex
	for (int i = 0; i < adj_edges_count; i++)
	{
		// resize edges data if needed
		if ((edges.top + 1) == edges.size)
		{
			edges.adj_ids = resize_array<int>(edges.adj_ids, edges.size, edges.size * 2);
			edges.weights = resize_array<_TEdgeWeight>(edges.weights, edges.size, edges.size * 2);
			edges.size    = edges.size * 2;
		}

		edges.adj_ids[edges.top] = _adj_ids[i];
		edges.weights[edges.top] = _adj_weights[i];
		edges.top++;
	}

	// change connections count
	vertices.edge_ptrs[vertices.top] = vertices.edge_ptrs[vertices.top - 1] + adj_edges_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::add_edge(int _src_id, int _dst_id, _TEdgeWeight _weight)
{
	throw "add_edge not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::empty()
{
	vertices.top = 0;
	edges.top = 0;

	vertices.ids[0] = 0;
	vertices.values[0] = 0;
	vertices.edge_ptrs[0] = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphContainerData<_TVertexValue, _TEdgeWeight> CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::get_container_data()
{
	return GraphContainerData<_TVertexValue, _TEdgeWeight>(vertices.ids, vertices.edge_ptrs, vertices.values, NULL, edges.adj_ids, edges.weights);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>* CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>::clone() const
{
	int vertices_count    = this->get_vertices_count();
	long long edges_count = this->get_edges_count();
	CompressedAdjacencyList<_TVertexValue, _TEdgeWeight> *clone =
		new CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>(vertices_count, edges_count, this->directed);

	universal_memcpy(clone->vertices.ids, vertices.ids, vertices_count + 1);
	universal_memcpy(clone->vertices.values, vertices.values, vertices_count + 1);
	universal_memcpy(clone->vertices.edge_ptrs, vertices.edge_ptrs, vertices_count + 1);

	universal_memcpy(clone->edges.adj_ids, edges.adj_ids, edges_count);
	universal_memcpy(clone->edges.weights, edges.weights, edges_count);

	clone->vertices.top = this->vertices.top;
	clone->edges.top = this->edges.top;

	return clone;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
