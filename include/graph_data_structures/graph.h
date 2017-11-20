#pragma once

#include "containers/compressed_adjacency_list.h"
#include "containers/adjacency_list.h"
#include "containers/edges_list.h"

#include <omp.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \enum ContainerType
//!  Defines type of data storage container in Graph class
enum ContainerType
{
	COMPRESSED_ADJACENCY_LIST,
	ADJACENCY_LIST,
	EDGES_LIST
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \class  Graph graph.h "src/data_structures/graph.h"
//! \brief  Implements graph storage model. Has the same interface for all storage formats
//! \author Ilya Afanasyev
//!
//! \details Now three different storage models are implemented: COMPRESSED_ADJACENCY_LIST, ADJACENCY_LIST, EDGES_LIST.
//!       Each container model has it's own advantages and disadvantages, for more details check each storage model container. \n
//!       Also stores directed and non-directed graphs. If graph is non-directed, stores edges when it's source_id < destination_id. \n
//!       Storing non-directed graph requires two times less memory. \n
//!
//!  Each container implements the same set of operations over vertices and edges: \n
//!  1. get vertex/edge by ID              \n
//!  2. iterate between all vertices/edges \n
//!  3. iterate between adjacent vertices  \n
//!  4. add vertex/edge                    \n
//!  5. get vertex/edges count             \n
//!  6. get vertex connections count       \n
//!  7. empty                              \n 
template <typename _TVertexValue, typename _TEdgeWeight>
class Graph
{
private:
	BaseContainer<_TVertexValue, _TEdgeWeight> *data; //!< stores all vertices and edges in conatainer
	ContainerType current_container_type;             //!< stores type of currently used container

	BaseContainer<_TVertexValue, _TEdgeWeight>* create_new_container(ContainerType _container_type, int _vertices_count, long long _edges_count,
		                                                             bool _directed, bool _empty);
public:
	//! \brief Creates new empty graph object
	//!
	//! \details this constructor creates empty directed graph object in ADJACENCY_LIST format with 0 vertices and 0 edges
	Graph();

	//! \brief Creates new graph object with specified vertices and edges count
	//!
	//! \details this constructor just allocates memory for vertices and edges storage, without  putting any data to data set \n
	//! \param _vertices_count Number of vertices in created graph
	//! \param _edges_count Number of edges in created graph
	//! \param _container_type Type of data set, used to store graph data. Possuble values: COMPRESSED_ADJACENCY_LIST, 
	//!                        ADJACENCY_LIST, EDGES_LIST
	//! \param _empty          Set false if data inside container should be considered as real vertices and edges
	Graph(int _vertices_count, long long _edges_count, ContainerType _container_type, bool _directed, bool _empty = true);

	//! \brief Destroys graph object
	//!
	//! \details Automatically calls container destructor \n
    ~Graph();

	//! \brief Change the size of current graph and delete all it's data
	//!
	//! \param _new_vertices_count New number of vertices
	//! \param _new_edges_count    New number of edges
	//! \param _empty              Set false if data inside container should be considered as real vertices and edges
	void resize(int _new_vertices_count, long long _new_edges_count, bool _empty = true);

	//! \brief Returns i-th vertex, positioned in container on i-th pos
	//!
	//! \details Use this function to traverse all vertices in graph one by one (from first to last). \n
	//! \param _pos Position(index) of vertex
	Vertex<_TVertexValue> iterate_vertices(int _vertex_pos) const;

	//! \brief Returns i-th edge, positioned in container on i-th pos
	//!
	//! \details Use this function to traverse all edges in graph one by one (from first to last). \n
	//! \param _pos Position(index) of edge
	Edge<_TEdgeWeight> iterate_edges(long long _edges_pos) const;

	//! \brief Returns i-th edge, adjacent to vertex with source ID
	//!
	//! \details Use this function to traverse all edges, adjacent to current one
	//! \param _src_id Vertex ID of source vertex
	//! \param _pos Edge position within curent vertex edges
	//!
	Edge<_TEdgeWeight> iterate_adjacent_edges(int _src_vertex_id, int _edge_pos) const;

	//! \brief Returns vertex with specified ID
	//!
	//! \details Use this function to get vertex with specified ID \n
	//! \param _ID vertex ID in data set
	Vertex<_TVertexValue> get_vertex_by_id(int _id) const;

	//! \brief Returns edge, connecting source and destination vertices
	//!
	//! \details Use this function to get particular edge with known adjacent vertices
	//! \param _src_id Vertex ID of source vertex
	//! \param _dst_id Vertex ID of destination vertex
	Edge<_TEdgeWeight> get_edge_by_id(int _src_id, int _dst_id) const;

	//! \brief Returns number of vertices in current graph
	int get_vertices_count() const;

	//! \brief Returns number of edges in current graph
	long long get_edges_count() const;

	//! \brief Returns number of edges, adjacent to vertex with specified ID
	//!
	//! \param _id Vertex ID of source vertex
	int get_vertex_connections_count(int _id) const;
    
    //! \brief Sets vertex value of specified already existing vertex
    //!
    //! \details WARNING: Uses vertex ID to specify the required vertex \n
    //! \param _id ID of vertex to change value
    //! \param _value New value of vertex
    void set_vertex(int _id, _TVertexValue _value);

	//! \brief Adds vertex <b>to the end</b> of graph data set without any edges
	//!	
	//! \details WARNING: In general this action can destroy the order in vertex indexing process inside the container. \n
	//!                   The best way to use it is when new vertex ID is equal to last vertex ID + 1. \n
	//! \param _id ID of vertex to add
	//! \param _value Value of vertex to add
	void add_vertex(int _id, _TVertexValue _value);

	//! \brief The same as general function, but vertex is represented as Vertex struct
	void add_vertex(Vertex<_TVertexValue> _vertex);

	//! \brief Adds vertex <b>to the end</b> of graph data set with a vector of edges
	//!
	//! \details WARNING: In general this action can destroy the order in vertex indexing process inside the container. \n
	//!                   The best way to use it is when new vertex ID is equal to last vertex ID + 1. \n
	//! \param _id ID of vertex to add
	//! \param _value Value of vertex to add
	//! \param _adj_ids Vector of adjacency vertices IDs to current one
	//! \param _value Vector of adjacency edges weights to current one
	void add_vertex(int _id, _TVertexValue _value, const vector<int> &_adj_ids, const vector<_TEdgeWeight> &_adj_weights);

	//! \brief Adds edge, connecting source and destination vertices with specified weight
	//!
	//! \details \n
	//! \param _src_id ID of source vertex
	//! \param _dst_id ID of destination vertex
	//! \param _weight Weight of current edge
	void add_edge(int _src_id, int _dst_id, _TEdgeWeight _weight);

	//! \brief The same as general function, but vertex is represented as Vertex struct
	void add_edge(Edge<_TEdgeWeight> _edge);

	//! \brief Empty graph data container
	//!
	//! \details Sets all data as unimportant, without free data functions. To free memory use destructor of current object. \n
	void empty();

	//! \brief Returns data structure with pointers to device data
	GraphContainerData<_TVertexValue, _TEdgeWeight> get_graph_data();

	//! \brief Returns memory usage in MB
	long long memory_usage();

	//! \brief Changes current storage format to ADJACENCY_LIST format
	//!
	//! \details Not impemented yet
	void convert_to_adjacency_list();

	//! \brief Changes current storage format to COMPRESSED_ADJACENCY_LIST format
	//!
	//! \details Not impemented yet
	void convert_to_compressed_adjacency_list();

	//! \brief Changes current storage format to EDGES_LIST format
	//!
	//! \details Uses O(|E|) algorithm
	void convert_to_edges_list();

	//! \brief Comparison operator, checks if two graphs are equal (vertices and edges)
	//!
	//! \details WARNING: now uses non-optimal n^2 algorithm \n
	//! \param _other Second graph to compare
	bool operator==(const Graph<_TVertexValue, _TEdgeWeight> &_other) const;

	//! \brief Copy operator, deep copy from one graph to another
	//!
	//! \param _other graph to copy
	void operator=(const Graph<_TVertexValue, _TEdgeWeight> &_other);

	//! \brief Checks whether graph is directed or not
	bool check_if_directed() { return data->check_if_directed(); };

	//! \brief Changes dircted mode (only flags for now!)
	void set_directed(bool _directed) { data->set_directed(_directed); };
    
    //! \brief Converts undirected graph to directed representation
    void convert_to_directed();

	//! \brief Returns format of currently used container
	ContainerType get_container_type();
    
    //! \brief Transposes the graph inside the object
    void transpose();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
