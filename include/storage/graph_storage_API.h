#pragma once

#include "include/graph_data_structures/graph.h"

#include <map>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \class  GraphStorageApi graph_storage_API.h "storage/graph_storage_API.h"
//! \brief  Saves and loads graph in different storage databases
//! \author Ilya Afanasyev
//!
//! \details Now supports only storage inside binary file. \n
template <typename _TVertexValue, typename _TEdgeWeight>
class GraphStorageAPI
{
public:
	//! \brief Saves graph object to binary file in adjacency list format
	//!
	//! \details Stores the entire graph in the selected file in adjacency list format                       \n
	//!          The graph is stored in following format:                                                    \n
	//!          <header>: (vertices count, edges_count)													 \n
	//!          <body>: (vertex id, vertex_value) adjacent edges: (destination vertex id, edge weight) ...  \n
	//!           Returns false if any errors occur while oppening the file (file already exist or others).  \n
	//!
	//! \param _graph       input graph object, required to store
	//! \param _file_name   string, containing name of the binary file
	static bool save_to_adjacency_list_bin_file(Graph<_TVertexValue, _TEdgeWeight> &_graph, string _file_name);
    
    //! \brief Saves graph object to binary file in edges list
    //!
    //! \details Stores the entire graph in the selected file in edges list format                          \n
    //!          The graph is stored in following format:                                                   \n
    //!          <body>: for each edge (source vertex id, destination vertex id, edge weight)               \n
    //!          Returns false if any errors occur while oppening the file (file already exist or others).  \n
    //!
    //! \param _graph       input graph object, required to store
    //! \param _file_name   string, containing name of the binary file
    static bool save_to_edges_list_bin_file(Graph<_TVertexValue, _TEdgeWeight> &_graph, string _file_name);

	//! \brief Loads graph object from binary file in adjacency list format
	//!
	//! \details Loads the entire graph from selected file in adjacency list format.                       \n
	//!          File should be in the format, described in save_to_adjacency_list_bin_file(               \n
	//!          Returns false if any errors occur while oppening the file (file doesn't exist or others). \n
	//!
	//! \param _graph     input graph object, required to store
	//! \param _file_name string, containing name of the binary file
	static bool load_from_adjacency_list_bin_file(Graph<_TVertexValue, _TEdgeWeight> &_graph, string _file_name);
    
    //! \brief Loads graph object from binary file in edges list format
    //!
    //! \details Loads the entire graph from selected file in edges list format.                           \n
    //!          File should be in the format, described in save_to_dges_list_bin_file(                   \n
    //!          Returns false if any errors occur while oppening the file (file doesn't exist or others). \n
    //!
    //! \param _graph     input graph object, required to store
    //! \param _file_name string, containing name of the binary file
    static bool load_from_edges_list_bin_file(Graph<_TVertexValue, _TEdgeWeight> &_graph, string _file_name);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_storage_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
