#pragma once

#include "include/graph_data_structures/graph.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \enum GraphvizMode
//!  Defines the mode of directed/undirected visualization for Graphviz file
enum GraphvizMode
{
    VISUALIZE_AS_DIRECTED,
    VISUALIZE_AS_UNDIRECTED,
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \class  GraphVisualizationAPI graph_visualization_API.h "include/vizualization/graph_visualization_API.h"
//! \author Ilya Afanasyev
//! \brief  Visualize graph using different tools
//!
//! \details Two ways of graph visualization are now supported:       \n
//!          1. Using standart console output (print graph to stdout) \n
//!          2. Create and save "dot" format file for Graphviz tool   \n
template <typename _TVertexValue, typename _TEdgeWeight>
class GraphVisualizationAPI
{
public:
	//! \brief Prints adjacency matrix of current graph to standart output (console) 
	//!
	//! \details Prints to standart output adjacency matrix of current graph it the following form: \n
	//!          a_ij = 1 if i-th vertex is connected to j-th vertex                                \n
	//!          a_ij = 0 if there is no edge, connection i-th and j-th vertices                    \n
	//!          DOESN'T SUPPORT EDGE_LIST FORMAT FOR NOW                                           \n
	//! \param _graph Graph object, requred for output
	static void print_vertices_matrix (Graph<_TVertexValue, _TEdgeWeight> &_graph);

	//! \brief Prints graph container structure (depending on container type) to standart output (console) 
	//!
	//! \details Edges list format not supported yet \n
	//! 
	//! \param _graph Graph object, requred for output
	static void print_container_data(Graph<_TVertexValue, _TEdgeWeight> &_graph);

	//! \brief Creates "dot" format file for graphviz tool 
	//!
	//! \param _graph             Graph object, requred for output
	//! \param _output_file_name  Name of output "dot" format file
	//! \param _mode              Enum, VISUALIZE_AS_DIRECTED or VISUALIZE_AS_UNDIRECTED
	static void create_graphviz_file(Graph<_TVertexValue, _TEdgeWeight> &_graph, string _output_file_name,
                                     GraphvizMode _mode = VISUALIZE_AS_DIRECTED);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_visualization_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
