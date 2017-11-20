#pragma once

#include "include/graph_data_structures/graph.h"
#include "gpu/boruvka.cuh"

#ifdef __USE_GPU__
#include "include/common/cuda_error_hadling.h"
#endif

#include "include/common_data_structures/union_find.h"

#include <list>
#include <set>
#include <algorithm> 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ThreadsManagerMST;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \class  MinimumSpanningTree mst.h "src/algorithm/mst/mst.h"
//! \author Ilya Afanasyev
//! \brief  Solves minimum spanning tree problem(MST)
//!
//! \details Now supports only single node sequential CPU version.                              \n
//!          Template parameters _TVertex and _TEdge should be the same, as in the input graph. \n
template <typename _TVertexValue, typename _TEdgeWeight>
class MinimumSpanningTree
{
private:
    int omp_threads;
    
    // boruvka implementations kernel
    void cpu_boruvka_kernel(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, bool *_mst_edges);
    
    #ifdef __USE_GPU__
    void gpu_boruvka_kernel(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, bool *_mst_edges);
    #endif
    
    void update_mst(Graph<_TVertexValue, _TEdgeWeight> &_graph, Graph<_TVertexValue, _TEdgeWeight> &_mst, bool *_mst_edges);
    
    // calculates MST weight
	static _TEdgeWeight compute_MST_weight(Graph<_TVertexValue, _TEdgeWeight> &_mst);
public:
    MinimumSpanningTree(int _omp_threads = 1) { omp_threads = _omp_threads; }
    void set_omp_threads(int _omp_threads) { omp_threads = _omp_threads; }

    // main CPU computational function, uses boruvka algorithm to compute MST
	void cpu_boruvka(Graph<_TVertexValue, _TEdgeWeight> &_graph, Graph<_TVertexValue, _TEdgeWeight> &_mst);
    
    // main GPU computational function, uses boruvka algorithm to compute MST
    #ifdef __USE_GPU__
    void gpu_boruvka(Graph<_TVertexValue, _TEdgeWeight> &_graph, Graph<_TVertexValue, _TEdgeWeight> &_mst);
    #endif

    // check if 2 msts are similar
	static bool check_mst_results(Graph<_TVertexValue, _TEdgeWeight> &_first_mst, Graph<_TVertexValue, _TEdgeWeight> &_second_mst);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "mst.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
