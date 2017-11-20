#pragma once

#include "include/graph_data_structures/graph.h"

#include "include/common/computational_mode.h"

#include <map>
#include <set>
#include <algorithm>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class SubgraphIsomorphism
{
private:
    bool DFS(Graph<_TVertexValue, _TEdgeWeight> &_g, int _v, int _w, Graph<_TVertexValue, _TEdgeWeight> &_h, bool **_T,
             std::set<int> &visited_g, std::map<int, int> &phi);
public:
    void ullman_algorithm(Graph<_TVertexValue, _TEdgeWeight> &_big_graph, Graph <_TVertexValue, _TEdgeWeight> &_small_graph,
                          ComputationalMode _computational_mode);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "subgraph_isomorphism.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
