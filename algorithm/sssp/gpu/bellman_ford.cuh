#pragma once

#include "include/node_data/node_data.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MAX_THREADS 1024

#define __USE_FERMI__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void init_distances_wrapper(_TEdgeWeight *_distances, int _vertices_count, int _source_vertex);

template <typename _TEdgeWeight>
void bellman_ford_wrapper(int *_src_ids, int *_dst_ids, _TEdgeWeight *_weights, int _vertices_count, long long _edges_count,
                          int _source_vertex, _TEdgeWeight *_distances);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////