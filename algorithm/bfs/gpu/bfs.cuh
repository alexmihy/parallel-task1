#pragma once

#define VISITED true
#define UNVISITED false

#define __USE_FERMI__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void bfs_wrapper(int *_src_ids, int *_dst_ids, int _vertices_count, long long _edges_count, bool *_visited, int _source_vertex);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
