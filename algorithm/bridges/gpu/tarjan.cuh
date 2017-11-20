#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void tarjan_bfs_wrapper(int *_device_src_ids, int *_device_dst_ids, bool *_device_in_trees, long long _edges_count,
                        int *_device_bfs_level, int _vertices_count, int &_max_level, int _root);

void compute_D_wrapper(int *_device_trees_src_ids, int *_device_trees_dst_ids, int _trees_edges_count, 
	                   int *_device_D, int *_device_N, int *_device_bfs_level, int _vertices_count, int _max_level);

void compute_L_H_wrapper(int *_device_src_ids, int *_device_dst_ids, bool *_device_in_trees, int _edges_count,
	                     int *_device_trees_src_ids, int *_device_trees_dst_ids, int _trees_edges_count,
	                     int *_device_L, int *_device_H, int *_device_N, int *_device_bfs_level, int _vertices_count, int _max_level);

void process_results_wrapper(int *_device_src_ids, int *_device_dst_ids, bool *_device_bridges, int _edges_count, 
	                         int *_device_L, int *_device_H, int *_device_D, int *_device_N);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
