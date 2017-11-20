#pragma once 

#include <vector>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define ERROR_IN_PIVOT -10
#define INIT_COMPONENT 1
#define INIT_TREE 1

#define __USE_FERMI__

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void init_fb_data_wrapper(int *_trees, bool *_active, int *_components, int _vertices_count);

void trim_wrapper(int *_src_ids, int *_dst_ids, int _vertices_count, long long _edges_count, int *_components,
	              int *_trees, bool *_active, int &_last_component);

void forward_backward_wrapper(int *_src_ids, int *_dst_ids, int _vertices_count, long long _edges_count, int *_components, int *_trees, int _tree_num,
							  bool *_active, int _last_component);

void large_graph_trim_wrapper(int *_host_src_ids, int *_host_dst_ids,  int *_device_src_ids, int *_device_dst_ids,
							  int _vertices_count, long long _edges_count, int *_components,int *_trees, bool *_active,
							  int &_last_component, long long _max_edges_in_gpu_partition);

void large_graph_forward_backward_wrapper(int *_host_src_ids, int *_host_dst_ids, int *_device_src_ids, int *_device_dst_ids,
										  int _vertices_count, long long _edges_count, int *_components, int *_trees, int _tree_num,
										  bool *_active, int _last_component, long long _max_edges_in_gpu_partition);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
