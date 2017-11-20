#pragma once

#include "include/architectures.h"
#include "include/knl_data.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define USE_EDGES_LIST_HEADER true

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// single node data structure
#include "include/graph_data_structures/graph.h"

// common functions
#include "include/common/file_system.h"

// graph visualization
#include "include/visualization/graph_visualization_API.h"

// graph generation
#include "include/test/generation/graph_generation_API.h"

// graph optimization
#include "include/test/optimization/graph_optimization_API.h"

// graph storage
#include "include/storage/graph_storage_API.h"

// system properties
#include "include/node_data/node_data.h"

// operations
#include "algorithm/sssp/sssp.h"
#include "algorithm/bfs/bfs.h"
#include "algorithm/scc/scc.h"
#include "algorithm/mst/mst.h"
#include "algorithm/transitive_closure/transitive_closure.h"
//#include "algorithm/bridges/bridges.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
