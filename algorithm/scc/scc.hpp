/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tarjan algorithm
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void StronglyConnectedComponents<_TVertexValue, _TEdgeWeight>::cpu_tarjan(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                          int *_components)
{
    // convert get graph data
    _graph.convert_to_compressed_adjacency_list();
    
    vertices_count = _graph.get_vertices_count();
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    vertices_to_edges_ptrs = graph_data.vertices_to_edges_ptrs;
    dst_ids = graph_data.edges_dst_ids;
    
    // allocate memory for Tarjan's algorithm computations
    int *disc = new int[vertices_count];
    int *low = new int[vertices_count];
    bool *stack_member = new bool[vertices_count];
    stack<int> st;
    
    // Initialize disc and low, and stackMember arrays
    for (int i = 0; i < vertices_count; i++)
    {
        disc[i] = -1;
        low[i] = -1;
        stack_member[i] = false;
    }
    
    // run algorithm
    for (int root = 0; root < vertices_count; root++)
        if (disc[root] == -1)
            tarjan_kernel(root, disc, low, st, stack_member, _components);
    
    // gree memory
    delete[] disc;
    delete[] low;
    delete[] stack_member;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void StronglyConnectedComponents<_TVertexValue, _TEdgeWeight>::tarjan_kernel(int _u,
                                                                             int *_disc,int *_low,
                                                                             stack<int> &_st,
                                                                             bool *_stack_member,
                                                                             int *_components)
{
    static int time = 0;
    static int comp = 0;
    
    struct SnapShotStruct
    {
        int u;
        int v;
        int stage;
    };
    
    stack<SnapShotStruct> snapshotStack;
    SnapShotStruct currentSnapshot;
    currentSnapshot.u = _u;
    currentSnapshot.v = -1;
    currentSnapshot.stage = 0;
    
    snapshotStack.push(currentSnapshot);
    
    while (!snapshotStack.empty()) // do DFS without recursion
    {
        currentSnapshot = snapshotStack.top();
        snapshotStack.pop();
        
        switch (currentSnapshot.stage)
        {
            case 0:
            {
                int u = currentSnapshot.u;
                _disc[u] = _low[u] = ++time;
                _st.push(u);
                _stack_member[u] = true;
                
                SnapShotStruct retSnapshot;
                retSnapshot.u = u;
                retSnapshot.v = 0;
                retSnapshot.stage = 2;
                snapshotStack.push(retSnapshot);
                
                int connections_count = vertices_to_edges_ptrs[u + 1] - vertices_to_edges_ptrs[u];
                for (int i = 0; i < connections_count; i++)
                {
                    long long pos = vertices_to_edges_ptrs[u] + i;
                    int v = dst_ids[pos];
                    
                    if (_disc[v] == -1)
                    {
                        currentSnapshot.u = u;
                        currentSnapshot.v = v;
                        currentSnapshot.stage = 1;
                        snapshotStack.push(currentSnapshot);
                        
                        SnapShotStruct newSnapshot;
                        newSnapshot.u = v;
                        newSnapshot.v = 0;
                        newSnapshot.stage = 0;
                        snapshotStack.push(newSnapshot);
                    }
                    else if (_stack_member[v] == true)
                    {
                        _low[u] = min(_low[u], _disc[v]);
                    }
                }
            }
                break;
            case 1:
            {
                int u = currentSnapshot.u;
                int v = currentSnapshot.v;
                _low[u] = min(_low[u], _low[v]);
            }
                break;
            case 2:
            {
                int u = currentSnapshot.u;
                int w = 0;  // To store stack extracted vertices
                if (_low[u] == _disc[u])
                {
                    while (_st.top() != u)
                    {
                        w = _st.top();
                        _stack_member[w] = false;
                        _st.pop();
                        _components[w] = comp;
                    }
                    w = _st.top();
                    _components[w] = comp;
                    _stack_member[w] = false;
                    _st.pop();
                    
                    comp++;
                }
            }
                break;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void StronglyConnectedComponents<_TVertexValue, _TEdgeWeight>::gpu_forward_backward(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                                    int *_components)
{
    // convert to necessary format
    _graph.convert_to_edges_list();
    
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    vertices_count = _graph.get_vertices_count();
    edges_count = _graph.get_edges_count();
    src_ids = graph_data.edges_src_ids;
    dst_ids = graph_data.edges_dst_ids;
    
    double t3 = omp_get_wtime();
    
    // allocate data on device for computations
    int *device_components, *device_trees;
    bool *device_active;
    SAFE_CALL(cudaMalloc((void**)&device_components, vertices_count * sizeof(int)));
    SAFE_CALL(cudaMalloc((void**)&device_trees, vertices_count * sizeof(int)));
    SAFE_CALL(cudaMalloc((void**)&device_active, vertices_count * sizeof(bool)));
    
    // run computations
    int last_component = 0;
    init_fb_data_wrapper(device_trees, device_active, device_components, vertices_count);
    
    int *device_src_ids, *device_dst_ids;
    
    // allocate data for graph storage
    SAFE_CALL(cudaMalloc((void**)&device_src_ids, edges_count * sizeof(int)));
    SAFE_CALL(cudaMalloc((void**)&device_dst_ids, edges_count * sizeof(int)));
    
    // copy graph edges data to device
    SAFE_CALL(cudaMemcpy(device_src_ids, src_ids, edges_count * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(device_dst_ids, dst_ids, edges_count * sizeof(int), cudaMemcpyHostToDevice));
    
    double t1 = omp_get_wtime();
    
    // computational steps
    trim_wrapper(device_src_ids, device_dst_ids, vertices_count, edges_count, device_components, device_trees,
                 device_active, last_component);
    last_component++;
    
    forward_backward_wrapper(device_src_ids, device_dst_ids, vertices_count, edges_count, device_components, device_trees, INIT_TREE,
                             device_active, last_component);
    
    double t2 = omp_get_wtime();
    cout << "time: " << t2 - t1 << " sec" << endl;
    cout << "GPU perf: " << edges_count / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
    
    // free edges data
    SAFE_CALL(cudaFree(device_src_ids));
    SAFE_CALL(cudaFree(device_dst_ids));
    
    // copy result back to host
    SAFE_CALL(cudaMemcpy(_components, device_components, vertices_count * sizeof(int), cudaMemcpyDeviceToHost));
    
    // free device memory
    SAFE_CALL(cudaFree(device_components));
    SAFE_CALL(cudaFree(device_trees));
    SAFE_CALL(cudaFree(device_active));
    
    double t4 = omp_get_wtime();
    cout << "full GPU perf: " << edges_count / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward-Backward algorithm
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void StronglyConnectedComponents<_TVertexValue, _TEdgeWeight>::cpu_forward_backward(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                                    int *_components)
{
 
    // get graph data
    _graph.convert_to_edges_list();
    
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    edges_count = _graph.get_edges_count();
    vertices_count = _graph.get_vertices_count();
    src_ids = graph_data.edges_src_ids;
    dst_ids = graph_data.edges_dst_ids;
    
    // set active and trees arrays
    bool *active = new bool[vertices_count];
    int *trees = new int[vertices_count];
    
    // set trees and components variables
    int last_component = 0;
    
    double t1 = omp_get_wtime();
    
    // trimming
    double t3 = omp_get_wtime();
    trim(_components, trees, active, &last_component);
    double t4 = omp_get_wtime();
    //cout << "trim time: " << t4 - t3 << " sec" << endl;
    //cout << "trim perf: " << _graph.get_edges_count() / ((t4 - t3) * 1e6) << " MTEPS" << endl << endl;
    
    t3 = omp_get_wtime();
    FB_on_host(_components, trees, INIT_TREE, active, &last_component);
    t4 = omp_get_wtime();
    //cout << "bfs time: " << t4 - t3 << " sec" << endl;
    
    double t2 = omp_get_wtime();
    //cout << "inner time: " << t2 - t1 << " sec" << endl;
    //cout << "perfomance: " << _graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
    
    // free algorithm data
    delete[] active;
    delete[] trees;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void StronglyConnectedComponents<_TVertexValue, _TEdgeWeight>::FB_on_host(int *_components, int *_trees, int _tree_num,
                                                                          bool *_active, int *_last_component)
{
    int pivot = select_pivot(_trees, _tree_num);
    if (pivot == ERROR_IN_PIVOT)
        return;
    
    bool *fwd_result = new bool[vertices_count];
    bool *bwd_result = new bool[vertices_count];
    
    bfs_reach(src_ids, dst_ids, pivot, fwd_result, _trees);
    bfs_reach(dst_ids, src_ids, pivot, bwd_result, _trees);
    
    int loc_last_trees[3] = { 0, 0, 0 };
    process_result(fwd_result, bwd_result, _components, _trees, _active, _last_component, loc_last_trees);
    
    delete[] fwd_result;
    delete[] bwd_result;
    
    // launch FB algorithm for 3 disjoint sets
    FB_on_host(_components, _trees, loc_last_trees[0], _active, _last_component);
    
    FB_on_host(_components, _trees, loc_last_trees[1], _active, _last_component);
    
    FB_on_host(_components, _trees, loc_last_trees[2], _active, _last_component);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// randomly select pivot
template <typename _TVertexValue, typename _TEdgeWeight>
int StronglyConnectedComponents<_TVertexValue, _TEdgeWeight>::select_pivot(int *_trees, int _tree_num)
{
    int result_pos = ERROR_IN_PIVOT;
    
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < vertices_count; i++)
        if (_trees[i] == _tree_num)
            result_pos = i;
    
    return result_pos;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// do trim step of FB algorithm (eliminates SCCs of size 1)
template <typename _TVertexValue, typename _TEdgeWeight>
void StronglyConnectedComponents<_TVertexValue, _TEdgeWeight>::trim(int *_components, int *_trees, bool *_active,
                                                                    int *_last_component)
{
    // allocate in and out degree arrays
    int *in_deg = new int[vertices_count];
    int *out_deg = new int[vertices_count];
    
    bool changes = false;
    #pragma omp parallel num_threads(omp_threads) shared(changes)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; i++)
        {
            _components[i] = INIT_COMPONENT;
            _trees[i] = INIT_TREE;
            _active[i] = true;
        }
        
        do
        {
            // clear data
            #pragma omp barrier
            
            #pragma omp single
            {
                changes = false;
            }
            
            #pragma omp for schedule(guided, 1024)
            for (int i = 0; i < vertices_count; i++)
            {
                in_deg[i] = 0;
                out_deg[i] = 0;
            }
            
            // compute vertices degrees
            #pragma omp for schedule(guided, 1024)
            for (long long idx = 0; idx < edges_count; idx++)
            {
                int src_id = src_ids[idx];
                int dst_id = dst_ids[idx];
                
                if (_active[src_id])
                    in_deg[dst_id]++;
                
                if (_active[dst_id])
                    out_deg[src_id]++;
            }
            
            // delete vertices in SCC with size 1
            #pragma omp for schedule(guided, 1024)
            for (int vertex_pos = 0; vertex_pos < vertices_count; vertex_pos++)
            {
                if (!_active[vertex_pos])
                    continue;
                
                if ((out_deg[vertex_pos] == 0) || (in_deg[vertex_pos] == 0))
                {
                    #pragma omp atomic capture
                    {
                        _components[vertex_pos] = (*_last_component);
                        (*_last_component)++;
                    }
                    
                    _trees[vertex_pos] = INIT_TREE - 2;
                    _active[vertex_pos] = false;
                    changes = true;
                }
            }
            
            #pragma omp barrier
        } while (changes);
    }
    
    // free arrays
    delete[] out_deg;
    delete[] in_deg;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// compute vertices, reachable from current pivot
template <typename _TVertexValue, typename _TEdgeWeight>
void StronglyConnectedComponents<_TVertexValue, _TEdgeWeight>::bfs_reach(int *_src_ids, int *_dst_ids, int _pivot,
                                                                         bool *_visited, int *_trees)
{
    // do BFS while no changes
    bool changes = true;
    
    #pragma omp parallel num_threads(omp_threads) shared(changes)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; i++)
            _visited[i] = false;
        _visited[_pivot] = true;

        do
        {
            #pragma omp barrier
            changes = false;
            #pragma omp barrier
        
            #pragma omp for schedule(guided, 1024)
            for (long long idx = 0; idx < edges_count; idx++)
            {
                int src_id = _src_ids[idx];
                int dst_id = _dst_ids[idx];
            
                // if they are in the same tree update visited
                if ((_visited[src_id] == true) && (_trees[src_id] == _trees[dst_id]) && (_visited[dst_id] == false))
                {
                    _visited[dst_id] = true;
                    changes = true;
                }
            }
        } while (changes == true);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// bfs function, visits all edges in parallel
template <typename _TVertexValue, typename _TEdgeWeight>
void StronglyConnectedComponents<_TVertexValue, _TEdgeWeight>::bfs_kernel(int *_src_ids, int *_dst_ids, bool *_visited,
                                                                          bool &_terminate, int *_trees, int _pivot)
{
    int omp_threads = omp_get_max_threads();
    
    #pragma omp parallel for num_threads(omp_threads)
    for (long long idx = 0; idx < edges_count; idx++)
    {
        int src_id = _src_ids[idx];
        int dst_id = _dst_ids[idx];
        
        // if they are in the same tree update visited
        if ((_visited[src_id] == true) && (_trees[src_id] == _trees[dst_id]) && (_visited[dst_id] == false))
        {
            _visited[dst_id] = true;
            _terminate = false;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// save results (assign vertices to different components)
template <typename _TVertexValue, typename _TEdgeWeight>
void StronglyConnectedComponents<_TVertexValue, _TEdgeWeight>::process_result(bool *_fwd_result, bool *_bwd_result,
                                                                              int *_components, int *_trees,
                                                                              bool *_active, int *_last_component, int *_last_trees)
{
    int omp_threads = omp_get_max_threads();
    
    #pragma omp parallel for num_threads(omp_threads)
    for (int i = 0; i < vertices_count; i++)
    {
        if (!_active[i])
            continue;
        
        int fwd_res = _fwd_result[i];
        int bwd_res = _bwd_result[i];
        
        if ((fwd_res == true) && (bwd_res == true))
        {
            _active[i] = false;
            _components[i] = *_last_component;
            _trees[i] = 4 * _trees[i];
        }
        else if ((fwd_res == false) && (bwd_res == false))
        {
            _trees[i] = 4 *_trees[i] + 1;
            _last_trees[0] = _trees[i];
        }
        else if ((fwd_res == true) && (bwd_res == false))
        {
            _trees[i] = 4 * _trees[i] + 2;
            _last_trees[1] = _trees[i];
        }
        else if ((fwd_res == false) && (bwd_res == true))
        {
            _trees[i] = 4 * _trees[i] + 3;
            _last_trees[2] = _trees[i];
        }
    }
    
    #pragma omp critical
    {
        (*_last_component)++;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
