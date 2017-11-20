/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BridgesDetection<_TVertexValue, _TEdgeWeight>::cpu_parallel_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                     int _root, vector<int> &_bfs_level, bool *_in_tree,
                                                                     int &_max_level, int _omp_threads)
{
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    long long *vertices_to_edges_ptrs = graph_data.vertices_to_edges_ptrs;
    int *dst_ids = graph_data.edges_dst_ids;
    
    // init queue and first vertex
    vector<int> global_queue;
    
    global_queue.push_back(_root);
    _bfs_level[_root] = 1;
    
    int current_level = 1;
    vector<vector<int>> thread_queues(_omp_threads);
    // arrays for merging
    vector<int> sizes(_omp_threads);
    vector<int> offsets(_omp_threads);
    
    // do bfs for each level
    #pragma omp parallel num_threads(_omp_threads)
    {
        int thread_id = omp_get_thread_num();
        vector<int> &local_queue = thread_queues[thread_id];
        do
        {
            // traverse single level
            #pragma omp for schedule(dynamic)
            for(int i = 0; i < global_queue.size(); i++)
            {
                int cur_vertex = global_queue[i];
                long long first = vertices_to_edges_ptrs[cur_vertex];
                long long last = vertices_to_edges_ptrs[cur_vertex + 1];
                for(long long cur_edge = first; cur_edge < last; cur_edge++)
                {
                    int dst_id = dst_ids[cur_edge];
                    int level = _bfs_level[dst_id];
                    while (level == -1)
                    {
                        if (atomic_compare_exchange_weak((atomic<int>*)&_bfs_level[dst_id], &level, current_level + 1))
                        {
                            _in_tree[cur_edge] = true;
                            local_queue.push_back(dst_id);
                            break;
                        }
                        level = _bfs_level[dst_id];
                    }
                }
            }
            
            #pragma omp barrier
            
            #pragma omp master
            {
                // compute
                int total_size = 0;
                for(int i = 0; i < _omp_threads; i++)
                {
                    sizes[i] = thread_queues[i].size();
                    offsets[i] = 0;
                    if(i > 0)
                        offsets[i] = offsets[i - 1] + sizes[i - 1];
                    total_size += sizes[i];
                }
                
                global_queue.clear();
                global_queue.resize(total_size);
                
                // increment level num
                current_level++;
            }
            
            #pragma omp barrier
            
            memcpy(&(global_queue[offsets[thread_id]]), &(local_queue[0]), sizes[thread_id] * sizeof(int));
            
            #pragma omp barrier
            
            thread_queues[thread_id].clear();
            
            #pragma omp barrier
        } while(global_queue.size() > 0);
    }

    if(current_level > _max_level)
        _max_level = current_level;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BridgesDetection<_TVertexValue, _TEdgeWeight>::cpu_sequential_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                       int _root, vector<int> &_bfs_level, bool *_in_tree,
                                                                       int &_max_level)
{
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    long long *vertices_to_edges_ptrs = graph_data.vertices_to_edges_ptrs;
    int *dst_ids = graph_data.edges_dst_ids;
    
    // init queue and first vertex
    int current_level = 1;
    std::queue<int> old_queue, new_queue;
    old_queue.push(_root);
    _bfs_level[_root] = current_level;
    
    do
    {
        while(old_queue.size() > 0)
        {
            int cur_vertex = old_queue.front();
            old_queue.pop();
        
            long long first = vertices_to_edges_ptrs[cur_vertex];
            long long last = vertices_to_edges_ptrs[cur_vertex + 1];
            for(long long i = first; i < last; i++)
            {
                int dst_id = dst_ids[i];
                if(_bfs_level[dst_id] == -1)
                {
                    _bfs_level[dst_id] = current_level + 1;
                    _in_tree[i] = true;
                    new_queue.push(dst_id);
                }
            }
        }
        
        current_level++;
        swap(old_queue, new_queue);
    } while (old_queue.size() > 0);
    
    if(current_level > _max_level)
        _max_level = current_level;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BridgesDetection<_TVertexValue, _TEdgeWeight>::DFS_pre_order(vector <int> &_vertices_to_edges_ptrs, vector <int> &_dst_ids,
                                                                  int _root, vector<int> &_pre_order_numbers, vector<bool> &_DFS_visited)
{
    stack<int> st;
    st.push(_root);
    
    while(st.size() > 0)
    {
        int cur_vertex = st.top();
        st.pop();
        _DFS_visited[cur_vertex] = true;
        _pre_order_numbers[cur_vertex] = DFS_counter;
        DFS_counter++;
        
        int first = _vertices_to_edges_ptrs[cur_vertex];
        int last = _vertices_to_edges_ptrs[cur_vertex + 1];
        for(int edge_pos = first; edge_pos < last; edge_pos++)
        {
            int dst_id = _dst_ids[edge_pos];
            if(!_DFS_visited[dst_id])
                st.push(dst_id);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BridgesDetection<_TVertexValue, _TEdgeWeight>::parallel_cpu_tarjan(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                        bool *_bridges, int _omp_threads_count)
{
    omp_set_num_threads(_omp_threads_count);
    memset(_bridges, 0, _graph.get_edges_count());
    
    // get pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    long long *vertices_to_edges_ptrs = graph_data.vertices_to_edges_ptrs;
    int *dst_ids = graph_data.edges_dst_ids;
    
    // convert to extra edges list array
    int *src_ids = new int [edges_count];
    #pragma omp parallel for schedule(guided) num_threads(_omp_threads_count)
    for (int cur_vertex = 0; cur_vertex < vertices_count; cur_vertex++)
    {
        long long first = vertices_to_edges_ptrs[cur_vertex];
        long long last = vertices_to_edges_ptrs[cur_vertex + 1];
        for (long long edge_pos = first; edge_pos < last; edge_pos++)
        {
            src_ids[edge_pos] = cur_vertex;
        }
    }
    
    // find tree forest using bfs on GPU
    vector<int>bfs_level(vertices_count, -1);
    bool *in_tree = new bool[edges_count];
    memset(in_tree, 0, edges_count * sizeof(bool));
    
    // do trim step
    vector<int> connections(vertices_count, 0);
    #pragma omp parallel for schedule(static) num_threads(_omp_threads_count)
    for(long long i = 0; i < edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        
        if(src_id == dst_id)
            continue;
        
        atomic_fetch_add((atomic<int>*)&connections[src_id], 1);
    }
    
    // init distances
    #pragma omp parallel for schedule(static) num_threads(_omp_threads_count)
    for(int i = 0; i < vertices_count; i++)
        bfs_level[i] = -1;
    
    // prepare
    #pragma omp parallel for schedule(static) num_threads(_omp_threads_count)
    for(int i = 0; i < vertices_count; i++)
    {
        if(connections[i] == 0)
        {
            bfs_level[i] = 1;
        }
    }
    
    // for all roots do bfs
    int max_level = 0;
    int times = 0;
    set<int> tree_roots;
    for(int root = 0; root < vertices_count; root++)
    {
        if(bfs_level[root] != -1)
            continue;
        tree_roots.insert(root);
        
        cpu_parallel_bfs(_graph, root, bfs_level, in_tree, max_level, _omp_threads_count);
    }
    
    // in tree fix
    #pragma omp parallel for schedule(guided) num_threads(_omp_threads_count)
    for(long long i = 0; i < edges_count; i++)
    {
        if(in_tree[i])
        {
            int src_id = src_ids[i];
            int dst_id = dst_ids[i];
            
            long long first = vertices_to_edges_ptrs[dst_id];
            long long last = vertices_to_edges_ptrs[dst_id + 1];
            for (long long edge_pos = first; edge_pos < last; edge_pos++)
            {
                if(dst_ids[edge_pos] == src_id)
                {
                    in_tree[edge_pos] = true;
                }
            }
        }
    }

    // build tree (from CPU data) on CPU sequential
    int prev_value = -1, pos = 0;
    vector<int> trees_vertices_to_edges_ptrs;
    vector<int> trees_src_ids, trees_dst_ids;
    for(long long i = 0; i < edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        
        if(in_tree[i])
        {
            if(src_id > prev_value)
            {
                for(int t = 0; t < (src_id - prev_value); t++)
                    trees_vertices_to_edges_ptrs.push_back(pos);
            }
            trees_src_ids.push_back(src_id);
            trees_dst_ids.push_back(dst_id);
            prev_value = src_id;
            pos++;
        }
    }
    trees_vertices_to_edges_ptrs.push_back(pos);
    long long trees_edges_count = trees_src_ids.size();
    
    // do pre-order on CPU sequential
    vector<int> N(vertices_count);
    vector<bool> DFS_visited(vertices_count, false);
    DFS_counter = 1;
    for(auto i: tree_roots)
    {
        DFS_pre_order(trees_vertices_to_edges_ptrs, trees_dst_ids, i, N, DFS_visited);
    }

    // compute D on CPU in parallel
    vector<int> D(vertices_count, 1);
    for(int level = max_level; level >= 0; level--)
    {
        #pragma omp parallel for schedule(static) num_threads(_omp_threads_count)
        for(long long i = 0; i < trees_edges_count; i++)
        {
            int src_id = trees_src_ids[i];
            int dst_id = trees_dst_ids[i];
            
            if((N[dst_id] > N[src_id]) && (bfs_level[src_id] == level))
            {
                atomic_fetch_add((atomic<int>*)&D[src_id], D[dst_id]);
            }
        }
    }
    
    // compute L and H on GPU in parallel
    vector<int> L(vertices_count, 0), H(vertices_count, 0);
    
    // use current vertex numbers to optimize
    #pragma omp parallel for schedule(static) num_threads(_omp_threads_count)
    for(int i = 0; i < vertices_count; i++)
    {
        L[i] = N[i];
        H[i] = N[i];
    }
    
    for(int level = max_level; level >= 1; level--)
    {
        // use tree to optimize
        #pragma omp parallel for schedule(static) num_threads(_omp_threads_count)
        for(long long i = 0; i < trees_edges_count; i++)
        {
            int src_id = trees_src_ids[i];
            int dst_id = trees_dst_ids[i];
            
            if(bfs_level[src_id] == level)
            {
                if(N[dst_id] > N[src_id])
                {
                    int old_val = L[src_id];
                    while (L[dst_id] < L[src_id])
                        if (atomic_compare_exchange_weak((atomic<int>*)&L[src_id], &old_val, L[dst_id]))
                            break;
                    
                    old_val = H[src_id];
                    while (H[dst_id] > H[src_id])
                        if (atomic_compare_exchange_weak((atomic<int>*)&H[src_id], &old_val, H[dst_id]))
                            break;
                }
            }
        }
        
        // use edges, not belonging to tree to optimize
        #pragma omp parallel for schedule(static) num_threads(_omp_threads_count)
        for(long long i = 0; i < edges_count; i++)
        {
            int src_id = src_ids[i];
            int dst_id = dst_ids[i];
            
            if(bfs_level[src_id] == level)
            {
                if(!in_tree[i])
                {
                    int old_val = L[src_id];
                    while (N[dst_id] < L[src_id])
                        if (atomic_compare_exchange_weak((atomic<int>*)&L[src_id], &old_val, N[dst_id]))
                            break;
                    
                    old_val = H[src_id];
                    while (N[dst_id] > H[src_id])
                        if (atomic_compare_exchange_weak((atomic<int>*)&H[src_id], &old_val, N[dst_id]))
                            break;
                }
            }
        }
    }
    
    // save result
    #pragma omp parallel for schedule(static) num_threads(_omp_threads_count)
    for(long long i = 0; i < edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        if(N[dst_id] > N[src_id])
        {
            if((L[dst_id] == N[dst_id]) && (H[dst_id] < (N[dst_id] + D[dst_id])))
            {
                _bridges[i] = true;
            }
        }
    }
    
    delete []in_tree;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BridgesDetection<_TVertexValue, _TEdgeWeight>::parallel_gpu_tarjan(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool *_bridges)
{
    double t1, t2;
    t1 = omp_get_wtime();
    int omp_threads_count = omp_get_max_threads();
    memset(_bridges, 0, _graph.get_edges_count());
    
    // get pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    long long *vertices_to_edges_ptrs = graph_data.vertices_to_edges_ptrs;
    int *dst_ids = graph_data.edges_dst_ids;
    
    // convert to extra edges list array
    int *src_ids = new int [edges_count];
    #pragma omp parallel for schedule(guided) num_threads(omp_threads_count)
    for (int cur_vertex = 0; cur_vertex < vertices_count; cur_vertex++)
    {
        long long first = vertices_to_edges_ptrs[cur_vertex];
        long long last = vertices_to_edges_ptrs[cur_vertex + 1];
        for (long long edge_pos = first; edge_pos < last; edge_pos++)
        {
            src_ids[edge_pos] = cur_vertex;
        }
    }
    
    // copy data to GPU
    int *device_src_ids, *device_dst_ids;
    int *device_bfs_level;
    bool *device_in_trees;
    
    SAFE_CALL(cudaMalloc((void**)&device_src_ids, sizeof(int) * edges_count));
    SAFE_CALL(cudaMalloc((void**)&device_dst_ids, sizeof(int) * edges_count));
    SAFE_CALL(cudaMalloc((void**)&device_bfs_level, sizeof(int) * vertices_count));
    SAFE_CALL(cudaMalloc((void**)&device_in_trees, sizeof(bool) * edges_count));
    
    SAFE_CALL(cudaMemcpy(device_src_ids, src_ids, sizeof(int) * edges_count, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(device_dst_ids, dst_ids, sizeof(int) * edges_count, cudaMemcpyHostToDevice));
    
    // find tree forest using bfs on GPU
    vector<int>bfs_level(vertices_count, -1);
    bool *in_tree = new bool[edges_count];
    memset(in_tree, 0, edges_count * sizeof(bool));
    
    // do trim step
    vector<int> connections(vertices_count, 0);
    #pragma omp parallel for schedule(static) num_threads(omp_threads_count)
    for(long long i = 0; i < edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        
        if(src_id == dst_id)
            continue;
        
        atomic_fetch_add((atomic<int>*)&connections[src_id], 1);
    }
    
    // init distances
    int max_degree = connections[0], max_vertex = 0;
    #pragma omp parallel for schedule(static) num_threads(omp_threads_count)
    for(int i = 0; i < vertices_count; i++)
    {
        bfs_level[i] = -1;
        if(connections[i] > max_degree)
            max_degree = connections[i], max_vertex = i;
    }
    
    cout << "max degree vertex: " << max_vertex << " with degree " << max_degree << endl;
    
    // prepare
    #pragma omp parallel for schedule(static) num_threads(omp_threads_count)
    for(int i = 0; i < vertices_count; i++)
    {
        if(connections[i] == 0)
        {
            bfs_level[i] = 1;
        }
    }
    
    // for all roots do bfs
    int max_level = 0;
    int times = 0;
    set<int> tree_roots;
    for(int root = 0; root < vertices_count; root++)
    {
        if(bfs_level[root] != -1)
            continue;
        tree_roots.insert(root);
        
        if (root == max_vertex)
        {
            tarjan_bfs_wrapper(device_src_ids, device_dst_ids, device_in_trees, edges_count, device_bfs_level,
                                      vertices_count, max_level, root);
        }
        else
        {
            cpu_parallel_bfs(_graph, root, bfs_level, in_tree, max_level, omp_threads_count);
        }
    }
    
    // in tree fix
    #pragma omp parallel for schedule(guided) num_threads(omp_threads_count)
    for(long long i = 0; i < edges_count; i++)
    {
        if(in_tree[i])
        {
            int src_id = src_ids[i];
            int dst_id = dst_ids[i];
            
            long long first = vertices_to_edges_ptrs[dst_id];
            long long last = vertices_to_edges_ptrs[dst_id + 1];
            for (long long edge_pos = first; edge_pos < last; edge_pos++)
            {
                if(dst_ids[edge_pos] == src_id)
                {
                    in_tree[edge_pos] = true;
                }
            }
        }
    }
    t2 = omp_get_wtime();
    cout << "first part: " << t2 - t1 << " sec" << endl;
    
    t1 = omp_get_wtime();
    // build tree (from GPU data) on CPU sequential
    int prev_value = -1, pos = 0;
    vector<int> trees_vertices_to_edges_ptrs;
    vector<int> trees_src_ids, trees_dst_ids;
    for(long long i = 0; i < edges_count; i++)
    {
        int src_id = src_ids[i];
        int dst_id = dst_ids[i];
        
        if(in_tree[i])
        {
            if(src_id > prev_value)
            {
                for(int t = 0; t < (src_id - prev_value); t++)
                    trees_vertices_to_edges_ptrs.push_back(pos);
            }
            trees_src_ids.push_back(src_id);
            trees_dst_ids.push_back(dst_id);
            prev_value = src_id;
            pos++;
        }
    }
    trees_vertices_to_edges_ptrs.push_back(pos);
    long long trees_edges_count = trees_src_ids.size();
    
    // do pre-order on CPU sequential
    vector<int> N(vertices_count);
    vector<bool> DFS_visited(vertices_count, false);
    DFS_counter = 1;
    for(auto i: tree_roots)
    {
        DFS_pre_order(trees_vertices_to_edges_ptrs, trees_dst_ids, i, N, DFS_visited);
    }
    t2 = omp_get_wtime();
    cout << "seq DFS: " << t2 - t1 << " sec" << endl;
    
    t1 = omp_get_wtime();
    // copy N(pre-order numbers) to device
    int *device_N;
    SAFE_CALL(cudaMalloc((void**)&device_N, sizeof(int) * vertices_count));
    SAFE_CALL(cudaMemcpy(device_N, &(N[0]), sizeof(int) * vertices_count, cudaMemcpyHostToDevice));
    
    // gpu memory for tree
    int *device_trees_src_ids, *device_trees_dst_ids;
    SAFE_CALL(cudaMalloc((void**)&device_trees_src_ids, sizeof(int) * trees_edges_count));
    SAFE_CALL(cudaMalloc((void**)&device_trees_dst_ids, sizeof(int) * trees_edges_count));
    
    SAFE_CALL(cudaMemcpy(device_trees_src_ids, &(trees_src_ids[0]), sizeof(int) * trees_edges_count, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(device_trees_dst_ids, &(trees_dst_ids[0]), sizeof(int) * trees_edges_count, cudaMemcpyHostToDevice));
    
    // compute D on GPU
    int *device_D;
    SAFE_CALL(cudaMalloc((void**)&device_D, sizeof(int) * vertices_count));
    compute_D_wrapper(device_trees_src_ids, device_trees_dst_ids, trees_edges_count, device_D, device_N, device_bfs_level,
                      vertices_count, max_level);
    
    // compute L and H on GPU
    int *device_L, *device_H;
    SAFE_CALL(cudaMalloc((void**)&device_L, sizeof(int) * vertices_count));
    SAFE_CALL(cudaMalloc((void**)&device_H, sizeof(int) * vertices_count));
    
    compute_L_H_wrapper(device_src_ids, device_dst_ids, device_in_trees, edges_count, device_trees_src_ids, device_trees_dst_ids,
                        trees_edges_count, device_L, device_H, device_N, device_bfs_level, vertices_count, max_level);
    
    // init GPU result
    bool *device_bridges;
    SAFE_CALL(cudaMalloc((void**)&device_bridges, sizeof(bool) * edges_count));
    SAFE_CALL(cudaMemset(device_bridges, 0, sizeof(bool) * edges_count));
    
    process_results_wrapper(device_src_ids, device_dst_ids, device_bridges, edges_count, device_L, device_H, device_D, device_N);
    
    // cope final result back to CPU
    SAFE_CALL(cudaMemcpy(_bridges, device_bridges, sizeof(bool) * edges_count, cudaMemcpyDeviceToHost));
    
    // free memory
    SAFE_CALL(cudaFree(device_bridges));
    
    SAFE_CALL(cudaFree(device_trees_src_ids));
    SAFE_CALL(cudaFree(device_trees_dst_ids));
    
    SAFE_CALL(cudaFree(device_N));
    SAFE_CALL(cudaFree(device_D));
    
    SAFE_CALL(cudaFree(device_L));
    SAFE_CALL(cudaFree(device_H));
    
    delete []in_tree;
    t2 = omp_get_wtime();
    cout << "final CUDA: " << t2 - t1 << " sec" << endl;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
