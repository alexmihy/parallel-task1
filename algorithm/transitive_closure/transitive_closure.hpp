/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct VertexPair
{
    VertexPair(int _first, int _second, int _old_index)
    {
        first = _first;
        second = _second;
        old_index = _old_index;
    }
    
    int first, second;
    int old_index;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void TransitiveClosure<_TVertexValue, _TEdgeWeight>::allocate_and_copy_device_arrays(Graph<_TVertexValue, _TEdgeWeight> &_graph)
{
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *host_src_ids = graph_data.edges_src_ids;
    int *host_dst_ids = graph_data.edges_dst_ids;
    
    SAFE_CALL(cudaMalloc((void**)&device_src_ids, sizeof(int) * edges_count));
    SAFE_CALL(cudaMalloc((void**)&device_dst_ids, sizeof(int) * edges_count));
    SAFE_CALL(cudaMalloc((void**)&device_result, sizeof(bool) * vertices_count));
    
    // copy graph to device
    SAFE_CALL(cudaMemcpy(device_src_ids, host_src_ids, sizeof(int) * edges_count, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(device_dst_ids, host_dst_ids, sizeof(int) * edges_count, cudaMemcpyHostToDevice));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void TransitiveClosure<_TVertexValue, _TEdgeWeight>::gpu_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool *_result,
                                                             int _source_vertex)
{
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    
    // perform computations
    bfs_wrapper(device_src_ids, device_dst_ids, vertices_count, edges_count, device_result, _source_vertex);
    
    // compy result back
    SAFE_CALL(cudaMemcpy(_result, device_result, vertices_count * sizeof(bool), cudaMemcpyDeviceToHost));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void TransitiveClosure<_TVertexValue, _TEdgeWeight>::free_device_arrays()
{
    // free memory
    SAFE_CALL(cudaFree(device_src_ids));
    SAFE_CALL(cudaFree(device_dst_ids));
    SAFE_CALL(cudaFree(device_result));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
double TransitiveClosure<_TVertexValue, _TEdgeWeight>::cpu_purdom(Graph<_TVertexValue, _TEdgeWeight> &_input_graph,
                                                                  vector<pair<int, int>> _pairs_to_check,
                                                                  vector<bool> &_answer)
{
    double t3 = omp_get_wtime();
    double t1 = omp_get_wtime();
   
    // resize result vector
    _answer.resize(_pairs_to_check.size());
    
    // get vertices and edges count
    int input_graph_vertices_count = _input_graph.get_vertices_count();
    long long input_graph_edges_count = _input_graph.get_edges_count();
    
    // compute strongly connected components for current graph
    int *scc_result = new int[input_graph_vertices_count];
    StronglyConnectedComponents<int, float> scc_operation(omp_threads);
    scc_operation.cpu_forward_backward(_input_graph, scc_result);
    //scc_operation.cpu_tarjan(_input_graph, scc_result);
    //_input_graph.convert_to_edges_list();
    
    double t2 = omp_get_wtime();
    cout << "scc time: " << t2 - t1 << " sec" << endl;
    
    t1 = omp_get_wtime();
    
    // add vertices to intermediate representation
    vector<bool> set(input_graph_vertices_count);
    int scc_count = 0;
    for(int i = 0 ; i < input_graph_vertices_count; i++)
    {
        if(!set[scc_result[i]])
        {
            set[scc_result[i]] = true;
            scc_count++;
        }
    }
    
    // add edges to intermediate representation
    vector<int> ir_src_ids;
    vector<int> ir_dst_ids;
    
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _input_graph.get_graph_data();
    int *src_ids = graph_data.edges_src_ids;
    int *dst_ids = graph_data.edges_dst_ids;
    
    vector<vector<int>> thread_ir_src_ids(omp_threads);
    vector<vector<int>> thread_ir_dst_ids(omp_threads);
    
    #pragma omp parallel num_threads(omp_threads) shared(ir_src_ids, ir_dst_ids)
    {
        int thread_id = omp_get_thread_num();
        
        #pragma omp for schedule(guided, 1024)
        for(long long i = 0; i < input_graph_edges_count; i++)
        {
            int left_component = scc_result[src_ids[i]];
            int right_component = scc_result[dst_ids[i]];
        
            if(left_component != right_component) // if edge connects different SCCs, add it
            {
                thread_ir_src_ids[thread_id].push_back(left_component);
                thread_ir_dst_ids[thread_id].push_back(right_component);
            }
        }
    }
    
    for (int i = 0; i < omp_threads; i++)
    {
        ir_src_ids.insert(ir_src_ids.end(), thread_ir_src_ids[i].begin(), thread_ir_src_ids[i].end());
        ir_dst_ids.insert(ir_dst_ids.end(), thread_ir_dst_ids[i].begin(), thread_ir_dst_ids[i].end());
    }
    
    int ir_vertices_count = scc_count;
    long long ir_edges_count = ir_src_ids.size();
    t2 = omp_get_wtime();
    cout << "ir time: " << t2 - t1 << " sec" << endl;
    
    t1 = omp_get_wtime();
    int c = 0;
    // compute answer for all pairs
    bool *bfs_result = new bool[ir_vertices_count];
    for(int pair_pos = 0; pair_pos < _pairs_to_check.size(); pair_pos++) // for all pairs
    {
        pair<int, int> cur_pair = _pairs_to_check[pair_pos];
        
        int left_component = scc_result[cur_pair.first]; // get scc number of vertex 1
        int right_component = scc_result[cur_pair.second]; // get scc number of vertex 2
        
        if(left_component == right_component) // if they have same scc numbers, belong to transitive closure
        {
            _answer[pair_pos] = true;
        }
        else // if not, do bfs
        {
            int left_old_id = cur_pair.first; // old vertex ID
            int right_old_id = cur_pair.second; // old vertex ID
            
            edges_list_bfs(&(ir_src_ids[0]), &(ir_dst_ids[0]), bfs_result, left_component, ir_vertices_count, ir_edges_count);
            c++;
            
            int right_component = scc_result[right_old_id];
            
            _answer[pair_pos] = bfs_result[right_component];
        }
    }
    delete []bfs_result;
    delete []scc_result;
    
    t2 = omp_get_wtime();
    cout << "check time: " << t2 - t1 << " sec" << endl;
    
    double t4 = omp_get_wtime();
    cout << "total time: " << t4 - t3 << " sec" << endl;
    cout << "total performance: "  << (5 * input_graph_edges_count + c * ir_edges_count) / ((t4 - t3) * 1e6) << " MTEPS" << endl << endl;
    return double(5 * input_graph_edges_count + c * ir_edges_count);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void TransitiveClosure<_TVertexValue, _TEdgeWeight>::edges_list_bfs(int *_src_ids,
                                                                    int *_dst_ids,
                                                                    bool *_result,
                                                                    int _source_vertex,
                                                                    int _vertices_count,
                                                                    long long _edges_count)
{
    bool changes = true;
    #pragma omp parallel num_threads(omp_threads) shared(changes)
    {
        // set initial distances
        #pragma omp for schedule(static)
        for (int i = 0; i < _vertices_count; i++)
        {
            _result[i] = UNVISITED;
        }
        _result[_source_vertex] = VISITED;
        
        // do bellman-ford algorithm
        while (changes)
        {
            #pragma omp barrier
            
            changes = false;
            
            #pragma omp barrier
            
            #pragma omp for schedule(guided, 1024)
            for (long long cur_edge = 0; cur_edge < _edges_count; cur_edge++)
            {
                int src_id = _src_ids[cur_edge];
                int dst_id = _dst_ids[cur_edge];
                
                if((_result[src_id] == VISITED) && (_result[dst_id] == UNVISITED))
                {
                    _result[dst_id] = VISITED;
                    changes = true;
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void TransitiveClosure<_TVertexValue, _TEdgeWeight>::adj_list_bfs(vector<vector<int>> &_adj_ids,
                                                                  bool *_result,
                                                                  int _source_vertex,
                                                                  int _vertices_count)
{
    for(int i = 0; i < _vertices_count; i++)
        _result[i] = UNVISITED;
    
    // init queue and first vertex
    std::queue<int> vertex_queue;
    vertex_queue.push(_source_vertex);
    _result[_source_vertex] = VISITED;
    
    // do bfs
    while(vertex_queue.size() > 0)
    {
        int cur_vertex = vertex_queue.front();
        vertex_queue.pop();
        
        int adj_count = _adj_ids[cur_vertex].size();
        for(int i = 0; i < adj_count; i++)
        {
            int dst_id = _adj_ids[cur_vertex][i];
            
            if(_result[dst_id] == UNVISITED)
            {
                _result[dst_id] = VISITED;
                vertex_queue.push(dst_id);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void TransitiveClosure<_TVertexValue, _TEdgeWeight>::cpu_purdom2(Graph<_TVertexValue, _TEdgeWeight> &_input_graph,
                                                                 vector<pair<int, int>> _pairs_to_check,
                                                                 vector<bool> &_answer)
{
    double t3 = omp_get_wtime();
    
    // resize result vector
    _answer.resize(_pairs_to_check.size());
    
    // get vertices and edges count
    int input_graph_vertices_count = _input_graph.get_vertices_count();
    long long input_graph_edges_count = _input_graph.get_edges_count();
    
    // compute strongly connected components for current graph
    double t1 = omp_get_wtime();
    int *scc_result = new int[input_graph_vertices_count];
    StronglyConnectedComponents<int, float> scc_operation(omp_threads);
    scc_operation.cpu_forward_backward(_input_graph, scc_result);
    double t2 = omp_get_wtime();
    cout << "scc time: " << t2 - t1 << " sec" << endl;
    
    t1 = omp_get_wtime();
    
    // add vertices to intermediate representation
    vector<bool> set(input_graph_vertices_count);
    int scc_count = 0;
    for(int i = 0 ; i < input_graph_vertices_count; i++)
    {
        if(!set[scc_result[i]])
        {
            set[scc_result[i]] = true;
            scc_count++;
        }
    }
    
    // add edges to intermediate representation
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _input_graph.get_graph_data();
    int *src_ids = graph_data.edges_src_ids;
    int *dst_ids = graph_data.edges_dst_ids;
    
    vector<vector<int>> adj_ids(scc_count);
    vector<vector<vector<int>>> local_adj_ids(omp_threads);
    
    #pragma omp parallel for num_threads(omp_threads) schedule(static)
    for(int i = 0; i < omp_threads; i++)
    {
        local_adj_ids[i].resize(scc_count);
    }
    
    #pragma omp parallel num_threads(omp_threads)
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(guided, 1024)
        for(long long i = 0; i < input_graph_edges_count; i++)
        {
            int left_component = scc_result[src_ids[i]];
            int right_component = scc_result[dst_ids[i]];
        
            if(left_component != right_component) // if edge connects different SCCs, add it
            {
                local_adj_ids[thread_id][left_component].push_back(right_component);
            }
        }
    }
    
    for(int thread = 0; thread < omp_threads; thread++)
    {
        #pragma omp parallel for schedule(guided, 1024)
        for(int vert = 0; vert < scc_count; vert++)
        {
            adj_ids[vert].insert(adj_ids[vert].end(), local_adj_ids[thread][vert].begin(), local_adj_ids[thread][vert].end());
        }
    }
    
    int ir_edges_count = 0;
    #pragma omp parallel for num_threads(omp_threads) schedule(static) reduction(+ : ir_edges_count)
    for (int i = 0; i < scc_count; i++)
    {
        ir_edges_count += adj_ids[i].size();
    }
    
    t2 = omp_get_wtime();
    cout << "ir time: " << t2 - t1 << " sec" << endl;
    
    t1 = omp_get_wtime();
    // compute answer for all pairs
    bool **bfs_result = new bool*[omp_threads];
    for(int i = 0; i < omp_threads; i++)
        bfs_result[i] = new bool[scc_count];
    
    int c = 0;
    #pragma omp parallel for num_threads(omp_threads) schedule(dynamic) reduction(+ : c)
    for(int pair_pos = 0; pair_pos < _pairs_to_check.size(); pair_pos++) // for all pairs
    {
        int thread_id = omp_get_thread_num();
        pair<int, int> cur_pair = _pairs_to_check[pair_pos];
        
        int left_component = scc_result[cur_pair.first]; // get scc number of vertex 1
        int right_component = scc_result[cur_pair.second]; // get scc number of vertex 2
        
        if(left_component == right_component) // if they have same scc numbers, belong to transitive closure
        {
            _answer[pair_pos] = true;
        }
        else // if not, do bfs
        {
            int left_old_id = cur_pair.first; // old vertex ID
            int right_old_id = cur_pair.second; // old vertex ID
            adj_list_bfs(adj_ids, bfs_result[thread_id], left_component, scc_count);
            int right_component = scc_result[right_old_id];
            _answer[pair_pos] = bfs_result[thread_id][right_component];
            
            c++;
        }
    }
    t2 = omp_get_wtime();
    cout << "check time: " << t2 - t1 << " sec" << endl;
    
    for(int i = 0; i < omp_threads; i++)
        delete []bfs_result[i];
    delete []bfs_result;
    delete []scc_result;
    
    double t4 = omp_get_wtime();
    cout << "total time: " << t4 - t3 << " sec" << endl;
    cout << "total performance: "  << (5 * input_graph_edges_count + c * ir_edges_count) / ((t4 - t3) * 1e6) << " MTEPS" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
