/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::cpu_sequential_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                          bool *_result, int _source_vertex)
{
    // convert to necessary format
    _graph.convert_to_compressed_adjacency_list();
    
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    long long *vertices_to_edges_ptrs = graph_data.vertices_to_edges_ptrs;
    int *dst_ids = graph_data.edges_dst_ids;
    
    // init distances
    for(int i = 0; i < vertices_count; i++)
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
        
        long long first = vertices_to_edges_ptrs[cur_vertex];
        long long last = vertices_to_edges_ptrs[cur_vertex + 1];
        
        for(long long i = first; i < last; i++)
        {
            int dst_id = dst_ids[i];
            
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
void BFS<_TVertexValue, _TEdgeWeight>::nec_sx_parallel_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                           bool *_result,
                                                           int _source_vertex)
{
    // convert to necessary format
    _graph.convert_to_edges_list();
    
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *src_ids = graph_data.edges_src_ids;
    int *dst_ids = graph_data.edges_dst_ids;
    
    double t1 = omp_get_wtime();
    
    int iterations_count = 0;
    bool changes = true;

    #pragma omp parallel num_threads(omp_threads) shared(changes)
    {
        // set initial distances
        #pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; i++)
        {
            _result[i] = UNVISITED;
        }
        _result[_source_vertex] = VISITED;
        
        // do bellman-ford algorithm
        while (changes)
        {
            #pragma omp barrier
            
            changes = false;
            #pragma omp single
            iterations_count++;
            
            #pragma omp barrier
            
            #pragma omp for schedule(guided, 1024)
            for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
            {
                int src_id = src_ids[cur_edge];
                int dst_id = dst_ids[cur_edge];
                
                if((_result[src_id] == VISITED) && (_result[dst_id] == UNVISITED))
                {
                    _result[dst_id] = VISITED;
                    changes = true;
                }
            }
        }
    }

    
    double t2 = omp_get_wtime();
    
    cout << "inner time: " << t2 - t1 << " sec" << endl;
    cout << "iterations: " << iterations_count << endl;
    cout << "throughput: " << (iterations_count * edges_count * (2 * sizeof(int) + 3*sizeof(bool))) / ((t2 - t1) * (1024 * 1024)) << " GB/s" << endl;
    cout << "perfomance: " << _graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::cpu_parallel_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                        bool *_result, int _source_vertex)
{
    // convert to necessary format
    _graph.convert_to_compressed_adjacency_list();
    
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    long long *vertices_to_edges_ptrs = graph_data.vertices_to_edges_ptrs;
    int *dst_ids = graph_data.edges_dst_ids;
    
    // init distances
    #pragma omp parallel for num_threads(omp_threads) schedule(static)
    for(int i = 0; i < vertices_count; i++)
        _result[i] = UNVISITED;
    
    // init queue and first vertex
    vector<int> global_queue;
    
    global_queue.push_back(_source_vertex);
    _result[_source_vertex] = VISITED;
    
    int omp_threads = omp_get_max_threads();
    int current_level = 1;
    vector<vector<int>> thread_queues(omp_threads);
    // arrays for merging
    vector<int> sizes(omp_threads);
    vector<int> offsets(omp_threads);
    
    // do bfs for each level
    #pragma omp parallel num_threads(omp_threads)
    {
        int thread_id = omp_get_thread_num();
        vector<int> &local_queue = thread_queues[thread_id];
        do
        {
            // traverse single level
            #pragma omp for schedule(guided)
            for(int i = 0; i < global_queue.size(); i++)
            {
                int cur_vertex = global_queue[i];
                
                long long first = vertices_to_edges_ptrs[cur_vertex];
                long long last = vertices_to_edges_ptrs[cur_vertex + 1];
                for(long long cur_edge = first; cur_edge < last; cur_edge++)
                {
                    int dst_id = dst_ids[cur_edge];
                    if(_result[dst_id] == UNVISITED)
                    {
                        _result[dst_id] = VISITED;
                        local_queue.push_back(dst_id);
                    }
                }
            }
            
            #pragma omp barrier
            
            #pragma omp master
            {
                // compute
                int total_size = 0;
                for(int i = 0; i < omp_threads; i++)
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
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::gpu_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                               bool *_result, int _source_vertex)
{
    int omp_threads = omp_get_max_threads();
    
    // convert to necessary format
    _graph.convert_to_edges_list();
    
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *host_src_ids = graph_data.edges_src_ids;
    int *host_dst_ids = graph_data.edges_dst_ids;
    
    // init device data
    int *device_src_ids, *device_dst_ids;
    bool *device_result;
    SAFE_CALL(cudaMalloc((void**)&device_src_ids, sizeof(int) * edges_count));
    SAFE_CALL(cudaMalloc((void**)&device_dst_ids, sizeof(int) * edges_count));
    SAFE_CALL(cudaMalloc((void**)&device_result, sizeof(bool) * vertices_count));
    
    // copy graph to device
    SAFE_CALL(cudaMemcpy(device_src_ids, host_src_ids, sizeof(int) * edges_count, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(device_dst_ids, host_dst_ids, sizeof(int) * edges_count, cudaMemcpyHostToDevice));
    
    // perform computations
    double t1 = omp_get_wtime();
    bfs_wrapper(device_src_ids, device_dst_ids, vertices_count, edges_count, device_result, _source_vertex);
    double t2 = omp_get_wtime();
    cout << "no memcpy performance: " << edges_count / ((t2 - t1) * 1e6) << " MTEPS" << endl;
    
    // compy result back
    SAFE_CALL(cudaMemcpy(_result, device_result, vertices_count * sizeof(bool), cudaMemcpyDeviceToHost));
    
    // free memory
    SAFE_CALL(cudaFree(device_src_ids));
    SAFE_CALL(cudaFree(device_dst_ids));
    SAFE_CALL(cudaFree(device_result));
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_KNL__
template <typename _TVertexValue, typename _TEdgeWeight>
void BFS<_TVertexValue, _TEdgeWeight>::knl_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                               bool *_result, int _source_vertex)
{
    // convert to necessary format
    _graph.convert_to_edges_list();
    
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *src_ids = graph_data.edges_src_ids;
    int *dst_ids = graph_data.edges_dst_ids;
    
    double t1 = omp_get_wtime();
    
    int iterations_count = 0;
    bool changes = true;
    #pragma omp parallel num_threads(omp_threads) shared(changes)
    {
        // set initial distances
        #pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; i++)
        {
            _result[i] = UNVISITED;
        }
        _result[_source_vertex] = VISITED;
        
        // do bellman-ford algorithm
        while (changes)
        {
            #pragma omp barrier
            
            changes = false;
            #pragma omp single
            iterations_count++;
            
            #pragma omp barrier
            
            #pragma omp for schedule(guided, 1024)
            for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
            {
                int src_id = src_ids[cur_edge];
                int dst_id = dst_ids[cur_edge];
                
                if((_result[src_id] == VISITED) && (_result[dst_id] == UNVISITED))
                {
                    _result[dst_id] = VISITED;
                    changes = true;
                }
            }
        }
    }
    
    double t2 = omp_get_wtime();
    
    cout << "inner time: " << t2 - t1 << " sec" << endl;
    cout << "iterations: " << iterations_count << endl;
    cout << "throughput: " << (iterations_count * edges_count * (2 * sizeof(int) + 3*sizeof(bool))) / ((t2 - t1) * (1024 * 1024)) << " GB/s" << endl;
    cout << "perfomance: " << _graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ParallelQueue
{
    int *queue;
    int size;
    int top;
    
    ParallelQueue(int _size = 0)
    {
        size = _size;
        top = 0;
        if(size > 0)
            queue = new int[size];
        else
            queue = NULL;
    }
    
    ~ParallelQueue()
    {
        if(queue != NULL)
            delete []queue;
        queue = NULL;
    }
    
    void allocate(int _size)
    {
        size = _size;
        top = 0;
        if(size > 0)
        {
            queue = new int[size];
        }
    }
    
    void free()
    {
        if(queue != NULL)
            delete []queue;
        queue = NULL;
    }
    
    void push(int val)
    {
        if (top >= size)
        {
            int *new_queue = new int[2*size];             // alocate new
            memcpy(new_queue, queue, size * sizeof(int)); // copy from old to new
            delete[] queue;                               // delete old
            queue = new_queue;                            // swap pointers
            size *= 2;                                    // update size
        }
        
        queue[top] = val;
        top++;
    }
    
    bool empty()
    {
        if(top > 0)
            return false;
        else
            return true;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

