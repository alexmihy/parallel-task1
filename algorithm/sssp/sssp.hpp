/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void SingleSourceShortestPaths<_TVertexValue, _TEdgeWeight>::cpu_bellman_ford(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                              _TEdgeWeight *_distances,
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
    _TEdgeWeight *weights = graph_data.edges_weights;
    
    double t1 = omp_get_wtime();
    
    double max_val = numeric_limits<_TEdgeWeight>::max();
    bool changes = true;
    int iterations = 0;
    #pragma omp parallel num_threads(omp_threads) shared(changes)
    {
        // set initial distances
        #pragma omp for schedule(static)
        for (int i = 0; i < vertices_count; i++)
        {
            _distances[i] = max_val;
        }
        _distances[_source_vertex] = 0;
        
        // do bellman-ford algorithm
        while (changes)
        {
            #pragma omp barrier
            
            #pragma omp single
            iterations++;
            
            changes = false;
            
            #pragma omp barrier
            
            #pragma omp for schedule(guided)
            for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
            {
                int src = src_ids[cur_edge];
                int dst = dst_ids[cur_edge];
                _TEdgeWeight weight = weights[cur_edge];
                
                _TEdgeWeight src_distance = _distances[src];
                _TEdgeWeight dst_distance = _distances[dst];
                
                if (dst_distance > src_distance + weight)
                {
                    _distances[dst] = src_distance + weight;
                    changes = true;
                }
            }
        }
    }
    
    double t2 = omp_get_wtime();
    
    //cout << "CPU iterations count: " << iterations << endl;
    
    //cout << "inner time: " << t2 - t1 << " sec" << endl;
    //cout << "perfomance: " << _graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_KNL__
template <typename _TVertexValue, typename _TEdgeWeight>
void SingleSourceShortestPaths<_TVertexValue, _TEdgeWeight>::knl_bellman_ford(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                              _TEdgeWeight *_distances,
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
    _TEdgeWeight *weights = graph_data.edges_weights;
    _TEdgeWeight *distances = _distances;
    
    while(edges_count % 16 != 0)
        edges_count++;
    
    int *aligned_src_ids = _mm_malloc(edges_count * sizeof(int), 64);
    int *aligned_dst_ids = _mm_malloc(edges_count * sizeof(int), 64);
    _TEdgeWeight *aligned_weights = _mm_malloc(edges_count * sizeof(_TEdgeWeight), 64);
    _TEdgeWeight *aligned_distances = _mm_malloc(vertices_count * sizeof(_TEdgeWeight), 64);
    
    for(long long i = 0; i < edges_count; i++)
    {
        if(i < _graph.get_edges_count())
        {
            aligned_src_ids[i] = src_ids[i];
            aligned_dst_ids[i] = dst_ids[i];
            aligned_weights[i] = weights[i];
        }
        else
        {
            aligned_src_ids[i] = 0;
            aligned_dst_ids[i] = 0;
            aligned_weights[i] = numeric_limits<_TEdgeWeight>::max();
        }
    }
    for(long long i = 0; i < vertices_count; i++)
    {
        aligned_distances[i] = numeric_limits<_TEdgeWeight>::max();
    }
    aligned_distances[_source_vertex] = 0;
    
    const __m512i* src_ids_ptrs = (__m512i*)aligned_src_ids;
    const __m512i* dst_ids_ptrs = (__m512i*)aligned_dst_ids;
    
    __mmask16 true_mask;
    true_mask = _mm512_kor(true_mask, _mm512_knot(true_mask));
    __mmask16 false_mask = _mm512_knot(true_mask);
    
    double t1 = omp_get_wtime();
    
    bool changes = true;
    int iterations = 0;
    
    int prefetch_distance=8;
    #pragma omp parallel num_threads(omp_threads)
    {
        while (changes)
        {
            #pragma omp single
            iterations++;
            
            #pragma omp barrier
            
            changes = false;
            __mmask16 changes_mask = false_mask;
            
            #pragma omp barrier
            
            #pragma omp for schedule(guided, 1024)
            for (int i = 0; i < (edges_count / 16 - prefetch_distance); i++)
            {
                __m512i src_prefetch = _mm512_load_si512(&src_ids_ptrs[i+prefetch_distance]);
                __m512i dst_prefetch = _mm512_load_si512(&dst_ids_ptrs[i+prefetch_distance]);
                _mm512_prefetch_i32extgather_ps (src_prefetch, aligned_distances, 0, 4, 2);
                _mm512_prefetch_i32extgather_ps (dst_prefetch, aligned_distances, 0, 4, 2);
                
                __m512i src_avx = _mm512_load_si512(&src_ids_ptrs[i]);
                __m512i dst_avx = _mm512_load_si512(&dst_ids_ptrs[i]);
                __m512 weight_avx = _mm512_load_ps(&aligned_weights[16*i]);
                
                __m512 src_distance_avx = _mm512_i32gather_ps(src_avx, aligned_distances, 4);
                __m512 dst_distance_avx = _mm512_i32gather_ps(dst_avx, aligned_distances, 4);
                
                __m512 new_distance_avx = _mm512_add_ps(src_distance_avx, weight_avx);
                
                __mmask16 cmp_result_mask = _mm512_cmp_ps_mask(new_distance_avx, dst_distance_avx, _MM_CMPINT_LT);
                
                changes_mask = _mm512_kor(changes_mask, cmp_result_mask);
                
                _mm512_mask_i32scatter_ps(aligned_distances, cmp_result_mask, dst_avx, new_distance_avx, 4);
            }
            
            
            #pragma omp for schedule(static)
            for (int i = edges_count / 16 - prefetch_distance; i < edges_count/16; i++)
            {
                
                __m512i src_avx = _mm512_load_si512(&src_ids_ptrs[i]);
                __m512i dst_avx = _mm512_load_si512(&dst_ids_ptrs[i]);
                __m512 weight_avx = _mm512_load_ps(&aligned_weights[16*i]);
                
                __m512 src_distance_avx = _mm512_i32gather_ps(src_avx, aligned_distances, 4);
                __m512 dst_distance_avx = _mm512_i32gather_ps(dst_avx, aligned_distances, 4);
                
                __m512 new_distance_avx = _mm512_add_ps(src_distance_avx, weight_avx);
                
                __mmask16 cmp_result_mask = _mm512_cmp_ps_mask(new_distance_avx, dst_distance_avx, _MM_CMPINT_LT);
                
                changes_mask = _mm512_kor(changes_mask, cmp_result_mask);
                
                _mm512_mask_i32scatter_ps(aligned_distances, cmp_result_mask, dst_avx, new_distance_avx, 4);
            }
            
            #pragma omp barrier
            
            CPUmask check_changes = { changes_mask };
            if((check_changes.a[0] != 0)|| (check_changes.a[1] != 0))
                changes = true;
            
            #pragma omp barrier
        }
    }
    double t2 = omp_get_wtime();
    cout << "inner avx time : " << t2 - t1 << " sec" << endl;
    cout << "itterations: " << iterations << endl;
    cout << "perfomance: " << _graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
    
    memcpy(_distances, aligned_distances, vertices_count * sizeof(_TEdgeWeight));
    
    _mm_free(aligned_src_ids);
    _mm_free(aligned_dst_ids);
    _mm_free(aligned_weights);
    _mm_free(aligned_distances);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
template <typename _TVertexValue, typename _TEdgeWeight>
double SingleSourceShortestPaths<_TVertexValue, _TEdgeWeight>::gpu_bellman_ford(Graph<_TVertexValue, _TEdgeWeight> &_graph,
                                                                              _TEdgeWeight *_distances,
                                                                              int _source_vertex)
{
    // convert to necessary format
    _graph.convert_to_edges_list();

    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _graph.get_graph_data();
    int vertices_count = _graph.get_vertices_count();
    long long edges_count = _graph.get_edges_count();
    int *host_src_ids = graph_data.edges_src_ids;
    int *host_dst_ids = graph_data.edges_dst_ids;
    _TEdgeWeight *host_weights = graph_data.edges_weights;
    _TEdgeWeight *host_distances = _distances;
    
    double t1 = omp_get_wtime();
    // device data
    int *device_src_ids;
    int *device_dst_ids;
    _TEdgeWeight *device_weights;
    _TEdgeWeight *device_distances;
    
    // memory for edges storage on device
    SAFE_CALL(cudaMalloc((void**)&device_src_ids, edges_count * sizeof(int)));
    SAFE_CALL(cudaMalloc((void**)&device_dst_ids, edges_count * sizeof(int)));
    SAFE_CALL(cudaMalloc((void**)&device_weights, edges_count * sizeof(_TEdgeWeight)));
    SAFE_CALL(cudaMalloc((void**)&device_distances, vertices_count * sizeof(_TEdgeWeight)));
    
    // copy edges data from host to device
    SAFE_CALL(cudaMemcpy(device_src_ids, host_src_ids, edges_count * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(device_dst_ids, host_dst_ids, edges_count * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(device_weights, host_weights, edges_count * sizeof(_TEdgeWeight), cudaMemcpyHostToDevice));

    // compute shortest paths on GPU
    double t2 = omp_get_wtime();
    init_distances_wrapper<float>(device_distances, vertices_count, _source_vertex);
    bellman_ford_wrapper<float>(device_src_ids, device_dst_ids, device_weights, vertices_count, edges_count, _source_vertex,
                         device_distances);
    SAFE_CALL(cudaDeviceSynchronize());
    
    double t3 = omp_get_wtime();
    
    // copy results back to host
    SAFE_CALL(cudaMemcpy(host_distances, device_distances, vertices_count * sizeof(_TEdgeWeight), cudaMemcpyDeviceToHost));
    
    // free all memory
    SAFE_CALL(cudaFree(device_src_ids));
    SAFE_CALL(cudaFree(device_dst_ids));
    SAFE_CALL(cudaFree(device_weights));
    SAFE_CALL(cudaFree(device_distances));
    double t4 = omp_get_wtime();
    
    cout << "GPU-only perfomance: " << _graph.get_edges_count() / ((t3 - t2) * 1e6) << " MTEPS" << endl;
    
    return _graph.get_edges_count() / ((t3 - t2) * 1e6);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
