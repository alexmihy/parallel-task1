/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// private functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int GraphOptimizationAPI<_TVertexValue, _TEdgeWeight>::get_segment_num(int _id, int _cache_line_size)
{
    return _id / _cache_line_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphOptimizationAPI<_TVertexValue, _TEdgeWeight>::reorder_edges_in_adjacency_list_format(Graph<_TVertexValue, _TEdgeWeight>
                                                                                               &_input_graph)
{
    // get vertices and edges count
    int vertices_count = _input_graph.get_vertices_count();
    long long edges_count = _input_graph.get_edges_count();
    
    // fill sort array
    Edge<_TEdgeWeight> *array_for_sort = new Edge<_TEdgeWeight>[edges_count];
    for(long long i = 0; i < edges_count; i++)
    {
        array_for_sort[i] = _input_graph.iterate_edges(i);
    }
    
    // sort edges in most optimal order for cache usage
    auto comp = [&](Edge<_TEdgeWeight> lhs, Edge<_TEdgeWeight> rhs)-> bool
    {
        if(lhs.src_id != rhs.src_id)
            return lhs.src_id < rhs.src_id;
        if(lhs.dst_id != rhs.dst_id)
            return lhs.dst_id < rhs.dst_id;
        return lhs.weight < rhs.weight;
    };
    std::sort(array_for_sort, array_for_sort + edges_count, comp);
    
    // create tmp graph
    Graph<_TVertexValue, _TEdgeWeight> tmp_graph(vertices_count, edges_count, EDGES_LIST, _input_graph.check_if_directed(), true);
    
    // add old vertices
    for(int i = 0; i < vertices_count; i++)
        tmp_graph.add_vertex(_input_graph.iterate_vertices(i));
    
    // add sorted edges in right order
    for(long long i = 0; i < edges_count; i++)
        tmp_graph.add_edge(array_for_sort[i]);
    
    // copy temporary graph to input
    _input_graph = tmp_graph;
    
    // delete memory for sort array
    delete[] array_for_sort;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphOptimizationAPI<_TVertexValue, _TEdgeWeight>::remove_loops_and_multiple_arcs(Graph<_TVertexValue, _TEdgeWeight> &_input_graph)
{
    // convert into format to remove edges easily
    reorder_edges_in_adjacency_list_format(_input_graph);
    
    // get vertices and edges count
    int vertices_count = _input_graph.get_vertices_count();
    long long edges_count = _input_graph.get_edges_count();
    
    // bool array which edges to keep
    vector<bool> edges_to_keep(edges_count, true);
    
    long long new_edges_count = 0;
    // remove loops and multiple arcs
    for(long long i = 0; i < edges_count; i++)
    {
        Edge<_TEdgeWeight> cur_edge = _input_graph.iterate_edges(i);
        Edge<_TEdgeWeight> prev_edge = _input_graph.iterate_edges(i - 1);
        
        // remove if edge is loop
        if(cur_edge.src_id == cur_edge.dst_id)
        {
            edges_to_keep[i] = false;
            continue;
        }
        
        // remove if edge is multiple arc
        if(i > 0)
        {
            if((cur_edge.src_id == prev_edge.src_id) && (cur_edge.dst_id == prev_edge.dst_id) && (cur_edge.weight >= prev_edge.weight))
            {
                edges_to_keep[i] = false;
                continue;
            }
        }
        
        // keep if edge is usual
        edges_to_keep[i] = true;
        new_edges_count++;
    }
    
    // create tmp graph
    Graph<_TVertexValue, _TEdgeWeight> tmp_graph(vertices_count, new_edges_count, EDGES_LIST, _input_graph.check_if_directed(), true);
    
    // add old vertices
    for(int i = 0; i < vertices_count; i++)
        tmp_graph.add_vertex(_input_graph.iterate_vertices(i));
    
    // add edges left after removal
    for(long long i = 0; i < edges_count; i++)
    {
        if(edges_to_keep[i])
            tmp_graph.add_edge(_input_graph.iterate_edges(i));
    }
    
    // copy temporary graph to input
    _input_graph = tmp_graph;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphOptimizationAPI<_TVertexValue, _TEdgeWeight>::reorder_edges_for_cache(Graph<_TVertexValue, _TEdgeWeight> &_input_graph,
                                                                                int _cache_size, int elem_size)
{
    // get vertices and edges count
    int vertices_count = _input_graph.get_vertices_count();
    long long edges_count = _input_graph.get_edges_count();
    
    int cache_line_size = _cache_size / elem_size;
    //int cache_line_size = vertices_count / _p;
    cout << "cache line size: " << cache_line_size * elem_size / 1024 << " KB" << endl;
    cout << "segments count: " << vertices_count / cache_line_size << endl;
    
    // fill sort array
    Edge<_TEdgeWeight> *array_for_sort = new Edge<_TEdgeWeight>[edges_count];
    for(long long i = 0; i < edges_count; i++)
    {
        array_for_sort[i] = _input_graph.iterate_edges(i);
    }
    
    // test segments
    auto comp = [&](Edge<_TEdgeWeight> lhs, Edge<_TEdgeWeight> rhs)-> bool
    {
        if(get_segment_num(lhs.src_id, cache_line_size) != get_segment_num(rhs.src_id, cache_line_size))
            return get_segment_num(lhs.src_id, cache_line_size) < get_segment_num(rhs.src_id, cache_line_size);
        return get_segment_num(lhs.dst_id, cache_line_size) < get_segment_num(rhs.dst_id, cache_line_size);
    };
    std::sort(array_for_sort, array_for_sort + edges_count, comp);
    
    // create tmp graph
    Graph<_TVertexValue, _TEdgeWeight> tmp_graph(vertices_count, edges_count, EDGES_LIST, _input_graph.check_if_directed(), true);
    
    // add old vertices
    for(int i = 0; i < vertices_count; i++)
        tmp_graph.add_vertex(_input_graph.iterate_vertices(i));
    
    // add sorted edges in right order
    for(long long i = 0; i < edges_count; i++)
        tmp_graph.add_edge(array_for_sort[i]);
    
    // copy temporary graph to input
    _input_graph = tmp_graph;
    
    // delete memory for sort array
    delete[] array_for_sort;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// public
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphOptimizationAPI<_TVertexValue, _TEdgeWeight>::optimize_graph_for_GPU(Graph<_TVertexValue, _TEdgeWeight> &_input_graph,
                                                                                int elem_size, GPU_ARCHITECTURE _arch)
{
    cout << "here" << endl;
    // convert to edges list graph format
    _input_graph.convert_to_edges_list();
    
    cout << "starts" << endl;

    // remove loops and multiple arcs
    remove_loops_and_multiple_arcs(_input_graph);

    cout << "loops" << endl;    
    // optimize graph storage format for GPU Kepler architecture
    if(_arch == KEPLER)
        reorder_edges_for_cache(_input_graph, GPU_KEPLER_L2_CACHE_SIZE, elem_size);
    else if(_arch == PASCAL)
        reorder_edges_for_cache(_input_graph, GPU_PASCAL_L2_CACHE_SIZE, elem_size);
    cout << "edges" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphOptimizationAPI<_TVertexValue, _TEdgeWeight>::optimize_graph_for_CPU(Graph<_TVertexValue, _TEdgeWeight> &_input_graph,
                                                                               int elem_size)
{
    // convert to edges list graph format
    _input_graph.convert_to_edges_list();
    
    // remove loops and multiple arcs
    remove_loops_and_multiple_arcs(_input_graph);
    
    // optimize graph storage format for multicore CPU architecture
    reorder_edges_for_cache(_input_graph, CPU_L2_CACHE_SIZE, elem_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void GraphOptimizationAPI<_TVertexValue, _TEdgeWeight>::optimize_graph_for_KNL(Graph<_TVertexValue, _TEdgeWeight> &_input_graph,
                                                                               int elem_size)
{
    // convert to edges list graph format
    _input_graph.convert_to_edges_list();
    
    // remove loops and multiple arcs
    remove_loops_and_multiple_arcs(_input_graph);
    
    // optimize graph storage format for multicore CPU architecture
    reorder_edges_for_cache(_input_graph, KNL_L2_CACHE_SIZE, elem_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
long long GraphOptimizationAPI<_TVertexValue, _TEdgeWeight>::check_gpu_memory_transactions_count(Graph<_TVertexValue, _TEdgeWeight> &
                                                                                                 _input_graph, int elem_size)
{
    // check storage format?
    
    // get graph pointers
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _input_graph.get_graph_data();
    int vertices_count = _input_graph.get_vertices_count();
    long long edges_count = _input_graph.get_edges_count();
    int *src_ids = graph_data.edges_src_ids;
    int *dst_ids = graph_data.edges_dst_ids;
    
    long long transactions_count = 0;
    map<int, int> src_transactions;
    map<int, int> dst_transactions;
        
    //#pragma omp for schedule(guided)
    int prev_warp = 0;
    for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
    {
        if((cur_edge / 32) != prev_warp)
        {
            prev_warp = (cur_edge / 32);
            transactions_count += src_transactions.size();
            transactions_count += dst_transactions.size();
            src_transactions.clear();
            dst_transactions.clear();
        }
        
        int src = src_ids[cur_edge];
        int dst = dst_ids[cur_edge];
        
        src_transactions[(src * elem_size)/128]++;
        dst_transactions[(dst * elem_size)/128]++;
    }

    return transactions_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
