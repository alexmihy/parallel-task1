/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// tries to find isomorpic subraphs in graph g, from vertex v.
template <typename _TVertexValue, typename _TEdgeWeight>
bool SubgraphIsomorphism<_TVertexValue, _TEdgeWeight>::DFS(Graph<_TVertexValue, _TEdgeWeight> &_g, // big graph
                                                           int _v, // root vertex in big graph
                                                           int _w, // root vertex in small graph
                                                           Graph<_TVertexValue, _TEdgeWeight> &_h, // small graph
                                                           bool **_T, std::set<int> &visited_g, // visited array
                                                           std::map<int, int> &phi) // isomorphic data
{
    if (_T[_v][_w] == false)
        return false;
    
    visited_g.insert(_v);
    phi[_w] = _v;
    
    for(int wp_i = 0; wp_i < _h.get_vertex_connections_count(_w); wp_i++)
    {
        Edge<_TEdgeWeight> h_edge = _h.iterate_adjacent_edges(_w, wp_i);
        int wp = h_edge.dst_id;
        
        bool wp_in_phi = false;
        if (phi.find(wp) == phi.end())
            wp_in_phi = false;
        else
            wp_in_phi = true;
        
        bool phi_wp_found = false;
        if(phi.find(wp) != phi.end())
        {
            for(int i = 0; i < _g.get_vertex_connections_count(_v); i++)
            {
                Edge<_TEdgeWeight> g_edge = _g.iterate_adjacent_edges(_v, i);
                int dst_id = g_edge.dst_id;
                
                if(phi[wp] == dst_id)
                    phi_wp_found = true;
            }
        }
        
        if(wp_in_phi && !phi_wp_found)
        {
            visited_g.erase(_v);
            phi.erase(_w);
            return false;
        }
    }
    
    if(phi.size() == _h.get_vertices_count())
    {
        #pragma omp critical
        {
            cout << "Found isomorphism" << endl;
            for(auto i: phi)
                cout << "(" << i.first << ", " << i.second << ") ";
            cout << endl;
        }
        visited_g.erase(_v);
        phi.erase(_w);
        return true;
    }
    
    for(int vp_i = 0; vp_i < _g.get_vertex_connections_count(_v); vp_i++)
    {
        Edge<_TEdgeWeight> g_edge = _g.iterate_adjacent_edges(_v, vp_i);
        int vp = g_edge.dst_id;
        
        if(visited_g.find(vp) == visited_g.end())
        {
            for(int wp_i = 0; wp_i < _h.get_vertex_connections_count(_w); wp_i++)
            {
                Edge<_TEdgeWeight> h_edge = _h.iterate_adjacent_edges(_w, wp_i);
                int wp = h_edge.dst_id;
                
                if(phi.find(wp) == phi.end())
                {
                    if(DFS(_g, vp, wp, _h, _T, visited_g, phi))
                    {
                        visited_g.erase(_v);
                        phi.erase(_w);
                        return true;
                    }
                }
            }
        }
    }
    
    visited_g.erase(_v);
    phi.erase(_w);
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// main computational function, uses ulman algorithm to check subgraph isomorphism
template <typename _TVertexValue, typename _TEdgeWeight>
void SubgraphIsomorphism<_TVertexValue, _TEdgeWeight>::ullman_algorithm(Graph<_TVertexValue, _TEdgeWeight> &_big_graph, // big graph
                                                                        Graph <_TVertexValue, _TEdgeWeight> &_small_graph, // subgraph
                                                                        ComputationalMode _computational_mode) // computatuonal mode
{
    // get num threads
    int threads_count = 1;
    if(_computational_mode == USE_CPU_MODE)
        threads_count = omp_get_max_threads();
    
    // get vertices count
    int v1_vertices_count = _big_graph.get_vertices_count(); // v1 = vertices count in big graph
    int v2_vertices_count = _small_graph.get_vertices_count(); // v2 = vertices count in subgraph
    
    // create and init table ( v1 * v2) of isomorphism
    bool **T = new bool*[v1_vertices_count];
    for(int i = 0; i < v1_vertices_count; i++)
    {
        T[i] = new bool[v2_vertices_count];
        for(int j = 0; j < v2_vertices_count; j++)
            T[i][j] = 1; //set all table to 1 in the beginning
    }
    
    // get data for both graphs (big and subgraph)
    GraphContainerData<_TVertexValue, _TEdgeWeight> graph_data = _big_graph.get_graph_data();
    long long *v1_vertices_to_edges_ptrs = graph_data.vertices_to_edges_ptrs;
    int *v1_dst_ids = graph_data.edges_dst_ids;
    int v1_edges_count = _big_graph.get_edges_count();
    graph_data = _small_graph.get_graph_data();
    long long *v2_vertices_to_edges_ptrs = graph_data.vertices_to_edges_ptrs;
    int *v2_dst_ids = graph_data.edges_dst_ids;
    int v2_edges_count = _small_graph.get_edges_count();
    
    // compute vertex degrees (of each vertex) for big graph
    vector<int> v1_connections_count(v1_vertices_count, 0);
    #pragma omp parallel for num_threads(threads_count)
    for(int i = 0; i < v1_vertices_count; i++)
    {
        int first = v1_vertices_to_edges_ptrs[i];
        int last = v1_vertices_to_edges_ptrs[i + 1];
        v1_connections_count[i] = (last - first);
    }
    
    // compute vertex degrees (of each vertex) for small graph
    vector<int> v2_connections_count(v2_vertices_count, 0);
    #pragma omp parallel for num_threads(threads_count)
    for(int i = 0; i < v2_vertices_count; i++)
    {
        int first = v2_vertices_to_edges_ptrs[i];
        int last = v2_vertices_to_edges_ptrs[i + 1];
        v2_connections_count[i] = (last - first);
    }
    
    // fill table data using degrees
    #pragma omp parallel for num_threads(threads_count)
    for(int v1_vert = 0; v1_vert < v1_vertices_count; v1_vert++)
    {
        for(int v2_vert = 0; v2_vert < v2_vertices_count; v2_vert++)
        {
            if(v1_connections_count[v1_vert] != v2_connections_count[v2_vert])
                T[v1_vert][v2_vert] = 0; // if vertices has different connections, they can't be merged later
        }
    }
    
    bool graph_found = false;
    // traverse and check izomorhism
    #pragma omp parallel for num_threads(threads_count) schedule(guided)
    for(int v1_vert = 0; v1_vert < v1_vertices_count; v1_vert++)
    {
        for(int v2_vert = 0; v2_vert < v2_vertices_count; v2_vert++)
        {
            if(T[v1_vert][v2_vert] == true)
            {
                std::set<int> visited;
                std::map<int, int> phy;
                // try to merge v1 and v2 vertices using DFS
                if(!DFS(_big_graph, v1_vert, v2_vert, _small_graph, T, visited, phy))
                {
                    atomic_store((atomic<bool>*)&T[v1_vert][v2_vert], false);
                    graph_found = true;
                }
            }
        }
    }
    // print if found any isomorphic subgraphs
    if(graph_found)
        cout << "Isomorphic graph found!" << endl;
    else
        cout << "Isomorphic graph NOT found!" << endl;
    
    // delete table
    for(int i = 0; i < v1_vertices_count; i++)
    {
        delete []T[i];
    }
    delete T;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
