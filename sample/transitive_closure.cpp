#include "include/graph_library.h"

//#include "boost_API.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void parse_cmd_params(int _argc, char **_argv, string &_input_file_name, string &_mode,
                      int &_pairs_count, bool &_check, int &_omp_threads, bool &_use_edges_reordering)
{
    _input_file_name  = "input.graph";
    _mode = "cpu";
    _pairs_count = 1;
    _check = false;
    _omp_threads = 1;
    _use_edges_reordering = false;
    
    for (int i = 1; i < _argc; i++)
    {
        string option(_argv[i]);
        
        if (option.compare("-infile") == 0)
        {
            _input_file_name = _argv[++i];
        }
        
        if (option.compare("-pairs_count") == 0)
        {
            _pairs_count = atoi(_argv[++i]);
        }
        
        if (option.compare("-mode") == 0)
        {
            _mode = _argv[++i];
        }
        
        if (option.compare("-check") == 0)
        {
            _check = true;
        }
        
        if (option.compare("-omp_threads") == 0)
        {
            _omp_threads = atoi(_argv[++i]);
        }
        
        if (option.compare("-use_edges_reordering") == 0)
        {
            _use_edges_reordering = true;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void bfs_check(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, vector<pair<int, int>> pairs_to_check,
               vector<bool> &_answer, int _omp_threads)
{
    cout << endl << "---------------- BFS check ----------------- " << endl;
    cout << "Checking results with BFS..." << endl;
    
    bool check_result = true;
    int pair_pos = 0;
    
    BFS<_TVertexValue, _TEdgeWeight> operation;
    operation.set_omp_threads(_omp_threads);
    bool *bfs_result = new bool[_input_graph.get_vertices_count()];
    for(auto cur_pair : pairs_to_check)
    {
        memset(bfs_result, 0, _input_graph.get_vertices_count() * sizeof(bool));
        
        int source_vertex = cur_pair.first;
        operation.cpu_sequential_bfs(_input_graph, bfs_result, source_vertex);
        
        if(_answer[pair_pos] != bfs_result[cur_pair.second])
        {
            cout << "Error in pair: " << cur_pair.first << " - " << cur_pair.second << endl;
            cout << _answer[pair_pos] << " " << bfs_result[cur_pair.second] << endl << endl;;
            check_result = false;
        }
        pair_pos++;
    }
    
    delete []bfs_result;
    
    if (check_result)
        cout << "Transitive closure is correct compared to APSP result." << endl;
    else
        cout << endl << "Error in transitive closure!" << endl;
    cout << "---------------- check complete ------------------- " << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void generate_test_input(vector<pair<int, int>> &_pairs_to_check, int _pairs_count, int _vertices_count)
{
    srand(time(NULL));
    
    _pairs_to_check.clear();
    
    for(int i = 0; i < _pairs_count; i++)
    {
        int left = rand() % _vertices_count;
        int right = rand() % _vertices_count;
        _pairs_to_check.push_back(pair<int, int>(left, right));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        // init library
        NodeData node_data;
        init_library(node_data);
        node_data.print_node_properties();
        double t1, t2;
        
        int scale = atoi(argv[1]);
        int pairs_count = atoi(argv[2]);
        string type = argv[3];
        
        int threads = omp_get_max_threads();
        
        // load graph
        Graph<int, float> graph(0, 0, EDGES_LIST, true);
        
        // generate graph
        cout << "Generating graph..." << endl;
        if(type == "RMAT")
            GraphGenerationAPI<int, float>::R_MAT_parallel(graph, pow(2.0, scale), 32, 45, 20, 20, 15, threads, true);
        else
            GraphGenerationAPI<int, float>::SSCA2(graph, pow(2.0, scale), 32, true);
        
        GraphOptimizationAPI<int, float>::optimize_graph_for_KNL(graph, sizeof(int));
        cout << "Graph Optimization complete!" << endl << endl;
        
        // generate test input vertices pairs
        vector<pair<int, int>> pairs_to_check;
        generate_test_input(pairs_to_check, pairs_count, graph.get_vertices_count());
        
        // perform computations
        TransitiveClosure<int, float> operation(threads);
        vector<bool> answer;
        double complexity = operation.cpu_purdom(graph, pairs_to_check, answer);
        cout << "Edges list purdom is done!" << endl << endl;
        
        vector<bool> answer2;
        t1 = omp_get_wtime();
        operation.cpu_purdom2(graph, pairs_to_check, answer2);
        t2 = omp_get_wtime();
        cout << "Adj list purdom is done!" << endl << endl;
        
        if(false)
        {
            for(int i = 0; i < answer.size(); i++)
            {
                if (answer[i] != answer2[i])
                    cout << "error in 1-2 check!" << endl;
                else
                    cout << "correct!" << endl;
            }
        }
        
        //cout << "new perfomance: " << pairs_count * graph.get_edges_count() / ((t2 - t1) * 1e6) << " MTEPS" << endl << endl;
        
        // perform checks if neccesary
        //if(check)
        //    bfs_check(graph, pairs_to_check, answer, omp_threads);
        
        cout << "Log file has been saved to perf.log" << endl << endl;
    }
    catch (const char *error)
    {
        cout << error << endl;
    }
    catch (...)
    {
        cout << "unknown error" << endl;
    }
    
    cout << "press any key to exit..." << endl;
    
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
