#include <vector>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VectorWrapper
{
private:
    vector<int> data;
public:
    void push_back(int _val) { data.push_back(_val); };
    int get_size() { return data.size(); };
    int get_elem(int _pos) { return data[_pos]; };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
