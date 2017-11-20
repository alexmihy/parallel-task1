#pragma once

#include <vector>

#ifndef __USE_NEC_SX__
#ifndef __NO_CPP_11_SUPPORT__
#include <atomic>
#endif
#endif

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \cond DOXYGEN_SHOULD_SKIP_THIS

template<class T>
struct UnionFindRecord
{
	T parent;
	T rank;
};

//! \endcond
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \cond DOXYGEN_SHOULD_SKIP_THIS

template<class _T>
class UnionFind
{
private:
	vector<UnionFindRecord<_T>> a;

	// See cited paper for explanation of update_root and merge.
	inline bool update_root(_T &_x, _T _old_rank, _T _y, _T _new_rank);

	inline void compress(_T _x);
public:
	UnionFind(_T _size);

	// initialize data set
	inline void clear(_T _p);

	// return parent of selected element
	inline _T get_parent(_T _x);

	// find operation without path compression.
	inline _T find_fast(_T _x);

	// flatten the union-find structure, such that each node's parent points to the root of the component.
	inline void flatten(_T _p);

	// retruns true if merge was successful (i.e. two components were actually merged)
	inline bool merge(_T _x, _T _y);
};

//! \endcond
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "union_find.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
