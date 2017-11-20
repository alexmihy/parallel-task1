/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// public interface
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
UnionFind<_T>::UnionFind(_T _size):
a(_size)
{

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void UnionFind<_T>::clear(_T _p)
{
	a[_p] = { _p, 0 };
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T UnionFind<_T>::get_parent(_T _x)
{
	return a[_x].parent;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T UnionFind<_T>::find_fast(_T _x)
{
	_T root = _x;
	while (root != a[root].parent)
		root = a[root].parent;
	while (_x != root)
	{
		int newp = a[_x].parent;
		a[_x].parent = root;
		_x = newp;
	}
	return root;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void UnionFind<_T>::flatten(_T p)
{
	if (a[p].parent != p)
	{
		compress(p);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
bool UnionFind<_T>::merge(_T _x, _T _y)
{
	_x = find_fast(_x);
	_y = find_fast(_y);
	if (_x != _y)
	{
		do
		{
			_T xRank = a[_x].rank, yRank = a[_y].rank;
			if (xRank > yRank)
			{
				if (update_root(_y, yRank, _x, yRank))
				{
					return true;
				}
			}
			else if (xRank == yRank)
			{
				if (_x < _y)
				{
					if (update_root(_y, yRank, _x, yRank))
					{
						update_root(_x, xRank, _x, xRank + 1);
						return true;
					}
				}
				else
				{
					if (update_root(_x, xRank, _y, xRank))
					{
						update_root(_y, yRank, _y, yRank + 1);
						return true;
					}
				}
			}
			else
			{
				if (update_root(_x, xRank, _y, xRank))
				{
					return true;
				}
			}
		} while (_x != _y);
	}
	return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// public interface
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void UnionFind<_T>::compress(_T x)
{
	_T y = a[x].parent;
	if (y != x)
	{
		_T z = a[y].parent;
		if (a[y].parent != y)
		{
			do
			{
				y = a[y].parent;
			} while (y != a[y].parent);
			a[x].parent = y;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
bool UnionFind<_T>::update_root(_T &x, _T oldRank, _T y, _T newRank)
{
	UnionFindRecord<_T> old_record = { x, oldRank };
	UnionFindRecord<_T> res_record = { y, newRank };
    
    #ifndef __NO_CPP_11_SUPPORT__
	if (atomic_compare_exchange_weak((atomic<UnionFindRecord<_T>> *)&a[x], &old_record, res_record))
	{
		return true;
	}
	else
	{
		// here we re-use actual parent value returned by the atomic operation
		x = old_record.parent;
		return false;
	}
    #else
    #pragma omp critical
    {
        a[x] = res_record;
    }
    return true;
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
