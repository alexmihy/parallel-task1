#pragma once
#include <iostream>
#include <sstream> 
#include <fstream>
#include <cstring>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void universal_malloc(_T **_ptr, long long _size)
{
    if (*_ptr == NULL && _size > 0)
    {
        *_ptr = new _T[_size];
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void universal_free(_T **_ptr)
{
    if (*_ptr != NULL)
    {
        delete[](*_ptr);
    }
    *_ptr = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
void universal_memcpy(_T *_dst, _T *_src, long long _size)
{
	memcpy((void*)_dst, (const void*)_src, (std::size_t)(_size * sizeof(_T)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T* resize_array(_T *_old_pointer, long long _old_size, long long _new_size)
{
	// compute new size

	_T *new_pointer = NULL;
	universal_malloc<_T>(&new_pointer, _new_size);
	universal_memcpy<_T>(new_pointer, _old_pointer, _old_size);
	universal_free<_T>(&_old_pointer);

	return new_pointer;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
