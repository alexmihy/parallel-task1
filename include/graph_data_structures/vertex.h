#pragma once

#include <iostream>
#include <limits>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \class Vertex vertex.h "src/data_structures/vertex.h"
//!
//! \brief Stores information about single vertex
//!
//! \author Ilya Afanasyev
//!
//! \details Stores information about vertex ID and vertex value \n
template <typename _TVertexValue>
struct Vertex
{
	int id;
	_TVertexValue value;

	//! \brief Sets all vertex values equal to 0
	Vertex() { id = 0; value = 0; };

	//! \brief Sets all vertex values according to input parameters
	Vertex(int _id, _TVertexValue _value) { id = _id; value = _value; };

	//! \brief Compare current edge with another Vertex
	bool operator==(const Vertex<_TVertexValue> &_other) const;

	//! \brief Checks if current edges is not equal to another Vertex
	bool operator!=(const Vertex<_TVertexValue> &_other) const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue>
bool Vertex<_TVertexValue>::operator == (const Vertex<_TVertexValue> &_other) const
{
	if (this->id != _other.id)
		return false;
	if (abs(this->value - _other.value) > numeric_limits<_TVertexValue>::epsilon())
		return false;

	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue>
bool Vertex<_TVertexValue>::operator != (const Vertex<_TVertexValue> &_other) const
{
	return !(*this == _other);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
