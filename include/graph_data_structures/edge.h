#pragma once

#include <iostream>
#include <cstdlib>
#include <limits>
#include <cmath>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \class Edge edge.h "src/data_structures/edge.h"
//!
//! \brief Stores information about single edge
//!
//! \author Ilya Afanasyev
//!
//! \details Stores information about source vertex ID, destination vertex ID and edge weight. \n
template <typename _TEdgeWeight>
struct Edge
{
	int src_id;          //!< source vertex ID
	int dst_id;          //!< destination vertex ID
	_TEdgeWeight weight; //!< weight of the edge

	//! \brief Sets all edge values equal to 0
	Edge() { src_id = 0; dst_id = 0; weight = 0; };

	//! \brief Sets all edge values according to input parameters
	Edge(int _src_id, int _dst_id, _TEdgeWeight _weight) { src_id = _src_id; dst_id = _dst_id, weight = _weight; };

	//! \brief Compare current edge with another Edge
	bool operator == (const Edge<_TEdgeWeight> &_other) const;

	//! \brief Compare current edge with int value (can be used like: edge == -1)
	bool operator == (const int _value) const;

	//! \brief Checks if current edges is not equal to another Edge
	bool operator != (const Edge<_TEdgeWeight> &_other) const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
bool Edge<_TEdgeWeight>::operator == (const Edge<_TEdgeWeight> &_other) const
{
	if (this->src_id != _other.src_id)
		return false;
	if (this->dst_id != _other.dst_id)
		return false;
	if (abs(this->weight - _other.weight) > numeric_limits<_TEdgeWeight>::epsilon())
		return false;

	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
bool Edge<_TEdgeWeight>::operator == (const int _value) const
{
	if (this->src_id != _value)
		return false;
	if (this->dst_id != _value)
		return false;
	if (abs(this->weight - _value) > numeric_limits<_TEdgeWeight>::epsilon())
		return false;

	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
bool Edge<_TEdgeWeight>::operator != (const Edge<_TEdgeWeight> &_other) const
{
	return !(*this == _other);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
bool edge_dst_comparator(Edge<_TEdgeWeight> _first, Edge<_TEdgeWeight> _second)
{
	if (_first.dst_id < _second.dst_id)
		return true;
	else
		return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
