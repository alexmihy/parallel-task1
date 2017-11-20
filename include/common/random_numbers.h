#pragma once
#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T rand_uniform_val(int _upper_border)
{
	return (_T)(rand() % _upper_border);
}

template <>
int rand_uniform_val(int _upper_border)
{
	return (int)(rand() % _upper_border);
}

template <>
float rand_uniform_val(int _upper_border)
{
	return (float)(rand() % _upper_border) / _upper_border;
}

template <>
double rand_uniform_val(int _upper_border)
{
	return (double)(rand() % _upper_border) / _upper_border;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float frand()
{
	return (float)rand() / RAND_MAX;
}

double drand()
{
	return (double)rand() / RAND_MAX;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
