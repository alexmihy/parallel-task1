#pragma once

#include <fstream>

#if defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <libgen.h>
#endif

#if defined(__unix__) || defined(__unix) || defined(unix)
#include <unistd.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <mach-o/dyld.h>
#endif

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

long long get_filesize(string _filename)
{
    ifstream in(_filename, ios::binary | ios::ate);
    return (long long)(in.tellg());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
