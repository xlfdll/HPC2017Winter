// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// Include.h - common and library include directives

#pragma once

#pragma comment(lib, "Shlwapi.lib") // Windows Shell API
#pragma comment(lib, "Gdiplus.lib") // Windows GDI+

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <mutex>
#include <cmath>

#include <Windows.h> // Windows API
#include <Shlwapi.h> // Windows Shell API
#include <gdiplus.h> // Windows GDI+
#include <strsafe.h> // Windows API Safe String Functions

#include <cuda_runtime.h>

using namespace std;
using namespace Gdiplus;

#define IMAGE_DIRECTORY_PATH ".\\images"
#define FEATURE_DIRECTORY_PATH ".\\features"

#define FEATURE_EXTENSION ".feature"

/**
 * Turn this on or off. If on, it enables the CUDA version of the
 * histogram calculation.
 */
#define CUDA_HISTOGRAM          1
/**
 * Turn this on or off. If on, it enables the CUDA version of the
 * search function.
 */
#define CUDA_SEARCH             1

#if CUDA_HISTOGRAM || CUDA_SEARCH
#define HANDLE_CUDA_ERROR(err) __handle_cuda_error((err), __LINE__, __FILE__)
#include "error.h"
#endif

typedef vector<wstring> StringVector;
typedef multimap<double, wstring> ResultMultiMap;
typedef pair<double, wstring> ResultPair;
