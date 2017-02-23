// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// Include.h - common and library include directives

#pragma once

#pragma comment(lib, "Gdiplus.lib") // Add Windows GDI+ library

#include <iostream>
#include <vector>

#include <Windows.h>
#include <gdiplus.h>
#include <strsafe.h>

using namespace std;
using namespace Gdiplus;

#define IMAGE_DIRECTORY_NAME ".\\images"
#define FEATURE_DIRECTORY_NAME ".\\features"

typedef vector<wstring> StringVector;