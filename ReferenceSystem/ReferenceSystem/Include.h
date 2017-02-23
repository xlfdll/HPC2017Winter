// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// Include.h - common and library include directives

#pragma once

#define cimg_display 0

#include <iostream>
#include <vector>

#include <Windows.h>
#include <strsafe.h>

#include "CImg.h"

using namespace std;
using namespace cimg_library;

#define IMAGE_DIRECTORY_NAME ".\\images"
#define FEATURE_DIRECTORY_NAME ".\\features"

typedef vector<wstring> StringVector;