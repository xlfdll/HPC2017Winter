// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// CBIR.h - content-based image retrieval algorithms functions (declaration)

#pragma once

#include "Include.h"

#define INTENSITY_BIN_COUNT 25
#define COLORCODE_BIN_COUNT 64

typedef struct imageFeatureData
{
	UINT width;
	UINT height;

	UINT *features;
	size_t featureCount;
} ImageFeatureData;

// Intensity color histogram functions
UINT *GetIntensityBins(Bitmap *image);
int GetIntensityBinIndex(BYTE r, BYTE g, BYTE b);

// Color-Code color histogram functions
UINT *GetColorCodeBins(Bitmap *image);
int GetColorCodeBinIndex(BYTE r, BYTE g, BYTE b);

// Distance functions
double GetManhattanDistance(const ImageFeatureData *featureA, const ImageFeatureData *featureB);

enum CBIRMethod { Intensity, ColorCode };