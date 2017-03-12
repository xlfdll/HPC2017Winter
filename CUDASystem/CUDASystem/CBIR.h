// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// CBIR.h - content-based image retrieval algorithms functions (declaration)

#pragma once

#include "Include.h"
#include "Helper.h"

#define INTENSITY_BIN_COUNT 25
#define COLORCODE_BIN_COUNT 64

// Structures

typedef struct simpleColor
{
	BYTE r;
	BYTE g;
	BYTE b;
	BYTE a; // Alpha channel, for padding only
} SimpleColor;

// CUDA kernel launch functions
void LaunchUpdateKernel
(
	cudaStream_t stream, cudaDeviceProp *cudaDeviceInfo,
	SimpleColor *pixels, UINT imageWidth, UINT imageHeight,
	UINT *intensityBins, UINT *colorCodeBins
);
void LaunchSearchKernel
(
	cudaStream_t stream,
	UINT *dbImageHistogramBins, UINT *dbImagePixelCounts, UINT dbImageCount,
	UINT refImagePixelCount, UINT histogramBinCount, double *distanceResults
);
void PrepareSearchKernel(UINT *refImageHistogramBins, UINT histogramBinCount);

// CUDA kernels
__global__ void UpdateHistogramBins
(
	SimpleColor *pixels, UINT imageWidth, UINT imageHeight,
	UINT *intensityBins, UINT *colorCodeBins
);
__global__ void GetHistogramDistance
(
	UINT *dbImageHistogramBins, UINT *dbImagePixelCounts,
	double refImagePixelCount, double *distanceResults
);

// Color histogram bin helper functions
__device__ int GetIntensityBinIndex(BYTE r, BYTE g, BYTE b);
__device__ int GetColorCodeBinIndex(BYTE r, BYTE g, BYTE b);

enum CBIRMethod { Intensity, ColorCode };