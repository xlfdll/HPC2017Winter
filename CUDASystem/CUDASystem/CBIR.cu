// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// CBIR.cu - content-based image retrieval algorithms functions (definition)

#include "CBIR.h"

// Structures

// Use maximum size due to the static characteristics of constant memory
__constant__ UINT refHistogramBins[COLORCODE_BIN_COUNT];

// CUDA kernel launch functions

void LaunchUpdateKernel
(
	cudaStream_t stream, cudaDeviceProp *cudaDeviceInfo,
	SimpleColor *pixels, UINT imageWidth, UINT imageHeight,
	UINT *intensityBins, UINT *colorCodeBins
)
{
	int tpb_width = (int)sqrt(cudaDeviceInfo->maxThreadsPerBlock);

	dim3 threads(tpb_width, tpb_width, 1);
	dim3 blocks((imageWidth + tpb_width - 1) / tpb_width, (imageHeight + tpb_width - 1) / tpb_width, 1);

	UpdateHistogramBins << <blocks, threads >> > (pixels, imageWidth, imageHeight, intensityBins, colorCodeBins);
}



void LaunchSearchKernel
(
	cudaStream_t stream,
	UINT *dbImageHistogramBins, UINT *dbImagePixelCounts, UINT dbImageCount,
	UINT refImagePixelCount, UINT histogramBinCount, double *distanceResults
)
{
	GetHistogramDistance << <dbImageCount, histogramBinCount, histogramBinCount * sizeof(double), stream >> > (dbImageHistogramBins, dbImagePixelCounts, refImagePixelCount, distanceResults);
}

void PrepareSearchKernel(UINT *refImageHistogramBins, UINT histogramBinCount)
{
	// Initialize constant memory with reference image histogram bins (only once)
	HANDLE_CUDA_ERROR(cudaMemcpyToSymbol(refHistogramBins, refImageHistogramBins, histogramBinCount * sizeof(UINT), 0, cudaMemcpyDeviceToDevice));
}

// CUDA kernels

__global__ void UpdateHistogramBins
(
	SimpleColor *pixels, UINT imageWidth, UINT imageHeight,
	UINT *intensityBins, UINT *colorCodeBins
)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * imageWidth + col;
	int numPixels = imageWidth * imageHeight;

	if (index < numPixels)
	{
		SimpleColor pixel = pixels[index];

		// Perform atomic addition
		UINT *binAddress = &(intensityBins[GetIntensityBinIndex(pixel.r, pixel.g, pixel.b)]);
		atomicAdd(binAddress, 1);
		binAddress = &(colorCodeBins[GetColorCodeBinIndex(pixel.r, pixel.g, pixel.b)]);
		atomicAdd(binAddress, 1);
	}
}

__global__ void GetHistogramDistance
(
	UINT *dbImageHistogramBins, UINT *dbImagePixelCounts,
	double refImagePixelCount, double *distanceResults
)
{
	extern __shared__ double sharedHistogramBins[];

	// Get all variables
	UINT imageIndex = blockIdx.x; // Current image index
	UINT binIndex = threadIdx.x; // Current histogram bin index

	UINT imageCount = gridDim.x; // # of images
	UINT binCount = blockDim.x; // # of histogram bins per image

	UINT totalBinCount = imageCount * binCount; // # of histogram bins for all images
	UINT totalBinIndex = imageIndex * binCount + binIndex; // Current histogram bin index within all images

	double pixelCount = dbImagePixelCounts[imageIndex]; // Current image pixel count

	// Load database image histogram bins into shared memory for current block
	if (totalBinIndex < totalBinCount)
	{
		sharedHistogramBins[binIndex] = dbImageHistogramBins[totalBinIndex];
	}

	__syncthreads(); // Wait for complete

	// Replace each bin with intermediate result for distance calculation
	if (binIndex < binCount)
	{
		sharedHistogramBins[binIndex] =
			fabs(sharedHistogramBins[binIndex] / pixelCount - refHistogramBins[binIndex] / refImagePixelCount);
	}

	__syncthreads(); // Wait for complete

	// Perform parallel scan to sum up and get final result
	for (UINT stride = 1; stride < binCount; stride *= 2)
	{
		__syncthreads();

		if (binIndex >= stride)
		{
			sharedHistogramBins[binIndex] += sharedHistogramBins[binIndex - stride];
		}
	}

	distanceResults[imageIndex] = sharedHistogramBins[binCount - 1]; // The last value in shared memory array is the result
}

// Color histogram bin helper functions

__device__ int GetIntensityBinIndex(BYTE r, BYTE g, BYTE b)
{
	int intensity = (int)(0.299 * r + 0.587 * g + 0.114 * b);
	int index = intensity / 10;

	if (index > (INTENSITY_BIN_COUNT - 1))
	{
		index = INTENSITY_BIN_COUNT - 1;
	}

	return index;
}

__device__ int GetColorCodeBinIndex(BYTE r, BYTE g, BYTE b)
{
	return ((r & 0xC0) >> 2 | (g & 0xC0) >> 4 | (b & 0xC0) >> 6);
}