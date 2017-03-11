// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// kernels.cu - Kernel functions

#include "kernels.h"
#include "device_launch_parameters.h"
#include "CBIR.h"

__constant__ UINT refHist[MAX(INTENSITY_BIN_COUNT, COLORCODE_BIN_COUNT)];

/*
 * The algorithm for computing a parallel histogram is as follows:
 * We are given a 2D grid of 2D blocks. Each block is as large as can be
 * allowed on the GPU device. Each thread in a block is responsible for exactly
 * one pixel.
 */
__global__ void histogram(UINT *histogramI,
                          UINT *histogramC,
                          UINT32 *pixels,
                          UINT imageWidth,
                          UINT imageHeight)
{
	/* Get this thread's pixel */
	const UINT row = blockIdx.y * blockDim.y + threadIdx.y;
	const UINT col = blockIdx.x * blockDim.x + threadIdx.x;
	const UINT pxIndex = row * imageWidth + col;
	const UINT numPixels = imageWidth * imageHeight;

	/* Calculate the histogram values for this thread's pixel */
	if (pxIndex < numPixels)
	{
		const UINT32 px = pixels[pxIndex];

		/* Break out the R, G, and B values from the pixel */
		const BYTE red = (BYTE)(((UINT32)0xFF000000 & px) >> 24);
		const BYTE green = (BYTE)(((UINT32)0x00FF0000 & px) >> 16);
		const BYTE blue = (BYTE)(((UINT32)0x0000FF00 & px) >> 8);

		/* Get color coded bin index */
		const UINT binIndexC = ((red & 0xC0) >> 2 |
                                        (green & 0xC0) >> 4 |
                                        (blue & 0xC0) >> 6);
		/* Get intensity bin index */
		const UINT intensity = (UINT)(0.299 * red +
                                              0.587 * green +
                                              0.114 * blue);
		const UINT binIndexI = ((intensity / 10) >
                                        (INTENSITY_BIN_COUNT - 1) ?
                                        (INTENSITY_BIN_COUNT - 1) :
                                        (intensity / 10));
		histogramI[binIndexI]++;
		histogramC[binIndexC]++;
	}


}

/*
 * Each block of threads only processes one histogram out of the list of
 * histograms. This won't be great for our GPU occupancy, but it will have to
 * do for now.
 */
__global__ void search_kernel(UINT *histograms,
                              UINT *widths,
                              UINT *heights,
                              double *results,
                              UINT refWidth,
                              UINT refHeight,
                              UINT refHistLength)
{
}

