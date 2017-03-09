// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// CBIR.cpp - content-based image retrieval algorithms functions (definition)

#include "CBIR.h"
#include "kernels.h"

// Intensity color histogram functions

# if CUDA_HISTOGRAM
/*
 * This function computes both the color and intensity version of the histogram
 * for the given image.
 *
 * @param image:       The image
 *
 * @param histogramsI: The array of intensity histograms that will be filled by
 *                     CUDA kernel calls.
 *
 * @param histogramsC: The array of color histograms that will be filled by
 *                     CUDA kernel calls.
 *
 * @param histIndex:   The index into the array of histograms can be computed
 *                     by multiplying this value with the width of a color of
 *                     intensity histogram.
 */
UINT * GetIntensityBins(Bitmap *image, UINT *histogramsI, UINT *histogramsC, histIndex)
{
	/*
	 * Collect the image into a flat array of UINT32s. Each UINT32 has
	 * the R as the MSB, the G, then B. The LSB is 0.
 	 * On the kernel side, we will extract these three values back out
 	 * using bit operations to compute the histogram.
	 */

	UINT imageWidth = image->GetWidth();
	UINT imageHeight = image->GetHeight();

	UINT32 *pixels = new UINT32[imageWidth * imageHeight];
	ZeroMemory(bins, imageWidth * imageHeight * sizeof(UINT32));

	Color pixelColor;
	for (UINT i = 0; i < imageWidth; i++)
	{
		for (UINT j = 0; j < imageHeight; j++)
		{
			image->GetPixel(i, j, &pixelColor);
			BYTE red = pixelColor.GetR();
			BYTE green = pixelColor.GetG();
			BYTE blue = pixelColor.GetB();
			const UINT32 px = ((UINT32)red) << 24 |
					  ((UINT32)green) << 16 |
					  ((UINT32)blue) << 8;
			pixels[j * imageWidth + i] = px;
		}
	}

	/* Kernel call will compute the histogram and store the results in
		the right spot in histogramsI and histogramsC */

	// TODO: Dump the pixels array into the stream
	histogram<<<whatever, whatever>>>(histogramsI, histogramsC, pixels,
                                          imageWidth, imageHeight, histIndex);
}
#else
UINT * GetIntensityBins(Bitmap *image)
{
	Color pixelColor;
	UINT *bins = new UINT[INTENSITY_BIN_COUNT];

	ZeroMemory(bins, INTENSITY_BIN_COUNT * sizeof(unsigned int));

	UINT imageWidth = image->GetWidth();
	UINT imageHeight = image->GetHeight();

	for (UINT i = 0; i < imageWidth; i++)
	{
		for (UINT j = 0; j < imageHeight; j++)
		{
			image->GetPixel(i, j, &pixelColor);

			int k = GetIntensityBinIndex
			(
				pixelColor.GetR(),
				pixelColor.GetG(),
				pixelColor.GetB()
			);

			bins[k]++;
		}
	}

	return bins;
}

int GetIntensityBinIndex(BYTE r, BYTE g, BYTE b)
{
	int intensity = (int)(0.299 * r + 0.587 * g + 0.114 * b);
	int index = intensity / 10;

	if (index > (INTENSITY_BIN_COUNT - 1))
	{
		index = INTENSITY_BIN_COUNT - 1;
	}

	return index;
}

// Color-Code color histogram functions
UINT *GetColorCodeBins(Bitmap *image)
{
	Color pixelColor;
	UINT *bins = new UINT[COLORCODE_BIN_COUNT];

	ZeroMemory(bins, COLORCODE_BIN_COUNT * sizeof(unsigned int));

	UINT imageWidth = image->GetWidth();
	UINT imageHeight = image->GetHeight();

	for (UINT i = 0; i < imageWidth; i++)
	{
		for (UINT j = 0; j < imageHeight; j++)
		{
			image->GetPixel(i, j, &pixelColor);

			int k = GetColorCodeBinIndex
			(
				pixelColor.GetR(),
				pixelColor.GetG(),
				pixelColor.GetB()
			);

			bins[k]++;
		}
	}

	return bins;
}

int GetColorCodeBinIndex(BYTE r, BYTE g, BYTE b)
{
	return ((r & 0xC0) >> 2 | (g & 0xC0) >> 4 | (b & 0xC0) >> 6);
}

double GetManhattanDistance(const ImageFeatureData *featureA, const ImageFeatureData *featureB)
{
	double distance = 0.0;

	for (size_t i = 0; i < featureA->featureCount; i++)
	{
		distance += abs(
			(double)(featureA->features)[i] / (featureA->width * featureA->height)
			- (double)(featureB->features)[i] / (featureB->width * featureB->height));
	}

	return distance;
}
#endif //CUDA_HISTOGRAMS
