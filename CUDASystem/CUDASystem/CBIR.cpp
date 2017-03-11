// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// CBIR.cpp - content-based image retrieval algorithms functions (definition)

#include "CBIR.h"

// Intensity color histogram functions

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

