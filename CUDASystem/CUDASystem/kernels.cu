// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// kernels.cu - Kernel functions

#include "kernels.cuh"
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
	/* Get this block's histogram index */
	const UINT histIndex = blockIdx.x;

	/* Get this thread's index into the big list of histograms */
	const UINT elIndex = histIndex * refHistLength + threadIdx.x;

	/* Get this block's histogram's width and height */
	const UINT width = widths[blockIdx.x];
	const UINT height = heights[blockIdx.x];

	/* Load this block's histogram into shared memory */
	__shared__ double blockHistogram[MAX(INTENSITY_BIN_COUNT, COLORCODE_BIN_COUNT)];
	if (elIndex < refHistLength)
		blockHistogram[threadIdx.x] = histograms[elIndex];
	__syncthreads();

	/* Do the manhatten distance in parallel */
	if (threadIdx.x < refHistLength)
	{
		const double elDist = abs(
		      (double)blockHistogram[threadIdx.x] / (width * height) -
		      (double)refHist[threadIdx.x] / (refWidth * refHeight));
		blockHistogram[threadIdx.x] = elDist;
	}

	/* Do the parallel scan over the histograms */
	for (UINT stride = 1; stride < refHistLength; stride *= 2)
	{
		__syncthreads();
		if (threadIdx.x >= stride)
			blockHistogram[threadIdx.x] += blockHistogram[threadIdx.x - stride];
	}

	/* Put the result into the correct place in the results array */
	results[blockIdx.x] = blockHistogram[refHistLength - 1];
}

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
 *
 * @param stream:      This thread's stream.
 */
void GetBins(Bitmap *image,
             UINT *histogramsI,
             UINT *histogramsC,
             UINT histIndex,
             cudaStream_t *stream)
{
	/*
	 * Collect the image into a flat array of UINT32s. Each UINT32 has
	 * the R as the MSB, the G, then B. The LSB is 0.
 	 * On the kernel side, we will extract these three values back out
 	 * using bit operations and then compute the histograms.
	 */

	UINT imageWidth = image->GetWidth();
	UINT imageHeight = image->GetHeight();

	UINT32 *pixels = new UINT32[imageWidth * imageHeight];
	//This should be pixels correct?
	ZeroMemory(pixels, imageWidth * imageHeight * sizeof(UINT32));

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

	cudaError_t err;

	/* Move the three buffers we need into pinned memory */

	UINT *pinned_histI, *pinned_histC, *pinned_pixels;

	err = cudaMallocHost((void **)&pinned_histI, INTENSITY_BIN_COUNT * sizeof(UINT));
	HANDLE_CUDA_ERROR(err);

	err = cudaMallocHost((void **)&pinned_histC, COLORCODE_BIN_COUNT * sizeof(UINT));
	HANDLE_CUDA_ERROR(err);

	err = cudaMallocHost((void **)&pinned_pixels, imageWidth * imageHeight * sizeof(UINT32));
	HANDLE_CUDA_ERROR(err);

	memset(pinned_histI, 0, INTENSITY_BIN_COUNT * sizeof(UINT));
	memset(pinned_histC, 0, COLORCODE_BIN_COUNT * sizeof(UINT));
	memcpy(pinned_pixels, pixels, imageWidth * imageHeight * sizeof(UINT32));

	UINT *dev_histogramI, *dev_histogramC, *dev_pixels;

	/* Allocate device memory for the three things */

	err = cudaMalloc((void **)&dev_histogramI, INTENSITY_BIN_COUNT * sizeof(UINT));
	HANDLE_CUDA_ERROR(err);

	err = cudaMalloc((void **)&dev_histogramC, COLORCODE_BIN_COUNT * sizeof(UINT));
	HANDLE_CUDA_ERROR(err);

	err = cudaMalloc((void **)&dev_pixels, imageWidth * imageHeight * sizeof(UINT32));
	HANDLE_CUDA_ERROR(err);

	/* Move the stuff to the device */

	err = cudaMemcpyAsync(pinned_histI,
                              dev_histogramI,
                              INTENSITY_BIN_COUNT * sizeof(UINT),
                              cudaMemcpyHostToDevice,
                              *stream);
	HANDLE_CUDA_ERROR(err);

	err = cudaMemcpyAsync(pinned_histC,
                              dev_histogramC,
                              COLORCODE_BIN_COUNT * sizeof(UINT),
                              cudaMemcpyHostToDevice,
                              *stream);
	HANDLE_CUDA_ERROR(err);

	err = cudaMemcpyAsync(pinned_pixels,
                              dev_pixels,
                              imageWidth * imageHeight * sizeof(UINT32),
                              cudaMemcpyHostToDevice,
                              *stream);
	HANDLE_CUDA_ERROR(err);

	/* Execute the kernel */

	dim3 grid, threads;
	threads.x = 64;
	threads.y = 64;
	grid.x = (int)ceil(imageWidth / (float)threads.x);
	grid.y = (int)ceil(imageHeight / (float)threads.y);
	histogram<<< grid, threads>>>(dev_histogramI,
                                     dev_histogramC,
                                     dev_pixels,
                                     imageWidth,
                                     imageHeight);

	/* Transfer the histograms back to the host */

	err = cudaMemcpyAsync(dev_histogramI,
                              pinned_histI,
                              INTENSITY_BIN_COUNT * sizeof(UINT),
                              cudaMemcpyDeviceToHost,
                              *stream);
	HANDLE_CUDA_ERROR(err);

	err = cudaMemcpyAsync(dev_histogramC,
                              pinned_histC,
                              COLORCODE_BIN_COUNT * sizeof(UINT),
                              cudaMemcpyDeviceToHost,
                              *stream);
	HANDLE_CUDA_ERROR(err);

	memcpy(&histogramsI[histIndex], pinned_histI, INTENSITY_BIN_COUNT * sizeof(UINT));
	memcpy(&histogramsC[histIndex], pinned_histC, COLORCODE_BIN_COUNT * sizeof(UINT));

	/* Wait until all the stuff we put into the stream has completed */

	cudaStreamSynchronize(*stream);

	/* Then free up the memory */
	cudaFree(dev_histogramI);
	cudaFree(dev_histogramC);
	cudaFree(dev_pixels);

	free(pinned_histI);
	free(pinned_histC);
	free(pixels);
}


DWORD WINAPI SearchThreadFunction(PVOID lpParam)
{
	// Read image feature data from feature files, and calculate distances with reference image, then output results

	SearchThreadData *data = (SearchThreadData *)lpParam;

	// Read reference image feature data
	Bitmap *image = new Bitmap(data->refPath);
	ImageFeatureData *refImageFeatureData = new ImageFeatureData;

	refImageFeatureData->width = image->GetWidth();
	refImageFeatureData->height = image->GetHeight();

	switch (data->method)
	{
	case Intensity:
		refImageFeatureData->features = GetIntensityBins(image);
		refImageFeatureData->featureCount = INTENSITY_BIN_COUNT;
		break;
	case ColorCode:
		refImageFeatureData->features = GetColorCodeBins(image);
		refImageFeatureData->featureCount = COLORCODE_BIN_COUNT;
		break;
	default:
		break;
	}

	delete image;

	// Read feature files
	StringVector &filelist = *(data->filelist);
	ResultMultiMap &result = *(data->result);
	ImageFeatureData *dbImageFeatureData;

	/*
         * Go through each of the files in this thread's section of files and
         * collect a histogram for each one. Put each histogram into a giant
         * list of histograms that we will process with a kernel call.
         * We will also need to know the width and height of the image that
         * created each histogram, so put that information into arrays as well.
         * Also, each thread will need to know the reference image's histogram,
         * so put that into constant memory.
         */

	const size_t numFiles = data->end - data->start;

	/* Declare the arrays (including a results array) */
	UINT *histograms, *imageWidths, *imageHeights, *pinned_features;
	double *results;
	cudaError_t err;

	/* Allocate space for each array in pinned memory */
	err = cudaMallocHost((void **)&histograms, refImageFeatureData->featureCount * numFiles * sizeof(UINT));
	HANDLE_CUDA_ERROR(err);

	err = cudaMallocHost((void **)&imageWidths, numFiles * sizeof(UINT));
	HANDLE_CUDA_ERROR(err);

	err = cudaMallocHost((void **)&imageHeights, numFiles * sizeof(UINT));
	HANDLE_CUDA_ERROR(err);

	err = cudaMallocHost((void **)&results, numFiles * sizeof(double));
	HANDLE_CUDA_ERROR(err);

	err = cudaMallocHost((void **)&pinned_features, refImageFeatureData->featureCount * numFiles * sizeof(UINT));

	/* Zero all the arrays */
	ZeroMemory(histograms, refImageFeatureData->featureCount * numFiles * sizeof(UINT)];
	ZeroMemory(imageWidths, numFiles * sizeof(UINT));
	ZeroMemory(imageHeights, numfiles * sizeof(UINT));
	ZeroMemory(results, numFiles * sizeof(double));

	/* Memcpy the reference histogram into the pinned memory we declared */
	memcpy(pinned_features, refImageFeatureData->features, refImageFeatureData->featureCount * numFiles * sizeof(UINT));

	/* Create a CUDA stream for this thread */
	cudaStream_t stream;
	err = cudaStreamCreate(&stream);

	/* Start an asynchronous memcpy to the GPU of the reference hist */
	err = cudaMemcpyToSymbolAsync((const void *)refHist,
                                      MAX(INTENSITY_BIN_COUNT, COLORCODE_BIN_COUNT) * sizeof(UINT),
                                      0,
                                      cudaMemcpyHostToDevice,
                                      stream);
	HANDLE_CUDA_ERROR(err);

	wstring *imageFileNames = new wstring[numFiles];

	/* Loop over our image files and collect the data we need */
	for (size_t i = (data->start); i < (data->end); i++)
	{
		dbImageFeatureData = new ImageFeatureData;

		switch (data->method)
		{
		case Intensity:
			dbImageFeatureData->features = new UINT[INTENSITY_BIN_COUNT];
			dbImageFeatureData->featureCount = INTENSITY_BIN_COUNT;
			break;
		case ColorCode:
			dbImageFeatureData->features = new UINT[COLORCODE_BIN_COUNT];
			dbImageFeatureData->featureCount = COLORCODE_BIN_COUNT;
			break;
		default:
			break;
		}

		wstring imageFileName, featureLine;
		wifstream featureStream;

		featureStream.open(filelist[i]);

		// Read image file name
		getline(featureStream, imageFileName);

		// Read image size information
		featureStream >> dbImageFeatureData->width;
		featureStream.ignore();
		featureStream >> dbImageFeatureData->height;

		// Read image size information into corresponding arrays
		imageWidths[i - data->start] = dbImageFeatureData->width;
		imageHeights[i - data->start] = dbImageFeatureData->height;

		getline(featureStream, featureLine); // Skip endline

		// Read image feature data
		switch (data->method)
		{
		case Intensity:
			getline(featureStream, featureLine); // Read intensity histogram feature data
			break;
		case ColorCode:
			getline(featureStream, featureLine); // Skip intensity histogram feature data
			getline(featureStream, featureLine); // Read color-code histogram feature data
			break;
		default:
			break;
		}

		featureStream.close();

		wistringstream wiss(featureLine);

		for (size_t j = 0; j < dbImageFeatureData->featureCount; j++)
		{
			wiss >> dbImageFeatureData->features[i];
			/* Put this number into the histogram we are currently
                         * working on */
			const size_t index = (i - data->start) * refImageFeatureData->featureCount + j;
			histograms[index] = dbImageFeatureData->features[i];

			if (j < dbImageFeatureData->featureCount - 1)
			{
				wiss.ignore();
			}
		}

		delete[] dbImageFeatureData->features;
		delete dbImageFeatureData;
	}

	/*
         * At this point, we have the array of histograms we need to give to
         * the kernel call. Let's load them onto the GPU.
         */
	UINT *devHistograms, *devWidths, *devHeights, *devResults;

	err = cudaMemcpyAsync(histograms,
                              devHistograms,
                              refImageFeatureData->featureCount * numFiles * sizeof(UINT),
                              cudaMemcpyHostToDevice,
                              stream);
	HANDLE_CUDA_ERROR(err);

	err = cudaMemcpyAsync(imageWidths,
                              devWidths,
                              numFiles * sizeof(UINT),
                              cudaMemcpyHostToDevice,
                              stream);
	HANDLE_CUDA_ERROR(err);

	err = cudaMemcpyAsync(imageHeights,
                              devHeights,
                              numFiles * sizeof(UINT),
                              cudaMemcpyHostToDevice,
                              stream);
	HANDLE_CUDA_ERROR(err);

	err = cudaMemcpyAsync(results,
                              devResults,
                              numFiles * sizeof(double),
                              cudaMemcpyHostToDevice,
                              stream);
	HANDLE_CUDA_ERROR(err);

	/*
         * Call the kernel with a naive number of blocks. Only histograms size
         * number of threads per block. This won't lead to great occupancy.
         * This makes it very easy to orient ourselves in the memory space of
         * the arrays though.
         */
	dim3 grid, threads;
	threads.x = refImageFeatureData->featureCount;
	grid.x = numFiles;//Each block will process one histogram
	histogram<<<grid, threads, stream>>>(devHistograms,
                                             devWidths,
                                             devHeights,
                                             devResults,
                                             refImageFeatureData->width,
                                             refImageFeatureData->height,
                                             refImageFeatureData->featureCount);

	/* Move the results back into pinned memory */
	err = cudaMemcpyAsync(devResults,
                              results,
                              numFiles * sizeof(double),
                              cudaMemcpyDeviceToHost,
                              stream);
	HANDLE_CUDA_ERROR(err);

	/* Sync with and then destroy the stream */
	cudaStreamSynchronize(stream);

	err = cudaStreamDestroy(stream);
	HANDLE_CUDA_ERROR(err);

	/* Dump the results into the result list */
	for (size_t i = 0; i < numFiles; i++)
	{
		const double distance = results[i];

		EnterCriticalSection(&CriticalSection);
		result.insert(ResultPair(distance, imageFileNames[i]));
		LeaveCriticalSection(&CriticalSection);
	}

	/* Clear up the memory */
	delete refImageFeatureData;
	delete results;
	delete histograms;
	delete imageWidths;
	delete imageHeights;
	delete imageFileNames;

	return EXIT_SUCCESS;
}

