// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// DBIO.cpp - image database I/O functions (definition)

#include "DBIO.h"

// Critical section for thread synchronization
CRITICAL_SECTION CriticalSection;

// Main functions

void InitializeCBIRDatabase()
{
	BOOL bImageDirectory = DirectoryExists(TEXT(IMAGE_DIRECTORY_PATH));
	BOOL bFeatureDirectory = DirectoryExists(TEXT(FEATURE_DIRECTORY_PATH));

	if (!bFeatureDirectory)
	{
		cout
			<< "Feature data directory does not exist."
			<< endl
			<< "Creating feature data directory ..."
			<< endl
			<< endl;

		CreateDirectory(TEXT(FEATURE_DIRECTORY_PATH), NULL);
	}

	if (!bImageDirectory)
	{
		cout
			<< "Image data directory does not exist."
			<< endl
			<< "Creating image data directory ..."
			<< endl
			<< endl;

		CreateDirectory(TEXT(IMAGE_DIRECTORY_PATH), NULL);
	}
	else
	{
		cout
			<< "Updating image features ..."
			<< endl;

		UpdateCBIRDatabase();
	}

	cout << "Done." << endl;
}

void UpdateCBIRDatabase()
{
	StringVector filelist = GetFileList(TEXT(IMAGE_DIRECTORY_PATH));

	cout << filelist.size() << " image file(s) in database." << endl;

	if (filelist.size() == 0) // No need to update if no image files
	{
		cout << "No need to update." << endl;
	}
	else
	{
		DWORD nCPU = GetSystemProcessorCount();
		cout << "Creating " << nCPU << " thread(s) for feature updating ..." << endl;

		// Initialize Windows GDI+ for image pixel extraction
		ULONG_PTR gdiplusToken;
		GdiplusStartupInput gdiplusStartupInput;

		GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

		/*
		 * Set up a stream of kernels, then iterate each thread
		 * over their alotment of images. Each image is converted
		 * into pixels and pushed into a kernel call in the stream.
		 * We collect the results in a giant array of histograms.
		 */
		UINT *histogramsI = new UINT[filelist.size() * INTENSITY_BIN_COUNT];
		UINT *histogramsC = new UINT[filelist.size() * COLORCODE_BIN_COUNT];

		// Initialize thread arguments
		size_t nFileCountPerThread = (filelist.size() + nCPU - 1) / nCPU;
		UpdateThreadData *thread_data = new UpdateThreadData[nCPU];

		for (size_t i = 0; i < nCPU; i++)
		{
			thread_data[i].id = i;
			thread_data[i].filelist = &filelist;
			thread_data[i].start = i * nFileCountPerThread;
			thread_data[i].end = thread_data[i].start + nFileCountPerThread;
			thread_data[i].intensityHistograms = histogramsI;
			thread_data[i].colorHistograms = histogramsC;

			if (thread_data[i].end > filelist.size())
			{
				thread_data[i].end = filelist.size();
			}
		}

		// Start threads
		HANDLE *hThreads = new HANDLE[nCPU];
		DWORD *dwThreadIDs = new DWORD[nCPU];

		for (size_t i = 0; i < nCPU; i++)
		{
			hThreads[i] = CreateThread(NULL, 0, UpdateThreadFunction, &thread_data[i], 0, &dwThreadIDs[i]);
		}

		WaitForMultipleObjects(nCPU, hThreads, TRUE, INFINITE);

		for (size_t i = 0; i < nCPU; i++)
		{
			CloseHandle(hThreads[i]);
		}

		// Shutdown Windows GDI+
		GdiplusShutdown(gdiplusToken);

		delete[] thread_data;
		delete[] hThreads;
		delete[] dwThreadIDs;
		delete[] histogramsI;
		delete[] histogramsC;
	}
}

void PerformCBIRSearch(PCTSTR pszPath, CBIRMethod method)
{
	StringVector featureFilelist;

	if (DirectoryExists(TEXT(FEATURE_DIRECTORY_PATH)))
	{
		featureFilelist = GetFileList(TEXT(FEATURE_DIRECTORY_PATH));
	}

	if (!PathFileExists(pszPath))
	{
		cout << "Reference image file do not exist." << endl;
	}
	else if (featureFilelist.size() == 0)
	{
		cout << "Image features do not exist." << endl;
		cout << "Run \"CBIRSystem -u\" to update the database." << endl;
	}
	else
	{
		cout << featureFilelist.size() << " image file(s) scanned in database." << endl;

		DWORD nCPU = GetSystemProcessorCount();
		cout << "Creating " << nCPU << " thread(s) for searching ..." << endl;

		// Initialize Windows GDI+ for image pixel extraction
		ULONG_PTR gdiplusToken;
		GdiplusStartupInput gdiplusStartupInput;

		GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

		InitializeCriticalSection(&CriticalSection);

		// Initialize thread arguments
		size_t nFileCountPerThread = (featureFilelist.size() + nCPU - 1) / nCPU;
		SearchThreadData *thread_data = new SearchThreadData[nCPU];
		ResultMultiMap result;

		for (size_t i = 0; i < nCPU; i++)
		{
			thread_data[i].id = i;
			thread_data[i].filelist = &featureFilelist;
			thread_data[i].start = i * nFileCountPerThread;
			thread_data[i].end = thread_data[i].start + nFileCountPerThread;
			thread_data[i].refPath = pszPath;
			thread_data[i].method = method;
			thread_data[i].result = &result;

			if (thread_data[i].end > featureFilelist.size())
			{
				thread_data[i].end = featureFilelist.size();
			}
		}

		// Start threads
		HANDLE *hThreads = new HANDLE[nCPU];
		DWORD *dwThreadIDs = new DWORD[nCPU];

		for (size_t i = 0; i < nCPU; i++)
		{
			hThreads[i] = CreateThread(NULL,
                                                   0,
                                                   SearchThreadFunction,
                                                   &thread_data[i],
                                                   0,
                                                   &dwThreadIDs[i]);
		}

		WaitForMultipleObjects(nCPU, hThreads, TRUE, INFINITE);

		for (size_t i = 0; i < nCPU; i++)
		{
			CloseHandle(hThreads[i]);
		}

		// Write results to results.txt file
		cout << "Writing results to results.txt file ..." << endl;

		wofstream resultStream;
		resultStream.open("results.txt");

		for (ResultMultiMap::const_iterator it = result.begin();
			it != result.end();
			it++)
		{
			resultStream << (*it).second << endl;
		}

		resultStream.close();

		// Shutdown Windows GDI+ and release resources
		GdiplusShutdown(gdiplusToken);

		DeleteCriticalSection(&CriticalSection);

		delete[] thread_data;
		delete[] hThreads;
		delete[] dwThreadIDs;

		cout << "Done." << endl;
	}
}

// Thread functions

DWORD WINAPI UpdateThreadFunction(PVOID lpParam)
{
	// Read image pixels and write feature data into feature files

	UpdateThreadData *data = (UpdateThreadData *)lpParam;
	StringVector &filelist = *(data->filelist);
	TCHAR szFeaturePath[MAX_PATH];

	cudaStream_t stream;
	cudaError_t err;
	err = cudaStreamCreate(&stream);
	HANDLE_CUDA_ERROR(err);
	/* Iterate over this thread's alotment of images, calling
 	 * GetBins on each image. This function call will put the correct
	 * histogram into each of the histogram arrays.
	 */
	for (size_t i = (data->start); i < (data->end); i++)
	{
		PCTSTR imagePath = filelist[i].c_str();
		Bitmap *image = new Bitmap(imagePath);

		GetBins(image,
	                data->intensityHistograms,
	                data->colorHistograms,
	                i,
	                &stream);
	}
	err = cudaStreamDestroy(stream);
	HANDLE_CUDA_ERROR(err);


	for (size_t i = (data->start); i < (data->end); i++)
	{
		// Extract features
		PCTSTR imagePath = filelist[i].c_str();
		PTSTR imageFileName = PathFindFileName(imagePath);
		SimplePathCombine(szFeaturePath, MAX_PATH, TEXT(FEATURE_DIRECTORY_PATH), imageFileName);
		StringCchCat(szFeaturePath, MAX_PATH, TEXT(FEATURE_EXTENSION));

		Bitmap *image = new Bitmap(imagePath);

		UINT *intensityBins = &(data->intensityHistograms[i * INTENSITY_BIN_COUNT]);
		UINT *colorCodeBins = &(data->colorHistograms[i * COLORCODE_BIN_COUNT]);

		// Write feature data into files
		wofstream featureStream;
		featureStream.open(szFeaturePath);

		featureStream << imageFileName << endl; // Write image file name
		featureStream << image->GetWidth() << ',' << image->GetHeight() << endl; // Write image size information

		// Write intensity color histogram feature data
		for (size_t i = 0; i < INTENSITY_BIN_COUNT; i++)
		{
			featureStream << intensityBins[i];

			if (i < INTENSITY_BIN_COUNT - 1)
			{
				featureStream << ',';
			}
		}

		featureStream << endl;

		// Write color-code color histogram feature data
		for (size_t i = 0; i < COLORCODE_BIN_COUNT; i++)
		{
			featureStream << colorCodeBins[i];

			if (i < COLORCODE_BIN_COUNT - 1)
			{
				featureStream << ',';
			}
		}

		featureStream.close();

		delete image;
	}

	return EXIT_SUCCESS;
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
		double distance = results[i];

		EnterCriticalSection(&CriticalSection);
		result.insert(ResultPair(distance, imageFileName));
		LeaveCriticalSection(&CriticalSection);
	}

	/* Clear up the memory */
	delete refImageFeatureData;
	delete results;
	delete histograms;
	delete imageWidths;
	delete imageHeights;

	return EXIT_SUCCESS;
}

