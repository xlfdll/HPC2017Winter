// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// DBIO.cpp - image database I/O functions (definition)

#include "DBIO.h"

// Critical section for CPU thread synchronization
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

		cudaDeviceProp cudaDeviceInfo;
		cudaGetDeviceProperties(&cudaDeviceInfo, 0);
		cout << "Using " << cudaDeviceInfo.name << " GPU for acceleration ..." << endl;

		// Initialize Windows GDI+ for image pixel extraction
		ULONG_PTR gdiplusToken;
		GdiplusStartupInput gdiplusStartupInput;

		GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

		// Initialize CPU thread arguments
		size_t nFileCountPerThread = (filelist.size() + nCPU - 1) / nCPU;
		UpdateThreadData *thread_data = new UpdateThreadData[nCPU];

		for (size_t i = 0; i < nCPU; i++)
		{
			thread_data[i].id = i;
			thread_data[i].filelist = &filelist;
			thread_data[i].start = i * nFileCountPerThread;
			thread_data[i].end = thread_data[i].start + nFileCountPerThread;

			if (thread_data[i].end > filelist.size())
			{
				thread_data[i].end = filelist.size();
			}

			thread_data[i].cudaDeviceInfo = &cudaDeviceInfo;
		}

		// Start CPU threads
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

		cudaDeviceProp cudaDeviceInfo;
		cudaGetDeviceProperties(&cudaDeviceInfo, 0);
		cout << "Using " << cudaDeviceInfo.name << " GPU for acceleration ..." << endl;

		// Initialize Windows GDI+ for image pixel extraction
		ULONG_PTR gdiplusToken;
		GdiplusStartupInput gdiplusStartupInput;

		GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

		InitializeCriticalSection(&CriticalSection);

		// Read reference image feature data
		Bitmap *image = new Bitmap(pszPath);

		UINT imageWidth = image->GetWidth();
		UINT imageHeight = image->GetHeight();
		UINT imagePixelCount = imageWidth * imageHeight;

		SimpleColor *pixels;
		HANDLE_CUDA_ERROR(cudaMallocHost(&pixels, imagePixelCount * sizeof(SimpleColor)));

		Color pixelColor;

		for (UINT i = 0; i < imageWidth; i++)
		{
			for (UINT j = 0; j < imageHeight; j++)
			{
				image->GetPixel(i, j, &pixelColor);

				pixels[j * imageWidth + i] = { pixelColor.GetR(), pixelColor.GetG(), pixelColor.GetB() };
			}
		}

		delete image;

		// Initialize a CUDA stream
		cudaStream_t stream;
		HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

		// Memory copy: host -> device
		SimpleColor *d_pixels;
		HANDLE_CUDA_ERROR(cudaMalloc(&d_pixels, imagePixelCount * sizeof(SimpleColor)));
		HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_pixels, pixels, imagePixelCount * sizeof(SimpleColor), cudaMemcpyHostToDevice, stream));

		// Prepare histogram bins on device
		UINT *d_intensityBins, *d_colorCodeBins;
		HANDLE_CUDA_ERROR(cudaMalloc(&d_intensityBins, INTENSITY_BIN_COUNT * sizeof(UINT)));
		HANDLE_CUDA_ERROR(cudaMalloc(&d_colorCodeBins, COLORCODE_BIN_COUNT * sizeof(UINT)));
		HANDLE_CUDA_ERROR(cudaMemsetAsync(d_intensityBins, 0, INTENSITY_BIN_COUNT * sizeof(UINT), stream));
		HANDLE_CUDA_ERROR(cudaMemsetAsync(d_colorCodeBins, 0, COLORCODE_BIN_COUNT * sizeof(UINT), stream));

		// Launch kernels
		LaunchUpdateKernel(stream, &cudaDeviceInfo, d_pixels, imageWidth, imageHeight, d_intensityBins, d_colorCodeBins);

		// Memory copy: device -> device (constant)
		UINT *d_histogramBins;
		UINT histogramBinCount;

		switch (method)
		{
		case Intensity:
			d_histogramBins = d_intensityBins;
			histogramBinCount = INTENSITY_BIN_COUNT;
			break;
		case ColorCode:
			d_histogramBins = d_colorCodeBins;
			histogramBinCount = COLORCODE_BIN_COUNT;
			break;
		default:
			break;
		}

		HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

		// Release CUDA stream resources
		HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

		// Copy reference image feature data to constant memory
		PrepareSearchKernel(d_histogramBins, histogramBinCount);

		HANDLE_CUDA_ERROR(cudaFree(d_pixels));
		HANDLE_CUDA_ERROR(cudaFree(d_intensityBins));
		HANDLE_CUDA_ERROR(cudaFree(d_colorCodeBins));

		HANDLE_CUDA_ERROR(cudaFreeHost(pixels));

		// Initialize CPU thread arguments
		size_t nFileCountPerThread = (featureFilelist.size() + nCPU - 1) / nCPU;
		SearchThreadData *thread_data = new SearchThreadData[nCPU];
		ResultMultiMap resultMap;

		for (size_t i = 0; i < nCPU; i++)
		{
			thread_data[i].id = i;
			thread_data[i].filelist = &featureFilelist;
			thread_data[i].start = i * nFileCountPerThread;
			thread_data[i].end = thread_data[i].start + nFileCountPerThread;

			if (thread_data[i].end > featureFilelist.size())
			{
				thread_data[i].end = featureFilelist.size();
			}

			thread_data[i].method = method;
			thread_data[i].refImagePixelCount = imagePixelCount;
			thread_data[i].resultMap = &resultMap;
			thread_data[i].cudaDeviceInfo = &cudaDeviceInfo;
		}

		// Start CPU threads
		HANDLE *hThreads = new HANDLE[nCPU];
		DWORD *dwThreadIDs = new DWORD[nCPU];

		for (size_t i = 0; i < nCPU; i++)
		{
			hThreads[i] = CreateThread(NULL, 0, SearchThreadFunction, &thread_data[i], 0, &dwThreadIDs[i]);
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

		for (ResultMultiMap::const_iterator it = resultMap.begin();
			it != resultMap.end();
			it++)
		{
#ifdef OUTPUT_FOR_DEBUG
			resultStream << (*it).second << ": " << (*it).first << endl;
#else
			resultStream << (*it).second << endl;
#endif
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

// CPU thread functions

DWORD WINAPI UpdateThreadFunction(PVOID lpParam)
{
	// Read image pixels and write feature data into feature files

	UpdateThreadData *data = (UpdateThreadData *)lpParam;
	StringVector &filelist = *(data->filelist);
	TCHAR szFeaturePath[MAX_PATH];

	cudaDeviceProp *cudaDeviceInfo = data->cudaDeviceInfo;

	for (size_t i = (data->start); i < (data->end); i++)
	{
		// Extract pixel colors
		PCTSTR imagePath = filelist[i].c_str();
		PTSTR imageFileName = PathFindFileName(imagePath);
		SimplePathCombine(szFeaturePath, MAX_PATH, TEXT(FEATURE_DIRECTORY_PATH), imageFileName);
		StringCchCat(szFeaturePath, MAX_PATH, TEXT(FEATURE_EXTENSION));

		Bitmap *image = new Bitmap(imagePath);

		UINT imageWidth = image->GetWidth();
		UINT imageHeight = image->GetHeight();
		UINT imagePixelCount = imageWidth * imageHeight;

		SimpleColor *pixels;
		HANDLE_CUDA_ERROR(cudaMallocHost(&pixels, imagePixelCount * sizeof(SimpleColor)));

		Color pixelColor;

		for (UINT i = 0; i < imageWidth; i++)
		{
			for (UINT j = 0; j < imageHeight; j++)
			{
				image->GetPixel(i, j, &pixelColor);

				pixels[j * imageWidth + i] = { pixelColor.GetR(), pixelColor.GetG(), pixelColor.GetB() };
			}
		}

		delete image;

		// Initialize a CUDA stream
		cudaStream_t stream;
		HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

		// Memory copy: host -> device
		SimpleColor *d_pixels;
		HANDLE_CUDA_ERROR(cudaMalloc(&d_pixels, imagePixelCount * sizeof(SimpleColor)));
		HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_pixels, pixels, imagePixelCount * sizeof(SimpleColor), cudaMemcpyHostToDevice, stream));

		// Prepare histogram bins on device
		UINT *d_intensityBins, *d_colorCodeBins;
		HANDLE_CUDA_ERROR(cudaMalloc(&d_intensityBins, INTENSITY_BIN_COUNT * sizeof(UINT)));
		HANDLE_CUDA_ERROR(cudaMalloc(&d_colorCodeBins, COLORCODE_BIN_COUNT * sizeof(UINT)));
		HANDLE_CUDA_ERROR(cudaMemsetAsync(d_intensityBins, 0, INTENSITY_BIN_COUNT * sizeof(UINT), stream));
		HANDLE_CUDA_ERROR(cudaMemsetAsync(d_colorCodeBins, 0, COLORCODE_BIN_COUNT * sizeof(UINT), stream));

		// Launch kernels
		LaunchUpdateKernel(stream, cudaDeviceInfo, d_pixels, imageWidth, imageHeight, d_intensityBins, d_colorCodeBins);

		// Prepare histogram bins on host
		UINT *intensityBins, *colorCodeBins;
		HANDLE_CUDA_ERROR(cudaMallocHost(&intensityBins, INTENSITY_BIN_COUNT * sizeof(UINT)));
		HANDLE_CUDA_ERROR(cudaMallocHost(&colorCodeBins, COLORCODE_BIN_COUNT * sizeof(UINT)));

		// Memory copy: device -> host
		HANDLE_CUDA_ERROR(cudaMemcpyAsync(intensityBins, d_intensityBins, INTENSITY_BIN_COUNT * sizeof(UINT), cudaMemcpyDeviceToHost, stream));
		HANDLE_CUDA_ERROR(cudaMemcpyAsync(colorCodeBins, d_colorCodeBins, COLORCODE_BIN_COUNT * sizeof(UINT), cudaMemcpyDeviceToHost, stream));

		HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

		// Release CUDA stream resources
		HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

		HANDLE_CUDA_ERROR(cudaFree(d_pixels));
		HANDLE_CUDA_ERROR(cudaFree(d_intensityBins));
		HANDLE_CUDA_ERROR(cudaFree(d_colorCodeBins));

		HANDLE_CUDA_ERROR(cudaFreeHost(pixels));

		// Write feature data into files
		wofstream featureStream;
		featureStream.open(szFeaturePath);

		featureStream << imageFileName << endl; // Write image file name
		featureStream << imageWidth << ',' << imageHeight << endl; // Write image size information

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

		HANDLE_CUDA_ERROR(cudaFreeHost(intensityBins));
		HANDLE_CUDA_ERROR(cudaFreeHost(colorCodeBins));
	}

	return EXIT_SUCCESS;
}

DWORD WINAPI SearchThreadFunction(PVOID lpParam)
{
	// Read image feature data from feature files, and calculate distances with reference image, then output results

	SearchThreadData *data = (SearchThreadData *)lpParam;

	cudaDeviceProp *cudaDeviceInfo = data->cudaDeviceInfo;

	// Read feature files
	StringVector &filelist = *(data->filelist);
	ResultMultiMap &resultMap = *(data->resultMap);

	size_t fileCount = data->end - data->start;

	UINT histogramBinCount;

	switch (data->method)
	{
	case Intensity:
		histogramBinCount = INTENSITY_BIN_COUNT;
		break;
	case ColorCode:
		histogramBinCount = COLORCODE_BIN_COUNT;
		break;
	default:
		break;
	}

	// Prepare histogram bins and image pixel count array on host
	UINT *histogramBins, *pixelCounts;
	HANDLE_CUDA_ERROR(cudaMallocHost(&histogramBins, fileCount * histogramBinCount * sizeof(UINT)));
	HANDLE_CUDA_ERROR(cudaMallocHost(&pixelCounts, fileCount * sizeof(UINT)));

	// Prepare result array on device
	double *d_distanceResults;
	HANDLE_CUDA_ERROR(cudaMalloc(&d_distanceResults, fileCount * sizeof(double)));

	for (size_t i = (data->start); i < (data->end); i++)
	{
		size_t fileID = i - data->start;

		UINT width, height;

		wstring imageFileName, featureLine;
		wifstream featureStream;

		featureStream.open(filelist[i]);

		// Read image file name
		getline(featureStream, imageFileName);

		// Read image size information
		featureStream >> width;
		featureStream.ignore();
		featureStream >> height;

		pixelCounts[fileID] = width * height;

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

		for (size_t i = 0; i < histogramBinCount; i++)
		{
			wiss >> histogramBins[fileID * histogramBinCount + i];

			if (i < histogramBinCount - 1)
			{
				wiss.ignore();
			}
		}
	}

	// Initialize a CUDA stream
	cudaStream_t stream;
	HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

	// Memory copy: host -> device
	UINT *d_histogramBins, *d_pixelCounts;
	HANDLE_CUDA_ERROR(cudaMalloc(&d_histogramBins, fileCount * histogramBinCount * sizeof(UINT)));
	HANDLE_CUDA_ERROR(cudaMalloc(&d_pixelCounts, fileCount * sizeof(UINT)));
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_histogramBins, histogramBins, fileCount * histogramBinCount * sizeof(UINT), cudaMemcpyHostToDevice, stream));
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_pixelCounts, pixelCounts, fileCount * sizeof(UINT), cudaMemcpyHostToDevice, stream));

	LaunchSearchKernel(stream, d_histogramBins, d_pixelCounts, fileCount, data->refImagePixelCount, histogramBinCount, d_distanceResults);

	// Prepare result array on host
	double *distanceResults;
	HANDLE_CUDA_ERROR(cudaMallocHost(&distanceResults, fileCount * sizeof(double)));

	// Memory copy: device -> host
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(distanceResults, d_distanceResults, fileCount * sizeof(double), cudaMemcpyDeviceToHost, stream));

	HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

	// Release CUDA stream resources
	HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

	HANDLE_CUDA_ERROR(cudaFree(d_histogramBins));
	HANDLE_CUDA_ERROR(cudaFree(d_pixelCounts));
	HANDLE_CUDA_ERROR(cudaFree(d_distanceResults));

	HANDLE_CUDA_ERROR(cudaFreeHost(histogramBins));
	HANDLE_CUDA_ERROR(cudaFreeHost(pixelCounts));

	// Get distance and construct result map
	for (size_t i = (data->start); i < (data->end); i++)
	{
		EnterCriticalSection(&CriticalSection);
		resultMap.insert(ResultPair(distanceResults[i - data->start], filelist[i]));
		LeaveCriticalSection(&CriticalSection);
	}

	HANDLE_CUDA_ERROR(cudaFreeHost(distanceResults));

	return EXIT_SUCCESS;
}