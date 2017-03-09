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

#if CUDA_HISTOGRAM
		/*
		 * If we are using CUDA for calculating the histograms,
		 * we set up a stream of kernels, then iterate each thread
		 * over their alotment of images. Each image is converted
		 * into pixels and pushed into a kernel call in the stream.
		 * We collect the results in a giant array of histograms.
		 */
		UINT *histogramsI = new UINT[filelist.size() * INTENSITY_BIN_COUNT];
		UINT *histogramsC = new UINT[filelist.size() * COLORCODE_BIN_COUNT];
#endif //CUDA_HISTOGRAM

		// Initialize thread arguments
		size_t nFileCountPerThread = (filelist.size() + nCPU - 1) / nCPU;
		UpdateThreadData *thread_data = new UpdateThreadData[nCPU];

		for (size_t i = 0; i < nCPU; i++)
		{
			thread_data[i].id = i;
			thread_data[i].filelist = &filelist;
			thread_data[i].start = i * nFileCountPerThread;
			thread_data[i].end = thread_data[i].start + nFileCountPerThread;
#if CUDA_HISTOGRAM
			thread_data[i].intensityHistograms = histogramsI;
			thread_data[i].colorHistograms = histogramsC;
#endif //CUDA_HISTOGRAM

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

#if CUDA_HISTOGRAM

	for (size_t i = (data->start); i < (data->end); i++)
	{
		// Extract features
		PCTSTR imagePath = filelist[i].c_str();
		PTSTR imageFileName = PathFindFileName(imagePath);
		SimplePathCombine(szFeaturePath, MAX_PATH, TEXT(FEATURE_DIRECTORY_PATH), imageFileName);
		StringCchCat(szFeaturePath, MAX_PATH, TEXT(FEATURE_EXTENSION));

		Bitmap *image = new Bitmap(imagePath);

		UINT *intensityBins = GetIntensityBins(image);
		UINT *colorCodeBins = GetColorCodeBins(image);

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
		delete[] intensityBins;
		delete[] colorCodeBins;
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

		for (size_t i = 0; i < dbImageFeatureData->featureCount; i++)
		{
			wiss >> dbImageFeatureData->features[i];

			if (i < dbImageFeatureData->featureCount - 1)
			{
				wiss.ignore();
			}
		}

		// Get distance and construct result map
		double distance = GetManhattanDistance(refImageFeatureData, dbImageFeatureData);

		EnterCriticalSection(&CriticalSection);
		result.insert(ResultPair(distance, imageFileName));
		LeaveCriticalSection(&CriticalSection);

		delete[] dbImageFeatureData->features;
		delete dbImageFeatureData;
	}

	delete refImageFeatureData;

	return EXIT_SUCCESS;
}
