// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// DBIO.cpp - image database I/O functions (definition)

#include "DBIO.h"

// Main functions

void InitializeCBIRDatabase()
{
	BOOL bImageDirectory = DirectoryExists(TEXT(IMAGE_DIRECTORY_NAME));
	BOOL bFeatureDirectory = DirectoryExists(TEXT(FEATURE_DIRECTORY_NAME));

	if (!bFeatureDirectory)
	{
		cout
			<< "Feature data directory does not exist."
			<< endl
			<< "Creating feature data directory ..."
			<< endl
			<< endl;

		CreateDirectory(TEXT(FEATURE_DIRECTORY_NAME), NULL);
	}

	if (!bImageDirectory)
	{
		cout
			<< "Image data directory does not exist."
			<< endl
			<< "Creating image data directory ..."
			<< endl
			<< endl;

		CreateDirectory(TEXT(IMAGE_DIRECTORY_NAME), NULL);
	}
	else
	{
		cout
			<< "Updating image features ..."
			<< endl;

		UpdateCBIRDatabase();
	}
}

void UpdateCBIRDatabase()
{
	StringVector filelist = GetFileList(TEXT(IMAGE_DIRECTORY_NAME));
	DWORD nCPU = GetSystemProcessorCount();

	cout << filelist.size() << " image file(s) in database." << endl;

	if (filelist.size() == 0) // No need to update if no image files
	{
		cout << "No need to update." << endl;
	}
	else
	{
		cout << "Creating " << nCPU << " thread(s) for feature updating ..." << endl;
		cout << endl;

		// Initialize Windows GDI+ for image pixel extraction
		ULONG_PTR gdiplusToken;
		GdiplusStartupInput gdiplusStartupInput;

		GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

		size_t nFileCountPerThread = (filelist.size() + nCPU - 1) / nCPU;
		UpdateThreadData *thread_data = new UpdateThreadData[nCPU];

		for (size_t i = 0; i < nCPU; i++)
		{
			thread_data[i].id = i;
			thread_data[i].filelist = &filelist;
			thread_data[i].start = i * nFileCountPerThread;
			thread_data[i].end = thread_data[i].start + nFileCountPerThread;
		}

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

		cout << "Done." << endl;
	}
}

void PerformCBIRSearch(LPCTSTR szPath)
{

}

// Thread functions

DWORD WINAPI UpdateThreadFunction(LPVOID lpParam)
{
	// Read image pixels and write feature data into feature files

	UpdateThreadData *data = (UpdateThreadData *)lpParam;
	StringVector &filelist = *(data->filelist);
	TCHAR szFeaturePath[MAX_PATH];

	// TODO: Construct image feature path

	for (size_t i = (data->start); i < (data->end); i++)
	{
		Bitmap *image = new Bitmap(filelist[i].c_str());

		
	}

	return 0;
}

DWORD WINAPI SearchThreadFunction(LPVOID lpParam)
{
	// Read image feature data from feature files, and calculate distances with reference image, then output results

	return 0;
}