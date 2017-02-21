// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// DBIO.cpp - image database I/O functions (definition)

#include "DBIO.h"

// Helper functions

BOOL DirectoryExists(LPCTSTR szPath)
{
	DWORD dwAttrib = GetFileAttributes(szPath);

	return (dwAttrib != INVALID_FILE_ATTRIBUTES
		&& (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

int GetSystemProcessorCount()
{
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);

	return sysinfo.dwNumberOfProcessors;
}

// Database I/O functions

void InitializeCBIRDatabase()
{
	BOOL bImageDirectory = DirectoryExists(TEXT(IMAGE_DIRECTORY_NAME));
	BOOL bFeatureDirectory = DirectoryExists(TEXT(FEATURE_DIRECTORY_NAME));

	if (!bFeatureDirectory)
	{
		CreateDirectory(TEXT(FEATURE_DIRECTORY_NAME), NULL);
	}

	if (!bImageDirectory)
	{
		CreateDirectory(TEXT(IMAGE_DIRECTORY_NAME), NULL);
	}
	else
	{
		UpdateCBIRDatabase();
	}
}

void UpdateCBIRDatabase()
{
	int nCPU = GetSystemProcessorCount();


}

void PerformCBIRSearch(LPCTSTR szPath)
{

}