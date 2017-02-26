// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// Helper.cpp - helper functions (definition)

#include "Helper.h"

// Error Handling

void HandleLastError()
{

}

// System

DWORD GetSystemProcessorCount()
{
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);

	return sysinfo.dwNumberOfProcessors;
}

// I/O

BOOL DirectoryExists(PCTSTR pszPath)
{
	DWORD dwAttrib = GetFileAttributes(pszPath);

	return (dwAttrib != INVALID_FILE_ATTRIBUTES
		&& (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

void SimplePathCombine(PTSTR pszPathBuffer, size_t sPathBufferSize, PCTSTR pszFirstElement, PCTSTR pszSecondElement)
{
	// Suggested by Windows API Security Cosideration
	StringCchCopy(pszPathBuffer, sPathBufferSize, pszFirstElement);
	StringCchCat(pszPathBuffer, sPathBufferSize, TEXT("\\"));
	StringCchCat(pszPathBuffer, sPathBufferSize, pszSecondElement);
}

StringVector GetFileList(PCTSTR pszDirectoryPath)
{
	TCHAR szSearchPattern[MAX_PATH];
	TCHAR szFullPath[MAX_PATH];

	SimplePathCombine(szSearchPattern, MAX_PATH, pszDirectoryPath, TEXT("*.*"));

	StringVector filelist;

	WIN32_FIND_DATA wfd;
	HANDLE hFind;

	hFind = FindFirstFile(szSearchPattern, &wfd);

	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if ((wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
			{
				SimplePathCombine(szFullPath, MAX_PATH, pszDirectoryPath, wfd.cFileName);

				filelist.push_back(wstring(szFullPath));
			}
		} while (FindNextFile(hFind, &wfd) != 0);

		FindClose(hFind);
	}

	return filelist;
}