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

int GetSystemProcessorCount()
{
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);

	return sysinfo.dwNumberOfProcessors;
}

// I/O

BOOL DirectoryExists(LPCTSTR szPath)
{
	DWORD dwAttrib = GetFileAttributes(szPath);

	return (dwAttrib != INVALID_FILE_ATTRIBUTES
		&& (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

StringVector GetFileList(LPCTSTR szDirectoryPath)
{
	TCHAR szSearchPattern[MAX_PATH];
	TCHAR szFullPath[MAX_PATH];

	// Suggested by Windows API Security Cosideration
	StringCchCopy(szSearchPattern, MAX_PATH, szDirectoryPath);
	StringCchCat(szSearchPattern, MAX_PATH, TEXT("\\*.*"));

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
				StringCchCopy(szFullPath, MAX_PATH, szDirectoryPath);
				StringCchCat(szFullPath, MAX_PATH, TEXT("\\"));
				StringCchCat(szFullPath, MAX_PATH, wfd.cFileName);

				filelist.push_back(wstring(szFullPath));
			}
		} while (FindNextFile(hFind, &wfd) != 0);

		FindClose(hFind);
	}

	return filelist;
}