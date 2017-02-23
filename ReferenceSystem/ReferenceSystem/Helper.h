// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// Helper.h - helper functions (declaration)

#pragma once

#include "Include.h"

// Error Handling
void HandleLastError();

// System
int GetSystemProcessorCount();

// I/O
BOOL DirectoryExists(LPCTSTR szPath);
StringVector GetFileList(LPCTSTR szDirectoryPath);