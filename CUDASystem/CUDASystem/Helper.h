// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// Helper.h - helper functions (declaration)

#pragma once

#include "Include.h"

// System
DWORD GetSystemProcessorCount();

// I/O
BOOL DirectoryExists(PCTSTR pszPath);
void SimplePathCombine(PTSTR pszPathBuffer, size_t sPathBufferSize, PCTSTR pszFirstElement, PCTSTR pszSecondElement);
StringVector GetFileList(PCTSTR pszDirectoryPath);

// CUDA
void HandleCUDAError(cudaError_t errorCode, unsigned int lineNumber, const char *fileName);

#define HANDLE_CUDA_ERROR(errorCode) HandleCUDAError((errorCode), __LINE__, __FILE__)