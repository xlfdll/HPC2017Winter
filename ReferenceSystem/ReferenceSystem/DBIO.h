// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// DBIO.h - image database I/O functions (declaration)

#pragma once

#include "Include.h"
#include "Helper.h"
#include "CBIR.h"

// Main functions

void InitializeCBIRDatabase();
void UpdateCBIRDatabase();
void PerformCBIRSearch(PCTSTR pszPath);

// Thread argument structures

typedef struct updateThreadData
{
	size_t id;

	StringVector *filelist;

	size_t start;
	size_t end;
} UpdateThreadData;

// Thread functions

DWORD WINAPI UpdateThreadFunction(PVOID lpParam);
DWORD WINAPI SearchThreadFunction(PVOID lpParam);