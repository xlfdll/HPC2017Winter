// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// DBIO.h - image database I/O functions (declaration)

#pragma once

#include "Include.h"

// Helper functions

BOOL DirectoryExists(LPCTSTR szPath);

// Database I/O functions

void InitializeCBIRDatabase();
void UpdateCBIRDatabase();
void PerformCBIRSearch(LPCTSTR szPath);