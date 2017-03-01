// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// CmdLine.h - command line handling functions (declaration)

#pragma once

#include "Include.h"
#include "DBIO.h"

BOOL ValidateArguments(PTSTR *szArgument, int nArgumentCount);
void HandleArguments(PTSTR *szArgument, int nArgumentCount);

void ShowArgumentHelp();