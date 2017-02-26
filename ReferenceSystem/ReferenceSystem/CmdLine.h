// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// CmdLine.h - command line handling functions (declaration)

#pragma once

#include "Include.h"
#include "DBIO.h"

void ShowHelp();
void HandleArguments(PCTSTR *szArgument, int nArgumentCount);