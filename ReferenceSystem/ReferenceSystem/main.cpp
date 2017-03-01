// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// Main.cpp - main function

#include "CmdLine.h"

int main(int argc, char *argv[])
{
	// Parameters:
	//
	// -u: If database folders do not exist, create them and quit; otherwise, update all image features and quit
	// <filename>: search similar images to the given one

	cout
		<< "Color Histogram CBIR System - Reference System (CPU)" << endl
		<< "CSS 535 Project - Winter 2017" << endl
		<< "Max Strange, Jeremy Albert, Longfei Xi" << endl
		<< endl;

	// Convert command arguments to Unicode
	int nArgs;
	PTSTR *szArgList = CommandLineToArgvW(GetCommandLine(), &nArgs);

	if (ValidateArguments(szArgList, nArgs))
	{
		HandleArguments(szArgList, nArgs);
	}
	else
	{
		ShowArgumentHelp();
	}

	LocalFree(szArgList); // Release memory for Unicode argument list

	return EXIT_SUCCESS;
}