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

	if (argc < 1)
	{
		ShowHelp();
	}
	else
	{
		HandleArguments(argv[1]);
	}
}