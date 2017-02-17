#include "Common.h"

void ShowHelp()
{
	cout
		<< "Color Histogram CBIR System" << endl
		<< "CSS 535 Project - Winter 2017" << endl
		<< endl
		<< "Usage:" << endl
		<< "CBIRSystem -u | <reference image filename>" << endl
		<< endl
		<< "Parameters:" << endl
		<< endl
		<< "-u: create image and feature directories if they do not exist; otherwise, update all image features." << endl
		<< "OR" << endl
		<< "<reference image filename>: search all images similar to given (reference) image file in image directory" << endl
		<< endl;
}

void HandleArguments(char *arg)
{

}