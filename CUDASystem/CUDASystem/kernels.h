// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// kernels.h - Kernel functions (declaration)

#pragma once

#include "Include.h"

/**
 * Computes the intensity- and color-based histograms for a given image.
 *
 * @param histogramI:     The intensity-based histogram to fill for the
 *                        given image.
 *
 * @param histogramC:     The color-based histogram to fill for the given
 *                        image.
 *
 * @param pixels:         The image, as pixels of the form RGBX, where each
 *                        letter corresponds to 8 bits, with R being the MSB
 *                        and X being the LSB and 0.
 *
 * @param imageWidth:     The width (x dimension) of the image.
 *
 * @param imageHeight:    The height (y dimension) of the image.
 *
 */
__global__ void histogram(UINT *histogramI,
                          UINT *histogramC,
                          UINT32 *pixels,
                          UINT imageWidth,
                          UINT imageHeight);


/**
 * Computes the manhatten distance between each histogram in the histograms
 * array and the refHist in constant memory. Returns the results in the results
 * array.
 *
 * @param histograms:     The array of histograms from the database.
 *
 * @param widths:         The widths of each image from the database, must be
 *                        the same order as the histograms.
 *
 * @param heights:        The heights of each image from the database, must be
 *                        the same order as the histograms.
 *
 * @param results:        The results array. The kernel call will fill this
 *                        array with the distances. The one that is smallest
 *                        from that list is the most likely match. These will
 *                        also be in the same order as the other arrays.
 *
 * @param refWidth:       The reference image width.
 *
 * @param refHeight:      The reference image height.
 *
 * @param refHistLength:  The length of each histogram.
 */
__global__ void search_kernel(UINT *histograms,
                              UINT *widths,
                              UINT *heights,
                              double *results,
                              UINT refWidth,
                              UINT refHeight,
                              UINT refHistLength);
