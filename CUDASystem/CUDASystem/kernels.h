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

