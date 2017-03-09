// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// kernels.h - Kernel functions (declaration)

#pragma once

#include "Include.h"

#if CUDA_HISTOGRAM
/**
 * Computes the intensity- and color-based histograms for a given image and
 * puts the resulting histograms into the correct spot in the histogramsI and
 * histogramsC arrays.
 *
 * @param histogramsI:    Array of intensity histograms. Each call to this
 *                        kernel will fill only one of the histograms in
 *                        this array.
 *
 * @param histogramsC:    Array of color histograms.
 *
 * @param pixels:         The image, as pixels of the form RGBX, where each
 *                        letter corresponds to 8 bits, with R being the MSB
 *                        and X being the LSB and 0.
 *
 * @param imageWidth:     The width (x dimension) of the image.
 *
 * @param imageHeight:    The height (y dimension) of the image.
 *
 * @param histIndex:      The index of the histogram we are computing. The
 *                        index into the histogramsI/C arrays is calculated
 *                        by multiplying this value with the length of the
 *                        corresponding type of histogram.
 */
__global__ void histogram(UINT *histogramsI,
                          UINT *histogramsC,
                          UINT32 *pixels,
                          UINT imageWidth,
                          UINT imageHeight,
                          UINT histIndex);
#endif //CUDA_HISTOGRAM

