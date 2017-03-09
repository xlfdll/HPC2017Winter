// CBIR System (Color Histogram based)
// CSS 535 Project - Winter 2017
// Max Strange, Jeremy Albert, Longfei Xi
//
// kernels.cu - Kernel functions

#include "kernels.h"

# if CUDA_HISTOGRAM
/*
 * The algorithm for computing a parallel histogram is as follows:
 * TODO
 */
__global__ void histogram(UINT *histogramsI,
                          UINT *histogramsC,
                          UINT32 *pixels,
                          UINT imageWidth,
                          UINT imageHeight,
                          UINT histIndex)
{
	// TODO: compute parallel histogram
}
#endif //CUDA_HISTOGRAM

