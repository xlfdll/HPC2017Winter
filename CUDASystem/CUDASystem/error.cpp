#include "error.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/*
 * Do not call this function directly. Call HANDLE_CUDA_ERROR(err) instead.
 */
void __handle_cuda_error(cudaError_t err,
                         unsigned int line_no,
                         const char *file_name)
{
	if (err != cudaSuccess)
	{
		printf("Function call returned %s (code %d)\n",
                       cudaGetErrorString(err),
                       err);
		printf("This is at file %s, line %d\n", file_name, line_no);
		exit(EXIT_FAILURE);
	}
}

