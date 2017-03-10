#pragma once

void __handle_cuda_error(cudaError_t err,
                         unsigned int line_no,
                         const char *file_name);
