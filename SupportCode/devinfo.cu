#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

int main(void)
{
	int driver_ver, runtime_ver;

	cudaDriverGetVersion(&driver_ver);
	cudaRuntimeGetVersion(&runtime_ver);

	printf("CUDA Driver Version: %d.%d\n", driver_ver / 1000, (driver_ver % 100) / 10);
	printf("CUDA Runtime Version: %d.%d\n", runtime_ver / 1000, (runtime_ver % 100) / 10);
	printf("\n");

	int dev_count;

	cudaGetDeviceCount(&dev_count);

	if (dev_count == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
		printf("\n");

		exit(EXIT_SUCCESS);
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", dev_count);
		printf("\n");
	}

	cudaDeviceProp dev_prop;

	for (int i = 0; i < dev_count; i++)
	{
		cudaGetDeviceProperties(&dev_prop, i);

		printf("--- Device %d ---\n", i);

		printf("Device: %s\n", dev_prop.name);
		printf("Type: %s\n", dev_prop.integrated ? "Integrated" : "Discrete");
		printf("Compute Capability Version: %d.%d\n", dev_prop.major, dev_prop.minor);
		printf("Driver Mode: %s\n", dev_prop.tccDriver ? "Tesla Compute Cluster (TCC)" : "Windows Display Driver Model (WDDM)");
		printf("\n");
		printf("Clock Rate: %d Mhz\n", dev_prop.clockRate / 1000);
		printf("Memory Clock Rate: %d Mhz\n", dev_prop.memoryClockRate / 1000);
		printf("\n");
		printf("Global Memory Size: %llu MB\n", dev_prop.totalGlobalMem / (1024 * 1024));
		printf("Constant Memory Size: %llu KB\n", dev_prop.totalConstMem / 1024);
		printf("L2 Cache Size: %d KB\n", dev_prop.l2CacheSize / 1024);
		printf("\n");
		printf("Memory Bandwidth: %d-bit\n", dev_prop.memoryBusWidth);
		printf("ECC Support: %s\n", dev_prop.ECCEnabled ? "Enabled" : "Disabled");
		printf("Unified Addressing: %s\n", dev_prop.unifiedAddressing ? "Yes" : "No");
		printf("\n");
		printf("L1 Cache for Globals: %s\n", dev_prop.globalL1CacheSupported ? "Yes" : "No");
		printf("L1 Cache for Locals: %s\n", dev_prop.localL1CacheSupported ? "Yes" : "No");
		printf("\n");
		printf("SM #: %d\n", dev_prop.multiProcessorCount);
		printf("Max Grid Size: X - %d, Y - %d, Z - %d\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
		printf("Max Block Size: X - %d, Y - %d, Z - %d\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
		printf("Wrap Size: %d\n", dev_prop.warpSize);
		printf("\n");
		printf("Max # of Threads per Block: %d\n", dev_prop.maxThreadsPerBlock);
		printf("Max # of Threads per SM: %d\n", dev_prop.maxThreadsPerMultiProcessor);
		printf("Registers per Block: %d\n", dev_prop.regsPerBlock);
		printf("Registers per SM: %d\n", dev_prop.regsPerMultiprocessor);
		printf("Shared Memory per Block: %llu KB\n", dev_prop.sharedMemPerBlock / 1024);
		printf("Shared Memory per SM: %llu KB\n", dev_prop.sharedMemPerMultiprocessor / 1024);
		printf("\n");
		printf("Single-to-Double Performance Ratio (in FLOPS): %d\n", dev_prop.singleToDoublePrecisionPerfRatio);
		printf("\n");
	}

	exit(EXIT_SUCCESS);
}