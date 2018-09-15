#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <memory>

namespace StreamCompaction
{
    namespace Efficient
    {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //taken from printArray in main
        void printArray(int n, int* a, bool abridged = false)
        {
            printf("    [ ");
            for (int i = 0; i < n; i++)
            {
                if (abridged && i + 2 == 15 && n > 16)
                {
                    i = n - 2;
                    printf("... ");
                }
                printf("%3d ", a[i]);
            }
            printf("]\n");
        }

        //pow function
        __device__ __host__ int kernPow2(int power_of)
        {
            int result = 1;
            for (int i = 0; i < power_of; i++)
            {
                result <<= 1;
            }
            return result;
        }

        //round to the nearest power of 2 (ceiling)
        void round_to_nearest_pow(int& n)
        {
            //round n to the nearest pow of n
            n = std::ceil(std::log2(n));
            n = kernPow2(n);
        }

        //up sweep function
        __global__ void kernUpSweep(int N, int* odata, int d)
        {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);

            //k is multiplied by pow to index the correct location
            k *= kernPow2(d + 1);
            if (k > N)
                return;

            //formula
            odata[k + kernPow2(d + 1) - 1] += odata[k + kernPow2(d) - 1];
        }

        __global__ void kernDownSweep(int N, int* odata, int d)
        {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            k *= kernPow2(d + 1);
            if (k > N)
                return;

            //formula
            int t = odata[k + kernPow2(d) - 1];
            odata[k + kernPow2(d) - 1] = odata[k + kernPow2(d + 1) - 1];
            odata[k + kernPow2(d + 1) - 1] += t;
        }

#define BLOCK_SIZE 128

        //actual implementation of scan
        //because timer().startGpuTimer() is called inside
        //scan(...) from scatter(...), causing an abort
        void scan_impl(int n, int* odata, const int* idata)
        {
            //round to nearest pow because n might not be a power of 2
            round_to_nearest_pow(n);

            dim3 kernelBlock((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* kern_odata;

            cudaMalloc(reinterpret_cast<void**>(&kern_odata), n * sizeof(int));

            cudaMemcpy(kern_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            //call upsweep
            for (int d = 0; static_cast<float>(d) < std::ceil(std::log2(n)); d++)
            {
                kernUpSweep<<<kernelBlock, BLOCK_SIZE>>>(n, kern_odata, d);
                //cudaMemcpy(odata, kern_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
                //printArray(n, odata, true);
                //printArray(n/2, odata + n/2, true);
            }

            //printf("=====================\n");

            //copy 0 to the end (from formula)
            int zero = 0;
            cudaMemcpy(kern_odata + n - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

            //down sweep
            for (int d = static_cast<int>(std::ceil(std::log2(n))) - 1; d >= 0; d--)
            {
                kernDownSweep<<<kernelBlock, BLOCK_SIZE>>>(n, kern_odata, d);
                //cudaMemcpy(odata, kern_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
                //printArray(n, odata, true);
                //printArray(n/2, odata + n/2, true);
            }

            //copy the result
            cudaMemcpy(odata, kern_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(kern_odata);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata)
        {
            timer().startGpuTimer();

            scan_impl(n, odata, idata);

            timer().endGpuTimer();
        }

        __global__ void kernScatter(int N, int* final_array, const int* bool_array, const int* scan_array,
                                    const int* unfiltered_array)
        {
            const int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index > N)
                return;

            //filter only elements that are not zero in the bool map.
            if (bool_array[index])
            {
                //get the index of where the element is suppose to be in the in the final array
                const int index_of_filtered_element = scan_array[index];
                final_array[index_of_filtered_element] = unfiltered_array[index];
            }
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata)
        {
            timer().startGpuTimer();

            dim3 kernelBlock((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            //make another variable that is the power of n (we need this since counter can't iterate through pow n)
            int rounded_n = n;
            round_to_nearest_pow(rounded_n);

            auto counters = std::make_unique<int[]>(rounded_n);
            memset(counters.get(), 0, rounded_n * sizeof(int));

            //idata
            int* unfiltered_array;
            cudaMalloc(reinterpret_cast<void**>(&unfiltered_array), n * sizeof(int));
            cudaMemcpy(unfiltered_array, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            //bool mapping (1 or 0)
            int* bool_array;
            cudaMalloc(reinterpret_cast<void**>(&bool_array), n * sizeof(int));

            Common::kernMapToBoolean<<<kernelBlock, BLOCK_SIZE>>>(n, bool_array, unfiltered_array);

            cudaMemcpy(counters.get(), bool_array, n * sizeof(int), cudaMemcpyDeviceToHost);

            int count = 0;

            //iterate through and count
            for (int i = 0; i < n; i++)
            {
                if (counters[i])
                {
                    count++;
                }
            }

            //now round to nearest pow
            round_to_nearest_pow(n);

            kernelBlock = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            auto scan_result = std::make_unique<int[]>(n);

            memcpy(scan_result.get(), counters.get(), n);

            //scan
            scan_impl(n, scan_result.get(), counters.get());

            int* final_array;
            int* scan_array;

            cudaMalloc(reinterpret_cast<void**>(&final_array), n * sizeof(int));
            cudaMalloc(reinterpret_cast<void**>(&scan_array), n * sizeof(int));

            cudaMemcpy(scan_array, scan_result.get(), n * sizeof(int), cudaMemcpyHostToDevice);

            //do scatter
            kernScatter<<<kernelBlock, BLOCK_SIZE>>>(n, final_array, bool_array, scan_array, unfiltered_array);

            //copy the result back
            cudaMemcpy(odata, final_array, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(final_array);
            cudaFree(bool_array);
            cudaFree(scan_array);
            cudaFree(unfiltered_array);

            timer().endGpuTimer();
            return count;
        }
    }
}
