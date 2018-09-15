#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <memory>

namespace StreamCompaction
{
    namespace Naive
    {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        //pow function
        __device__ int kernPow2(int power_of)
        {
            int result = 1;
            for (int i = 0; i < power_of; i++)
            {
                result <<= 1;
            }
            return result;
        }

        __global__ void kernNaive(int N, int* odata, const int* idata, int d)
        {
            const int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k > N)
                return;

            //follow the formula
            if (k >= kernPow2(d - 1))
            {
                odata[k] = idata[k - kernPow2(d - 1)] + idata[k];
            }
        }

        //swap pointers
        template <typename T, typename = std::enable_if<std::is_pointer<T>::value>::type>
        void swap_pointers(T& a, T& b)
        {
            T c = a;
            a = b;
            b = c;
        }


#define BLOCK_SIZE 128

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata)
        {
            timer().startGpuTimer();

            dim3 kernelBlock((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* kern_odata;
            int* kern_odata2;

            //allocate memory
            cudaMalloc(reinterpret_cast<void**>(&kern_odata), n * sizeof(int));
            cudaMalloc(reinterpret_cast<void**>(&kern_odata2), n * sizeof(int));

            //copy the start data
            cudaMemcpy(kern_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            for (int d = 1; static_cast<float>(d) <= std::ceil(std::log2(n)); d++)
            {
                //make sure we copy over the ones that are not part of offset d
                cudaMemcpy(kern_odata2, kern_odata, n * sizeof(int), cudaMemcpyHostToDevice);

                //call the naive impl of kernel
                kernNaive<<<kernelBlock, BLOCK_SIZE>>>(n, kern_odata2, kern_odata, d);

                //ping pong
                swap_pointers(kern_odata, kern_odata2);
            }

            //copy the result
            cudaMemcpy(odata, kern_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(kern_odata);
            cudaFree(kern_odata2);

            //shift right
            auto temp = std::make_unique<int[]>(n);
            memcpy(temp.get(), odata, n * sizeof(int));

            //shift right by 1
            for (int i = 1; i < n; i++)
            {
                odata[i] = temp[i - 1];
            }

            //set first element to 0
            odata[0] = 0;

            timer().endGpuTimer();
        }
    }
}
