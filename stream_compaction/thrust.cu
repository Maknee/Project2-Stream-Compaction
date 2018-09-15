#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            thrust::host_vector<int> hv_in(idata, idata + n);
            thrust::host_vector<int> hv_out(odata, odata + n);
            thrust::device_vector<int> dv_in = hv_in;
            thrust::device_vector<int> dv_out = hv_out;
            
            timer().startGpuTimer();
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            
            thrust::exclusive_scan(std::begin(dv_in),
                                   std::end(dv_in),
                                   std::begin(dv_out));
            // thrust::exclusive_scan(idata,
            //     idata + n,
            //     odata);

            timer().endGpuTimer();
            
            hv_out = dv_out;
            memcpy(odata, &hv_out[0], n * sizeof(int));

        }
    }
}
