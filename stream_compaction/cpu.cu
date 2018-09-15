#include <cstdio>
#include "cpu.h"

#include "common.h"
#include <memory>
#include <iostream>

namespace StreamCompaction
{
	namespace CPU
	{
		using StreamCompaction::Common::PerformanceTimer;

		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		//actual implementation of scan
		//because timer().startCpuTimer() is called inside
		//scan(...) from scatter(...), causing an abort
		void scan_impl(int n, int* odata, const int* idata)
		{
			/// super naive cpu implementation ///
			// memset(odata, 0, n * sizeof(int));
			//
			//  for(int k = 1; k < n; k++)
			//  {
			//  	odata[k] = odata[k - 1] + idata[k - 1];
			//  }

			/// psuedo parallel implementation ///

			//make sure the data is set first before beginning
			memcpy(odata, idata, sizeof(int) * n);

			for (int d = 1; static_cast<float>(d) <= std::ceil(std::log2(n)); d++)
			{
				//make a copy, because naive can't be done in place
				auto temp = std::make_unique<int[]>(n);
				memcpy(temp.get(), odata, n * sizeof(int));
				for (int k = 0; k < n; k++)
				{
					//follow the formula
					if (k >= static_cast<int>(std::pow(2, d - 1)))
					{
						odata[k] = temp[k - static_cast<int>(std::pow(2, d - 1))] + temp[k];
					}
				}
			}

			//copy the data back
			auto temp = std::make_unique<int[]>(n);
			memcpy(temp.get(), odata, n * sizeof(int));

			//shift right by 1
			for (int i = 1; i < n; i++)
			{
				odata[i] = temp[i - 1];
			}
			//set first element to 0
			odata[0] = 0;
		}

		/**
		 * CPU scan (prefix sum).
		 * For performance analysis, this is supposed to be a simple for loop.
		 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
		 */
		void scan(int n, int* odata, const int* idata)
		{
			timer().startCpuTimer();

			scan_impl(n, odata, idata);

			timer().endCpuTimer();
		}

		/**
		 * CPU stream compaction without using the scan function.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithoutScan(int n, int* odata, const int* idata)
		{
			timer().startCpuTimer();

			memset(odata, 0, n * sizeof(int));

			int index = 0;

			//iterate through and count
			for (int i = 0; i < n; i++)
			{
				if (idata[i])
				{
					odata[index] = idata[i];
					index++;
				}
			}

			timer().endCpuTimer();
			return index;
		}

		/**
		 * CPU stream compaction using scan and scatter, like the parallel version.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithScan(int n, int* odata, const int* idata)
		{
			timer().startCpuTimer();

			auto counters = std::make_unique<int[]>(n);

			int count = 0;

			//iterate through and count
			for (int i = 0; i < n; i++)
			{
				counters[i] = idata[i] ? 1 : 0;
				if (counters[i])
				{
					count++;
				}
			}

			auto indicies = std::make_unique<int[]>(n);

			memcpy(indicies.get(), counters.get(), n);

			//scan
			scan_impl(n, indicies.get(), counters.get());

			//now set the scanned result to the correct index
			for (int i = 0; i < n; i++)
			{
				if (counters[i])
				{
					odata[indicies[i]] = idata[i];
				}
			}

			timer().endCpuTimer();
			return count;
		}
	}
}
