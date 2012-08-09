/*
 * @file
 * This file is part of SWE.
 *
 * @author Alexander Breuer (breuera AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer)
 *
 * @section LICENSE
 *
 * SWE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SWE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SWE.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * @section DESCRIPTION
 *
 * SWE_Block in CUDA, which uses solvers in the wave propagation formulation.
 */

#include "SWE_WavePropagationBlockCuda.hh"
#include "SWE_BlockCUDA.hh"

#include "SWE_WavePropagationBlockCuda_kernels.hh"

#include <cassert>

#ifndef STATICLOGGER
#define STATICLOGGER
#include "tools/Logger.hpp"
static tools::Logger s_sweLogger;
#endif

#define TIMEKERNELS (0)

// system time includes
#ifdef TIMEKERNELS
#include <sys/time.h>
#endif /* TIMEKERNELS */

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime_api.h>

// Thrust library (used for the final maximum reduction in the method computeNumericalFluxes(...))
#include <thrust/device_vector.h>

/**
 * Constructor of a SWE_WavePropagationBlockCuda.
 *
 * Allocates the variables for the simulation:
 *   Please note: The definition of indices changed in contrast to the CPU-Implementation.
 *
 *   unknowns hd,hud,hvd,bd stored on the CUDA device are defined for grid indices [0,..,nx+1]*[0,..,ny+1] (-> Abstract class SWE_BlockCUDA)
 *     -> computational domain is [1,..,nx]*[1,..,ny]
 *     -> plus ghost cell layer
 *
 *   net-updates are defined for edges with indices [0,..,nx]*[0,..,ny] for horizontal and vertical edges for simplicity (one layer is not necessary).
 *
 *   A left/right net update with index (i-1,j) is located on the edge between
 *   cells with index (i-1,j) and (i,j):
 * <pre>
 *   *********************
 *   *         *         *
 *   * (i-1,j) *  (i,j)  *
 *   *         *         *
 *   *********************
 *
 *             *
 *            ***
 *           *****
 *             *
 *             *
 *   NetUpdatesLeft(i-1,j)
 *             or
 *   NetUpdatesRight(i-1,j)
 * </pre>
 *
 *   A below/above net update with index (i, j-1) is located on the edge between
 *   cells with index (i, j-1) and (i,j):
 * <pre>
 *   ***********
 *   *         *
 *   * (i, j)  *   *
 *   *         *  **  NetUpdatesBelow(i,j-1)
 *   *********** *****         or
 *   *         *  **  NetUpdatesAbove(i,j-1)
 *   * (i,j-1) *   *
 *   *         *
 *   ***********
 * </pre>
 * @param i_offsetX spatial offset of the block in x-direction.
 * @param i_offsetY spatial offset of the offset in y-direction.
 * @param i_cudaDevice ID of the CUDA-device, which should be used.
 */
SWE_WavePropagationBlockCuda::SWE_WavePropagationBlockCuda( const float i_offsetX,
                                                            const float i_offsetY,
                                                            const int i_cudaDevice ): SWE_BlockCUDA(i_offsetX, i_offsetY, i_cudaDevice) {
  // compute the size of one 1D net-update array.
  int sizeOfNetUpdates = (nx+1)*(ny+1)*sizeof(float);

  // allocate CUDA memory for the net-updates
  cudaMalloc((void**)&hNetUpdatesLeftD, sizeOfNetUpdates);
  checkCUDAError("allocate device memory for hNetUpdatesLeftD");

  cudaMalloc((void**)&hNetUpdatesRightD, sizeOfNetUpdates);
  checkCUDAError("allocate device memory for hNetUpdatesRightD");

  cudaMalloc((void**)&huNetUpdatesLeftD, sizeOfNetUpdates);
  checkCUDAError("allocate device memory for huNetUpdatesLeftD");

  cudaMalloc((void**)&huNetUpdatesRightD, sizeOfNetUpdates);
  checkCUDAError("allocate device memory for huNetUpdatesRightD");

  cudaMalloc((void**)&hNetUpdatesBelowD, sizeOfNetUpdates);
  checkCUDAError("allocate device memory for hNetUpdatesBelowD");

  cudaMalloc((void**)&hNetUpdatesAboveD, sizeOfNetUpdates);
  checkCUDAError("allocate device memory for hNetUpdatesAboveD");

  cudaMalloc((void**)&hvNetUpdatesBelowD, sizeOfNetUpdates);
  checkCUDAError("allocate device memory for hvNetUpdatesBelowD");

  cudaMalloc((void**)&hvNetUpdatesAboveD, sizeOfNetUpdates);
  checkCUDAError("allocate device memory for hNetUpdatesAboveD");

  // Create the CUDA streams
  for(int i = 0;i<streamnum;i++) {
	  cudaStreamCreate(&stream[i]);
  }

  // Initialize the CUBLAS handles
  for(int i = 0;i<streamnum;i++) {
	  cublas_status = cublasCreate(&cu_handle[i]);
	  if(cublas_status != CUBLAS_STATUS_SUCCESS) {
		  if(cublas_status == CUBLAS_STATUS_NOT_INITIALIZED) printf("CUBLAS initialization failure: CUDA not initialized.\n");
		  else if(cublas_status == CUBLAS_STATUS_ALLOC_FAILED) printf("CUBLAS initialization failure:  could not allocate resources.\n");
		  else printf("CUBLAS initialization failure:  unspecified reason.\n");
	  }

	  cublas_status = cublasSetStream(cu_handle[i], stream[i]);
	  if(cublas_status != CUBLAS_STATUS_SUCCESS) {
		  printf("CUBLAS error:  could not map a CUDA stream to a CUBLAS handle.\n");
	  }
  }
}

/**
 * Destructor of a SWE_WavePropagationBlockCuda.
 *
 * Frees all of the memory, which was allocated within the constructor.
 * Resets the CUDA device: Useful if error occured and printf is used on the device (buffer).
 */
SWE_WavePropagationBlockCuda::~SWE_WavePropagationBlockCuda() {
  // free the net-updates memory
  cudaFree(hNetUpdatesLeftD);
  cudaFree(hNetUpdatesRightD);
  cudaFree(huNetUpdatesLeftD);
  cudaFree(huNetUpdatesRightD);

  cudaFree(hNetUpdatesBelowD);
  cudaFree(hNetUpdatesAboveD);
  cudaFree(hvNetUpdatesBelowD);
  cudaFree(hvNetUpdatesAboveD);

  // Free the cuda streams
  for(int i=0;i<streamnum;i++) {
	  cudaStreamDestroy(stream[i]);
  }

  // Free the CUBLAS handles
  for(int i=0;i<streamnum;i++) {
	  cublas_status = cublasDestroy(cu_handle[i]);
	  if(cublas_status != CUBLAS_STATUS_SUCCESS) {
		  printf("Could not destroy the CUBLAS handle.\n");
	  }
  }

  // reset the cuda device
  s_sweLogger.printString("Resetting the CUDA devices");
  cudaDeviceReset();
}

/**
 * Compute a single global time step of a given time step width.
 * Remark: The user has to take care about the time step width. No additional check is done. The time step width typically available
 *         after the computation of the numerical fluxes (hidden in this method).
 *
 * First the net-updates are computed.
 * Then the cells are updated with the net-updates and the given time step width.
 *
 * @param i_dT time step width in seconds.
 */
__host__
void SWE_WavePropagationBlockCuda::simulateTimestep(float i_dT) {
 // Compute the numerical fluxes/net-updates in the wave propagation formulation.
 computeNumericalFluxes();

 // Update the unknowns with the net-updates.
 updateUnknowns(i_dT);
}

/**
 * perform forward-Euler time steps, starting with simulation time tStart,:
 * until simulation time tEnd is reached;
 * device-global variables hd, hud, hvd are updated;
 * unknowns h, hu, hv in main memory are not updated.
 * Ghost layers and bathymetry sources are updated between timesteps.
 * intended as main simulation loop between two checkpoints
 */
__host__
float SWE_WavePropagationBlockCuda::simulate(float tStart, float tEnd) {
  float t = tStart;
  do {
     // set values in ghost cells:
     setGhostLayer();

     // Compute the numerical fluxes/net-updates in the wave propagation formulation.
     computeNumericalFluxes();

     // Update the unknowns with the net-updates.
     updateUnknowns(maxTimestep);

	 t += maxTimestep;
  } while(t < tEnd);

  return t;
}


/**
 * Compute the numerical fluxes (net-update formulation here) on all edges.
 *
 * The maximum wave speed is computed within the net-updates kernel for each CUDA-block.
 * To finalize the method the Thrust-library is called, which does the reduction over all blockwise maxima.
 *   In the wave speed reduction step the actual cell width in x- and y-direction is not taken into account.
 */
void SWE_WavePropagationBlockCuda::computeNumericalFluxes() {
  /*
   * Initialization.
   */

	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid(nx/TILE_SIZE, ny/TILE_SIZE);

  // "2D array" which holds the blockwise maximum wave speeds
  float* l_maximumWaveSpeedsD;

  // size of the maximum wave speed array (dimension of the grid + ghost layers, without the top right block), sizeof(float) not included
  int l_sizeMaxWaveSpeeds = ((dimGrid.x+1)*(dimGrid.y+1)-1);
  cudaMalloc((void**)&l_maximumWaveSpeedsD, (l_sizeMaxWaveSpeeds*sizeof(float)) );


  /*
   * Compute the net updates for the 'main part and the two 'boundary' parts.
   */


#ifdef TIMEKERNELS
struct timeval start_time;
struct timeval end_time;
long diff1 = 0;

// Run all four kernels 20 times, print average time of execution
for(int ii=0;ii<20;ii++) {
cudaDeviceSynchronize();
gettimeofday(&start_time, NULL);
#endif /* TIMEKERNELS */

	computeNetUpdatesKernel<<<dimGrid, dimBlock, 0 , stream[0]>>>(
		hd,
		hud,
		hvd,
		bd,
		hNetUpdatesLeftD,
		hNetUpdatesRightD,
		huNetUpdatesLeftD,
		huNetUpdatesRightD,
    		hNetUpdatesBelowD,
		hNetUpdatesAboveD,
		hvNetUpdatesBelowD,
		hvNetUpdatesAboveD,
		l_maximumWaveSpeedsD,
		nx,
		ny,
		0,
		0,
		0,
		0);

	dimBlock.x = TILE_SIZE; dimBlock.y = 1;
	dimGrid.x = nx/TILE_SIZE; dimGrid.y = 1;
	computeNetUpdatesKernel<<<dimGrid, dimBlock, 0, stream[1]>>>(
		hd,
		hud,
		hvd,
		bd,
		hNetUpdatesLeftD,
		hNetUpdatesRightD,
		huNetUpdatesLeftD,
		huNetUpdatesRightD,
    		hNetUpdatesBelowD,
		hNetUpdatesAboveD,
		hvNetUpdatesBelowD,
		hvNetUpdatesAboveD,
		l_maximumWaveSpeedsD,
		nx,
		ny,
		0,
		ny,
		0,
		ny/TILE_SIZE);

	dimBlock.x = 1; dimBlock.y = TILE_SIZE;
	dimGrid.x = 1; dimGrid.y = ny/TILE_SIZE;
	computeNetUpdatesKernel<<<dimGrid, dimBlock, 0, stream[2]>>>(
		hd,
		hud,
		hvd,
		bd,
		hNetUpdatesLeftD,
		hNetUpdatesRightD,
		huNetUpdatesLeftD,
		huNetUpdatesRightD,
    		hNetUpdatesBelowD,
		hNetUpdatesAboveD,
		hvNetUpdatesBelowD,
		hvNetUpdatesAboveD,
		l_maximumWaveSpeedsD,
		nx,
		ny,
		nx,
		0,
		nx/TILE_SIZE,
		0);


#ifdef TIMEKERNELS
cudaDeviceSynchronize();
gettimeofday(&end_time, NULL);
diff1 += ((int)end_time.tv_sec - (int)start_time.tv_sec)*1000000 + ((int)end_time.tv_usec - (int)start_time.tv_usec);
}
printf("computeNetUpdatesKernel: %ld s\n", diff1);
#endif /* TIMEKERNELS */

  /*
   * Finalize (max reduction of the maximumWaveSpeeds-array.)
   *
   * The Thrust library is used in this step.
   * An optional kernel could be written for the maximum reduction.
   */
  // Thrust pointer to the device array


  thrust::device_ptr<float> l_thrustDevicePointer(l_maximumWaveSpeedsD);
  //thrust::device_ptr<float> l_thrustDevicePointer = thrust::device_pointer_cast(l_maximumWaveSpeedsD);

  // use Thrusts max_element-function for the maximum reduction
  thrust::device_ptr<float> l_thrustDevicePointerMax = thrust::max_element(l_thrustDevicePointer, l_thrustDevicePointer+l_sizeMaxWaveSpeeds);

  // get the result from the device
  float l_maximumWaveSpeed = l_thrustDevicePointerMax[0];

  // free the max wave speeds array on the device
  cudaFree(l_maximumWaveSpeedsD);

  // set the maximum time step for this SWE_WavePropagationBlockCuda
  maxTimestep = std::min( dx/l_maximumWaveSpeed, dy/l_maximumWaveSpeed );

  // CFL = 0.5
  maxTimestep *= (float)0.4;

}
/**
 * Update the cells with a given global time step.
 *
 * @param i_deltaT time step size.
 */
void SWE_WavePropagationBlockCuda::updateUnknowns(const float i_deltaT) {
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid(nx/TILE_SIZE, ny/TILE_SIZE);

#ifdef TIMEKERNELS
struct timeval start_time;
struct timeval end_time;
long diff2 = 0;
long diff3 = 0;
long diff4 = 0;
for(int ii=0;ii<20;ii++) {
cudaDeviceSynchronize();
gettimeofday(&start_time, NULL);
#endif /* TIMEKERNELS */


	updateUnknownsKernel<<<dimGrid, dimBlock>>>(
		hNetUpdatesLeftD,
		hNetUpdatesRightD,
		huNetUpdatesLeftD,
		huNetUpdatesRightD,
		hNetUpdatesBelowD,
		hNetUpdatesAboveD,
    		hvNetUpdatesBelowD,
		hvNetUpdatesAboveD,
    		hd,
		hud,
		hvd,
		i_deltaT/dx,
		i_deltaT/dy,
   		nx,
		ny);


#ifdef TIMEKERNELS
cudaDeviceSynchronize();
gettimeofday(&end_time, NULL);
diff2 += ((int)end_time.tv_sec - (int)start_time.tv_sec)*1000000 + ((int)end_time.tv_usec - (int)start_time.tv_usec);
gettimeofday(&start_time, NULL);
#endif /* TIMEKERNELS */

/*
	updateUnknownsCUBLAS(
		hNetUpdatesLeftD,
		hNetUpdatesRightD,
		huNetUpdatesLeftD,
		huNetUpdatesRightD,
		hNetUpdatesBelowD,
		hNetUpdatesAboveD,
		hvNetUpdatesBelowD,
		hvNetUpdatesAboveD,
		hd,
		hud,
		hvd,
		i_deltaT/dx,
		i_deltaT/dy,
		nx,
		ny,
		cu_handle);
*/

#ifdef TIMEKERNELS
cudaDeviceSynchronize();
gettimeofday(&end_time, NULL);
diff3 += ((int)end_time.tv_sec - (int)start_time.tv_sec)*1000000 + ((int)end_time.tv_usec - (int)start_time.tv_usec);
gettimeofday(&start_time, NULL);
#endif /* TIMEKERNELS */

/*
	updateUnknownsCUBLASOld(
		hNetUpdatesLeftD,
		hNetUpdatesRightD,
		huNetUpdatesLeftD,
		huNetUpdatesRightD,
		hNetUpdatesBelowD,
		hNetUpdatesAboveD,
		hvNetUpdatesBelowD,
		hvNetUpdatesAboveD,
		hd,
		hud,
		hvd,
		i_deltaT/dx,
		i_deltaT/dy,
		nx,
		ny,
		cu_handle);
/*

#ifdef TIMEKERNELS
cudaDeviceSynchronize();
gettimeofday(&end_time, NULL);
diff4 += ((int)end_time.tv_sec - (int)start_time.tv_sec)*1000000 + ((int)end_time.tv_usec - (int)start_time.tv_usec);
}
printf("updateUnknownsKernel: %ld \nupdateUnknownsCUBLAS: %ld\nupdateUnknownsCUBLASOld: %ld\n\n", diff2, diff3, diff4);
#endif /* TIMEKERNELS */


  // synchronize the copy layer for MPI communication
  #ifdef USEMPI
  synchCopyLayerBeforeRead();
  #endif
}
