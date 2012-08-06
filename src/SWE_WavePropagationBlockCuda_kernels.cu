/**
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
 * CUDA Kernels for a SWE_Block, which uses solvers in the wave propagation formulation.
 */

#include "SWE_BlockCUDA.hh"
#include "SWE_WavePropagationBlockCuda_kernels.hh"

#include <cmath>
#include <cstdio>

#include "solvers/FWaveCuda.h"

#define MAX(a,b) (a > b ? a : b)
#define G 9.81

/**
 * The compute net-updates kernel calls the solver for a defined CUDA-Block and does a reduction over the computed wave speeds within this block.
 *
 * Remark: In overall we have nx+1 / ny+1 edges.
 *         Therefore the edges "simulation domain"/"top ghost layer" and "simulation domain"/"right ghost layer"
 *         will not be computed in a typical call of the function:
 *           computeNetUpdatesKernel<<<dimGrid,dimBlock>>>( hd, hud, hvd, bd,
 *                                                          hNetUpdatesLeftD,  hNetUpdatesRightD,
 *                                                          huNetUpdatesLeftD, huNetUpdatesRightD,
 *                                                          hNetUpdatesBelowD, hNetUpdatesAboveD,
 *                                                          hvNetUpdatesBelowD, hvNetUpdatesAboveD,
 *                                                          l_maximumWaveSpeedsD,
 *                                                          i_nx, i_ny
 *                                                        );
 *         To reduce the effect of branch-mispredictions the kernel provides optional offsets, which can be used to compute the missing edges.
 *
 *
 * @param i_h water heights (CUDA-array).
 * @param i_hu momentums in x-direction (CUDA-array).
 * @param i_hv momentums in y-direction (CUDA-array).
 * @param i_b bathymetry values (CUDA-array).
 * @param o_hNetUpdatesLeftD left going net-updates for the water height (CUDA-array).
 * @param o_hNetUpdatesRightD right going net-updates for the water height (CUDA-array).
 * @param o_huNetUpdatesLeftD left going net-updates for the momentum in x-direction (CUDA-array).
 * @param o_huNetUpdatesRightD right going net-updates for the momentum in x-direction (CUDA-array).
 * @param o_hNetUpdatesBelowD downwards going net-updates for the water height (CUDA-array).
 * @param o_hNetUpdatesAboveD upwards going net-updates for the water height (CUDA-array).
 * @param o_hvNetUpdatesBelowD downwards going net-updates for the momentum in y-direction (CUDA-array).
 * @param o_hvNetUpdatesAboveD upwards going net-updates for the momentum in y-direction (CUDA-array).
 * @param o_maximumWaveSpeeds maximum wave speed which occurred within the CUDA-block is written here (CUDA-array).
 * @param i_nx number of cells within the simulation domain in x-direction (excludes ghost layers).
 * @param i_ny number of cells within the simulation domain in y-direction (excludes ghost layers).
 * @param i_offsetX cell/edge offset in x-direction.
 * @param i_offsetY cell/edge offset in y-direction.
 */
__global__
void computeNetUpdatesKernel(
    const float* i_h, const float* i_hu, const float* i_hv, const float* i_b,
    float* o_hNetUpdatesLeftD,   float* o_hNetUpdatesRightD,
    float* o_huNetUpdatesLeftD,  float* o_huNetUpdatesRightD,
    float* o_hNetUpdatesBelowD,  float* o_hNetUpdatesAboveD,
    float* o_hvNetUpdatesBelowD, float* o_hvNetUpdatesAboveD,
    float* o_maximumWaveSpeeds,
    const int i_nX, const int i_nY,
    const int i_offsetX, const int i_offsetY,
    const int i_blockOffsetX, const int i_blockOffsetY
) {
	// Recover absolute array indices from CUDA thread constants
	int i = blockIdx.x * TILE_SIZE + threadIdx.x;
	int j = blockIdx.y * TILE_SIZE + threadIdx.y;

	// T is from ./solvers/FWaveCuda.h
	// this implements "typedef float T;"  by default
	T netUpdates[5];

	
	// computeOneDPo...(arg0,arg1,arg2) = arg0*arg2 + arg1
	int oneDPosition = computeOneDPositionKernel(i + i_offsetX + i_blockOffsetX,
		 j + i_offsetY + i_blockOffsetY,
		 i_nY + 2);
	int maxWaveSpeedPos = computeOneDPositionKernel(i, j, i_nY+1);

	// Returns the values of net updates in T netUpdates[5] with values
	// corresponding to ("h_left","h_right","hu_left","hu_right","MaxWaveSpeed").
	fWaveComputeNetUpdates(G,
		i_h[oneDPosition - i_nY - 2],
		i_h[oneDPosition],
		i_hu[oneDPosition - i_nY - 2],
		i_hu[oneDPosition],
		i_b[oneDPosition - i_nY - 2],
		i_b[oneDPosition],
		netUpdates);

	o_hNetUpdatesLeftD[oneDPosition - i_nY - 2] = netUpdates[0];
	o_hNetUpdatesRightD[oneDPosition] = netUpdates[1];
	o_huNetUpdatesLeftD[oneDPosition - i_nY - 2] = netUpdates[2];
	o_huNetUpdatesRightD[oneDPosition] = netUpdates[3];
	o_maximumWaveSpeeds[maxWaveSpeedPos] = netUpdates[4];

	fWaveComputeNetUpdates(G,
		i_h[oneDPosition - 1],
		i_h[oneDPosition],
		i_hv[oneDPosition - 1],
		i_hv[oneDPosition],
		i_b[oneDPosition - 1],
		i_b[oneDPosition],
		netUpdates);

	o_hNetUpdatesBelowD[oneDPosition - 1] = netUpdates[0];
	o_hNetUpdatesAboveD[oneDPosition] = netUpdates[1];
	o_hvNetUpdatesBelowD[oneDPosition - 1] = netUpdates[2];
	o_hvNetUpdatesAboveD[oneDPosition] = netUpdates[3];
	o_maximumWaveSpeeds[maxWaveSpeedPos] = MAX(o_maximumWaveSpeeds[maxWaveSpeedPos], netUpdates[4]);
}

/**
 * The "update unknowns"-kernel updates the unknowns in the cells with precomputed net-updates.
 *
 * @param i_hNetUpdatesLeftD left going net-updates for the water height (CUDA-array).
 * @param i_hNetUpdatesRightD right going net-updates for the water height (CUDA-array).
 * @param i_huNetUpdatesLeftD left going net-updates for the momentum in x-direction (CUDA-array).
 * @param i_huNetUpdatesRightD right going net-updates for the momentum in x-direction (CUDA-array).
 * @param i_hNetUpdatesBelowD downwards going net-updates for the water height (CUDA-array).
 * @param i_hNetUpdatesAboveD upwards going net-updates for the water height (CUDA-array).
 * @param i_hvNetUpdatesBelowD downwards going net-updates for the momentum in y-direction (CUDA-array).
 * @param i_hvNetUpdatesAboveD upwards going net-updates for the momentum in y-direction (CUDA-array).
 * @param io_h water heights (CUDA-array).
 * @param io_hu momentums in x-direction (CUDA-array).
 * @param io_hv momentums in y-direction (CUDA-array).
 * @param i_updateWidthX update width in x-direction.
 * @param i_updateWidthY update width in y-direction.
 * @param i_nx number of cells within the simulation domain in x-direction (excludes ghost layers).
 * @param i_ny number of cells within the simulation domain in y-direction (excludes ghost layers).
 */
__global__
void updateUnknownsKernel(
    const float* i_hNetUpdatesLeftD,   const float* i_hNetUpdatesRightD,
    const float* i_huNetUpdatesLeftD,  const float* i_huNetUpdatesRightD,
    const float* i_hNetUpdatesBelowD,  const float* i_hNetUpdatesAboveD,
    const float* i_hvNetUpdatesBelowD, const float* i_hvNetUpdatesAboveD,
    float* io_h, float* io_hu, float* io_hv,
    const float i_updateWidthX, const float i_updateWidthY,
    const int i_nX, const int i_nY )
{
	int i = blockIdx.x * TILE_SIZE + threadIdx.x;
	int j = blockIdx.y * TILE_SIZE + threadIdx.y;

	// TODO I think we also need the blockOffset here ...
	int oneDPosition = computeOneDPositionKernel(i, j, i_nY + 2);

	io_h[oneDPosition] -= i_updateWidthX * (i_hNetUpdatesRightD[oneDPosition - i_nY - 2]
			+ i_hNetUpdatesLeftD[oneDPosition])
		+ i_updateWidthY * (i_hNetUpdatesAboveD[oneDPosition - i_nY - 2]
			+ i_hNetUpdatesBelowD[oneDPosition]);
	io_hu[oneDPosition] -= i_updateWidthX * (i_huNetUpdatesRightD[oneDPosition - i_nY - 2]
		+ i_huNetUpdatesLeftD[oneDPosition]);
	io_hv[oneDPosition] -= i_updateWidthY * (i_hvNetUpdatesAboveD[oneDPosition - i_nY - 2]
		+ i_hvNetUpdatesBelowD[oneDPosition]);
}

/**
 * Compute the position of 2D coordinates in a 1D array.
 *   array[i][j] -> i * ny + j
 *
 * @param i_i row index.
 * @param i_j column index.
 * @param i_ny #(cells in y-direction).
 * @return 1D index.
 */
__device__
inline int computeOneDPositionKernel(const int i_i, const int i_j, const int i_ny) {
  return i_i*i_ny + i_j;
}
