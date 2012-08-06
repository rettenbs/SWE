/**
 * @file
 * This file is part of SWE.
 *
 * @author Alexander Breuer (breuera AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer)
 *         Michael Bader (bader AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Univ.-Prof._Dr._Michael_Bader)
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
 * Basic setting of SWE, which uses a wave propagation solver and an artificial or ASAGI scenario on a single block.
 */

#include <cassert>
#include <cstdlib>
#include <string>
#include <iostream>
#include "../tools/help.hh"

#include "../SWE_Block.hh"

#ifndef CUDA
#include "../SWE_WavePropagationBlock.hh"
#else
#include "../SWE_WavePropagationBlockCuda.hh"
#endif

#ifdef WRITENETCDF
#include "../tools/NetCdfWriter.hh"
#endif

#ifdef ASAGI
#include "../scenarios/SWE_AsagiScenario.hpp"
#else
#include "../scenarios/SWE_simple_scenarios.h"
#endif

#ifdef READXML
#include "../tools/CXMLConfig.hpp"
#endif

#ifndef STATICLOGGER
#define STATICLOGGER
#include "../tools/Logger.hpp"
static tools::Logger s_sweLogger;
#endif

/**
 * Main program for the simulation on a single SWE_WavePropagationBlock.
 */
int main( int argc, char** argv ) {
  /**
   * Initialization.
   */
  // check if the necessary command line input parameters are given
  #ifndef READXML
  if(argc != 4) {
    std::cout << "Aborting .. please provide proper input parameters." << std::endl
              << "Example: ./SWE_parallel 200 300 /work/openmp_out" << std::endl
              << "\tfor a single block of size 200 * 300" << std::endl;
    return 1;
  }
  #endif

  //! number of grid cells in x- and y-direction.
  int l_nX, l_nY;

  //! l_baseName of the plots.
  std::string l_baseName;

  // read command line parameters
  #ifndef READXML
  l_nY = l_nX = atoi(argv[1]);
  l_nY = atoi(argv[2]);
  l_baseName = std::string(argv[3]);
  #endif

  // read xml file
  #ifdef READXML
  assert(false); //TODO: not implemented.
  if(argc != 2) {
    s_sweLogger.printString("Aborting. Please provide a proper input file.");
    s_sweLogger.printString("Example: ./SWE_gnu_debug_none_augrie config.xml");
    return 1;
  }
  s_sweLogger.printString("Reading xml-file.");

  std::string l_xmlFile = std::string(argv[1]);
  s_sweLogger.printString(l_xmlFile);

  CXMLConfig l_xmlConfig;
  l_xmlConfig.loadConfig(l_xmlFile.c_str());
  #endif

  #ifdef ASAGI
  /* Information about the example bathymetry grid (tohoku_gebco_ucsb3_500m_hawaii_bath.nc):
   *
   * Pixel node registration used [Cartesian grid]
   * Grid file format: nf = GMT netCDF format (float)  (COARDS-compliant)
   * x_min: -500000 x_max: 6500000 x_inc: 500 name: x nx: 14000
   * y_min: -2500000 y_max: 1500000 y_inc: 500 name: y ny: 8000
   * z_min: -6.48760175705 z_max: 16.1780223846 name: z
   * scale_factor: 1 add_offset: 0
   * mean: 0.00217145586762 stdev: 0.245563641735 rms: 0.245573241263
   */

  //simulation area
  float simulationArea[4];
  simulationArea[0] = -450000;
  simulationArea[1] = 6450000;
  simulationArea[2] = -2450000;
  simulationArea[3] = 1450000;

  SWE_AsagiScenario l_scenario( "/work/breuera/workspace/geo_information/output/tohoku_gebco_ucsb3_500m_hawaii_bath.nc",
                                "/work/breuera/workspace/geo_information/output/tohoku_gebco_ucsb3_500m_hawaii_displ.nc",
                                (float) 28800., simulationArea);
  #else
  // create a simple artificial scenario
  SWE_BathymetryDamBreakScenario l_scenario;
  #endif

  //! number of checkpoints for visualization (at each checkpoint in time, an output file is written).
  int l_numberOfCheckPoints = 20;

  //! size of a single cell in x- and y-direction
  float l_dX, l_dY;

  // compute the size of a single cell
  l_dX = (l_scenario.getBoundaryPos(BND_RIGHT) - l_scenario.getBoundaryPos(BND_LEFT) )/l_nX;
  l_dY = (l_scenario.getBoundaryPos(BND_TOP) - l_scenario.getBoundaryPos(BND_BOTTOM) )/l_nY;

  // initialize the grid data and the corresponding static variables
  SWE_Block::initGridData(l_nX,l_nY,l_dX,l_dY);

  //! origin of the simulation domain in x- and y-direction
  float l_originX, l_originY;

  // get the origin from the scenario
  l_originX = l_scenario.getBoundaryPos(BND_LEFT);
  l_originY = l_scenario.getBoundaryPos(BND_BOTTOM);

  // create a single wave propagation block
  #ifndef CUDA
  SWE_WavePropagationBlock l_wavePropgationBlock(l_originX, l_originY);
  #else
  SWE_WavePropagationBlockCuda l_wavePropgationBlock(l_originX, l_originY);
  #endif

  // initialize the wave propgation block
  l_wavePropgationBlock.initScenario(l_scenario);


  //! time when the simulation ends.
  float l_endSimulation = l_scenario.endSimulation();

  //! checkpoints when output files are written.
  float* l_checkPoints = new float[l_numberOfCheckPoints+1];

  // compute the checkpoints in time
  for(int cp = 0; cp <= l_numberOfCheckPoints; cp++) {
     l_checkPoints[cp] = cp*(l_endSimulation/l_numberOfCheckPoints);
  }


  // write the output at time zero
  s_sweLogger.printOutputTime((float) 0.);
  #ifdef WRITENETCDF
  //boundary size of the ghost layers
  int l_boundarySize[4];
  l_boundarySize[0] = l_boundarySize[1] = l_boundarySize[2] = l_boundarySize[3] = 1;

  //construct a NetCdfWriter
  std::string l_fileName = l_baseName;
  io::NetCdfWriter l_netCdfWriter( l_fileName, l_nX, l_nY );
  //create the netCDF-file
  l_netCdfWriter.createNetCdfFile(l_dX, l_dY, l_originX, l_originY);
  l_netCdfWriter.writeBathymetry(l_wavePropgationBlock.getBathymetry(), l_boundarySize);
  l_netCdfWriter.writeUnknowns( l_wavePropgationBlock.getWaterHeight(),
                                l_wavePropgationBlock.getDischarge_hu(),
                                l_wavePropgationBlock.getDischarge_hv(),
                                l_boundarySize, (float) 0.);
  #else
  l_wavePropgationBlock.writeVTKFileXML(generateFileName(l_baseName,0,0,0), l_nX, l_nY);
  #endif


  /**
   * Simulation.
   */
  // print the start message and reset the wall clock time
  s_sweLogger.printStartMessage();
  s_sweLogger.initWallClockTime(time(NULL));

  //! simulation time.
  float l_t = 0.0;

  // loop over checkpoints
  for(int c=1; c<=l_numberOfCheckPoints; c++) {
    // reset the cpu clock
    s_sweLogger.resetCpuClockToCurrentTime();

    // do time steps until next checkpoint is reached
    while( l_t < l_checkPoints[c] ) {
      // set values in ghost cells:
      l_wavePropgationBlock.setGhostLayer();
      
      // approximate the maximum time step
      // TODO: This calculation should be replaced by the usage of the wave speeds occuring during the flux computation
      // Remark: The code is executed on the CPU, therefore a "valid result" depends on the CPU-GPU-synchronization.
//      l_wavePropgationBlock.computeMaxTimestep();

      // compute numerical flux on each edge
      l_wavePropgationBlock.computeNumericalFluxes();

      //! maximum allowed time step width.
      float l_maxTimeStepWidth = l_wavePropgationBlock.getMaxTimestep();

      // update the cell values
      l_wavePropgationBlock.updateUnknowns(l_maxTimeStepWidth);

      // update simulation time with time step width.
      l_t += l_maxTimeStepWidth;

      // print the current simulation time
      s_sweLogger.printSimulationTime(l_t);
    }

    // update the cpu time in the logger
    s_sweLogger.updateCpuTime();

    // print current simulation time of the output
    s_sweLogger.printOutputTime(l_t);

    // write output
#ifdef WRITENETCDF
    l_netCdfWriter.writeUnknowns( l_wavePropgationBlock.getWaterHeight(),
                                  l_wavePropgationBlock.getDischarge_hu(),
                                  l_wavePropgationBlock.getDischarge_hv(),
                                  l_boundarySize, l_t);
#else
    l_wavePropgationBlock.writeVTKFileXML(generateFileName(l_baseName,c,0,0), l_nX, l_nY);
#endif
  }

  /**
   * Finalize.
   */
  // write the statistics message
  s_sweLogger.printStatisticsMessage();

  // print the cpu time
  s_sweLogger.printCpuTime();

  // print the wall clock time (includes plotting)
  s_sweLogger.printWallClockTime(time(NULL));

  return 0;
}
