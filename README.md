SWE
===

Shallow Water Equations teaching code.

Implementation Notes
--------------------
* Kernel timing can be enabled by uncommenting a #define statement in ./src/WE_WavePropagationBlockCuda.cu

* Returned timing data is aggregate (not average) time over 20 executions of a kernel, measured in microseconds.  These timings wrap around kernel executions and include the overhead of initializing the kernels

* Timing mode breaks the simulation, since it runs many implementations of the same kernel

* Users on the NPS mathgpu cluster will need to modify their $PATH:

        PATH=$PATH:$HOME/bin:/tmp/software_SWE/software/scons/scons-2.1.0/bin/:/home/student27/sanchez/Downloads/pdtoolkit-3.18/x86_64/bin
        LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/software_SWE/software/netcdf/lib/:/tmp/software_SWE/software/asagi/nompi/lib/
        export PATH
        export LD_LIBRARY_PATH

* Once configured, the project can be built from the SWE directory by executing:
* 
        scons buildVariablesFile=build/options/npsgpu_SWE_gnu_cuda_asagi.py

* This default build must reference a netcdf file for writing;
* 
        ./build/SWE_gnu_release_cuda_fwave 6400 6400 ./build/output.nc