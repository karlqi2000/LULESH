README for LULESH 2.0 DPCPP version.

More information on LULESH https://codesign.llnl.gov/lulesh.php

Original code is http://github.com/LLNL/LULESH.git

Version here is migrated by the Intel(r) DPCPP Compatibility Tool from 2.0.2-dev branch's cuda version.

Using Intel(r) oneAPI base toolkit gold release 2021.1.0.




Running   lulesh -s 30 (or other values)  may cause problems with older level zero GPU back end and CPU backend..
May consider using OpenCL  (GPU) backend if problems arise.
set environment variables SYCL_BE=PI_OPENCL   and     SYCL_DEVICE_TYPE=gpu
