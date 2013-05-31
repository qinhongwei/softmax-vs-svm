These files demonstrate using the CUDA accelerated FFT libraries and compiling
and accessing user CUDA code from MATLAB.

This has been tested using MATLAB 7.3.0 (R2006b) and MATLAB 7.5.0 (R2007b).

---- Setup ----

Install MATLAB and CUDA.  An installation into /usr/local/matlab and
/usr/local/cuda respectively are assumed but can be changed in 'Makefile'.


---- Run native MATLAB simulations ----

>> ls
fft2_cuda.c	   FS_2Dturb.m	 Makefile   README.txt	 Szeta.m
fft2_cuda_sp_dp.c  FS_vortex.m	 nvmex	    speed_fft.m
FS_2Dflow.pdf	   ifft2_cuda.c  nvopts.sh  Szeta.cu

>> which Szeta
/home/cuda/CUDA/Szeta.m
>> tic; FS_2Dturb(128,1,1,1); toc;

CFL =

    0.1017


Gsqav =

    1.1995

Elapsed time is 8.506012 seconds.
>> tic; FS_vortex; toc;

ans =

   512

Elapsed time is 216.061310 seconds.


---- Compile the CUDA files and test the CUFFT interface ----

>> ls
fft2_cuda.c        FS_2Dturb.m   Makefile   README.txt   Szeta.m
fft2_cuda_sp_dp.c  FS_vortex.m   nvmex      speed_fft.m
FS_2Dflow.pdf      ifft2_cuda.c  nvopts.sh  Szeta.cu

>> unix('make');
/usr/local/matlab/bin/mex CFLAGS='-fPIC -D_GNU_SOURCE -pthread -fexceptions'
COPTIMFLAGS='-O3 -funroll-loops -msse2' fft2_cuda.c \
        -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcufft
-Wl,-rpath,/usr/local/cuda/lib
/usr/local/matlab/bin/mex CFLAGS='-fPIC -D_GNU_SOURCE -pthread -fexceptions'
COPTIMFLAGS='-O3 -funroll-loops -msse2' fft2_cuda_sp_dp.c \
        -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcufft
-Wl,-rpath,/usr/local/cuda/lib
/usr/local/matlab/bin/mex CFLAGS='-fPIC -D_GNU_SOURCE -pthread -fexceptions'
COPTIMFLAGS='-O3 -funroll-loops -msse2' ifft2_cuda.c \
        -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcufft
-Wl,-rpath,/usr/local/cuda/lib
./nvmex -f nvopts.sh Szeta.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib
-lcufft -Wl,-rpath,/usr/local/cuda/lib
>> ls
fft2_cuda.c		FS_2Dturb.m	   nvmex	Szeta.m
fft2_cuda.mexa64	FS_vortex.m	   nvopts.sh	Szeta.mexa64
fft2_cuda_sp_dp.c	ifft2_cuda.c	   README.txt
fft2_cuda_sp_dp.mexa64	ifft2_cuda.mexa64  speed_fft.m
FS_2Dflow.pdf		Makefile	   Szeta.cu

>> speed_fft


[Note: The extension for MATLAB mex files will be .mexglx for 32-bit
installations and .mexa64 for 64-bit installations.]


---- Rerun the simulations with acceleration ----

>> which Szeta
/home/cuda/CUDA/Szeta.mexa64
>> tic; FS_2Dturb(128,1,1,1); toc;

CFL =

    0.1017


Gsqav =

    1.1995

Elapsed time is 2.228646 seconds.
>> tic; FS_vortex; toc;

ans =

   512

Elapsed time is 15.164892 seconds.




MATLAB scripts available for download from:
  http://www.amath.washington.edu/courses/571-winter-2006/matlab.html
  Professor Chris Bretherton, Atmospheric Science Department, University of Washington

Last modified: 6/26/2007
