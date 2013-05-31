# CT 7/2011 based on Makefile from Matlab_Cuda_1.1
# https://developer.nvidia.com/matlab-cuda
# this script only compiles .c and .cu files, using nvcc, see nvmex
#
# Define installation location for CUDA and compilation flags compatible
# with the CUDA include files.
CUDAHOME    = /u/tang/local/cuda/cuda-5.0
CUDASDKHOME = /u/tang/local/cuda/cuda-5.0/samples
MEXCUDAHOME = ../../Matlab_Cuda_1.1
CUDA_UT_HOME = ../

INCLUDEDIR  += 	-I$(CUDAHOME)/include -I$(CUDASDKHOME)/common/inc -I$(CUDA_UT_HOME)/include \

INCLUDELIB  += -L$(CUDAHOME)/lib64 -Wl,-rpath,$(CUDAHOME)/lib64 \
				-L$(CUDASDKHOME)/common/lib \
				-L$(CUDA_UT_HOME)/lib -Wl,-rpath,$(CUDA_UT_HOME)/lib

CFLAGS      += -fPIC -D_GNU_SOURCE -pthread -fexceptions
COPTIMFLAGS += -O3 -funroll-loops -msse2

# Define installation location for MATLAB.
export MATLAB = /pkgs/matlab-80

MEX           = $(MATLAB)/bin/mex
MEXEXT        = .$(shell $(MATLAB)/bin/mexext)

# nvmex is a modified mex script that knows how to handle CUDA .cu files.
NVMEX = $(MEXCUDAHOME)/nvmex
OUTBINDIR = $(CUDA_UT_HOME)/bin

INCLUDELIB += -lcublas -lcurand -lcudart
#INCLUDELIB += -lcufft
#INCLUDELIB += -lnpp
#INCLUDELIB += -lcutil_x86_64

# NVCCFLAGS += -gencode=arch=compute_13,code=\"sm_13,compute_13\"
# NVCCFLAGS += -gencode=arch=compute_20,code=\"sm_20,compute_20\"

ifeq ($(USECULA),1)
  INCLUDEDIR += -I$(CULAHOME)/include
  INCLUDELIB += -lcula_core -lcula_lapack -L$(CULAHOME)/lib64 -Wl,-rpath,$(CULAHOME)/lib64  
endif

ifeq ($(USEBOOST),1)
	INCLUDEDIR += -I/u/tang/local/boost_1_48_0
endif

ifeq ($(DOUBLE),1)
	NVCCFLAGS += -DDOUBLE_PRECISION
	DBL_EXT=_dbl
else
	DBL_EXT=
endif

# CT 7/2011 this is defined in individual Makefiles in subfolders
# List the mex files to be built.  The .mex extension will be replaced with the
# appropriate extension for this installation of MATLAB, e.g. .mexglx or
# .mexa64.
#MEXFILES = fft2_cuda.mex       \
#           fft2_cuda_sp_dp.mex \
#           ifft2_cuda.mex      \
#           Szeta.mex

############################################################
# costume options
#  add this: -DCLMATCUDANOSAFE flag to not do error checking during runtime


all: $(MEXFILES:.mex=$(MEXEXT)) 

clean:
	rm -f $(MEXFILES:.mex=$(MEXEXT))

.SUFFIXES: .cu .cu_o .mexglx .mexa64 .mexmaci

.c.mexglx:
	$(MEX) CFLAGS='$(CFLAGS)' COPTIMFLAGS='$(COPTIMFLAGS)' $< \
        $(INCLUDEDIR) $(INCLUDELIB)

.cu.mexglx:
	$(NVMEX) -f $(MEXCUDAHOME)/nvopts.sh $< $(INCLUDEDIR) $(INCLUDELIB) \
	verbose=1 OUTDIR='$(OUTBINDIR)' user_flags='$(NVCCFLAGS)'

.c.mexa64:
	$(MEX) CFLAGS='$(CFLAGS)' COPTIMFLAGS='$(COPTIMFLAGS)' $< \
        $(INCLUDEDIR) $(INCLUDELIB)

%.mexa64: %.cu
	$(NVMEX) -f $(MEXCUDAHOME)/nvopts.sh $< $(INCLUDEDIR) $(INCLUDELIB) \
	verbose=1 OUTDIR='$(OUTBINDIR)' user_flags='$(NVCCFLAGS)' -o $(patsubst %.mexa64,%$(DBL_EXT),$@) 

.c.mexmaci:
	$(MEX) CFLAGS='$(CFLAGS)' COPTIMFLAGS='$(COPTIMFLAGS)' $< \
        $(INCLUDEDIR) $(INCLUDELIB)

.cu.mexmaci:
	$(NVMEX) -f $(MEXCUDAHOME)/nvopts.sh $< $(INCLUDEDIR) $(INCLUDELIB) \
	verbose=1 OUTDIR='$(OUTBINDIR)' user_flags='$(NVCCFLAGS)'


