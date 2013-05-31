# Define installation location for CUDA and compilation flags compatible
# with the CUDA include files.
CUDAHOME    = /usr/local/cuda
INCLUDEDIR  = -I$(CUDAHOME)/include
INCLUDELIB  = -L$(CUDAHOME)/lib -lcufft -Wl,-rpath,$(CUDAHOME)/lib
CFLAGS      = -fPIC -D_GNU_SOURCE -pthread -fexceptions
COPTIMFLAGS = -O3 -funroll-loops -msse2

# Define installation location for MATLAB.
export MATLAB = /usr/local/matlab
#export MATLAB = /Applications/MATLAB_R2007b
MEX           = $(MATLAB)/bin/mex
MEXEXT        = .$(shell $(MATLAB)/bin/mexext)

# nvmex is a modified mex script that knows how to handle CUDA .cu files.
NVMEX = ./nvmex

# List the mex files to be built.  The .mex extension will be replaced with the
# appropriate extension for this installation of MATLAB, e.g. .mexglx or
# .mexa64.
MEXFILES = fft2_cuda.mex       \
           fft2_cuda_sp_dp.mex \
           ifft2_cuda.mex      \
           Szeta.mex

all: $(MEXFILES:.mex=$(MEXEXT))

clean:
	rm -f $(MEXFILES:.mex=$(MEXEXT))

.SUFFIXES: .cu .cu_o .mexglx .mexa64 .mexmaci

.c.mexglx:
	$(MEX) CFLAGS='$(CFLAGS)' COPTIMFLAGS='$(COPTIMFLAGS)' $< \
        $(INCLUDEDIR) $(INCLUDELIB)

.cu.mexglx:
	$(NVMEX) -f nvopts.sh $< $(INCLUDEDIR) $(INCLUDELIB)

.c.mexa64:
	$(MEX) CFLAGS='$(CFLAGS)' COPTIMFLAGS='$(COPTIMFLAGS)' $< \
        $(INCLUDEDIR) $(INCLUDELIB)

.cu.mexa64:
	$(NVMEX) -f nvopts.sh $< $(INCLUDEDIR) $(INCLUDELIB)

.c.mexmaci:
	$(MEX) CFLAGS='$(CFLAGS)' COPTIMFLAGS='$(COPTIMFLAGS)' $< \
        $(INCLUDEDIR) $(INCLUDELIB)

.cu.mexmaci:
	$(NVMEX) -f nvopts.sh $< $(INCLUDEDIR) $(INCLUDELIB)
