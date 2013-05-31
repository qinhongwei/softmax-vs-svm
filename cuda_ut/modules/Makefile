###############################################################################
# CT 7.2011
# modules build script
#
###############################################################################

###############################################################################

# add additional include and lib, specific for these mex files
INCLUDEDIR = 
INCLUDELIB = -lCudaUT

# additional compiler flags
# e.g. NVCCFLAGS = -arch=sm_20
# e.g. make it faster  -use_fast_math
#				       -ftz=true
#				       -prec-div=false	 				
#				       -prec-sqrt=false
# add this: -DCLMATCUDANOSAFE flag to not do error checking during runtime
NVCCFLAGS = -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false
#NVCCFLAGS =

# List the mex files to be built.  The .mex extension will be replaced with the
# appropriate extension for this installation of MATLAB, e.g. .mexglx or
# .mexa64.

MEXFILES = $(mexf)

include ../nvmex.common.mk

