#ifndef _CU_UTIL_H_
#define _CU_UTIL_H_

//this is a general purpose include header for all .cu files

#include "stdio.h"
#include "cublas_v2.h"
#include <vector>
/*
Copyright (C) 2013 Yichuan Tang. 
contact: tang at cs.toronto.edu
http://www.cs.toronto.edu/~tang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cstdlib>
#include <stdint.h>


#define IN
#define OUT
#define CL_MALLOC
#define CL_PREMALLOC
#define OPTIONAL

#define MAX_GRIDS 			65535
#define MAX_THREADS 		512
#define MEDIUM_NUM_THREADS 	256
#define BLOCK_DIM			16
#define MAX_FLOAT_EXP 		80
#define FLOAT_EXP_SATURATE	50    //above this number we consider a sigmoid to be 1.0f, below -50 we consider sigmoid output of 0.0f

#define MAX_DOUBLE_EXP 		300
#define DOUBLE_EXP_SATURATE	80   //above this number we consider a sigmoid to be 1.0f, below -100 we consider sigmoid output of 0.0f



#ifndef MIN
#define MIN(a,b) ((a) < (b) ?  (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ?  (a) : (b))
#endif


//if we are compiling a mex file, use mxASSERT instead of the default cASSERT
#ifdef MATLAB_MEX_FILE

	#include "mex_util.h"

	inline void clASSERT(bool b, const char* msg){
		mxASSERT(b, msg);
	}

	inline void clPrintf( char* msg ){
		mexPrintf( msg );
		mexEvalString("drawnow;");
	}

#else
	inline void clASSERT(bool b, const char* msg){

		if (!b){
			char errmsg[50];
			sprintf( errmsg, "\nAssertion Failed:%s\n", msg);
			printf( "%s", errmsg );
			exit(-1);
		}
	}

	inline void clPrintf( char* msg ){
		printf( "%s", msg );
	}

	//to make sure error code returned by cuda function calls are 0
	#ifndef clCheckErr
	#define clCheckErr( err ) \
		{ int my_err_code = (err); \
		if ( my_err_code != 0){ \
			char errmsg[100];\
			sprintf(errmsg, "CUDA function call failed![%d] %s:%d", my_err_code, __FILE__, __LINE__);\
			clPrintf( errmsg ); \
			exit(-1); \
		} }
	#endif
#endif

// short form for clCheckErr
#define cr( err ) \
	clCheckErr(err)




inline bool cublasCheckError(const cublasStatus_t & stat, char* message)
{
	if (stat != CUBLAS_STATUS_SUCCESS){
		clPrintf(message);
		return false;
	}
	return true;
}

inline bool cublas_cr(const cublasStatus_t & stat, char* message)
{
	if (stat != CUBLAS_STATUS_SUCCESS){
		clPrintf(message);
		return false;
	}
	return true;
}

inline bool cublas_cr(const cublasStatus_t & stat )
{
	if (stat != CUBLAS_STATUS_SUCCESS){
		clPrintf("cublas Function Failed!");
		return false;
	}
	return true;
}


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }



// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

void DisplayGPUMemory(int);
void cuTic();
float cuToc();

ptrdiff_t randperm_myrandom (ptrdiff_t i);

void randperm(int n, std::vector<int>& inds, unsigned int seed=0);


uint64_t prod(const std::vector<int>& inds );
double prod(const std::vector<double>& inds );


////convert from an index to a coordinate
//inline int ind2coord( int ){
//
//}
//
////convert from coordinate to an index
//inline coord2ind( std::vector<int>&   ){
//}


#endif



