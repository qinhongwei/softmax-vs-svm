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

//this file is for generating random numbers right from the GPU
#include "cu_gpu_rand.h"


////////////////////////////////////////////////////////////////////
/*
 *  To use: include this file, and
 * 1. int gpu_id = SelectGPUFromServer( "GPUID" );
 * 2. GpuRandInit(seed, GetGPUArchitecture( "FERMI" ) );

 *  given y or y2 clMatrix<T> objects,
 *  clCheckErr( cuda_clMatSetRand( y ) );
	clCheckErr( cuda_clMatSetRandn( y2, (mean)1.59, (std)3.14 ) );
 *
 * GpuRandDestroy();
 */
////////////////////////////////////////////////////////////////////


/*  below is unfinished coded using device side generation
  #include <curand_kernel.h>

int GPU_CURAND_INITIALIZED = 0; //toggle to indicate if we initialized or not

//assume 1D layout
//gives 1 state for every
__global__ void setup_kernel(unsigned int seed, curandState * pstates )
{
	int x = threadIdx.x;
	curand_init(seed, x, 0, &pstates[x] ); //see curand_library.pdf in cuda docs

}

//my wrapper function for initialization
//assume that pCuRandStates is pointer to nMaxI*nMaxJ allocated memory
void GpuRandInit(unsigned int seed, curandState * pcurandstates, int nMaxI, int nMaxJ)
{
	dim3 dim_block(MAXTHREADS);
	dim3 dim_grid((nMaxI*nMaxJ+dim_block.x-1)/dim_block.x);
	setup_kernel<<< dim_grid, dim_block >>>( seed, pcurandstates, nMaxI*nMaxJ);

	GPU_CURAND_INITIALIZED = 1;
}


//assumes 2D topography
//could also consider using smaller # of threads to generate many more
__global__ void mat_set_rand( float * pmat, int nI, int nJ, curandState * pcurandstates)
{
	int j = threadIdx.x+blockIdx.x*blockDim.x;
	int i = threadIdx.y+blockIdx.y*blockDim.y;

	if (j < nJ && i < nI){
		int id = j*nI+i; //unique id for this thread
		pmat[id] = curand_uniform ( &pcurandstates[id%MAX_WRAP_STATE_ID_DEV] );
	}
}


//a column of "generators will generate for an entire column"
__global__ void mat_set_rand_1d( float * pmat, int nI, int nJ, curandState * pcurandstates)
{
	//not completed yet
	int j = threadIdx.x+blockIdx.x*blockDim.x;
	int i = threadIdx.y+blockIdx.y*blockDim.y;

	for( int k = 0; k < nI; ++k)
	{
		pmat[id] = curand_uniform ( &pcurandstates[id] );
	}

	if (j < nJ && i < nI){
		int id = j*nI+i; //unique id for this thread

	}
}
*/

//host side controlled generation
int GPU_CURAND_INITIALIZED = 0; //toggle to indicate if we initialized or not
curandGenerator_t gen; //global generator


//function to initialize the random number generator
void GpuRandInit(unsigned long long seed, bool bFermi){

	clCheckErr(curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT));
	clCheckErr(curandSetPseudoRandomGeneratorSeed( gen, seed));
	curandGenerateSeeds(gen);

	if (bFermi)
		clCheckErr(cudaDeviceSetLimit(cudaLimitStackSize, 1024));
	//above is a hack that needs for Fermi and later (with cuda sdk 3.2) may not be necessary with cuda 4.x

	GPU_CURAND_INITIALIZED = 1;
}


//this function sets the matrix to random values
int cuda_clMatSetRand( clMatrix<float>& mat ){

	if (GPU_CURAND_INITIALIZED == 0) //means we didn't initialize properly
		return -1;

	clCheckErr(curandGenerateUniform( gen, mat.pData, mat.nI*mat.nJ ));
	return 0;
}
//double version
int cuda_clMatSetRand( clMatrix<double>& mat ){

	if (GPU_CURAND_INITIALIZED == 0) //means we didn't initialize properly
		return -1;

	clCheckErr(curandGenerateUniformDouble( gen, mat.pData, mat.nI*mat.nJ ));
	return 0;
}

//this function sets the matrix to random values
int cuda_clMatSetRandn( clMatrix<float>&  mat,  float mean, float std){

	if (GPU_CURAND_INITIALIZED == 0) //means we didn't initialize properly
		return -1;

	if (mat.nI*mat.nJ % 2 != 0) //must be even
		return -2;

	clCheckErr(curandGenerateNormal( gen, mat.pData, mat.nI*mat.nJ, mean, std ));
	return 0;
}

//double version
int cuda_clMatSetRandn( clMatrix<double>&  mat, double mean, double std){

	if (GPU_CURAND_INITIALIZED == 0) //means we didn't initialize properly
		return -1;

	if (mat.nI*mat.nJ % 2 != 0) //must be even
		return -2;

	clCheckErr(curandGenerateNormalDouble( gen, mat.pData, mat.nI*mat.nJ, mean, std ));
	return 0;
}


//this function sets the matrix to random values
int cuda_clMatSetRand( float* pdata, int nI, int nJ ){

	if (GPU_CURAND_INITIALIZED == 0) //means we didn't initialize properly
		return -1;

	clCheckErr(curandGenerateUniform( gen, pdata, nI*nJ ));
	return 0;
}

//double version
int cuda_clMatSetRand( double* pdata, int nI, int nJ ){

	if (GPU_CURAND_INITIALIZED == 0) //means we didn't initialize properly
		return -1;

	clCheckErr(curandGenerateUniformDouble( gen, pdata, nI*nJ ));
	return 0;
}

//this function sets the matrix to random values
int cuda_clMatSetRandn( float* pdata, int nI, int nJ,  float mean, float std){

	if (GPU_CURAND_INITIALIZED == 0) //means we didn't initialize properly
		return -1;

	if (nI*nJ % 2 != 0) //must be even
		return -2;

	clCheckErr(curandGenerateNormal( gen, pdata, nI*nJ, mean, std ));
	return 0;
}

//double version
int cuda_clMatSetRandn( double* pdata, int nI, int nJ,  double mean, double std){

	if (GPU_CURAND_INITIALIZED == 0) //means we didn't initialize properly
		return -1;

	if (nI*nJ % 2 != 0) //must be even
		return -2;

	clCheckErr(curandGenerateNormalDouble( gen, pdata, nI*nJ, mean, std ));
	return 0;
}

void GpuRandDestroy(){
	clCheckErr(curandDestroyGenerator(gen));
}

