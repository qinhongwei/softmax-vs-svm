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
#ifndef _CU_GPU_RAND_H_
#define _CU_GPU_RAND_H_

#include "cu_util.h"
#include "cu_clmatrix.h"
#include "curand.h"

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


//host side controlled generation functions

//function to initialize the random number generator
void GpuRandInit(unsigned long long seed, bool bFermi);

//this function sets the matrix to random values
int cuda_clMatSetRand( clMatrix<float>& mat );
//double version
int cuda_clMatSetRand( clMatrix<double>& mat );

//this function sets the matrix to random values
int cuda_clMatSetRandn( clMatrix<float>&  mat,  float mean, float std);

//double version
int cuda_clMatSetRandn( clMatrix<double>&  mat, double mean, double std);

//this function sets the matrix to random values
int cuda_clMatSetRand( float* pdata, int nI, int nJ );
//double version
int cuda_clMatSetRand( double* pdata, int nI, int nJ );

//this function sets the matrix to random values
int cuda_clMatSetRandn( float* pdata, int nI, int nJ,  float mean, float std);

//double version
int cuda_clMatSetRandn( double* pdata, int nI, int nJ,  double mean, double std);

void GpuRandDestroy();


#endif
