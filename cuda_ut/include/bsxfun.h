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

#ifndef _BSXFUN_H_
#define _BSXFUN_H_

#include "cu_util.h"
#include "cu_clmatrix.h"

/***********************************************************************************************************
 * @brief:		this function performs a matrix + col. vector operation *
 * @param[in]:	 pA and pOut: nI by nJ matrix
 * 				 pB is a column vector nI by 1
 * 				 nInJ is the total dimensionality of the matrix pA
 *
 * @param[out]:
 * @topology:	assumes a 1D block layout in x direction and covers the entire matrix pA
 * @note:		assume column-major
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template<class O, typename T>
__global__ void bsxfun_colvec_1dkernel( const T* pA,  const T* pVec, T* pOut,
										int nI, int nJ, int nInJ, O op)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (int i = ind; i < nInJ; i += totalThreads)
		pOut[i] = op(pA[i], pVec[i % nI]);
}


/***********************************************************************************************************
 * @brief:		this function performs a matrix + row. vector operation
 * @param[in]:	 pA and pOut: nI by nJ matrix
 * 				 pVec is a row vector 1 by nJ
 * 				 nInJ is the total dimensionality of the matrix pA
 *
 * @param[out]:
 * @topology:	assumes a 1D block layout in x direction and covers the entire matrix pA
 * @note:		assume column-major
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template<class O, typename T>
__global__ void bsxfun_rowvec_1dkernel(  const T* pA,  const T* pVec, T* pOut,
										 int nI, int nJ, int nInJ, O op)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (int i = ind; i < nInJ; i += totalThreads)
		pOut[i] = op(pA[i], pVec[i / nI]);
}


//alpha beta version
template<class O, typename T>
__global__ void bsxfun_colvec_1dkernel( T alpha, const T* pA,  T beta, const T* pVec, T* pOut,
										int nI, int nJ, int nInJ, O op)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (int i = ind; i < nInJ; i += totalThreads)
		pOut[i] = op(pA[i], alpha, pVec[i % nI], beta);
}

template<class O, typename T>
__global__ void bsxfun_rowvec_1dkernel(  T alpha, const T * pA,  T beta, const T* pVec, T* pOut,
										 int nI, int nJ, int nInJ, O op)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (int i = ind; i < nInJ; i += totalThreads)
		pOut[i] = op(pA[i], alpha, pVec[i / nI], beta);
}


/***********************************************************************************************************
* @brief: function similar to bsxfun of matlab
* A op B ---> Out
* @param[in]:	op - type of operation
* 				A - first matrix
* 				B - col/row vector, one dimension must be 1
* @param[out]:
     			if Out is set to A, the operation is inplace, overwrites A
*
* @topology:
* @note:
* @change:
* @tested:
* @to_do:		switch to shared memory operators to see if we can achieve speedup?!
***********************************************************************************************************
*/
template<class O, typename T>
int Bsxfun( const clMatrix<T>& A, O op, const clMatrix<T>& B, clMatrix<T>& Out){

	if (! (B.nI == 1 || B.nJ == 1) )
		return -1;
	if ( ( B.nI == 1 && B.nJ != A.nJ) || ( B.nJ == 1 && B.nI != A.nI) ){

		if (!(B.nI == 1 && B.nJ == 1))  //special case
			return -2;
	}
	if ( A.nI != Out.nI || A.nJ != Out.nJ)
		return -3;

	const unsigned int datadim = A.nJ*A.nI;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	if (B.nJ == 1 && B.nI != 1){
		bsxfun_colvec_1dkernel<<<dim_grid, dim_block>>>( A.pData, B.pData, Out.pData,
															A.nI, A.nJ, datadim, op);
	}else if (B.nJ != 1 && B.nI == 1){
		bsxfun_rowvec_1dkernel<<<dim_grid, dim_block>>>( A.pData, B.pData, Out.pData,
																  A.nI, A.nJ, datadim, op );
	}else{ // when B is 1x1
		if (A.nI == 1){
			bsxfun_colvec_1dkernel<<<dim_grid, dim_block>>>( A.pData, B.pData, Out.pData,
															A.nI, A.nJ, datadim, op);
		}else if (A.nJ == 1){
			bsxfun_rowvec_1dkernel<<<dim_grid, dim_block>>>( A.pData, B.pData, Out.pData,
																  A.nI, A.nJ, datadim, op );
		}else{
			return -4;  //invalid case
		}

	}
	return 0;
}

//alpha beta version
template<class O, typename T>
int Bsxfun(T alpha, const clMatrix<T>& A, O op, T beta, const clMatrix<T>& B, clMatrix<T>& Out){

	if (! (B.nI == 1 || B.nJ == 1) )
		return -1;
	if ( ( B.nI == 1 && B.nJ != A.nJ) || ( B.nJ == 1 && B.nI != A.nI) ){

		if (!(B.nI == 1 && B.nJ == 1))  //special case
			return -2;
	}
	if ( A.nI != Out.nI || A.nJ != Out.nJ)
		return -3;

	const uint64_t datadim = A.nJ*A.nI;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	if (B.nJ == 1 && B.nI != 1){
		bsxfun_colvec_1dkernel<<<dim_grid, dim_block>>>( alpha, A.pData, beta, B.pData, Out.pData,
																  A.nI, A.nJ, datadim, op);
	}else if (B.nJ != 1 && B.nI == 1){
		bsxfun_rowvec_1dkernel<<<dim_grid, dim_block>>>( alpha, A.pData, beta, B.pData, Out.pData,
																  A.nI, A.nJ, datadim, op );
	}else{
		if (A.nI == 1){
			bsxfun_colvec_1dkernel<<<dim_grid, dim_block>>>(alpha, A.pData, beta, B.pData, Out.pData,
															A.nI, A.nJ, datadim, op);
		}else if (A.nJ == 1){
			bsxfun_rowvec_1dkernel<<<dim_grid, dim_block>>>(alpha, A.pData, beta, B.pData, Out.pData,
																  A.nI, A.nJ, datadim, op );
		}else{
			return -4;  //invalid case
		}

	}

	return 0;
}








#endif
