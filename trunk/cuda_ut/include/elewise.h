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

#ifndef _ELEWISE_H_
#define _ELEWISE_H_

#define IN
#define OUT

#include "cu_util.h"
#include "cu_clmatrix.h"

//functions tested at: mexcuTestElewise
/***********************************************************************************************************
 * @brief:		this function performs an operation on 2 matrices: pA op pB ---> pOut
 * @param[in]:	 pA and pB: dim1 by dim2 matrix
 * 			     O is a functor class what operates on 2 floats
 * 			     dim == dim1*dim2
 * @param[out]:
 * @topology:	assumes a 1D block layout in x direction
 * @note:       templated function to perform and element wise operation
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template<class O, typename T>
__global__ void elewise_matmat_1dkernel(const T * pA,  const T* pB, T * pOut, int dim, O op) {
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;
	for (int i = ind; i < dim; i += totalThreads)
		pOut[i] = op(pA[i], pB[i]);
}

template<class O, typename T>
int EleWisefun( const clMatrix<T>& A, O op, const clMatrix<T>& B, clMatrix<T>& Out ){

	if (A.nI != B.nI || A.nJ != B.nJ || A.nI != Out.nI || A.nJ != Out.nJ )
		return -1;

	const unsigned int datadim = A.nI*A.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	elewise_matmat_1dkernel<<<dim_grid, dim_block>>>( A.pData, B.pData, Out.pData, datadim, op);

	return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////


/***********************************************************************************************************
* @brief: function to do inplace element wise binary operations on A with a scalar "val"
* A op val ---> Out
* @param[in]:	op - type of operation
* 				A - first matrix
* 				val - a scalar variable
* @param[out]:  if Out == A, then the operation is inplace, overwrites A
* @topology:
* @note:
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template<class O, typename T>
__global__ void elewise_matmat_1dkernel(IN const T * pA, IN T val, OUT T * pOut, int dim, O op) {
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;
	for (int i = ind; i < dim; i += totalThreads)
		pOut[i] = op(pA[i], val);
}

template<class O, typename T>
int EleWisefun(IN const clMatrix<T>& A, O op, IN T val, OUT clMatrix<T>& Out){

	if ( A.nI != Out.nI || A.nJ != Out.nJ )
		return -1;

	const unsigned int datadim = A.nI*A.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	elewise_matmat_1dkernel<<<dim_grid, dim_block>>>( A.pData, val, Out.pData, datadim, op);

	return 0;
}

/***********************************************************************************************************
* @brief: function to do inplace element wise unary operation on A
* op(A) ---> Out
* @param[in]:	op - type of operation
* 				A - first matrix
* @param[out]:  if Out is set to be A, then the operation is inplace, overwrites A
* @topology:
* @note:
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/

template<class O, typename T>
__global__ void elewise_matmat_1dkernel(IN const T * pA, OUT T * pOut, int dim, O op) {
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;
	for (int i = ind; i < dim; i += totalThreads)
		pOut[i] = op(pA[i]);
}

template<class O, typename T>
int EleWisefun(O op, IN const clMatrix<T>& A, OUT clMatrix<T>& Out){

	if ( A.nI != Out.nI || A.nJ != Out.nJ )
		return -1;

	const unsigned int datadim = A.nI*A.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	elewise_matmat_1dkernel<<<dim_grid, dim_block>>>( A.pData, Out.pData, datadim, op);

	return 0;
}



/***********************************************************************************************************
* @brief: function to do inplace element-wise binary operation on A, B
* op(alpha*A, beta*B) ---> Out
* @param[in]:	op - type of operation
* 				A - first matrix
* 				alpha - multiplier infront of A
* 				B - second matrix
* 				beta - multiplier infront of B
* @param[out]:  if Out is set to be A/B, then the operation is inplace, overwrites A/B
* @topology:
* @note:
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template<class O, typename T>
__global__ void elewise_matmat_1dkernel(const T* pA,  T alpha,
										const T* pB, T beta,
										T* pOut, int dim, O op) {
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;
	for (int i = ind; i < dim; i += totalThreads)
		pOut[i] = op(pA[i], alpha, pB[i], beta);
}

template<class O, typename T>
int EleWisefun(T alpha, IN const clMatrix<T>& A, O op, T beta, IN const clMatrix<T>& B, OUT clMatrix<T>& Out){

	if (A.nI != B.nI || A.nJ != B.nJ || A.nI != Out.nI || A.nJ != Out.nJ )
		return -1;

	const unsigned int datadim = A.nI*A.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	elewise_matmat_1dkernel<<<dim_grid, dim_block>>>( A.pData, alpha, B.pData, beta, Out.pData, datadim, op);

	return 0;
}





/***********************************************************************************************************
* @brief: function to do inplace element-wise binary operation on A, B
* op(alpha*A, beta*B)+gamma*OUT ---> Out
* @param[in]:	op - type of operation
* 				A - first matrix
* 				alpha - multiplier infront of A
* 				B - second matrix
* 				beta - multiplier infront of B
* @param[out]:  if Out is set to be A/B, then the operation is inplace, overwrites A/B
* @topology:
* @note:
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template<class O, typename T>
__global__ void elewise_matmat_1dkernel(const T* pA,  T alpha,
										const T* pB, T beta,
										T* pOut, T gamma,
										int dim, O op) {
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;
	for (int i = ind; i < dim; i += totalThreads)
		pOut[i] = gamma*pOut[i]+op(pA[i], alpha, pB[i], beta);
}

template<class O, typename T>
int EleWisefun(T alpha, IN const clMatrix<T>& A,
				O op, T beta, IN const clMatrix<T>& B,
				T gamma, OUT clMatrix<T>& Out){

	if (A.nI != B.nI || A.nJ != B.nJ || A.nI != Out.nI || A.nJ != Out.nJ )
		return -1;

	const unsigned int datadim = A.nI*A.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	elewise_matmat_1dkernel<<<dim_grid, dim_block>>>( A.pData, alpha, B.pData, beta,
			Out.pData, gamma, datadim, op);

	return 0;
}




/***********************************************************************************************************
* @brief: function to do inplace element-wise binary operation on A, B
* op(alpha*A, ['n' or 't'] beta*B) ---> Out
* @param[in]:	op - type of operation
* 				A - first matrix
* 				alpha - multiplier infront of A
* 				B - second matrix
* 				beta - multiplier infront of B
* @param[out]:  if Out is set to be A/B, then the operation is inplace, overwrites A/B
* @topology:
* @note:
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template<class O, typename T>
__global__ void elewise_matmat_2dkernel(
		const T* pA,  T alpha, const T* pB, T beta,
		T* pOut, int dim1, int dim2, O op)
{
	const unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

	const int ind = j*dim1+i;
	if (i < dim1 && j < dim2){
		pOut[ind] = op(pA[ind], alpha, pB[ind], beta);
	}
}

//pB should be transposed
template<class O, typename T>
__global__ void elewise_matmat_2dkernel_b_trans(
		const T* pA,  T alpha, const T* pB, T beta,
		T* pOut, int dim1, int dim2, O op)
{
	const unsigned int i1 = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int j1 = blockIdx.x*blockDim.x + threadIdx.x;
	const int ind1 = j1*dim1+i1;

	//note the switch of dim1 and dim2
	const int ind2 = i1*dim2+j1;

	if (i1 < dim1 && j1 < dim2){
		pOut[ind1] = op(pA[ind1], alpha, pB[ind2], beta);
	}
}

//this allows for transpose for matrix B
template<class O, typename T>
int EleWisefun2D(T alpha, IN const clMatrix<T>& A, O op,
				 T beta, char Btr, IN const clMatrix<T>& B,
				 OUT clMatrix<T>& Out){

	if (A.nI > BLOCK_DIM*MAX_GRIDS || A.nJ > BLOCK_DIM*MAX_GRIDS)
		return -1;
	if (A.nI != Out.nI || A.nJ != Out.nJ )
		return -2;

	if (Btr== 'n'){
		if (A.nI != B.nI || A.nJ != B.nJ )
			return -3;
	}else{
		if (A.nI != B.nJ || A.nJ != B.nI )
			return -4;
	}

	dim3 dim_block(BLOCK_DIM, BLOCK_DIM);
	dim3 dim_grid( ( A.nJ + dim_block.x-1)/dim_block.x,
			       ( A.nI + dim_block.y-1)/dim_block.y );

	if (Btr== 'n'){
		elewise_matmat_2dkernel<<<dim_grid, dim_block>>>( A.pData, alpha, B.pData, beta, Out.pData, A.nI, A.nJ, op);
	}else{
		elewise_matmat_2dkernel_b_trans<<<dim_grid, dim_block>>>( A.pData, alpha, B.pData, beta, Out.pData, A.nI, A.nJ, op);
	}

	return 0;
}













/***********************************************************************************************************
* @brief: function to do inplace element-wise binary operation on rows of A, B
* op(alpha*A(rowA,:), B(rowB,:) ) ---> Out(rowOut,:)
* @param[in]:	op - type of operation
* 				A - first matrix
* 				alpha - multiplier infront of A
* 				B - second matrix
* @param[out]:  if Out is set to be A/B, then the operation is inplace, overwrites A/B
* @topology:
* @note:
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template<class O, typename T>
__global__ void rowelewise_matmat_1dkernel(T alpha, const T * pA,  int AnI, int rowA,
														const T* pB, int BnI, int rowB,
														T * pOut, int OutnI, int rowOut,
														int total_nJ, O op){

	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;
	for (int j = ind; j < total_nJ; j += totalThreads)
		pOut[j*OutnI+rowOut] = op(pA[j*AnI+rowA], alpha, pB[j*BnI+rowB]);
}

template<class O, typename T>
int RowEleWisefun(T alpha, IN const clMatrix<T>& A, int rowA, O op,
				IN const clMatrix<T>& B, int rowB, OUT clMatrix<T>& Out, int rowOut){

	if ( A.nJ != B.nJ || A.nJ != Out.nJ)
		return -1;
	if ( rowA >= A.nI || rowB >= B.nI || rowOut >= Out.nI)
		return -2;

	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (A.nJ + dim_block.x-1)/dim_block.x) );

	rowelewise_matmat_1dkernel<<<dim_grid, dim_block>>>( alpha, A.pData, A.nI, rowA,
																B.pData, B.nI, rowB,
																Out.pData, Out.nI, rowOut,
																A.nJ, op);

	return 0;
}


/***********************************************************************************************************
* @brief: function to do inplace element-wise binary operation on rows of A, B
* op(alpha*A(rowA,:), beta*B(rowB,:) ) ---> Out(rowOut,:)
* @param[in]:	op - type of operation
* 				A - first matrix
* 				alpha - multiplier infront of A
* 				B - second matrix
* @param[out]:  if Out is set to be A/B, then the operation is inplace, overwrites A/B
* @topology:
* @note:
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template<class O, typename T>
__global__ void rowelewise_matmat_1dkernel(T alpha, const T * pA,  int AnI, int rowA,
										   T beta,	const T* pB, int BnI, int rowB,
														T * pOut, int OutnI, int rowOut,
														int total_nJ, O op){

	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;
	for (int j = ind; j < total_nJ; j += totalThreads)
		pOut[j*OutnI+rowOut] = op(pA[j*AnI+rowA], alpha, pB[j*BnI+rowB], beta);
}

template<class O, typename T>
int RowEleWisefun(T alpha, IN const clMatrix<T>& A, int rowA, O op,
				  T beta,  IN const clMatrix<T>& B, int rowB,
				  OUT clMatrix<T>& Out, int rowOut){

	if ( A.nJ != B.nJ || A.nJ != Out.nJ)
		return -1;
	if ( rowA >= A.nI || rowB >= B.nI || rowOut >= Out.nI)
		return -2;

	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (A.nJ + dim_block.x-1)/dim_block.x) );

	rowelewise_matmat_1dkernel<<<dim_grid, dim_block>>>( alpha, A.pData, A.nI, rowA,
														 beta,  B.pData, B.nI, rowB,
																Out.pData, Out.nI, rowOut,
																A.nJ, op);

	return 0;
}



/***********************************************************************************************************
 * @brief:		this function performs an operation on 2 matrices: pA(:,:) op pB(:,:) ---> pOut(:,:)

 * @param[in]:	 pA and pB: can be different size
 * 				 nA_nI is the height of A matrix
 * 				 nA_i0 and nA_j0 are the starting index on the A matrix
 * 				 nI, nJ are the size of the submatrix
 * @param[out]:
 * @topology:	assumes a 2D block layout
 * @note:
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template<class O, typename T>
__global__ void submat_elewise_2dkernel( T alpha, IN const T * pA, int nA_nI, int nA_i0, int nA_j0,
										 O op, 	T beta, const T* pB, int nB_nI, int nB_i0, int nB_j0,
										 OUT 	T * pOut, int nO_nI, int nO_i0, int nO_j0, int nI, int nJ)
{
	const int i = blockIdx.y*blockDim.y + threadIdx.y;
	const int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < nI && j < nJ){
		const int ind_A = (j+nA_j0)*nA_nI+i+nA_i0;
		const int ind_B = (j+nB_j0)*nB_nI+i+nB_i0;
		const int ind_O = (j+nO_j0)*nO_nI+i+nO_i0;

		pOut[ind_O] = op( pA[ind_A], alpha, pB[ind_B], beta);
	}
}


//for elewise functions on sub-matrix of matrices A, B and put it in C
//clIndex is all zero based, and inclusive
//e.g. for a matrix of dim dxd, clIndex(0,d-1,0,d-1) index the complete
//matrix
template<class O, typename T>
int SubmatEleWisefun(T alpha, IN const clMatrix<T>& A, clIndex inds_A,
					  O op, T beta, IN const clMatrix<T>& B, clIndex inds_B,
				 	  OUT clMatrix<T>& Out, clIndex inds_O)
{

	const int nI_A = inds_A.i1-inds_A.i0+1;  //submatrix size
	const int nJ_A = inds_A.j1-inds_A.j0+1;
	const int nI_B = inds_B.i1-inds_B.i0+1;
	const int nJ_B = inds_B.j1-inds_B.j0+1;
	const int nI_O = inds_O.i1-inds_O.i0+1;
	const int nJ_O = inds_O.j1-inds_O.j0+1;

	if(nI_A != nI_B || nI_A != nI_O)
		return -1;
	if(nJ_A != nJ_B || nJ_A != nJ_O)
		return -2;

	//make sure that the indices are in-bounds!
	if (  inds_A.i0 < 0 || inds_A.i1 >= A.nI || inds_A.i1 < inds_A.i0
		 || inds_A.j0 < 0 || inds_A.j1 >= A.nJ || inds_A.j1 < inds_A.j0 )
		return -3;
	if (  inds_B.i0 < 0 || inds_B.i1 >= B.nI || inds_B.i1 < inds_B.i0
		 || inds_B.j0 < 0 || inds_B.j1 >= B.nJ || inds_B.j1 < inds_B.j0 )
		return -4;
	if (  inds_O.i0 < 0 || inds_O.i1 >= Out.nI || inds_O.i1 < inds_O.i0
		 || inds_O.j0 < 0 || inds_O.j1 >= Out.nJ || inds_O.j1 < inds_O.j0 )
		return -5;

	if (nI_A > BLOCK_DIM*MAX_GRIDS || nJ_A > BLOCK_DIM*MAX_GRIDS) //to big to handle
		return -6;

	if (A.pData == NULL || B.pData == NULL || Out.pData == NULL)
		return -7;

	dim3 dim_block(BLOCK_DIM, BLOCK_DIM);
	dim3 dim_grid( ( nJ_A + dim_block.x-1)/dim_block.x,
			       ( nI_A + dim_block.y-1)/dim_block.y );

	submat_elewise_2dkernel<<<dim_grid, dim_block>>>(alpha, A.pData, A.nI, inds_A.i0, inds_A.j0,
														 op, beta, B.pData, B.nI, inds_B.i0, inds_B.j0,
														 Out.pData, Out.nI, inds_O.i0, inds_O.j0, nI_A, nJ_A);

	return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
// legacy compatible code to make some compilation still possible
//////////////////////////////////////////////////////////////////////////////////////////////////////////




#endif
