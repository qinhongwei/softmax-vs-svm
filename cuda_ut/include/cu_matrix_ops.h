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

// CT 3/2012 - new version of all code that deals with basic matrix operations
// a replacement for cu_matrix_util.cu

#ifndef _CU_MATRIX_OPS_h_
#define _CU_MATRIX_OPS_h_

#define IN
#define OUT

#include <stdint.h>
#include "cublas_v2.h"
#include "cu_clmatrix.h"
#include "elewise.h"


template <typename T>
bool inline clMatrixSizeEqual( const clMatrix<T>& mat1, const clMatrix<T>& mat2 ){
	return mat1.nI == mat2.nI && mat1.nJ == mat2.nJ;
}

template <typename T>
bool inline clMatrixSizeEqual( const clMatrix<T>& mat, int nI, int nJ ){
	return mat.nI == nI && mat.nJ == nJ;
}


// *** Device <---- Device *** copy data from one clMat onto another clMat
template <typename T>
void  cuda_clMatrixCpy( OUT clMatrix<T> & dest, IN const clMatrix<T> & orig );

// *** Device <---- host *** copy data from host to device
template <typename T>
void cuda_clMatrixCpy( OUT clMatrix<T> & dest, IN const void* const data_orig);

// *** Device <---- device void pointer ***
template <typename T>
void cuda_clMatrixCpy_d2d( OUT clMatrix<T> & dest, IN const void* const data_orig);

//*** host <---- Device ***     copy data from device to host
template <typename T>
void cuda_clMatrixCpy( OUT void* data_dest, IN const clMatrix<T> & orig);

//get the diagonal of one matrix and put it in a column vector
template <typename T>
int GetDiag( IN const clMatrix<T>& mat, OUT clMatrix<T>& diag);

template <>
int GetDiag(IN const clMatrix<double>& mat, OUT clMatrix<double>& diag);

// diag should be a column vector, we put it in the diag
//of a matrix
template <typename T>
int SetDiag( IN const clMatrix<T>& diag, OUT clMatrix<T>& mat);

template <>
int SetDiag( IN const clMatrix<double>& diag, OUT clMatrix<double>& mat);




/***********************************************************************************************************
 * @brief:   makes cublasSgemm much easier to work with
 * 			C is replaced with alpha*op(A)*op(B)+beta*C
 * 			assumes all matrix are column major nI by nJ
 * @param[in]:
 * @param[out]:  0 means ok, -1 means error
 *
 * @topology:   n/a
 * @note: //if we need ot use data which are in float/double pointers
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
int _cl_cublas_matmul( const T* devpA, char Atr, int AnI, int AnJ,
						const T* devpB, char Btr, int BnI, int BnJ,
					   T* devpC, int CnI, int CnJ, T alpha, T beta);

//another wrapper using clMatrix as input
template <typename T>
int cl_cublas_matmul( T alpha, const clMatrix<T>& matA, char Atr,
					 const clMatrix<T>& matB, char Btr,
					  T beta, clMatrix<T>& matC);

//additional wrappers to emphasize that we are overwriting C
//alpha*A*B = C, C's value is erased or set ot the new value
template <typename T>
inline int ABeqC( T alpha, const clMatrix<T>& matA, char Atr, const clMatrix<T>& matB, char Btr, clMatrix<T>& matC ){
	return cl_cublas_matmul<T>( alpha, matA, Atr, matB, Btr, T(0), matC);
}

//additional wrappers to emphasize that we are adding to C
//alpha*A*B + C = C, C's value is added to
template <typename T>
inline int ABpCeqC(  T alpha, const clMatrix<T>& matA, char Atr, const clMatrix<T>& matB, char Btr, clMatrix<T>& matC ){
	return cl_cublas_matmul<T>( alpha, matA, Atr, matB, Btr, T(1), matC);
}


//y = alpha*y
template <typename T>
int cl_cublas_scal(T alpha, clMatrix<T>& mat);

template <>
int cl_cublas_scal(double alpha, clMatrix<double>& mat);

//y <---- y + a*x
//assuming all dimension of mat_y is to be used
template <typename T>
int cl_cublas_axpy( T alpha, const clMatrix<T>& mat_x, clMatrix<T>& mat_y);

template <>
int cl_cublas_axpy( double alpha, const clMatrix<double>& mat_x, clMatrix<double>& mat_y);

//absolue sum
template <typename T>
T cl_cublas_asum( clMatrix<T>& mat);

template <>
double cl_cublas_asum( clMatrix<double>& mat);


template<class O, typename T>
int ReduceInplace(O op, clMatrix<T>& mat, int dim);

//dim: 1 for up-down 2 for left-right
template<typename T>
inline int SumInplace( clMatrix<T>& mat, int dim){
	return ReduceInplace( fctrPlus<T>(), mat, dim);
}

template <typename T>
T Sum2DInplace ( clMatrix<T>& mat);


/***********************************************************************************************************
* @brief: calculates the sum of squared error between A and B
* 		  and return the sum
* @param[in]:	A B, Out must all be same size
* @param[out]:
* @topology:
* @note:		Out is a buffer which will be overwritten with junk
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template <typename T>
T SumSqErr(IN const clMatrix<T>& A, IN const clMatrix<T>& B, OUT clMatrix<T>& Out);


template <typename T>
int KhatriRao2Transpose(IN const clMatrix<T>& mat1, IN const clMatrix<T>& mat2,
				OUT clMatrix<T>& BigMat);

template <typename T>
int KhatriRao2TransposeDeriv(IN const clMatrix<T>& BigMat_pd,
							IN const clMatrix<T>& mat1, IN const clMatrix<T>& mat2,
							T gamma,
							OUT clMatrix<T>& mat1_pd, OUT clMatrix<T>& mat2_pd );

template <typename T>
int SumKhatriRao2Transpose(IN const clMatrix<T>& mat1, IN const clMatrix<T>& mat2,
				OUT clMatrix<T>& BigMat);

template <typename T>
int SumKhatriRao2TransposeDeriv(IN const clMatrix<T>& BigMat_pd,
							IN const clMatrix<T>& mat1, IN const clMatrix<T>& mat2,
							OUT clMatrix<T>& mat1_pd, OUT clMatrix<T>& mat2_pd );

#endif
