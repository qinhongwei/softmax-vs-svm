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
#include "cu_matrix_ops.h"



extern cublasHandle_t cbh;

/////////////////////////////////////////////////////////////////////////////////////////////////////
// basic matrix operations
/////////////////////////////////////////////////////////////////////////////////////////////////////

// *** Device <---- Device *** copy data from one clMat onto another clMat
template <typename T>
void  cuda_clMatrixCpy( OUT clMatrix<T> & dest, IN const clMatrix<T> & orig ){

	clASSERT(dest.nI == orig.nI && dest.nJ == orig.nJ, "\n dest and orig dim not same!\n");
	clASSERT( dest.pData != NULL && orig.pData != NULL, "cuda_clMatrixCpy: one of data ptr is NULL");

	checkCudaErrors( cudaMemcpy( dest.pData,  orig.pData,
			sizeof(T)*orig.nI*orig.nJ, cudaMemcpyDeviceToDevice));
}

// *** Device <---- host *** copy data from host to device
template <typename T>
void cuda_clMatrixCpy( OUT clMatrix<T> & dest, IN const void* const data_orig){

	clASSERT( dest.pData != NULL && data_orig != NULL, "cuda_clMatrixCpy: one of data ptr is NULL");

	checkCudaErrors( cudaMemcpy(dest.pData, data_orig,
			sizeof(T)*dest.nI*dest.nJ, cudaMemcpyHostToDevice));
}

// *** Device <---- device void pointer ***
template <typename T>
void cuda_clMatrixCpy_d2d( OUT clMatrix<T> & dest, IN const void* const data_orig){

	clASSERT( dest.pData != NULL && data_orig != NULL, "cuda_clMatrixCpy_d2d: one of data ptr is NULL");

	checkCudaErrors( cudaMemcpy(dest.pData, data_orig,
			sizeof(T)*dest.nI*dest.nJ, cudaMemcpyDeviceToDevice));
}

//*** host <---- Device ***     copy data from device to host
template <typename T>
void cuda_clMatrixCpy( OUT void* data_dest, IN const clMatrix<T> & orig){

	clASSERT( orig.pData != NULL && data_dest != NULL, "cuda_clMatrixCpy: one of data ptr is NULL");

	checkCudaErrors( cudaMemcpy(data_dest, orig.pData,
			sizeof(T)*orig.nI*orig.nJ, cudaMemcpyDeviceToHost));
}


//get the diagonal of one matrix and put it in a column vector
template <typename T>
int GetDiag( IN const clMatrix<T>& mat, OUT clMatrix<T>& diag)
{
	if (diag.nJ != 1 || diag.nI != mat.nI || mat.nI != mat.nJ)
		return -1;
	if (diag.pData == NULL || mat.pData == NULL)
		return -2;

	cublas_cr( cublasScopy(cbh, diag.nI, mat.pData, mat.nI+1, diag.pData, 1) );
	return 0;
}

template <>
int GetDiag(IN const clMatrix<double>& mat, OUT clMatrix<double>& diag)
{
	if (diag.nJ != 1 || diag.nI != mat.nI || mat.nI != mat.nJ)
		return -1;
	if (diag.pData == NULL || mat.pData == NULL)
		return -2;

	cublas_cr( cublasDcopy(cbh, diag.nI, mat.pData, mat.nI+1, diag.pData, 1) );
	return 0;
}


// diag should be a column vector, we put it in the diag
//of a matrix
template <typename T>
int SetDiag( IN const clMatrix<T>& diag, OUT clMatrix<T>& mat){
	if (diag.nJ != 1 || diag.nI != mat.nI || mat.nI != mat.nJ)
		return -1;
	if (diag.pData == NULL || mat.pData == NULL)
		return -2;

	cublas_cr( cublasScopy(cbh, diag.nI, diag.pData, 1, mat.pData, mat.nI+1) );
	return 0;
}

template <>
int SetDiag( IN const clMatrix<double>& diag, OUT clMatrix<double>& mat){
	if (diag.nJ != 1 || diag.nI != mat.nI || mat.nI != mat.nJ)
		return -1;
	if (diag.pData == NULL || mat.pData == NULL)
		return -2;

	cublas_cr( cublasDcopy(cbh, diag.nI, diag.pData, 1, mat.pData, mat.nI+1) );
	return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////
// cublas wrappers
/////////////////////////////////////////////////////////////////////////////////////////////////////

// default: float type wrapper for cublasSgemm
template <typename T>
void cublas_gemm(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
		const T* alpha,
		const T* A, int lda,
		const T* B, int ldb,
		const T* beta, T* C, int ldc)
{
	cublas_cr( cublasSgemm( cbh, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)  );
}

// double specialization
template <>
void cublas_gemm(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
		const double* alpha,
		const double* A, int lda,
		const double* B, int ldb,
		const double* beta, double* C, int ldc)
{
	cublas_cr( cublasDgemm( cbh, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)  );
}


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
int _cl_cublas_matmul( const T* devpA, char Atr, int AnI, int AnJ, const T* devpB, char Btr, int BnI, int BnJ,
					   T* devpC, int CnI, int CnJ, T alpha, T beta)
{
	bool noErr;

	if (devpA == devpC || devpB == devpC || devpA == devpB){
		noErr = false;
		clASSERT(false, "cl_cublas_matmul: Can't do inplace multiplications!");
	}

	cublasOperation_t transa, transb;

	int nn; //numbers needed for cublasSgemm, CnI by CnJ is result, with nn dimensions summed out
	if (Atr == 'n' && Btr == 'n'){
		noErr = AnJ == BnI && AnI == CnI && BnJ == CnJ;
		nn = AnJ;
		transa = CUBLAS_OP_N;
		transb = CUBLAS_OP_N;

	}else if (Atr == 't' && Btr == 'n'){
		noErr = AnI == BnI && AnJ == CnI && BnJ == CnJ;
		nn = AnI;
		transa = CUBLAS_OP_T;
		transb = CUBLAS_OP_N;

	}else if (Atr == 'n' && Btr == 't'){
		noErr = AnJ == BnJ && AnI == CnI && BnI == CnJ;
		nn = AnJ;
		transa = CUBLAS_OP_N;
		transb = CUBLAS_OP_T;

	}else{ // Atr == 't' && Btr == 't'
		noErr = AnI == BnJ && AnJ == CnI && BnI == CnJ;
		nn = AnI;
		transa = CUBLAS_OP_T;
		transb = CUBLAS_OP_T;
	}

	if (noErr){
		cublas_gemm<T>(transa, transb,  CnI, CnJ, nn,
					&alpha, devpA, AnI,
					devpB, BnI,
					&beta, devpC, CnI);
	}else{
		char errmsg[100];
		sprintf( errmsg, "\n_cl_cublas_matmul in place failed! %dx%d %c %dx%d %c = %dx%d", AnI, AnJ, Atr, BnI, BnJ, Btr, CnI, CnJ);
		clASSERT(false, errmsg );
	}
	return 0;
}

//another wrapper using clMatrix as input
template <typename T>
int cl_cublas_matmul( T alpha, const clMatrix<T>& matA, char Atr, const clMatrix<T>& matB, char Btr, T beta, clMatrix<T>& matC)
{
	bool noErr = true;

	if (matA.pData == matC.pData || matB.pData == matC.pData ){
		noErr = false;
		clASSERT(false, "cl_cublas_matmul: Can't do inplace multiplications!");
	}

	cublasOperation_t transa, transb;

#ifndef CLMATCUDANOSAFE
	if (Atr == 'n' && Btr == 'n'){
		noErr = matA.nJ == matB.nI && matA.nI == matC.nI && matB.nJ == matC.nJ;
		transa = CUBLAS_OP_N;
		transb = CUBLAS_OP_N;
	}else if (Atr == 't' && Btr == 'n'){
		noErr = matA.nI == matB.nI && matA.nJ == matC.nI && matB.nJ == matC.nJ;
		transa = CUBLAS_OP_T;
		transb = CUBLAS_OP_N;
	}else if (Atr == 'n' && Btr == 't'){
		noErr = matA.nJ == matB.nJ && matA.nI == matC.nI && matB.nI == matC.nJ;
		transa = CUBLAS_OP_N;
		transb = CUBLAS_OP_T;
	}else{ // Atr == 't' && Btr == 't'
		noErr = matA.nI == matB.nJ && matA.nJ == matC.nI && matB.nI == matC.nJ;
		transa = CUBLAS_OP_T;
		transb = CUBLAS_OP_T;
	}
#endif

	int nn; //numbers needed for cublasSgemm, CnI by CnJ is result, with nn dimensions summed out
	if (Atr == 'n')
		nn = matA.nJ;
	else
		nn = matA.nI;

	if (noErr){
		cublas_gemm<T>(transa, transb,  matC.nI, matC.nJ, nn,
					&alpha, matA.pData, matA.nI,
					matB.pData, matB.nI,
					&beta, matC.pData, matC.nI);
	}else{
		char errmsg[100];
		sprintf( errmsg, "\ncl_cublas_matmul in place failed!  %dx%d %c %dx%d %c = %dx%d", matA.nI, matA.nJ, Atr, matB.nI, matB.nJ, Btr,  matC.nI, matC.nJ);
		clASSERT(false, errmsg );
	}
	return 0;
}



/*
// default: float type wrapper for cublasSgemm
template <typename T>
void cublas_gemm_batched(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
		const T* alpha,
		const T* A, int lda,
		const T* B, int ldb,
		const T* beta, T* C, int ldc)
{
	cublas_cr( cublasSgemm( cbh, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)  );
}

// double specialization
template <>
void cublas_gemm_batched(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
		const double* alpha,
		const double* A, int lda,
		const double* B, int ldb,
		const double* beta, double* C, int ldc)
{
	cublas_cr( cublasDgemm( cbh, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)  );
}

//another wrapper using clMatrix as input
template <typename T>
int cl_cublas_matmul_batched( T alpha, const clMatrix<T>& matA, char Atr, const clMatrix<T>& matB, char Btr, T beta, clMatrix<T>& matC)
{
	bool noErr = true;

	if (matA.pData == matC.pData || matB.pData == matC.pData ){
		noErr = false;
		clASSERT(false, "cl_cublas_matmul: Can't do inplace multiplications!");
	}

	cublasOperation_t transa, transb;

#ifndef CLMATCUDANOSAFE
	if (Atr == 'n' && Btr == 'n'){
		noErr = matA.nJ == matB.nI && matA.nI == matC.nI && matB.nJ == matC.nJ;
		transa = CUBLAS_OP_N;
		transb = CUBLAS_OP_N;
	}else if (Atr == 't' && Btr == 'n'){
		noErr = matA.nI == matB.nI && matA.nJ == matC.nI && matB.nJ == matC.nJ;
		transa = CUBLAS_OP_T;
		transb = CUBLAS_OP_N;
	}else if (Atr == 'n' && Btr == 't'){
		noErr = matA.nJ == matB.nJ && matA.nI == matC.nI && matB.nI == matC.nJ;
		transa = CUBLAS_OP_N;
		transb = CUBLAS_OP_T;
	}else{ // Atr == 't' && Btr == 't'
		noErr = matA.nI == matB.nJ && matA.nJ == matC.nI && matB.nI == matC.nJ;
		transa = CUBLAS_OP_T;
		transb = CUBLAS_OP_T;
	}
#endif

	int nn; //numbers needed for cublasSgemm, CnI by CnJ is result, with nn dimensions summed out
	if (Atr == 'n')
		nn = matA.nJ;
	else
		nn = matA.nI;

	if (noErr){
		cublas_gemm<T>(transa, transb,  matC.nI, matC.nJ, nn,
					&alpha, matA.pData, matA.nI,
					matB.pData, matB.nI,
					&beta, matC.pData, matC.nI);
	}else{
		char errmsg[100];
		sprintf( errmsg, "\ncl_cublas_matmul in place failed!  %dx%d %c %dx%d %c = %dx%d", matA.nI, matA.nJ, Atr, matB.nI, matB.nJ, Btr,  matC.nI, matC.nJ);
		clASSERT(false, errmsg );
	}
	return 0;
}
*/




//y = alpha*y
template <typename T>
int cl_cublas_scal(T alpha, clMatrix<T>& mat){
	if (mat.nI*mat.nJ <= 0 || mat.pData == NULL)
		return -1;

	cublas_cr( cublasSscal(cbh, mat.nI*mat.nJ, &alpha, mat.pData, 1) );
	return 0;
}
template <>
int cl_cublas_scal(double alpha, clMatrix<double>& mat){
	if (mat.nI*mat.nJ <= 0 || mat.pData == NULL)
		return -1;

	cublas_cr( cublasDscal(cbh, mat.nI*mat.nJ, &alpha, mat.pData, 1) );
	return 0;
}

//y <---- y + a*x
//assuming all dimension of mat_y is to be used
template <typename T>
int cl_cublas_axpy( T alpha, const clMatrix<T>& mat_x, clMatrix<T>& mat_y)
{
	if ( mat_x.nI != mat_y.nI || mat_x.nJ != mat_y.nJ || mat_x.pData == NULL || mat_y.pData == NULL)
		return -1;

	cublas_cr( cublasSaxpy(cbh, mat_y.nI*mat_y.nJ, &alpha, mat_x.pData, 1, mat_y.pData, 1) );
	return 0;
}

template <>
int cl_cublas_axpy( double alpha, const clMatrix<double>& mat_x, clMatrix<double>& mat_y)
{
	if ( mat_x.nI != mat_y.nI || mat_x.nJ != mat_y.nJ || mat_x.pData == NULL || mat_y.pData == NULL)
		return -1;

	cublas_cr( cublasDaxpy(cbh, mat_y.nI*mat_y.nJ, &alpha, mat_x.pData, 1, mat_y.pData, 1) );
	return 0;
}

//absolue sum
template <typename T>
T cl_cublas_asum( clMatrix<T>& mat){
	if (mat.nI*mat.nJ <= 0 || mat.pData == NULL)
		return -1;
	T res;
	cublas_cr( cublasSasum( cbh, mat.nI*mat.nJ, mat.pData, 1, &res) );

	return res;
}

template <>
double cl_cublas_asum( clMatrix<double>& mat){
	if (mat.nI*mat.nJ <= 0 || mat.pData == NULL)
		return -1;
	double res;
	cublas_cr( cublasDasum( cbh, mat.nI*mat.nJ, mat.pData, 1, &res) );
	return res;
}




//CT 8/2011 "reduce" functions Copied from sum_dim_1, sum_dim_2 and sum_inplace
/***********************************************************************************************************
 * @brief:  performs REDUCTION in the first dimension
 * 			given a large pMat matrix of total dim1 = mat_dim1, we want to compute a partial sum
 * 			and write it in the top row of the submatrix (this block is) assigned to.
 * @param[in]: mat_dim1 - real dim1 of pMat
 * 			   dim1 - the dim1 of the submatrix we are currently working with (independent of blocks)
 * 			   dim1step - the step size in dim1 which we should effectively skip
 * @param[out]:
 * @topology: blockDim.x = BLOCKDIM assigned in sum_inplace;
 *            gridDim.x = dim2 or big pMat
 *            gridDim.y = depends on which iteration inside sum_inplace
 *
 *            (one column of pMat)
 *            ...
 *            ...
 *            ...
 *            me
 *            me
 *            me - responsible for
 *            me - responsible for
 *            ...
 *            ...
 *
 * @note:   note, need blockDim.x or number of threads to be exponents of 2, i.e. 32, or 64, ... or 256
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template<class O, typename T>
__global__ void reduce_dim1(T* pMat, int mat_dim1, int dim1, int dim1step, O op)
{
	T * sdata1 = SharedMemory<T>();

	int tid = threadIdx.x;
	int i = blockIdx.y*(blockDim.x*2)+threadIdx.x; //note the multiplication by 2 is needed because we also sum the skipped section
	int i_global = i*dim1step; //the global i (across blocks) within dim1 "sub matrix"
	int col = blockIdx.x; //which column do we belong to in the global pMat matrix

	//copy data over to local shared memory
	if (i+blockDim.x < dim1){
		sdata1[tid] = op(pMat[col*mat_dim1+i_global], pMat[col*mat_dim1+i_global+blockDim.x*dim1step]);
	}else if( i < dim1) {
		sdata1[tid] = pMat[col*mat_dim1+i_global];
	}
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s>0; s >>= 1)
	{
		//i+s is ok because it will skip this conditional until s is small enough
		//have to use i here because dim1 is across blocks
		if (tid < s && i+s < dim1 ){
			sdata1[tid] = op( sdata1[tid+s], sdata1[tid] );
		}
		__syncthreads();
	}

	if (tid == 0){
		pMat[col*mat_dim1+i_global] = sdata1[0];
	}
}


/***********************************************************************************************************
 * @brief:  performs REDUCTION in the 2nd dimension
 * 			given a large pMat matrix of total dim2 = mat_dim2, we want to compute a partial sum
 * 			and write it in the leftmost column of the submatrix (this block is) assigned to.
 * @param[in]: mat_dim1 - real dim1 of pMat
 * 			   dim2 - the dim2 of the submatrix we are currently working with (independent of blocks)
 * 			   dim2step - this is the step size in pMat
 * @param[out]:
 * @topology: blockDim.x = BLOCKDIM assigned in sum_inplace;
 *            gridDim.x = depends on which iteration inside sum_inplace
 *            gridDim.y = dim1 or big pMat
 *
 *            (one row of pMat)
 *            (...)  (...)  (...)  me me (me - responsible for) (me - responsible for) (...) (...)
 * @note:   note, need blockDim.x or number of threads to be exponents of 2, i.e. 32, or 64, ... or 256
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template<class O, typename T>
__global__ void reduce_dim2(T* pMat, int mat_dim1, int dim2, int dim2step, O op)
{
	T * sdata2 = SharedMemory<T>();

	int tid = threadIdx.x;
	int j = blockIdx.x*(blockDim.x*2)+threadIdx.x; //note the multiplication by 2 is needed because we also sum the skipped section
	int j_global = j*dim2step;	// the global j (across blocks) within dim2 "sub matrix"
	int row = blockIdx.y; //which row do we belong to in the global pMat matrix

	//copy data over to local shared memory
	if (j+blockDim.x < dim2){
		sdata2[tid] = op(pMat[j_global*mat_dim1+row], pMat[(j+blockDim.x)*dim2step*mat_dim1+row]);
	}else if( j < dim2) {
		sdata2[tid] = pMat[j_global*mat_dim1+row];
	}
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s>0; s >>= 1)
	{
		//j+s is ok because it will skip this conditional until s is small enough
		//have to use j here because dim2 is across blocks
		if (tid < s && j+s < dim2 ){
			sdata2[tid] = op( sdata2[tid+s], sdata2[tid] );
		}
		__syncthreads();
	}

	if (tid == 0){
		pMat[ j_global*mat_dim1 + row] = sdata2[0]; //copy to global memory
	}
}

/***********************************************************************************************************
 * @brief: perform Reduction inplace on pMat
 * @param[in]:  pMat - matrix of dim1 by dim2
 * 				sum_dim - along which dimensions, 1 or 2
 *
 * @param[out]: row 0 or pMat for sum_dim ==1 or col 0 for sum_dim == 2;
 * @topology:   divides pMat into grids of 2*BLOCKDIM and dim2, each block have 1D threads
 * 				012345
 * 				....
 * 				012345
 * 				skip //will be added by 0 to 5 blocks
 *              ..
 *              skip
 *              6789 10
 *              ...
 *              6789 10 *
 *              skip //will be added by the 6 to 10 blocks
 *              ..
 *              skip
 * @note:
 * @change:  use pMat as a "writing pad" and give it junk values except for the first row or column,
 * 			 (depending on the dimension of summing 1 or 2)
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template<class O, typename T>
int _reduce_inplace(T* pMat, int dim1, int dim2, int sum_dim, O op)
{
	if (sum_dim > 2 || sum_dim < 1)
		return -1;

	int BLOCKDIM = MEDIUM_NUM_THREADS;

	//makesure size is NOT too big!
	if (sum_dim == 1){
		if (dim2 > MAX_GRIDS)
			return -2;
		if ( ceil(dim1/(double(BLOCKDIM*2))) > MAX_GRIDS)
			return -3;
	}else{
		if (dim1 > MAX_GRIDS)
			return -2;
		if ( ceil(dim2/(double(BLOCKDIM*2))) > MAX_GRIDS)
			return -3;
	}

	if (sum_dim ==1){
		int dimvar = dim1;
		unsigned int ns = sizeof(T)*BLOCKDIM; //shared memory needed
		int nBlocks, dim1step = 1;
		do{
			nBlocks = ceil(dimvar/(double(BLOCKDIM*2)));
			dim3 dim_block(BLOCKDIM);
			dim3 dim_grid(dim2, nBlocks);

			reduce_dim1<<<dim_grid, dim_block, ns>>>( pMat, dim1, dimvar, dim1step, op);  //comput partial sum in their own subblocks
			dimvar = nBlocks; //next loop we sum over the partial sum of the subblocks
			dim1step *= BLOCKDIM*2;
		}while (nBlocks > 1);
	}else{
		int dimvar =  dim2;
		unsigned int ns = sizeof(T)*BLOCKDIM; //shared memory needed

		int nBlocks, dim2step=1;
		do{
			nBlocks = ceil(dimvar/(double(BLOCKDIM*2)));
			//printf("\n nBlocks:%d dimvar:%d ", nBlocks, dimvar);
			dim3 dim_block(BLOCKDIM);
			dim3 dim_grid(nBlocks, dim1);

			reduce_dim2<<<dim_grid, dim_block, ns>>>( pMat, dim1, dimvar, dim2step, op);  //compute partial sum in their own subblocks
			dimvar = nBlocks; //next loop we sum over the partial sum of the subblocks
			dim2step *= BLOCKDIM*2;
		}while (nBlocks > 1);
	}
	return 0;
}


//see cu_matrix_util.cu for original implementation with a buggy code
//dim has matlab format, 1 means top down, 2 means left to right
template<class O, typename T>
int ReduceInplace(O op, clMatrix<T>& mat, int dim)
{
	if (dim != 1 && dim != 2)
		return -1;
	return _reduce_inplace( mat.pData, mat.nI, mat.nJ, dim, op);
}


//will sum inplace and destroy the original matrix
template <typename T>
T Sum2DInplace ( clMatrix<T>& mat){
	cr( _reduce_inplace( mat.pData, mat.nI, mat.nJ, 2, fctrPlus<T>() ) )
	cr( _reduce_inplace( mat.pData, mat.nI, 1, 1, fctrPlus<T>() ) )
	return mat.GetElem(0,0);
}


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
T SumSqErr(IN const clMatrix<T>& A, IN const clMatrix<T>& B, OUT clMatrix<T>& Out)
{
	if (!clMatrixSizeEqual(A, B) || !clMatrixSizeEqual(A, Out))
		return -1;

	cr( EleWisefun(A, fctrDiffSq<T>(), B, Out) )
	return Sum2DInplace(Out);
}


//////////////////////////////////////////////////////////////////////////////////////////////
//Khatri-Rao product
//////////////////////////////////////////////////////////////////////////////////////////////
//p1 is d1 by N, p2 is d2 by N, p_big is d1*d2 by N
//num_elems, total number of elements in p_big
template <typename T>
__global__ void khatri_rao2_kernel(IN const T* p1, IN const T* p2,
									IN uint64_t d1, IN uint64_t d2,
									OUT T* p_big, uint64_t num_elems)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (uint64_t i = ind; i < num_elems; i += totalThreads){

		uint64_t myN = i/(d1*d2);
		uint64_t rem = i % (d1*d2);
		int myd1 = rem / d2;
		int myd2 = rem % d2;

		p_big[i] = p1[myd1+myN*d1]*p2[myd2+myN*d2];
	}
}

//calculate the Khatri-Rao product of multiple matrices
//mats[k] must be nI(k) by N, where N is constant for all elements of mats
//BigMat must be Cross(all mats' nI) by N, and pre-allocated
template <typename T>
int KhatriRao2(IN const clMatrix<T>& mat1, IN const clMatrix<T>& mat2,
				OUT clMatrix<T>& BigMat)
{
	//first do some error checking
	if (mat1.nJ != mat2.nJ || mat1.nJ != BigMat.nJ)
		return -1;
	if (mat1.nI*mat2.nI != BigMat.nI)
		return -2;

	const uint64_t num_elems = BigMat.nI*BigMat.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (num_elems + dim_block.x-1)/dim_block.x) );

	khatri_rao2_kernel<<<dim_grid, dim_block>>>( mat1.pData, mat2.pData, mat1.nI, mat2.nI, BigMat.pData, num_elems);

	return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////
//Khatri-Rao product
//////////////////////////////////////////////////////////////////////////////////////////////
//p1 is N by d1, p2 is N by d2, p_big is N by d1*d2
//num_elems, total number of elements in p_big
//note that p2's index is first, then p1's index in the big matrix
template <typename T>
__global__ void khatri_rao2_transpose_kernel(IN const T* p1, IN const T* p2,
											IN uint64_t d2, IN uint64_t N,
											OUT T* p_big, uint64_t num_elems)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (uint64_t i = ind; i < num_elems; i += totalThreads){

		uint64_t myN = i % N;
		uint64_t r = i / N;
		int myd1 = r / d2;
		int myd2 = r % d2;

		p_big[i] = p1[myN + myd1*N]*p2[ myN + myd2*N ];
	}
}

//calculate the Khatri-Rao product of multiple matrices
//mats[k] must be N by nI(k), where N is constant for all elements of mats
//BigMat must be N by Cross(all mats' nI), and pre-allocated
template <typename T>
int KhatriRao2Transpose(IN const clMatrix<T>& mat1, IN const clMatrix<T>& mat2,
				OUT clMatrix<T>& BigMat)
{
	//first do some error checking
	if (mat1.nI != mat2.nI || mat1.nI != BigMat.nI)
		return -1;
	if (mat1.nJ*mat2.nJ != BigMat.nJ)
		return -2;

	const uint64_t num_elems = BigMat.nI*BigMat.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (num_elems + dim_block.x-1)/dim_block.x) );

	khatri_rao2_transpose_kernel<<<dim_grid, dim_block>>>( mat1.pData, mat2.pData, mat2.nJ, BigMat.nI, BigMat.pData, num_elems);

	return 0;
}

//derivative w.r.t. the first small matrix
//p1 is N by d1, p2 is N by d2, p_big is N by d1*d2
//num_elems, total number of elements in p_big
//NOTE important: p_big_pd is column major with index of N first, then d2, then d1
template <typename T>
__global__ void khatri_rao2_transpose_deriv1_kernel(IN const T* p_big_pd, IN const T* p2,
											IN T gamma, OUT T* p1_pd,
											IN uint64_t d1, IN uint64_t d2, IN uint64_t N, uint64_t num_elems)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	//threads are for elements of p1_pd
	for (uint64_t i = ind; i < num_elems; i += totalThreads){

		uint64_t myN = i % N;
		uint64_t myd1 = i / N;


		T temp = 0;
		uint64_t big_ind = myN+0*N+myd1*d2*N;


		for (int k = 0; k < d2; ++k){
			temp += p_big_pd[big_ind+k*N]*p2[myN+k*N];
		}
		p1_pd[i] = temp+gamma*p1_pd[i];
	}
}


//derivative w.r.t. the second small matrix
//p1 is N by d1, p2 is N by d2, p_big is N by d1*d2
//num_elems, total number of elements in p_big
//NOTE important: p_big_pd is column major with index of N first, then d2, then d1
template <typename T>
__global__ void khatri_rao2_transpose_deriv2_kernel(IN const T* p_big_pd, IN const T* p1,
											IN T gamma, OUT  T* p2_pd,
											IN uint64_t d1, IN uint64_t d2, IN uint64_t N, uint64_t num_elems)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	//threads are for elements of p2_pd
	for (uint64_t i = ind; i < num_elems; i += totalThreads){

		uint64_t myN = i % N;
		uint64_t myd2 = i / N;


		T temp = 0;
		uint64_t big_ind = myN+myd2*N+0*d2*N;

		for (int k = 0; k < d1; ++k){
			temp += p_big_pd[big_ind+k*d2*N]*p1[myN+k*N];
		}
		p2_pd[i] = temp+gamma*p2_pd[i];
	}
}



//calculate the Khatri-Rao product of multiple matrices
//mats[k] must be N by nI(k), where N is constant for all elements of mats
//BigMat must be N by Cross(all mats' nI), and pre-allocated
//mat1_pd is the partial derivative w.r.t. mat1, same size as mat1
//mat2_pd is the partial derivative w.r.t. mat2, same size as mat2

//gamma = 1 means we add to what's already exist in mat1_pd and mat2_pd
//gamma = 0 means we overwrite it
//mat1_pd = derivative + gamma*mat1_pd(old)

template <typename T>
int KhatriRao2TransposeDeriv(IN const clMatrix<T>& BigMat_pd,
							IN const clMatrix<T>& mat1, IN const clMatrix<T>& mat2,
							T gamma,
							OUT clMatrix<T>& mat1_pd, OUT clMatrix<T>& mat2_pd )
{
	//first do some error checking
	if (mat1.nI != mat2.nI || mat1.nI != BigMat_pd.nI)
		return -1;
	if (mat1.nJ*mat2.nJ != BigMat_pd.nJ)
		return -2;
	if (mat1.nI != mat1_pd.nI || mat1.nI != mat2_pd.nI)
		return -3;
	if (mat1.nJ != mat1_pd.nJ || mat2.nJ != mat2_pd.nJ)
		return -4;

	{
	const uint64_t num_elems = mat1.nI*mat1.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (num_elems + dim_block.x-1)/dim_block.x) );

	khatri_rao2_transpose_deriv1_kernel<<<dim_grid, dim_block>>>( BigMat_pd.pData, mat2.pData, gamma,
							mat1_pd.pData, mat1.nJ, mat2.nJ, mat1.nI, num_elems);
	}

	{
	const uint64_t num_elems = mat2.nI*mat2.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (num_elems + dim_block.x-1)/dim_block.x) );

	khatri_rao2_transpose_deriv2_kernel<<<dim_grid, dim_block>>>( BigMat_pd.pData, mat1.pData, gamma,
							mat2_pd.pData, mat1.nJ, mat2.nJ, mat1.nI, num_elems);
	}

	return 0;
}



//p1 is d1 by N, p2 is d2 by N, p_big is d1*d2 by N
//p3, same as above
//num_elems, total number of elements in p_big
template <typename T>
__global__ void khatri_rao3_kernel(IN const T* p1, IN const T* p2, IN const T* p3,
									IN uint64_t d1, IN uint64_t d2, IN uint64_t d3,
									OUT T* p_big, uint64_t num_elems)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (uint64_t i = ind; i < num_elems; i += totalThreads){

		uint64_t total123 = d1*d2*d3;
		uint64_t total23 = d2*d3;

		uint64_t myN = i/total123;
		uint64_t rem = i % total123;

		int myd1 = rem / total23;
		rem = rem % total23;
		int myd2 = rem / d3;
		int myd3 = rem % d3;

		p_big[i] = p1[myd1+myN*d1]*p2[myd2+myN*d2]*p3[myd3+myN*d3];
	}
}

//calculate the Khatri-Rao product of multiple matrices
//mats[k] must be nI(k) by N, where N is constant for all elements of mats
//BigMat must be Cross(all mats' nI) by N, and pre-allocated
template <typename T>
int KhatriRao3(IN const clMatrix<T>& mat1, IN const clMatrix<T>& mat2,
		 	 	 IN const clMatrix<T>& mat3, OUT clMatrix<T>& BigMat)
{
	//first do some error checking
	if (mat1.nJ != mat2.nJ || mat1.nJ != mat3.nJ ||  mat1.nJ != BigMat.nJ)
		return -1;
	if (mat1.nI*mat2.nI*mat3.nI != BigMat.nI)
		return -2;

	const uint64_t num_elems = BigMat.nI*BigMat.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (num_elems + dim_block.x-1)/dim_block.x) );

	khatri_rao3_kernel<<<dim_grid, dim_block>>>( mat1.pData, mat2.pData, mat3.pData, mat1.nI, mat2.nI, mat3.nI, BigMat.pData, num_elems);

	return 0;
}



//instead of khatri_rao product, we sum all the cross entries
//p1 is N by d1, p2 is N by d2, p_big is N by d1*d2
//num_elems, total number of elements in p_big
//note that p2's index is first, then p1's index in the big matrix
template <typename T>
__global__ void sum_khatri_rao2_transpose_kernel(IN const T* p1, IN const T* p2,
											IN uint64_t d2, IN uint64_t N,
											OUT T* p_big, uint64_t num_elems)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (uint64_t i = ind; i < num_elems; i += totalThreads){

		uint64_t myN = i % N;
		uint64_t r = i / N;
		int myd1 = r / d2;
		int myd2 = r % d2;

		p_big[i] = p1[myN + myd1*N]+p2[ myN + myd2*N ];
	}
}

//calculate the Khatri-Rao sums of multiple matrices,
//instead of khatri_rao product, we sum all the cross entries
//mats[k] must be N by nI(k), where N is constant for all elements of mats
//BigMat must be N by Cross(all mats' nI), and pre-allocated
template <typename T>
int SumKhatriRao2Transpose(IN const clMatrix<T>& mat1, IN const clMatrix<T>& mat2,
				OUT clMatrix<T>& BigMat)
{
	//first do some error checking
	if (mat1.nI != mat2.nI || mat1.nI != BigMat.nI)
		return -1;
	if (mat1.nJ*mat2.nJ != BigMat.nJ)
		return -2;

	const uint64_t num_elems = BigMat.nI*BigMat.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (num_elems + dim_block.x-1)/dim_block.x) );

	sum_khatri_rao2_transpose_kernel<<<dim_grid, dim_block>>>( mat1.pData, mat2.pData, mat2.nJ, BigMat.nI, BigMat.pData, num_elems);

	return 0;
}


//derivative w.r.t. the first small matrix
//p1 is N by d1, p2 is N by d2, p_big is N by d1*d2
//num_elems, total number of elements in p_big
//NOTE important: p_big_pd is column major with index of N first, then d2, then d1
template <typename T>
__global__ void sum_khatri_rao2_transpose_deriv1_kernel(IN const T* p_big_pd, IN const T* p2,
											OUT T* p1_pd,
											IN uint64_t d1, IN uint64_t d2, IN uint64_t N, uint64_t num_elems)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	//threads are for elements of p1_pd
	for (uint64_t i = ind; i < num_elems; i += totalThreads){

		uint64_t myN = i % N;
		uint64_t myd1 = i / N;


		T temp = 0;
		uint64_t big_ind = myN+0*N+myd1*d2*N;


		for (int k = 0; k < d2; ++k){
			temp += p_big_pd[big_ind+k*N];
		}
		p1_pd[i] = temp;
	}
}


//derivative w.r.t. the second small matrix
//p1 is N by d1, p2 is N by d2, p_big is N by d1*d2
//num_elems, total number of elements in p_big
//NOTE important: p_big_pd is column major with index of N first, then d2, then d1
template <typename T>
__global__ void sum_khatri_rao2_transpose_deriv2_kernel(IN const T* p_big_pd, IN const T* p1,
											OUT  T* p2_pd,
											IN uint64_t d1, IN uint64_t d2, IN uint64_t N, uint64_t num_elems)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	//threads are for elements of p2_pd
	for (uint64_t i = ind; i < num_elems; i += totalThreads){

		uint64_t myN = i % N;
		uint64_t myd2 = i / N;


		T temp = 0;
		uint64_t big_ind = myN+myd2*N+0*d2*N;

		for (int k = 0; k < d1; ++k){
			temp += p_big_pd[big_ind+k*d2*N];
		}
		p2_pd[i] = temp;
	}
}



//calculate the Khatri-Rao product of multiple matrices
//mats[k] must be N by nI(k), where N is constant for all elements of mats
//BigMat must be N by Cross(all mats' nI), and pre-allocated
//mat1_pd is the partial derivative w.r.t. mat1, same size as mat1
//mat2_pd is the partial derivative w.r.t. mat2, same size as mat2

template <typename T>
int SumKhatriRao2TransposeDeriv(IN const clMatrix<T>& BigMat_pd,
							IN const clMatrix<T>& mat1, IN const clMatrix<T>& mat2,
							OUT clMatrix<T>& mat1_pd, OUT clMatrix<T>& mat2_pd )
{
	//first do some error checking
	if (mat1.nI != mat2.nI || mat1.nI != BigMat_pd.nI)
		return -1;
	if (mat1.nJ*mat2.nJ != BigMat_pd.nJ)
		return -2;
	if (mat1.nI != mat1_pd.nI || mat1.nI != mat2_pd.nI)
		return -3;
	if (mat1.nJ != mat1_pd.nJ || mat1.nJ != mat2.nJ || mat1.nJ != mat2_pd.nJ)
		return -4;

	{
	const uint64_t num_elems = mat1.nI*mat1.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (num_elems + dim_block.x-1)/dim_block.x) );

	sum_khatri_rao2_transpose_deriv1_kernel<<<dim_grid, dim_block>>>( BigMat_pd.pData, mat2.pData, mat1_pd.pData,
													mat1.nJ, mat2.nJ, mat1.nI, num_elems);
	}

	{
	const uint64_t num_elems = mat2.nI*mat2.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (num_elems + dim_block.x-1)/dim_block.x) );

	sum_khatri_rao2_transpose_deriv2_kernel<<<dim_grid, dim_block>>>( BigMat_pd.pData, mat1.pData, mat2_pd.pData,
													mat1.nJ, mat2.nJ, mat1.nI, num_elems);
	}

	return 0;
}



//explicit template instantiation
template void cuda_clMatrixCpy<float>( OUT clMatrix<float> & dest, IN const clMatrix<float> & orig );
template void cuda_clMatrixCpy<float>( OUT clMatrix<float> & dest, IN const void* const data_orig);
template void cuda_clMatrixCpy_d2d<float>( OUT clMatrix<float> & dest, IN const void* const data_orig);
template void cuda_clMatrixCpy<float>( OUT void* data_dest, IN const clMatrix<float> & orig);
template int GetDiag<float>( IN const clMatrix<float>& mat, OUT clMatrix<float>& diag);
template int SetDiag<float>( IN const clMatrix<float>& diag, OUT clMatrix<float>& mat);
template int _cl_cublas_matmul<float>( const float* devpA, char Atr, int AnI, int AnJ,
		const float* devpB, char Btr, int BnI, int BnJ,
		float* devpC, int CnI, int CnJ, float alpha, float beta);

template int cl_cublas_matmul<float>( float alpha, const clMatrix<float>& matA, char Atr,
		const clMatrix<float>& matB, char Btr, float beta, clMatrix<float>& matC);

template int cl_cublas_scal<float>(float alpha, clMatrix<float>& mat);
template int cl_cublas_axpy<float>( float alpha, const clMatrix<float>& mat_x, clMatrix<float>& mat_y);
template float cl_cublas_asum<float>( clMatrix<float>& mat);

template int ReduceInplace< fctrPlus<float>, float>( fctrPlus<float> op, clMatrix<float>& mat, int dim);
template int ReduceInplace< fctrMax<float>, float>( fctrMax<float> op, clMatrix<float>& mat, int dim);
template int ReduceInplace< fctrMin<float>, float>( fctrMin<float> op, clMatrix<float>& mat, int dim);

template float Sum2DInplace<float>( clMatrix<float>& mat);
template float SumSqErr(IN const clMatrix<float>& A, IN const clMatrix<float>& B, OUT clMatrix<float>& Out);
template int KhatriRao2Transpose<float>(IN const clMatrix<float>& mat1, IN const clMatrix<float>& mat2,
				OUT clMatrix<float>& BigMat);

template int KhatriRao2TransposeDeriv<float>(IN const clMatrix<float>& BigMat_pd,
							IN const clMatrix<float>& mat1, IN const clMatrix<float>& mat2,
							float gamma,
							OUT clMatrix<float>& mat1_pd, OUT clMatrix<float>& mat2_pd );

template int SumKhatriRao2Transpose<float>(IN const clMatrix<float>& mat1,
		IN const clMatrix<float>& mat2,	OUT clMatrix<float>& BigMat);

template int SumKhatriRao2TransposeDeriv<float>(IN const clMatrix<float>& BigMat_pd,
							IN const clMatrix<float>& mat1, IN const clMatrix<float>& mat2,
							OUT clMatrix<float>& mat1_pd, OUT clMatrix<float>& mat2_pd );

//double
template void cuda_clMatrixCpy<double>( OUT clMatrix<double> & dest, IN const clMatrix<double> & orig );
template void cuda_clMatrixCpy<double>( OUT clMatrix<double> & dest, IN const void* const data_orig);
template void cuda_clMatrixCpy_d2d<double>( OUT clMatrix<double> & dest, IN const void* const data_orig);
template void cuda_clMatrixCpy<double>( OUT void* data_dest, IN const clMatrix<double> & orig);
template int GetDiag<double>( IN const clMatrix<double>& mat, OUT clMatrix<double>& diag);
template int SetDiag<double>( IN const clMatrix<double>& diag, OUT clMatrix<double>& mat);
template int _cl_cublas_matmul<double>( const double* devpA, char Atr, int AnI, int AnJ,
		const double* devpB, char Btr, int BnI, int BnJ,
		double* devpC, int CnI, int CnJ, double alpha, double beta);

template int cl_cublas_matmul<double>( double alpha, const clMatrix<double>& matA, char Atr,
		const clMatrix<double>& matB, char Btr, double beta, clMatrix<double>& matC);

template int cl_cublas_scal<double>(double alpha, clMatrix<double>& mat);
template int cl_cublas_axpy<double>( double alpha, const clMatrix<double>& mat_x, clMatrix<double>& mat_y);
template double cl_cublas_asum<double>( clMatrix<double>& mat);

template int ReduceInplace< fctrPlus<double>, double>( fctrPlus<double> op, clMatrix<double>& mat, int dim);
template int ReduceInplace< fctrMax<double>, double>( fctrMax<double> op, clMatrix<double>& mat, int dim);
template int ReduceInplace< fctrMin<double>, double>( fctrMin<double> op, clMatrix<double>& mat, int dim);

template double Sum2DInplace<double>( clMatrix<double>& mat);
template double SumSqErr(IN const clMatrix<double>& A, IN const clMatrix<double>& B, OUT clMatrix<double>& Out);
template int KhatriRao2Transpose<double>(IN const clMatrix<double>& mat1, IN const clMatrix<double>& mat2,
				OUT clMatrix<double>& BigMat);

template int KhatriRao2TransposeDeriv<double>(IN const clMatrix<double>& BigMat_pd,
							IN const clMatrix<double>& mat1, IN const clMatrix<double>& mat2,
							double gamma,
							OUT clMatrix<double>& mat1_pd, OUT clMatrix<double>& mat2_pd );

template int SumKhatriRao2Transpose<double>(IN const clMatrix<double>& mat1,
		IN const clMatrix<double>& mat2,	OUT clMatrix<double>& BigMat);

template int SumKhatriRao2TransposeDeriv<double>(IN const clMatrix<double>& BigMat_pd,
							IN const clMatrix<double>& mat1, IN const clMatrix<double>& mat2,
							OUT clMatrix<double>& mat1_pd, OUT clMatrix<double>& mat2_pd );

