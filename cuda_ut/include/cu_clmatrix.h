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

#ifndef _CU_CLMATRIX_H_
#define _CU_CLMATRIX_H_

#include "cu_util.h"
#include "cu_math.h"
#include <helper_cuda.h>

#define IN
#define OUT


//zero based index
class clIndex{
public:
	//constructor
	clIndex():
		i0(-1),
		i1(-1),
		j0(-1),
		j1(-1)
	{}

	clIndex(int _i0, int _i1, int _j0, int _j1) :
		i0(_i0),
		i1(_i1),
		j0(_j0),
		j1(_j1)
	{}

	int i0, i1, j0, j1;
};


//kernels
template <typename T>
__global__ void clMatSetValue_kernel(IN OUT T* pA, IN T val, int dim) {
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;
	for (int i = ind; i < dim; i += totalThreads)
		pA[i] = val;
}


//kernels
template <typename T>
__global__ void clMatLinspace_kernel(IN OUT T* pA, IN T a, IN T d, int dim) {
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;
	for (int i = ind; i < dim; i += totalThreads)
		pA[i] = a+i*d;
}



//class capable of single or double types
template<class T>
class clMatrix{
public:

	//constructor, don't create data
	clMatrix() :
		pData( NULL ),
		nI(0),
		nJ(0),
		nI_inc(1),
		nJ_inc(1),
		_owner(false)
	{}

	//constructor, create data
	clMatrix(int __nI, int __nJ){
		pData = NULL;
		_owner= false;
		nI = nJ = 0;
		CreateData(__nI, __nJ);
	}

	//constructor, create data, set to a certain value
	clMatrix(int __nI, int __nJ, T val){
		pData = NULL;
		_owner= false;
		nI = nJ = 0;
		CreateData(__nI, __nJ);
		SetVal( val );
	}

	//copy constructor, owner will be false if we pass a clMatrix thru a function
	clMatrix(const clMatrix<T> & mat) :
		_owner(false),
		nI(mat.nI),
		nJ(mat.nJ),
		nI_inc(mat.nI_inc),
		nJ_inc(mat.nJ_inc),
		pData(mat.pData)
	{}

	//destructor
	~clMatrix(){
		if (_owner && pData != NULL){
			checkCudaErrors( cudaFree( pData ));
			pData = NULL;
			//causes bug on nvidia driver 290. since it needs cudaThreadExit()
			//also: if this is rasied, need to have {  ... } to bound a clmat declaration
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//member functions
	//////////////////////////////////////////////////////////////////////////

	//manual destructor, needs to be called for all clMat objects in the Main Program!
	//such that the destructor is not called twice, else cuda 4.x sdk crashes for some reason
	//and we get a seg fault
	void Free(){
		if (_owner && pData != NULL){
			checkCudaErrors( cudaFree( pData ));
			pData = NULL;
		}
	};

	//creates data
	void CreateData(int __nI, int __nJ){
		Free();
		nI = __nI;
		nJ = __nJ;
		nI_inc = nJ_inc =1;

		size_t debugfree, debugtotal;
		cudaMemGetInfo (&debugfree, &debugtotal);
		float mem_free_ratio = float(debugfree)/(nI*nJ*sizeof(T));

		if (mem_free_ratio < 1.5f){
			clASSERT(false, "cuda_clMatrix - Not Enough Memory on GPU!!");
			//clPrintf("\nBoard Memory free:%.4f total:%.4f",  float(debugfree)/1048576, float(debugtotal)/1048576);
		}
		checkCudaErrors( cudaMalloc((void**) &( pData), sizeof(T)*nI*nJ ));
		SetVal( 0.0 ); //set all values to zero
		_owner = true;
	}

	//sets its data to a particular value
	void SetVal(T val){
		if (pData != NULL){
			const unsigned int datadim = nI*nJ;
			dim3 dim_block( MEDIUM_NUM_THREADS );
			dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

			clMatSetValue_kernel<T><<<dim_grid, dim_block>>>( pData, val, datadim);
		}
	}

	void Linspace(T val0, T val1 ){
		if (pData != NULL){
			const unsigned int datadim = nI*nJ;
			dim3 dim_block( MEDIUM_NUM_THREADS );
			dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

			clMatLinspace_kernel<T><<<dim_grid, dim_block>>>( pData, val0, (val1-val0)/(datadim-1), datadim);
		}
	}

	int CopyFrom(const clMatrix<T> & );
	int CopySubmatFrom(clIndex, const clMatrix<T>&, clIndex);
	int CopySubmatFrom(const clMatrix<T>&, clIndex );
	int CopySubmatFrom(clIndex, const clMatrix<T>& );
	int CopyRowFrom(int, const clMatrix<T>&, int );

	clMatrix<T> 	TrView();		//get the transpose view
	int 			CopyFromTr(const clMatrix<T>& );
	clMatrix<T> 	TrDeep();		//get the transpose deep version

	clMatrix<T> 	ColView(int, int);		//get several columns
	clMatrix<T> 	ColView(int );			//get one column
	clMatrix<T> 	ReshapeView(int, int ); //reshape to a different size

	T		GetElem(int i, int j);				//retrieve an element
	int		SetElem(int i, int j, T val);		//Set an element to a value

	//////////////////////////////////////////////////////////////////////////
	//member variables
	//////////////////////////////////////////////////////////////////////////

	int 	nI, nJ; //dimension of the matrix
	T* 		pData;  //data pointer
	bool 	_owner; //whether or not we own the data in pData
	int 	nI_inc, nJ_inc; //stride
};

template<typename T>
T clMatrix<T>::GetElem( int i, int j ){

	if (i >= nI || j >= nJ || pData ==NULL || i < 0 || j < 0)
		return -9999;

	T val;
	checkCudaErrors( cudaMemcpy( &val, pData+j*nI+i, sizeof(T), cudaMemcpyDeviceToHost));
	return val;
}

template<typename T>
int clMatrix<T>::SetElem( int i, int j, T val){

	if (i >= nI || j >= nJ || pData ==NULL || i < 0 || j < 0)
		return -1;

	checkCudaErrors( cudaMemcpy( pData+j*nI+i, &val,
			sizeof(T), cudaMemcpyHostToDevice));
	return 0;
}



////////////////////////////////////////////////////////////////////////////
//Deep copy from another matrix
template <typename T>
int clMatrix<T>::CopyFrom(const clMatrix<T>& othermat)
{
	if (nI_inc == 1 && nJ_inc ==1 && othermat.nI_inc == 1 && othermat.nJ_inc ==1){
		cuda_clMatrixCpy( *this, othermat);
	}else{
		clASSERT(false, "clMatrix<T>::CopyFrom don't support other stridings yet!");
	}

	return 0;
}



 //Deep copy a submatrix from another matrix to myself
//copy a different submatrix to a submatrix of myself
template <typename T>
int clMatrix<T>::CopySubmatFrom(clIndex myinds, const clMatrix<T>& othermat, clIndex otherinds)
{
	return  SubmatEleWisefun(T(1.0), othermat, otherinds, fctrAlphaPlusBeta<T>(), T(0), othermat, otherinds, *this, myinds);
}

//copy a sub matrix from another matrix, must be continous blocks
//assume self's dimensionality is same as otherinds
template <typename T>
int clMatrix<T>::CopySubmatFrom(const clMatrix<T>& othermat, clIndex otherinds )
{
	clIndex myinds(0, this->nI-1, 0, this->nJ-1);

	//check to see if dimension matches will be inside this function
	return  SubmatEleWisefun(T(1), othermat, otherinds, fctrAlphaPlusBeta<T>(), T(0), othermat, otherinds, *this, myinds);
}

//copy the complete matrix to my submatrix
//assume othermat's size is same as specified in myinds.
template <typename T>
int clMatrix<T>::CopySubmatFrom(clIndex myinds, const clMatrix<T>& othermat)
{
	clIndex otherinds(0, othermat.nI-1, 0, othermat.nJ-1);
	//check to see if dimension matches will be inside this function
	return  SubmatEleWisefun(T(1), othermat, otherinds, fctrAlphaPlusBeta<T>(), T(0), othermat, otherinds, *this, myinds);
}

//copy the complete matrix to my submatrix
//assume othermat's size is same as specified in myinds.
template <typename T>
int clMatrix<T>::CopyRowFrom(int row, const clMatrix<T>& othermat, int other_row)
{
	clIndex myinds(row, row, 0, this->nJ-1);
	clIndex otherinds(other_row, other_row, 0, othermat.nJ-1);

	//check to see if dimension matches will be inside this function
	return  SubmatEleWisefun(T(1), othermat, otherinds, fctrAlphaPlusBeta<T>(), T(0), othermat, otherinds, *this, myinds);
}



/* NOT yet implemented

//get a transposed of self, deep copying, very expensive
clMat clMat::TrDeep()
{
	if (nI > BLOCK_DIM*MAX_GRIDS || nJ > BLOCK_DIM*MAX_GRIDS) //to big to handle
	{
		clASSERT( false, "TrDeep(): too big to handle");
	}
	if (pData == NULL || !_owner || nI_inc != 1 || nJ_inc != 1)
	{
		clASSERT( false, "TrDeep(): pData == NULL, or not owner, or incs");
	}

	clMat out(nJ, nI); //expensive to create another matrix!
	clMat eye(nI, nI);
	clMat ones(nI, 1, 1.0f);
	cr( SetDiag(ones, eye) );

	cr( ABeqC(1.0f, *this, 't', eye, 'n', out) )

	int temp = nI;
	nI = nJ;
	nJ = temp;

	cr( this->CopyFrom(out) )
	return *this;
}


//copy a the transpose of othermat to self
//expensive, needs a better solution!
int clMat::CopyFromTr(const clMat & othermat)
{
	if (nI > BLOCK_DIM*MAX_GRIDS || nJ > BLOCK_DIM*MAX_GRIDS) //to big to handle
	{
		clASSERT( false, "CopyFromTr(): too big to handel");
	}
	if (othermat.pData == NULL || nI_inc != 1 || nJ_inc != 1)
	{
		clASSERT( false, "CopyFromTr(): pData == NULL, or incs");
	}
	if ( othermat.nI != nJ || othermat.nJ != nI){
		clASSERT( false, "CopyFromTr(): dimension mismatch");
	}

	clMat eye(othermat.nJ, othermat.nJ);
	clMat ones(othermat.nJ, 1, 1.0f);
	cr( SetDiag(ones, eye) );

	cr( ABeqC(1.0f, eye, 'n', othermat, 't', *this) )

	return 0;
}
*/



//get a column view
template <typename T>
clMatrix<T> clMatrix<T>::ColView(int c_start, int c_end)
{
	clASSERT(c_start >=0 && c_end < nJ && c_end >= c_start, "clMatrix<T>::ColView() index wrong!");
	clMatrix<T> col;
	col._owner = false;
	col.nI = nI;
    col.nJ = c_end-c_start+1;
	col.pData = pData+c_start*nI;

	return col;
}

//get a column view
template <typename T>
clMatrix<T> clMatrix<T>::ColView(int c)
{
	clASSERT(c >=0 && c < nJ, "clMatrix<T>::ColView() index wrong!");
	clMatrix<T> col;
	col._owner = false;
	col.nI = nI;
    col.nJ = 1;
	col.pData = pData+c*nI;

	return col;
}

//reshape to a new matrix, with same pointer
template <typename T>
clMatrix<T> clMatrix<T>::ReshapeView(int new_i, int new_j)
{
	clASSERT(new_i*new_j == nI*nJ, "clMatrix<T>::ReshapeView() dimensions wrong!");
	clMatrix<T> newmat;
	newmat._owner = false;
	newmat.nI = new_i;
    newmat.nJ = new_j;
	newmat.pData = pData;
	return newmat;
}


//get a transposed view of a row or col vector!
template <typename T>
clMatrix<T> clMatrix<T>::TrView()
{
	clASSERT(nI == 1 || nJ == 1, "clMatrix::TrView() must operate on a vector!");
	clMatrix<T> tr;
	tr._owner = false;
	tr.nI = nJ;
    tr.nJ = nI;
	tr.pData = pData;

	return tr;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix with 3 dimensions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

//class capable of single or double types
template<class T>
class clMatrix3{
public:

	//constructor, don't create data
	clMatrix3() :
		pData( NULL ),
		nI(0),
		nJ(0),
		nK(0),
		_owner(false)
	{}

	//constructor, create data
	clMatrix3(int __nI, int __nJ, int __nK){
		pData = NULL;
		_owner= false;
		nI = nJ = nK = 0;
		CreateData(__nI, __nJ, __nK);
	}

	//constructor, create data, set to a certain value
	clMatrix3(int __nI, int __nJ, int __nK, T val){
		pData = NULL;
		_owner= false;
		nI = nJ = nK = 0;
		CreateData(__nI, __nJ, __nK);
		SetVal( val );
	}

	//copy constructor, owner will be false if we pass a clMatrix thru a function
	clMatrix3(const clMatrix3<T> & mat) :
		_owner(false),
		nI(mat.nI),
		nJ(mat.nJ),
		nK(mat.nK),
		pData(mat.pData)
	{}

	//destructor
	~clMatrix3(){
		if (_owner && pData != NULL){
			checkCudaErrors( cudaFree( pData ));
			pData = NULL; //causes bug on nvidia driver 290. since it needs cudaThreadExit()
		}
	};

	//////////////////////////////////////////////////////////////////////////
	//member functions
	//////////////////////////////////////////////////////////////////////////

	//manual destructor, needs to be called for all clMat objects in the Main Program!
	//such that the destructor is not called twice, else cuda 4.x sdk crashes for some reason
	//and we get a seg fault
	void Free(){
		if (_owner && pData != NULL){
			checkCudaErrors( cudaFree( pData ));
			pData = NULL;
		}
	};

	//creates data
	void CreateData(int __nI, int __nJ, int __nK){
		Free();
		nI = __nI;
		nJ = __nJ;
		nK = __nK;

		size_t debugfree, debugtotal;
		cudaMemGetInfo (&debugfree, &debugtotal);
		float mem_free_ratio = float(debugfree)/(nI*nJ*nK*sizeof(T));

		if (mem_free_ratio < 1.5f){
			clASSERT(false, "cuda_clMatrix - Not Enough Memory on GPU!!");
			//clPrintf("\nBoard Memory free:%.4f total:%.4f",  float(debugfree)/1048576, float(debugtotal)/1048576);
		}
		checkCudaErrors( cudaMalloc((void**) &( pData), sizeof(T)*nI*nJ*nK ));
		SetVal( 0.0 ); //set all values to zero
		_owner = true;
	}

	//sets its data to a particular value
	void SetVal(T val){
		if (pData != NULL){
			const unsigned int datadim = nI*nJ*nK;
			dim3 dim_block( MEDIUM_NUM_THREADS );
			dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

			clMatSetValue_kernel<T><<<dim_grid, dim_block>>>( pData, val, datadim);
		}
	}

	int CopyFrom(const clMatrix3<T> & );

	clMatrix<T> 	SliceView(int);		//get one slice in depth or 'k' dimension or 3rd dimension

	//////////////////////////////////////////////////////////////////////////
	//member variables
	//////////////////////////////////////////////////////////////////////////

	int 	nI, nJ, nK; //dimension of the matrix
	T* 		pData;  //data pointer
	bool 	_owner; //whether or not we own the data in pData
};

// *** Device <---- Device *** copy data from one clMat onto another clMat
template <typename T>
void  cuda_clMatrix3Cpy( OUT clMatrix3<T> & dest, IN const clMatrix3<T> & orig ){

	clASSERT(dest.nI == orig.nI && dest.nJ == orig.nJ && dest.nK == orig.nK, "\n dest and orig dim not same!\n");
	clASSERT( dest.pData != NULL && orig.pData != NULL, "cuda_clMatrix3Cpy: one of data ptr is NULL");

	checkCudaErrors( cudaMemcpy( dest.pData,  orig.pData,
			sizeof(T)*orig.nI*orig.nJ*orig.nK, cudaMemcpyDeviceToDevice));
}

// *** Device <---- host *** copy data from host to device
template <typename T>
void cuda_clMatrix3Cpy( OUT clMatrix3<T> & dest, IN const void* const data_orig){

	clASSERT( dest.pData != NULL && data_orig != NULL, "cuda_clMatrix3Cpy: one of data ptr is NULL");

	checkCudaErrors( cudaMemcpy(dest.pData, data_orig,
			sizeof(T)*dest.nI*dest.nJ*dest.nK, cudaMemcpyHostToDevice));
}


////////////////////////////////////////////////////////////////////////////
//Deep copy from another matrix
template <typename T>
int clMatrix3<T>::CopyFrom(const clMatrix3<T>& othermat)
{
	cuda_clMatrix3Cpy( *this, othermat);
	return 0;
}

//this function simply returns a matrix (shallow) view of a clMat3
//k is the zero-based index of which slice in the 3rd dimension
template <typename T>
clMatrix<T> clMatrix3<T>::SliceView(int k)
{
	clASSERT( k >= 0 && k < nK, "Slice3View k out of bounds!");

	clMatrix<T> sliceview;
	sliceview._owner = false;
	sliceview.nI = nI;
	sliceview.nJ = nJ;
	sliceview.pData = pData+k*nI*nJ;

	return sliceview;
}


#endif
