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

#ifndef _CU_CLTENSOR_H_
#define _CU_CLTENSOR_H_

#include <vector>
#include <inttypes.h>

#include "cu_util.h"
#include "cu_math.h"
#include "cu_clmatrix.h"


#define IN
#define OUT

using namespace std;

//zero based index
class clTensorIndex{
public:
	//constructor
	clTensorIndex():
		i0(-1),
		i1(-1),
		j0(-1),
		j1(-1)
	{}

	clTensorIndex(int _i0, int _i1, int _j0, int _j1) :
		i0(_i0),
		i1(_i1),
		j0(_j0),
		j1(_j1)
	{}

	int i0, i1, j0, j1;
};


//kernels
template <typename T>
__global__ void clTensorSetValue_kernel(IN OUT T* pA, IN T val, uint64_t dim) {
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;
	for (uint64_t i = ind; i < dim; i += totalThreads)
		pA[i] = val;
}


//class capable of single or double types
template<class T>
class clTensor{
public:

	//constructor, don't create data
	clTensor():
		pData( NULL ),
		nModes(0),
		total_dim(0),
		_owner(false)
	{
		dims.resize(0);
		cumuel.resize(0);
		dev_cumuel = NULL;
	}

	//constructor, create data
	clTensor(const std::vector<int>& __nDim ){
		pData = NULL;
		_owner= false;
		nModes = 0;
		total_dim = 0;
		dev_cumuel = NULL;
		CreateData( __nDim );
	}

	//constructor, shallow wrapper around a matrix
	clTensor(const std::vector<int>& __nDim , const clMatrix<T>& mat){

		dims = __nDim;
		nModes = dims.size();
		clASSERT(nModes > 0, "clTensor::CreateData - nModes > 0");

		total_dim = prod(dims);
		clASSERT( total_dim == mat.nI*mat.nJ, "clTensor total_dim == mat.nI*mat.nJ");

		_owner= false;
		dev_cumuel = NULL;

		pData = mat.pData;
		/////////////////////////////////////////////////////////////////////////
		UpdateCumuelAlloc();
	}

	//constructor, create data, set to a certain value
	clTensor( std::vector<int>& __nDim, T val){
		pData = NULL;
		_owner= false;
		nModes = 0;
		total_dim = 0;
		dev_cumuel = NULL;
		CreateData( __nDim );
		SetVal( val );
	}

	//copy constructor, owner will be false if we pass a clTensor thru a function
	clTensor(const clTensor<T> & ten) :
		_owner(false),
		dims(ten.dims),
		total_dim(ten.total_dim),
		nModes(ten.nModes),
		pData(ten.pData),
		cumuel(ten.cumuel),
		dev_cumuel(NULL)
	{
		UpdateCumuelAlloc(); //always have its own version of dev_cumuel
	}

	//destructor
	~clTensor(){
		if (_owner && pData != NULL){
			checkCudaErrors( cudaFree( pData ));
			pData = NULL;
			//causes bug on nvidia driver 290. since it needs cudaThreadExit()
		}

		if (dev_cumuel != NULL){
			checkCudaErrors( cudaFree( dev_cumuel ));
			dev_cumuel = NULL;
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
		if (dev_cumuel != NULL){
			checkCudaErrors( cudaFree( dev_cumuel ));
			dev_cumuel = NULL;
		}
	};

	//creates data
	void CreateData(const std::vector<int>& __nDim ){
		Free();
		dims = __nDim;
		nModes = dims.size();

		clASSERT(nModes > 0, "clTensor::CreateData - nModes > 0");

		total_dim = prod(dims);

		size_t debugfree, debugtotal;
		cudaMemGetInfo (&debugfree, &debugtotal);
		float mem_free_ratio = float(debugfree)/(total_dim*sizeof(T));

		if (mem_free_ratio < 1.5f){
			clASSERT(false, "clTensor::CreateData - Not Enough Memory on GPU!!");
			//clPrintf("\nBoard Memory free:%.4f total:%.4f",  float(debugfree)/1048576, float(debugtotal)/1048576);
		}
		checkCudaErrors( cudaMalloc((void**) &( pData), sizeof(T)*total_dim ));
		SetVal( T(0) ); //set all values to zero
		_owner = true;
		dev_cumuel = NULL;

		/////////////////////////////////////////////////////////////////////////
		UpdateCumuelAlloc();
	}

	//sets its data to a particular value
	void SetVal(T val){
		if (pData != NULL){
			const uint64_t datadim = total_dim;
			dim3 dim_block( MEDIUM_NUM_THREADS );
			dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

			clTensorSetValue_kernel<T><<<dim_grid, dim_block>>>( pData, val, datadim);
		}
	}

	int CopyFrom(const clTensor<T> & );

	clTensor<T> 	ReshapeView( const std::vector<int>& ); //reshape to a different size
	clMatrix<T> 	ReshapeMatrixView( int nI, int nJ ); //reshape to a different size
	clMatrix<T> 	CastMatrixView();	//shallow matrix view, if tensor is already 2 modes
	clMatrix<T> 	VecView();			//serialize the tensor, shallow view
	clTensor<T> 	ChunkView(int);		//return the chunk given the index of last dimension


	T		GetElem(const std::vector<int>& inds ); //retrieve an element
	int		SetElem(const std::vector<int>& inds, T val);		   //Set an element to a value

	clMatrix<T>		t2m( int dim, OUT clTensor<T> & ); 			//deep matrixcization
	int				m2t( int dim, IN const clTensor<T> & ); 	//convert back from a matricized matrix


	void    UpdateCumuel() //assume the dims.size() hasn't changed, simply update the values of dev_cumuel
	{
		clASSERT(nModes == dims.size(), "clTensor::UpdateCumuel - nModes == dims.size()");
		clASSERT(nModes == cumuel.size(), "clTensor::UpdateCumuel - nModes == cumuel.size()");
		clASSERT(dev_cumuel != NULL, "clTensor::UpdateCumuel - dev_cumuel != NULL");

		cumuel[0] = 1;
		for ( unsigned int d = 1; d < nModes; ++d ){
			cumuel[d] = cumuel[d-1]*dims[d-1];
		}
		checkCudaErrors( cudaMemcpy(dev_cumuel, &cumuel[0], sizeof(int)*nModes, cudaMemcpyHostToDevice));
	}

	void    UpdateCumuelAlloc() //after change to dims, we update this, re-allocate
	{
		clASSERT(nModes == dims.size(), "clTensor::UpdateCumuel - nModes == dims.size()");

		//free memory first
		if (dev_cumuel != NULL){
			checkCudaErrors( cudaFree( dev_cumuel ));
			dev_cumuel = NULL;
		}
		checkCudaErrors( cudaMalloc((void**) &( dev_cumuel), sizeof(int)*nModes ));
		cumuel.resize(nModes);

		UpdateCumuel();
	}

	//////////////////////////////////////////////////////////////////////////
	//member variables
	//////////////////////////////////////////////////////////////////////////

	std::vector<int> 		dims;   //if this is changed, cumuel must also change!
	int 					nModes;  //the mode of the tensor
	T* 						pData;  //data pointer
	bool 					_owner; //whether or not we own the data in pData
	uint64_t				total_dim;  // == prod(dims);

	std::vector<int> 		cumuel; //on host side
	int*    				dev_cumuel; //array same length as dims, but on device side, e.g. [1 4 20 60]
};

template<typename T>
T clTensor<T>::GetElem( const std::vector<int>& inds ){

	if (inds.size() != nModes || pData == NULL)
		return -9999;

	uint64_t offset = 0ULL;
	for (int i = inds.size()-1; i >= 0; --i){

		if (inds[i] >= dims[i] ||  inds[i] < 0 )
			return -9999;

		if (i > 0){
			offset = (offset+inds[i])*dims[i-1];
		}else{
			offset += inds[i];
		}
	}

	T val;
	checkCudaErrors( cudaMemcpy( &val, pData+offset, sizeof(T), cudaMemcpyDeviceToHost));
	return val;
}

template<typename T>
int clTensor<T>::SetElem( const std::vector<int>& inds, T val){

	if (inds.size() != nModes || pData == NULL)
		return -9999;

	uint64_t offset = 0ULL;
	for (int i = inds.size()-1; i >= 0; --i){

		if (inds[i] >= dims[i] ||  inds[i] < 0 )
			return -9999;

		if (i > 0){
			offset = (offset+inds[i])*dims[i-1];
		}else{
			offset += inds[i];
		}
	}

	checkCudaErrors( cudaMemcpy( pData+offset, &val, sizeof(T), cudaMemcpyHostToDevice));
	return 0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////
// basic matrix operations
/////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename T>
bool inline clTensorSizeEqual( const clTensor<T>& mat1, const clTensor<T>& mat2 ){

	if ( mat1.nModes != mat2.nModes )
		return false;

	for (unsigned int i = 0; i < mat1.nModes; ++i){
		if (mat1.dims[i] != mat2.dims[i])
			return false;
	}
	return true;
}

template <typename T>
bool inline clTensorSizeEqual( const clTensor<T>& mat, const std::vector<int>& inds ){

	if ( mat.nModes != inds.size() )
		return false;

	for (unsigned int i = 0; i < mat.nModes; ++i){
		if (mat.dims[i] != inds[i])
			return false;
	}
	return true;
}



// *** Device <---- Device *** copy data from one clMat onto another clMat
template <typename T>
void  cuda_clTensorCpy( OUT clTensor<T> & dest, IN const clTensor<T> & orig ){

	clASSERT( clTesnorSizeEqual(dest, orig), "\n dest and orig dim not same!\n");
	clASSERT( dest.pData != NULL && orig.pData != NULL, "cuda_clTensorCpy: one of data ptr is NULL");

	checkCudaErrors( cudaMemcpy( dest.pData,  orig.pData,
			sizeof(T)*dest.total_dim, cudaMemcpyDeviceToDevice));
}

// *** Device <---- host *** copy data from host to device
template <typename T>
void cuda_clTensorCpy( OUT clTensor<T> & dest, IN const void* const data_orig){

	clASSERT( dest.pData != NULL && data_orig != NULL, "cuda_clTensorCpy: one of data ptr is NULL");

	checkCudaErrors( cudaMemcpy(dest.pData, data_orig,
			sizeof(T)*dest.total_dim, cudaMemcpyHostToDevice));
}

// *** Device <---- device void pointer ***
template <typename T>
void cuda_clTensorCpy_d2d( OUT clTensor<T> & dest, IN const void* const data_orig){

	clASSERT( dest.pData != NULL && data_orig != NULL, "cuda_clTensorCpy_d2d: one of data ptr is NULL");

	checkCudaErrors( cudaMemcpy(dest.pData, data_orig,
			sizeof(T)*dest.total_dim, cudaMemcpyDeviceToDevice));
}

//*** host <---- Device ***     copy data from device to host
template <typename T>
void cuda_clTensorCpy( OUT void* data_dest, IN const clTensor<T> & orig){

	clASSERT( orig.pData != NULL && data_dest != NULL, "cuda_clTensorCpy: one of data ptr is NULL");

	checkCudaErrors( cudaMemcpy(data_dest, orig.pData,
			sizeof(T)*orig.total_dim, cudaMemcpyDeviceToHost));
}


//return a view of the matrix, wrapped in a tensor class
template <typename T>
clTensor<T> CastTensorView(const clMatrix<T>& mat){

	clASSERT(mat.pData != NULL, "clTensor<T>::CastTensorView() pData==Null!");
	clASSERT(mat.nI*mat.nJ > 0, "clTensor<T>::CastTensorView() mat dimension wrong!");

	clTensor<T> ten;
	ten.nModes = 2;
	ten.dims.resize(ten.nModes);
	ten.dims[0] = mat.nI;
	ten.dims[1] = mat.nJ;
	ten.total_dim = prod(ten.dims);
	ten.UpdateCumuelAlloc();
	ten.pData = mat.pData;
	ten._owner= false;
	return ten;
}


////////////////////////////////////////////////////////////////////////////
//Deep copy from another matrix
template <typename T>
int clTensor<T>::CopyFrom(const clTensor<T>& othermat)
{
	cuda_clTensorCpy( *this, othermat);
	return 0;
}


//tensor vectorization
template <typename T>
clMatrix<T> clTensor<T>::VecView(){

	clASSERT(pData != NULL, "clTensor<T>::VecView() pData==Null!");

	clMatrix<T> vec;
	vec._owner = false;
	vec.nI = prod(dims);
    vec.nJ = 1;
	vec.pData = pData;
	return vec;
}

//reshape a tensor into another tensor
template <typename T>
clTensor<T> clTensor<T>::ReshapeView(IN const vector<int>& new_dims){

	clASSERT(pData != NULL, "clTensor<T>::Reshape() pData==Null!");
	clASSERT(total_dim == prod(new_dims), "clTensor<T>::Reshape() prod dim error");

	clTensor<T> ten;
	ten.dims = new_dims;
	ten.nModes = ten.dims.size();
	ten.total_dim = prod(ten.dims);
	ten.UpdateCumuelAlloc();

	ten.pData = pData;
	ten._owner= false;
	return ten;
}

//the chunk given the last dimension index
//same as frontal slice for a Mode-3 tensor
//e.g. for a 3x4x5x6 tensor, ChunkView() will return a tensor of 3x4x5
template <typename T>
clTensor<T> clTensor<T>::ChunkView( int index ){

	clASSERT(pData != NULL, "clTensor<T>::ChunkView() pData==Null!");
	clASSERT(index >=0 && index < dims[dims.size()-1], "ChunkView() index oob!");

	uint64_t offset = index*cumuel[dims.size()-1];

	clTensor<T> ten;
	ten.dims = dims;
	ten.dims.pop_back();
	ten.nModes = ten.dims.size();
	ten.total_dim = prod(ten.dims);
	ten.UpdateCumuelAlloc();

	ten.pData = pData+offset;
	ten._owner= false;
	return ten;
}



//reshape a tensor into a matrix of nI by nJ
template <typename T>
clMatrix<T> clTensor<T>::ReshapeMatrixView(int nI, int nJ){

	clASSERT(pData != NULL, "clTensor<T>::ReshapeMatrixView() pData==Null!");
	clASSERT(total_dim == nI*nJ, "clTensor<T>::ReshapeMatrixView() dim error");

	clMatrix<T> mat;
	mat._owner = false;
	mat.nI = nI;
    mat.nJ = nJ;
	mat.pData = pData;
	return mat;
}


//Convert Tensor to a Matrix, if tensor is already mode==2
template <typename T>
clMatrix<T> clTensor<T>::CastMatrixView(){

	clASSERT(pData != NULL, "clTensor<T>::CastMatrixView() pData==Null!");
	clASSERT( nModes == 2 && dims.size() ==2, "clTensor<T>::CastMatrixView() not Mode-2 !");

	clMatrix<T> mat;
	mat._owner = false;
	mat.nI = dims[0];
    mat.nJ = dims[1];
	mat.pData = pData;
	return mat;
}



//matricization
//dim - which dim should the tensor fibre span?  0-based
//ten2 - is the tensor underlying the matrix *view* that is returned
//assuming ten2 is already correct dimensionality (can be same as original tensor)
template <typename T>
clMatrix<T>	clTensor<T>::t2m( int dim, OUT clTensor<T> & ten2){

	clASSERT( !(dim >= nModes || dim < 0 || nModes <= 0), "t2m error!");
	clASSERT ( !(pData == NULL || ten2.pData == NULL || ten2.total_dim != total_dim), "t2m error!");

	if (dim == 0){
		cuda_clTensorCpy_d2d( ten2, pData);
		return ten2.ReshapeMatrixView(dims[0], total_dim/dims[0] );
	}else{
		vector<int> perm(nModes);

		perm[0] = dim;
		int ind = 1;
		for (int i = 0; i < perm.size(); ++i){
			if (i != dim){
				perm[ind++] = i;
			}
		}

		TensorPermute(*this, perm, NULL, ten2 );
		return ten2.ReshapeMatrixView(dims[dim], total_dim/dims[dim] );
	}
}


//copy what's in the matrix back into my tensor
//dim - which dim should the tensor fibre span?  0-based
//ten2 tensor has the exact correct permuted dimensionality as calculated from t2m
template <typename T>
int clTensor<T>::m2t( int dim, IN const clTensor<T> & ten2 ){

	clASSERT( !(dim >= nModes || dim < 0 || nModes <= 0), "m2t error!");
	clASSERT ( !(pData == NULL || ten2.pData == NULL || ten2.total_dim != total_dim), "m2t error!");

	if (dim == 0){ //no permutation needed
		cuda_clTensorCpy_d2d( *this, ten2.pData);
	}else{

		vector<int> perm(nModes);
		perm[0] = dim;
		int ind = 1;
		for (int i = 0; i < perm.size(); ++i){
			if (i != dim){
				perm[ind++] = i;
			}
		}

		TensorIPermute(ten2, perm, NULL, *this );
	}

	return 0;
}


// nModes - how many mode does tensor has
// num_elems - the total 1D length of the tensor elements
// p_perm[0:nModes-1] - zero-based permutation of the orginal order
// but from the perspective of the output order
// e.g. [0 3 1 2] says that the 2nd mode of pB is the 4th mode of pA.
// p_m_out - buffer, the number of elements in the lower order: e.g. [1 4 20 60]
// if p_outdim == [4 5 3 any]
// coord[0:nModes-1] - buffer for the coordinates
template <typename T>
__global__ void clTensorPermute_kernel(IN T* pA, OUT T* pB, uint64_t num_elems, int nModes,
		const int* dev_perm, const int* dev_cumuel_in, const int* dev_cumuel_out)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (uint64_t i = ind; i < num_elems; i += totalThreads){

		//find output coordinates, perm to find input coordinates,
		//and find output index in one for loop
		uint64_t rem = i;
		uint64_t a_ind = 0;
		for ( int d = nModes-1; d > 0; d-- ){
			unsigned int coord_out = rem/dev_cumuel_out[d];
			rem = rem % dev_cumuel_out[d];

			unsigned int coord_in_p_perm_d = coord_out;
			a_ind += coord_in_p_perm_d*dev_cumuel_in[ dev_perm[d] ];
		}
		//now, coord_out_0 == rem;
		a_ind += rem*dev_cumuel_in[ dev_perm[0] ];
		pB[i] = pA[a_ind];
	}

}


//perm is a vector describing the permutation
//e.g. out tensor has perm == [0 3 1 2],
//means the last mode of orig tensor is shifted to the 2nd mode
//assume that out has been allocated and equal in size to orig, but .dims is not necessarily correct
//dev_perm_buf could be NULL, in that case, device memory will be allocated, and freed
template <typename T>
int TensorPermute(IN const clTensor<T>& orig, IN const vector<int>& perm, IN int* dev_perm_buf,
			      OUT clTensor<T>& out ){

	if ( orig.total_dim != out.total_dim || orig.pData == NULL || out.pData == NULL)
		return -1;

	if (perm.size() != orig.nModes || perm.size() != out.nModes ||
		perm.size() != orig.dims.size() || perm.size() != out.dims.size() )
		return -2;

	if (out.dev_cumuel == NULL)
		return -3;

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//compute proper output dimensionality
	for ( unsigned int d = 0; d < orig.nModes; ++d ){
		out.dims[d] = orig.dims[ perm[d] ];
	}
	out.UpdateCumuel();
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	bool bAllocate = (dev_perm_buf == NULL);
	if ( bAllocate ){
		checkCudaErrors( cudaMalloc((void**) &(dev_perm_buf), sizeof(int)*perm.size() ));
	}//else we assume dev_perm_buf has buffer of length perm.size()
	checkCudaErrors( cudaMemcpy( dev_perm_buf, &perm[0], sizeof(int)*perm.size(), cudaMemcpyHostToDevice));

	const uint64_t num_elems = orig.total_dim;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (num_elems + dim_block.x-1)/dim_block.x) );

	clTensorPermute_kernel<<<dim_grid, dim_block>>>( orig.pData, out.pData, num_elems, orig.nModes,
													dev_perm_buf, orig.dev_cumuel, out.dev_cumuel);

	if ( bAllocate ){ //free memory
		checkCudaErrors( cudaFree(dev_perm_buf));
	}

	return 0;
}

//inverse permutation, same as matlab's ipermute
template <typename T>
int TensorIPermute(IN const clTensor<T>& permuted, IN const vector<int>& perm, IN int* dev_perm_buf,
				   OUT clTensor<T>& orig ){

	vector<int> iperm(perm.size());

	for (int i =0; i < perm.size(); ++i){
		iperm[perm[i]] =i;
	}

	return TensorPermute( permuted, iperm, dev_perm_buf, orig);
}




/////////////////////////////////////////////////////////////////////////////////////////////////////
//tensor-vector multiplication
/////////////////////////////////////////////////////////////////////////////////////////////////////




/////////////////////////////////////////////////////////////////////////////////////////////////////
//tensor matrix multiplications
/////////////////////////////////////////////////////////////////////////////////////////////////////






//t = alpha*t
template <typename T>
inline int cl_cublas_scal(T alpha, clTensor<T>& ten){

	clMatrix<T> v = ten.VecView();
	return cl_cublas_scal( alpha, v );
}


#endif
