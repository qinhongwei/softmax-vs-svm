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

//this file includes some common utility functions needed when mixing cuda code and mex functions
//CT 3/2012

#ifndef _CU_MATRIX_MEX_H_
#define _CU_MATRIX_MEX_H_

#include <vector>
#include <iostream>

#include "cu_util.h"
#include "mex_util.h"
#include "cu_clmatrix.h"
#include "cu_cltensor.h"
#include "cu_matrix_ops.h"



/* this function is for selecting a GPU based on server
 *  globalGPUIDvar is a variable in matlab workspace w/ the id
 *
 *  Example of use:
	int gpu_id = SelectGPUFromServer( "GPUID" );
	GpuRandInit(nSeed, GetGPUArchitecture( "FERMI", true/false ) );

	//initialize cublas
	cublasCheckError(cublasInit(), "cublas failed to initialize");

	...

	GpuRandDestroy();
	cublasCheckError(cublasShutdown(), "cublas shutdown error");
 */
int SelectGPUFromServer(char * globalGPUIDvar)
{
	mxArray * pGPUID = mexGetVariable("global", globalGPUIDvar);
	mxASSERT( pGPUID != NULL && !mxIsEmpty(pGPUID), "NO Global GPUID variable!");
	int gpu_id = mxGetScalar(pGPUID);
	mxASSERT( gpu_id >= 0, "\nNO GPU Boards Available.");
	int err=cudaSetDevice(gpu_id);
	mxASSERT(err == cudaSuccess || err ==  cudaErrorSetOnActiveProcess, "cudaSetDevice Error!");
	if (err == cudaErrorSetOnActiveProcess) //we discard this error because we don't want other code (alex Kriz) to complain
		cudaGetLastError();

	return gpu_id;
}

//this function finds out from Matlab if the current architecture is Fermi or not!
bool GetGPUArchitecture(char * globalGPUArchVar, bool bPrintf = true){

	mxArray * pGPU = mexGetVariable("global", globalGPUArchVar);
	mxASSERT( pGPU != NULL && !mxIsEmpty(pGPU), "NO Global GPU ARCH (FERMI) variable!");
	int nFERMI = mxGetScalar(pGPU);
	bool bFERMI;
	if (nFERMI == 1){
		bFERMI = true;
		if (bPrintf)
			mexPrintf("\n~~FERMI~~"); mexEvalString("drawnow;");
	}
	else{
		bFERMI = false;
		if (bPrintf)
			mexPrintf("\n~~NOT FERMI~~"); mexEvalString("drawnow;");
	}
	return bFERMI;
}


//for debuggin purposes, call this function
template <typename T>
void PrintfInMatlab( const clMatrix<T> & mat){

	mxArray* matdat[1];
	matdat[0] =  mxCreateNumericMatrix( mat.nI, mat.nJ, mxSINGLE_CLASS, mxREAL); //create some memory
	cuda_clMatrixCpy<T>( mxGetData(matdat[0]), mat);

	mexCallMATLAB( 0, NULL, 1, matdat, "MexPrintfInMatlab" );
	mxDestroyArray(matdat[0]); //memory cleanup
}

template <>
void PrintfInMatlab( const clMatrix<double> & mat ){

	mxArray* matdat[1];
	matdat[0] =  mxCreateNumericMatrix( mat.nI, mat.nJ, mxDOUBLE_CLASS, mxREAL); //create some memory
	cuda_clMatrixCpy<double>( mxGetData(matdat[0]), mat);

	mexCallMATLAB( 0, NULL, 1, matdat, "MexPrintfInMatlab" );
	mxDestroyArray(matdat[0]); //memory cleanup
}


/*
//call a function in Matlab C= func(A,B)
int CallInMatlab(char * funcname, const clMat A, const clMat B, clMat C){

	mxArray* visualdata[2];
	visualdata[0] =  mxCreateNumericMatrix( n, nXdim, mxSINGLE_CLASS, mxREAL);
	visualdata[1] =  mxCreateNumericMatrix( n, nYdim, mxSINGLE_CLASS, mxREAL);
}
*/


//assume that cell is always 1 by nJ
//load it from a mxArray object
//do not need to allocate any memory before hand, but need to free later!
//nJ is number of expected cell elements
template <typename T>
void load_mxCellObj( std::vector< clMatrix<T> >& vec, int nJ, const mxArray * ptr, char * str){

	clCheckCellFormat( ptr,	1, nJ, str);
	vec.resize(nJ);

	for (int k = 0; k < nJ; ++k){
		int nMatI = size(mxGetCell(ptr, k), 0);
		int nMatJ = size(mxGetCell(ptr, k), 1);

		clASSERT(nMatI > 0 && nMatJ > 0, "cell k size not geq 0.");

		vec[k].CreateData(nMatI, nMatJ);
		cuda_clMatrixCpy<T>( vec[k],  mxGetData(mxGetCell(ptr, k)) );
	}
}

//assume that cell is always nI by nJ
//load it from a mxArray object
//do not need to allocate any memory before hand, but need to free later!
//nJ is number of expected cell elements
template <typename T>
void load_mxCellObj( std::vector< std::vector<clMatrix<T> > >& vec, int nI, int nJ,
					const mxArray * ptr, char * str){

	clCheckCellFormat( ptr,	nI, nJ, str);
	vec.resize(nI);

	for (int ii =0; ii<nI; ++ii){
		vec[ii].resize(nJ);
		for (int jj = 0; jj < nJ; ++jj){

			int nMatI = size(mxGetCell(ptr, ii+jj*nI), 0);
			int nMatJ = size(mxGetCell(ptr, ii+jj*nI), 1);

			clASSERT(nMatI > 0 && nMatJ > 0, "cell ii,k size not geq 0.");

			vec[ii][jj].CreateData(nMatI, nMatJ);
			cuda_clMatrixCpy<T>( vec[ii][jj],  mxGetData(mxGetCell(ptr,  ii+jj*nI)) );
		}
	}
}




////////////////////////////////////////////////////////////////////////////////////////////
//additional overloaded functions for loading clMatrix objects
////////////////////////////////////////////////////////////////////////////////////////////
//2 dim matrix
template <typename T>
void load_mxMatObj( clMatrix<T> & mat, int nI, int nJ, const mxArray * ptr, char * str){
	clCheckMatrixFormat( ptr, nI, nJ, str);

	mat.CreateData( size(ptr,0), size(ptr,1));
	cuda_clMatrixCpy<T>( mat, mxGetData(ptr) );
}

template <>
void load_mxMatObj( clMatrix<double> & mat, int nI, int nJ, const mxArray * ptr, char * str){
	clCheckMatrixFormatDouble( ptr, nI, nJ, str);

	mat.CreateData( size(ptr,0), size(ptr,1));
	cuda_clMatrixCpy<double>( mat, mxGetData(ptr) );
}

//allocate and copy to an mxArray output variable
mxArray* mxArrayOutputFrom( const clMatrix<float>& mat ){
	mxArray* pOUT = mxCreateNumericMatrix( mat.nI, mat.nJ, mxSINGLE_CLASS, mxREAL);
	cuda_clMatrixCpy<float>( mxGetData(pOUT), mat);
	return pOUT;
}

mxArray* mxArrayOutputFrom( const clMatrix<double>& mat ){
	mxArray* pOUT = mxCreateNumericMatrix( mat.nI, mat.nJ, mxDOUBLE_CLASS, mxREAL);
	cuda_clMatrixCpy<double>( mxGetData(pOUT), mat);
	return pOUT;
}



////////////////////////////////////////////////////////////////////////////////////////////
//additional overloaded functions for loading clTensor objects
////////////////////////////////////////////////////////////////////////////////////////////
//n dim tensor
template <typename T>
void load_mxTensorObj( clTensor<T> & ten, const vector<int>& dims, const mxArray * ptr, char * str){
	clCheckTensorFormat(ptr, dims, str);

	const mwSize* tendims = mxGetDimensions(ptr);
	vector<int> dimsv( tendims, tendims+dims.size() );
	ten.CreateData( dimsv );
	cuda_clTensorCpy<T>( ten, mxGetData(ptr) );
}

template <>
void load_mxTensorObj( clTensor<double> & ten,  const vector<int>& dims, const mxArray * ptr, char * str){
	clCheckTensorFormatDouble(ptr, dims, str);

	const mwSize* tendims = mxGetDimensions(ptr);
	vector<int> dimsv( tendims, tendims+dims.size() );
	ten.CreateData( dimsv );
	cuda_clTensorCpy<double>( ten, mxGetData(ptr) );
}

//allocate and copy to an mxArray output variable
mxArray* mxArrayOutputFrom( const clTensor<float>& ten ){

	const vector<mwSize> dims = ten.dims;
	mxArray* pOUT = mxCreateNumericArray(dims.size(), &dims[0], mxSINGLE_CLASS, mxREAL);
	cuda_clTensorCpy<float>( mxGetData(pOUT), ten);
	return pOUT;
}

mxArray* mxArrayOutputFrom( const clTensor<double>& ten ){

	const vector<mwSize> dims = ten.dims;
	mxArray* pOUT = mxCreateNumericArray(dims.size(), &dims[0], mxDOUBLE_CLASS, mxREAL);
	cuda_clTensorCpy<double>( mxGetData(pOUT), ten);
	return pOUT;
}


//output to a cell
template<typename T>
mxArray* mxArrayOutputFrom( std::vector< std::vector<clMatrix<T> > >& array ){

	int nI = array.size();
	int nJ = array[0].size();

	int ndim = 2;
	int dims[2];
	dims[0] = nI;
	dims[1] = nJ;

	mxArray* pOUT = mxCreateCellArray(ndim, dims);

	for (int ii =0; ii<nI; ++ii){

		for (int jj = 0; jj < nJ; ++jj){

			mxSetCell(pOUT, ii+jj*nI, mxArrayOutputFrom(array[ii][jj]));

		}
	}
	return pOUT;

}

#endif













