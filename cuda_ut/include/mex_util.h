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

//this is a header include useful functions for writing
//mex files

#ifndef _MEX_UTIL_H_
#define _MEX_UTIL_H_

#include "mex.h"
#include "matrix.h"

#ifdef __cplusplus
#include <string>
#include <vector>
#else
#include <stdlib.h>
#include <vector.h>
#endif

#define IN
#define OUT
#define CL_MALLOC
#define CL_PREMALLOC

#ifndef MIN
#define MIN(a,b) ((a) < (b) ?  (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ?  (a) : (b))
#endif


//checks if all dimensions of the two array are equal
inline bool eq_dim_all(const mxArray* p1, const mxArray* p2)
{
	int n;
	if ((n = mxGetNumberOfDimensions(p1)) != mxGetNumberOfDimensions(p2))
		return false;
	const mwSize* s1 = mxGetDimensions(p1);
	const mwSize* s2 = mxGetDimensions(p2);

	for (int i = 0; i < n; ++i)
		if (s1[i] != s2[i])
			return false;

	return true;
}


#ifdef __cplusplus

inline void mxASSERT(bool b, std::string msg)
{
	if (!b){		
		std::string errmsg = "Assertion Failed: "+msg;
		mexErrMsgTxt( errmsg.c_str());
	}
}

#else

inline void mxASSERT(bool b, char* msg)
{
	char errmsg[50];
	sprintf( errmsg, "Assertion Failed:%s", msg);
	if (!b){	
		mexErrMsgTxt( errmsg);
	}
}
#endif



//becareful below not to use "err" multiple times or we will call the expensive function multiple times
#ifndef clCheckErr
#define clCheckErr( err ) \
	{ \
	int mex_err_code = (err); \
	if ( mex_err_code != 0){ \
		char errmsg[100];\
		sprintf(errmsg, "function call failed![%d] %s:%d", mex_err_code, __FILE__, __LINE__);\
		mexPrintf( errmsg ); \
		mexEvalString("drawnow;"); \
		throw mex_err_code; \
	} }
#endif

//dim is zero based
int size(const mxArray* array, int dim){
	return mxGetDimensions(array)[dim];
}

//boolean  function to check if an array has the same dimensions
bool size_2d_eq(const mxArray* array, int nI, int nJ){
	return mxGetDimensions(array)[0] == nI && mxGetDimensions(array)[1] == nJ;
}


void clCheckMatrixFormat(const mxArray* pMat, int nI, int nJ, std::string str){

	char errstring[1000];

	sprintf(errstring, "%s: NULL or empty" , str.c_str());
	mxASSERT( pMat != NULL && !mxIsEmpty(pMat), errstring);

	sprintf(errstring, "%s: number of dimension is not 2!!" , str.c_str());
	mxASSERT(mxGetNumberOfDimensions(pMat) == 2, errstring);

	sprintf(errstring, "%s Not Single or Not Numeric!!", str.c_str());
	mxASSERT( mxIsSingle(pMat) && mxIsNumeric(pMat),  errstring);

	sprintf(errstring, "%s size mismatch!!", str.c_str());
	mxASSERT( (size( pMat,0) == nI || nI == -1)
				&& (size( pMat, 1) == nJ || nJ == -1),
			  errstring );
}

void clCheckMatrixFormatDouble(const mxArray* pMat, int nI, int nJ, std::string str){

	char errstring[1000];

	sprintf(errstring, "%s: NULL or empty" , str.c_str());
	mxASSERT( pMat != NULL && !mxIsEmpty(pMat), errstring);

	sprintf(errstring, "%s: number of dimension is not 2!!" , str.c_str());
	mxASSERT(mxGetNumberOfDimensions(pMat) == 2, errstring);

	sprintf(errstring, "%s Not Double or Not Numeric!!", str.c_str());
	mxASSERT( mxIsDouble(pMat) && mxIsNumeric(pMat),  errstring);

	sprintf(errstring, "%s size mismatch!!", str.c_str());
	mxASSERT( (size( pMat,0) == nI || nI == -1)
				&& (size( pMat, 1) == nJ || nJ == -1),
			  errstring );
}

//below works for all types
template <typename T>
void clCheckMatrixFormatT(const mxArray* pMat, int nI, int nJ, std::string str){
	clCheckMatrixFormat(pMat, nI, nJ, str);
}

template <>
void clCheckMatrixFormatT<double>(const mxArray* pMat, int nI, int nJ, std::string str){
	clCheckMatrixFormatDouble(pMat, nI, nJ, str);
}


//3D matrix check
void clCheckMatrixFormat(const mxArray* pMat, int nI, int nJ, int nK, std::string str){

	char errstring[1000];

	sprintf(errstring, "%s: NULL or empty" , str.c_str());
	mxASSERT( pMat != NULL && !mxIsEmpty(pMat), errstring);

	sprintf(errstring, "%s: number of dimension is not 3!!" , str.c_str());
	mxASSERT(mxGetNumberOfDimensions(pMat) == 3  ||
			  (mxGetNumberOfDimensions(pMat) == 2 && nK == 1), errstring);

	sprintf(errstring, "%s Not Single or Not Numeric!!", str.c_str());
	mxASSERT( mxIsSingle(pMat) && mxIsNumeric(pMat),  errstring);


	if (mxGetNumberOfDimensions(pMat) == 2)
	{
		sprintf(errstring, "%s size mismatch!! %d %d %d", str.c_str(), nI, nJ, nK);
		mxASSERT( (size( pMat,0) == nI || nI == -1)
					&& (size( pMat, 1) == nJ || nJ == -1)
					&&  nK == 1, errstring);
	}else{
		sprintf(errstring, "%s size mismatch!!!", str.c_str() );
		mxASSERT( (size( pMat,0) == nI || nI == -1)
					&& (size( pMat, 1) == nJ || nJ == -1)
					&&  (size( pMat, 2) == nK || nK == -1),
				  errstring );
	}

}


//3D matrix check
void clCheckMatrixFormatDouble(const mxArray* pMat, int nI, int nJ, int nK, std::string str){

	char errstring[1000];

	sprintf(errstring, "%s: NULL or empty" , str.c_str());
	mxASSERT( pMat != NULL && !mxIsEmpty(pMat), errstring);

	sprintf(errstring, "%s: number of dimension is not 3!!" , str.c_str());
	mxASSERT(mxGetNumberOfDimensions(pMat) == 3  ||
			  (mxGetNumberOfDimensions(pMat) == 2 && nK == 1), errstring);

	sprintf(errstring, "%s Not Double or Not Numeric!!", str.c_str());
	mxASSERT( mxIsDouble(pMat) && mxIsNumeric(pMat),  errstring);


	if (mxGetNumberOfDimensions(pMat) == 2)
	{
		sprintf(errstring, "%s size mismatch!! %d %d %d", str.c_str(), nI, nJ, nK);
		mxASSERT( (size( pMat,0) == nI || nI == -1)
					&& (size( pMat, 1) == nJ || nJ == -1)
					&&  nK == 1, errstring);
	}else{
		sprintf(errstring, "%s size mismatch!!!", str.c_str() );
		mxASSERT( (size( pMat,0) == nI || nI == -1)
					&& (size( pMat, 1) == nJ || nJ == -1)
					&&  (size( pMat, 2) == nK || nK == -1),
				  errstring );
	}

}


//below works for all types
template <typename T>
void clCheckMatrixFormatT(const mxArray* pMat, int nI, int nJ,  int nK,  std::string str){
	clCheckMatrixFormat(pMat, nI, nJ, nK, str);
}

template <>
void clCheckMatrixFormatT<double>(const mxArray* pMat, int nI, int nJ,  int nK, std::string str){
	clCheckMatrixFormatDouble(pMat, nI, nJ, nK, str);
}


void clCheckCellFormat(const mxArray* pCell, int nI, int nJ, std::string str){

	char errstring[1000];

	sprintf(errstring, "%s: NULL or empty" , str.c_str());
	mxASSERT( pCell != NULL && !mxIsEmpty(pCell), errstring);

	sprintf(errstring, "%s: number of dimension is not 2!!" , str.c_str());
	mxASSERT(mxGetNumberOfDimensions(pCell) == 2, errstring);

	sprintf(errstring, "%s size mismatch!!", str.c_str());
	mxASSERT( (size( pCell, 0) == nI || nI == -1)
			&&  (size( pCell, 1) == nJ || nJ == -1),
			errstring );
}

//returns a double
double clGetMatlabStructField(const mxArray* pParams, const char* fieldname){

	char errstring[1000];
	sprintf(errstring, "pParams  NULL or empty" );
	mxASSERT( pParams != NULL && !mxIsEmpty(pParams), errstring);

	const mxArray* pField = mxGetField(pParams, 0, fieldname);
	sprintf(errstring, "%s field missing!!", fieldname);
	mxASSERT(pField != NULL, errstring);
	return mxGetScalar(pField);
}

inline bool clIsStruct( const mxArray* pField){
	if (pField == NULL)
		return false;
	return mxIsStruct(pField);
}

//when a cell appears empty in matlab, mxIsEmpty() will return false, causing segfaults!!
//use this function below intead!!
//7/2011 CT
bool mxIsCellEntryValid(const mxArray* pMat){
	return pMat != NULL && !mxIsEmpty( pMat );
}


//from host
mxArray* mxArrayOutputFrom( const float* p_mat, int nI, int nJ){
	mxArray* pOUT = mxCreateNumericMatrix( nI, nJ, mxSINGLE_CLASS, mxREAL);
	memcpy( mxGetData(pOUT), p_mat, sizeof(float)*nI*nJ );
	return pOUT;
}

//from host
mxArray* mxArrayOutputFrom( const float scalar ){
	mxArray* pOUT = mxCreateNumericMatrix( 1, 1, mxSINGLE_CLASS, mxREAL);
	float* pdata = (float*) mxGetData(pOUT);
	pdata[0] = scalar;
	return pOUT;
}

mxArray* mxArrayOutputFrom( const double scalar ){
	mxArray* pOUT = mxCreateNumericMatrix( 1, 1, mxDOUBLE_CLASS, mxREAL);
	double* pdata = (double*) mxGetData(pOUT);
	pdata[0] = scalar;
	return pOUT;
}

void PrintfInMatlab( mxArray* mat ){
	//create some memory
	mxArray* matdat[1];
	matdat[0] =  mat;
	//callback
	//mexCallMATLAB( 0, NULL, 0, NULL, "eval('dbstop in MexPrintfInMatlab')" ); //set breakpoint
	mexCallMATLAB( 0, NULL, 1, matdat, "MexPrintfInMatlab" );
}


//returns true if we want visible node to be gaussian, otherwise false
bool clGetVNG(const mxArray* pParams, float * fv_var, float * fsqrt_v_var ){

	const mxArray* pField = mxGetField(pParams, 0, "VNG");

	if (pField == NULL){
		*fv_var = *fsqrt_v_var = -1;
		return false;
	} else{
		float fVNG = (float) mxGetScalar(pField);

		if (fVNG == 0){
			*fv_var = *fsqrt_v_var = -1;
			return false;
		}
		else if (fVNG != 1){
			mexErrMsgTxt("Invalid VNG value: must be 0 or 1!");
			return false;
		}
	}

	//fVNG = 1;
	pField = mxGetField(pParams, 0, "v_var");
	mxASSERT( pField != 0, "v_var missing!");

	*fv_var = (float) mxGetScalar(pField);
	*fsqrt_v_var = sqrtf(*fv_var);

	mxASSERT( *fv_var > 0, "v_var value needs to be valid!!");
	return true;
}


void clCheckTensorFormat(const mxArray* pMat, const std::vector<int>& dims, char * str){

	char errstring[1000];

	sprintf(errstring, "%s: NULL or empty" , str);
	mxASSERT( pMat != NULL && !mxIsEmpty(pMat), errstring);

	sprintf(errstring, "%s: number of dimension is not correct!!" , str);
	mxASSERT(mxGetNumberOfDimensions(pMat) == dims.size(), errstring);

	sprintf(errstring, "%s Not Single or Not Numeric!!", str);
	mxASSERT( mxIsSingle(pMat) && mxIsNumeric(pMat),  errstring);

	for (unsigned int i = 0; i < dims.size(); ++i){
		sprintf(errstring, "%s size mismatch!! %d and %d", str, size(pMat, i), dims[i]);
		mxASSERT( (size( pMat,i) == dims[i] || dims[i] == -1), errstring );
	}
}

void clCheckTensorFormatDouble(const mxArray* pMat, const std::vector<int>& dims, char * str){

	char errstring[1000];

	sprintf(errstring, "%s: NULL or empty" , str);
	mxASSERT( pMat != NULL && !mxIsEmpty(pMat), errstring);

	sprintf(errstring, "%s: number of dimension is not correct!!" , str);
	mxASSERT(mxGetNumberOfDimensions(pMat) == dims.size(), errstring);

	sprintf(errstring, "%s Not Double or Not Numeric!!", str);
	mxASSERT( mxIsDouble(pMat) && mxIsNumeric(pMat),  errstring);

	for (unsigned int i = 0; i < dims.size(); ++i){
		sprintf(errstring, "%s size mismatch!! %d and %d", str, size(pMat, i), dims[i]);
		mxASSERT( (size( pMat,i) == dims[i] || dims[i] == -1), errstring );
	}
}


void load_mxDoubleVec( std::vector<double>& vec, const mxArray * ptr, char * str){

	clCheckMatrixFormatDouble( ptr, 1, -1, str);
	vec.resize( size(ptr,1) );

	memcpy(&vec[0], mxGetData(ptr), sizeof(double)*vec.size() );
}

void load_mxDoubleVec( std::vector<double>& vec, const mxArray * ptr, int N, char * str){

	clCheckMatrixFormatDouble( ptr, 1, N, str);
	vec.resize( size(ptr,1) );

	memcpy(&vec[0], mxGetData(ptr), sizeof(double)*vec.size() );
}


void load_mxSingleVec( std::vector<float>& vec, const mxArray * ptr, char * str){

	clCheckMatrixFormat( ptr, 1, -1, str);
	vec.resize( size(ptr,1) );

	memcpy(&vec[0], mxGetData(ptr), sizeof(float)*vec.size() );
}

void load_mxSingleVec( std::vector<float>& vec, const mxArray * ptr, int N, char * str){

	clCheckMatrixFormat( ptr, 1, N, str);
	vec.resize( size(ptr,1) );

	memcpy(&vec[0], mxGetData(ptr), sizeof(float)*vec.size() );
}


std::string clGetMatlabString(const mxArray* pMat){

	mxASSERT( pMat != NULL, "pMat is NULL");
	char cstr[1000];
	mxASSERT( mxIsChar(pMat), "clGetMatlabString:: matlab obj is not a string!");
	clCheckErr(mxGetString( pMat, cstr, 1000));

	std::string str(cstr);
	return str;
}

double clGetScalar(const mxArray* pMat){
	mxASSERT( pMat != NULL, "pMat is NULL");
	mxASSERT( mxIsNumeric(pMat), "pMat is not numeric" );
	return mxGetScalar(pMat);
}

#endif
