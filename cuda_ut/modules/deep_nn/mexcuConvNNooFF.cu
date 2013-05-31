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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CT 1/2013 FF version of mexcuConvNNoo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <numeric>



#include "cu_clmatrix.h"
#include "cu_matrix_mex.h"
#include "elewise.h"
#include "bsxfun.h"
#include "cu_dnn_ops.h"
#include "cu_gpu_rand.h"
#include "cu_matrix_ops.h"
#include "mlp.h"

#include "convNNoo_common.cu"


#define NARGIN 4
#define IN_W		pRhs[0]
#define IN_params	pRhs[1]
#define IN_X		pRhs[2]
#define IN_Y		pRhs[3]


#define NARGOUT 2
#define OUT_f			pLhs[0]
#define OUT_yest		pLhs[1]


extern cublasHandle_t cbh;

void mexFunction(int nLhs, mxArray *pLhs[], int nRhs, const mxArray *pRhs[])
{
	try{
	///////////////////////////////////////////////////////////////
	//make sure the input and outputs arguments are correct
	//check the number of inputs and outputs as well as their types
	mxASSERT(nRhs == NARGIN && nLhs == NARGOUT, "Number of input and/or output arguments not correct!" );
	mxASSERT(mxIsStruct(IN_params), "Input arg params not a structure!");

	//get parameters
	int 	nSamples = (int) clGetMatlabStructField(IN_params, "nSamples");
	int 	nValidBatches = (int) clGetMatlabStructField(IN_params, "nValidBatches");
	int 	nDataSources = (int) clGetMatlabStructField(IN_params, "nDataSources");
	int		nObjectiveSinks	=  (int) clGetMatlabStructField(IN_params, "nObjectiveSinks");
	int     nVerbose 	=  (int) clGetMatlabStructField(IN_params, "nVerbose");
	int		nLayerEst    =  (int) clGetMatlabStructField(IN_params, "nLayerEst");
	int		nTestingMode  =  (int) clGetMatlabStructField(IN_params, "nTestingMode");
	//do we want classification error or training objective

	//which layer do we want activations from

	///////////////////////////////////////////////////////
	//GPU locking code
	int gpu_id = SelectGPUFromServer( "GPUID" );
	GpuRandInit( 1234, GetGPUArchitecture( "FERMI", nVerbose==1 ) );
	///////////////////////////////////////////////////////

	///////////////////////////////////////////////////////
	//initialize cublas
	cublas_cr( cublasCreate( &cbh ) );
	///////////////////////////////////////////////////////


	clCheckMatrixFormatT<myf>( IN_W, -1, 1, "IN_X not a column vector or not single/double");
	clASSERT( size( IN_W, 0) > 0, "size( IN_W, 0) > 0");

	clmat W;
	load_mxMatObj( W, -1, 1, IN_W, "IN_W error!");

	clmat dW(W.nI, W.nJ);

	///// ------------------------ net initialization --------------------------------------------
	mxArray* nl = mxGetField(IN_params, 0, "net_layers");
	int nNodeLayers, nMaxNodeSize;

	clNet<myf>* pnet = create_net_from_matlab(nl, NULL /*nws*/, W, dW, nDataSources, nObjectiveSinks,
											 nSamples, nVerbose, nNodeLayers, nMaxNodeSize );

	std::vector<clmat> v_DataX(nDataSources);
	std::vector<int> v_nVisNodes(nDataSources);
	for (int i = 0; i < nDataSources; ++i){
		v_nVisNodes[i] = pnet->layers[i+pnet->nSourceStart]->nHidNodes;
		v_DataX[i].CreateData(nSamples, v_nVisNodes[i] );
	}

	std::vector<clmat> 	v_Targets(nObjectiveSinks);
	std::vector<clmat> 	v_TargetsTemp(nObjectiveSinks);
	std::vector<int> 	v_nTargetNodes(nObjectiveSinks); //nTargetNodes is the label dimensionality

	for (int i = 0; i < nObjectiveSinks; ++i){
		if (   pnet->layers[i+pnet->nSinkStart]->nNeuronType==5
			|| get_neuron_class(pnet->layers[i+pnet->nSinkStart]->nNeuronType) == 6 ){
			v_nTargetNodes[i] = 1;
		}else{
			v_nTargetNodes[i] = pnet->layers[i+pnet->nSinkStart]->nHidNodes;
		}

		v_Targets[i].CreateData(nSamples, v_nTargetNodes[i] );

		if ( pnet->layers[i+pnet->nSinkStart]->nNeuronType == 63 ){
			v_TargetsTemp[i].CreateData(nSamples, pnet->layers[i+pnet->nSinkStart]->nHidNodes*3);
		}else if ( get_neuron_class(pnet->layers[i+pnet->nSinkStart]->nNeuronType) == 6){
			v_TargetsTemp[i].CreateData(nSamples, pnet->layers[i+pnet->nSinkStart]->nHidNodes );
		}else{
			v_TargetsTemp[i].CreateData(nSamples, v_nTargetNodes[i] );
		}
	}

	clmat dummy; dummy.nI = 0; dummy.nJ = 0;

	clASSERT(nLayerEst > pnet->nSourceEnd && nLayerEst <= pnet->nSinkEnd, "nLayerEst not valid!");

	char notestr[1000];
	if (nVerbose==1){
		DisplayGPUMemory(gpu_id);
		sprintf(notestr, "\n[%d] ConvNNoo Feefforward: Samples:%d nValidBatches:%d "
						 "nDataSources:%d nObjectiveSinks:%d nSourceStartEnd:{%d %d}, "
						 "nSinkStartEnd:{%d %d} nNodeLayers:%d",
						 gpu_id, nSamples, nValidBatches,
						 nDataSources, nObjectiveSinks, pnet->nSourceStart, pnet->nSourceEnd, pnet->nSinkStart, pnet->nSinkEnd,
						 nNodeLayers);
		clPrintf(notestr);
	}


	////////////////////////////////////////////////////////////////////////////////////////////////
	//Get Training Data
	vp_validdata.resize(nDataSources);

	clASSERT( mxIsCellEntryValid(IN_X), " IN_X is == NULL!");
	clASSERT( mxIsCell(IN_X), " failed mxIsCell(IN_X) ");
	clASSERT(nDataSources == mxGetNumberOfElements(IN_X), "CallbackGetData: nDataSource mismatch");

	for (int nn = 0; nn < nDataSources; ++nn)
	{
		mxArray* p_valid_nn	= mxGetCell( IN_X, nn);

		if (nValidBatches == 1){
			clCheckMatrixFormatT<myf>(  p_valid_nn, -1, -1, "conv ValidData: callback's Data must be single/double!");
			//valid
			mxASSERT( mxGetNumberOfDimensions( p_valid_nn ) == 2
					&& size(p_valid_nn, 0) == nSamples
					&& size(p_valid_nn, 1) == v_nVisNodes[nn], "conv batchdata dimension mismatch! ValidX nBatches==1");

		}else{

			clCheckMatrixFormatT<myf>(  p_valid_nn, -1, -1, -1, "conv Data: callback's Data must be single/double!");

			mxASSERT( mxGetNumberOfDimensions( p_valid_nn ) == 3
					&& size(p_valid_nn, 0) == nSamples
					&& size(p_valid_nn, 1) == v_nVisNodes[nn]
					&& size(p_valid_nn, 2) == nValidBatches, "conv batchdata dimension mismatch! ValidX");
		}
		vp_validdata[nn] 	= (myf*) mxGetData(p_valid_nn);
	}// nn



	clASSERT( mxIsCellEntryValid(IN_Y), " IN_Y is == NULL!");
	clASSERT( mxIsCell(IN_Y), " failed mxIsCell(IN_Y) ");
	clASSERT(nObjectiveSinks == mxGetNumberOfElements(IN_Y), "CallbackGetData: nObjectiveSinks mismatch");

	vp_validdata_y.resize(nObjectiveSinks);

	for (int nn = 0; nn < nObjectiveSinks; ++nn)
	{

		mxArray* p_valid_y_nn	= mxGetCell( IN_Y, nn);

		if (nValidBatches == 1){
			clCheckMatrixFormatT<myf>(  p_valid_y_nn, -1, -1,  "conv ValidY: callback's Target must be single/double!");

			mxASSERT( mxGetNumberOfDimensions( p_valid_y_nn ) == 2
					&& size(p_valid_y_nn, 0) == nSamples
					&& size(p_valid_y_nn, 1) == v_nTargetNodes[nn], "conv batchdata dimension mismatch! ValidY nBatches==1");

		}else{
			clCheckMatrixFormatT<myf>( p_valid_y_nn, -1, -1, -1, "conv ValidY: callback's Target must be single/double!");

			mxASSERT( mxGetNumberOfDimensions(p_valid_y_nn ) == 3
					&& size(p_valid_y_nn, 0) == nSamples
					&& size(p_valid_y_nn, 1) == v_nTargetNodes[nn]
					&& size(p_valid_y_nn, 2) == nValidBatches, "conv batchdata dimension mismatch! ValidY");
		}
		vp_validdata_y[nn] 	= (myf*) mxGetData(p_valid_y_nn);
	}// nn
	////////////////////////////////////////////////////////////////////////////////////////////////


	myf fValidLoss;
	std::vector<myf> v_validloss(nObjectiveSinks+1, 0.0);


	myf* h_est;
	int n_Y_EST = 1;
	OUT_yest = mxCreateCellMatrix(1, n_Y_EST );
	for (int i = 0; i < n_Y_EST; ++i){

		int nI = pnet->layers[nLayerEst]->nodes.nI;
		int nJ = pnet->layers[nLayerEst]->nodes.nJ;

		mwSize sz[3] = {nI, nJ, nValidBatches};
		mxArray* pp = mxCreateNumericArray( 3, sz, mxSINGLE_CLASS, mxREAL);
		mxSetCell( OUT_yest, i, pp );

		h_est = (myf*) mxGetData( pp );
	}

	if (nTestingMode){
	cr( evaluate_loss( v_nVisNodes, v_DataX, v_nTargetNodes, v_Targets, v_TargetsTemp,
			vp_validdata, vp_validdata_y, nSamples, nValidBatches, dummy, pnet, false,
			v_validloss, fValidLoss, nLayerEst, h_est) )
	}else{
		cr( evaluate_loss_training_err( v_nVisNodes, v_DataX, v_nTargetNodes, v_Targets, v_TargetsTemp,
			vp_validdata, vp_validdata_y, nSamples, nValidBatches, dummy, pnet,
			v_validloss, fValidLoss, nLayerEst, h_est) )
	}

	//copy to output
	OUT_f = mxArrayOutputFrom( fValidLoss );

	sprintf( notestr, " TrnErr:"); clPrintf(notestr);
	for (int uu = 0; uu < v_validloss.size();++uu){
		sprintf( notestr, " %.5f", v_validloss[uu]); clPrintf(notestr);
	}
	clPrintf("\n");

	delete pnet;

	GpuRandDestroy();
	cublas_cr( cublasDestroy( cbh ) );

	}catch (int){
	}
	cudaThreadExit();

}

