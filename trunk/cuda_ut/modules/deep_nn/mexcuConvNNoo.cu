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

CT 11/2012

CUDA implementation of SGD of stochastic feedforward neural network

//matlab function header
function [f (W or dW)] = mexcuConvNNoo( W, params, Dim, Callback)

INPUT:
	params:

		HIDDENTYPE - 0 - linear
					 1 - sigmoid
					 2 - tanh
					 3 - relu					
					 5 - softmax


		CHECKGRAD - 0 standard stochastic gradients
		CHECKGRAD - 1 return the gradients

		W is a nWtsDim by 1 vector of all the weights serialized
		Dim is a vector specifying the dimension of the neural network, e.g. [784 500 200 500 784]


output:
		f - scalar value of the function evaluation
		dW - nWtsDim by 1 the partial derivatives

note:   the serialization and deserialization of the weights must be row major,
		meaning that each row of the weight matrix W (nVis by nHid)
		The augmented Weight matrix with the biases on the last row is (nVis+1 by nHid)
		So to serialize in matlab, we must take the transpose of it and then use (:)
		after transpose, the bias row become the last column.

		OR, simply serialize W' using matlab's (:), W' is nHid by nVis+1

CHANGELOG:
Todo:
	   -for logistic output nodes, can use more stable version	  

Be Careful:
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

#define NARGIN 3
#define IN_W			pRhs[0]
#define IN_params		pRhs[1]
#define IN_callback 	pRhs[2]

#define NARGOUT 1
#define OUT_f			pLhs[0]
#define OUT_W			pLhs[1]
#define OUT_dW			pLhs[1]


#ifdef __cplusplus
    extern "C" bool utIsInterruptPending();
#else
    extern bool utIsInterruptPending();
#endif


extern cublasHandle_t cbh;

void mexFunction(int nLhs, mxArray *pLhs[], int nRhs, const mxArray *pRhs[])
{
	try{
	///////////////////////////////////////////////////////////////
	//make sure the input and outputs arguments are correct
	//check the number of inputs and outputs as well as their types
	mxASSERT(nRhs == NARGIN && nLhs >= NARGOUT, "Number of input and/or output arguments not correct!" );
	mxASSERT(mxIsStruct(IN_params), "Input arg params not a structure!");

	//get parameters
	int 	nMaxEpochs = (int) clGetMatlabStructField(IN_params, "maxepoch");
	int 	nSamples = (int) clGetMatlabStructField(IN_params, "nSamples");
	int 	nValidBatches = (int) clGetMatlabStructField(IN_params, "nValidBatches");
	int 	nBatches = (int) clGetMatlabStructField(IN_params, "nBatches");

	int 	BEST = (int) clGetMatlabStructField(IN_params, "BEST"); //saving best weights according to validation or not?
	int 	CHECKGRAD = (int) clGetMatlabStructField(IN_params, "CHECKGRAD");
	bool    bRefill = (bool) clGetMatlabStructField(IN_params, "bRefill");

	int 	nDataSources = (int) clGetMatlabStructField(IN_params, "nDataSources");
	int		nObjectiveSinks	=  (int) clGetMatlabStructField(IN_params, "nObjectiveSinks");
	int     nEpochsLookValid =  (int) clGetMatlabStructField(IN_params, "nEpochsLookValid");

	//get vector parameters
	std::vector<float> p_rate, p_momen, p_noise, p_adagrad, save_wts_epochs, p_llv, p_llh, p_hiddentype, p_dropout;

	load_mxSingleVec( p_rate, mxGetField(IN_params, 0, "rates"), nMaxEpochs, "params.rates missing or not single or dim wrong!");
	load_mxSingleVec( p_momen, mxGetField(IN_params, 0, "momen"), nMaxEpochs, "params.momen missing or not single or dim wrong!");
	load_mxSingleVec( p_noise, mxGetField(IN_params, 0, "noise"), nMaxEpochs, "params.noise missing or not single or dim wrong!");
	load_mxSingleVec( p_adagrad, mxGetField(IN_params, 0, "adagrad"), nMaxEpochs, "params.adagrad missing or not single or dim wrong!");
	load_mxSingleVec( save_wts_epochs, mxGetField(IN_params, 0, "save_wts_epochs"), "params.save_wts_epochs missing or not single or dim wrong!");

	char BatchdataFetchCallbackName[100];
	clASSERT( mxIsChar(IN_callback), "Callback function must be a string");
	clCheckErr(mxGetString( IN_callback, BatchdataFetchCallbackName, 100));

	///////////////////////////////////////////////////////
	//GPU locking code
	int gpu_id = SelectGPUFromServer( "GPUID" );
	GpuRandInit( 1234, GetGPUArchitecture( "FERMI" ) );
	///////////////////////////////////////////////////////

	///////////////////////////////////////////////////////
	//initialize cublas
	cublas_cr( cublasCreate( &cbh ) );
	///////////////////////////////////////////////////////


	clCheckMatrixFormatT<myf>( IN_W, -1, 1, "IN_X not a column vector or not single/double");
	clASSERT( size( IN_W, 0) > 0, "size( IN_W, 0) > 0");

	clmat W;
	load_mxMatObj( W, -1, 1, IN_W, "IN_W error!");

	clmat dW		( W.nI, W.nJ );
	clmat dWinc		( W.nI, W.nJ );
	clmat W2		( W.nI, W.nJ);  //needed for dropouts
	clmat dW_temp	( W.nI, W.nJ);  //needed for adagrad
	clmat dW_var	( W.nI, W.nJ);  //needed for adagrad

	///// ------------------------ net initialization --------------------------------------------
	mxArray* nl = mxGetField(IN_params, 0, "net_layers");
	mxArray* nws = mxGetField(IN_params, 0, "net_wt_shares");
	clASSERT( nws==NULL || CHECKGRAD==0, "Cant' CHECKGRAD when net_wt_shares is present");

	int nNodeLayers, nMaxNodeSize;

	clNet<myf>* pnet = create_net_from_matlab(nl, nws, W, dW, nDataSources, nObjectiveSinks,
												nSamples, 1, nNodeLayers, nMaxNodeSize );

	std::vector<clmat> v_DataX(nDataSources);
	std::vector<int> v_nVisNodes(nDataSources);
	for (int i = 0; i < nDataSources; ++i){
		v_nVisNodes[i] = pnet->layers[i+pnet->nSourceStart]->nHidNodes;
		v_DataX[i].CreateData(nSamples, v_nVisNodes[i] );
	}

	std::vector<clmat> v_Targets(nObjectiveSinks);
	std::vector<clmat> v_TargetsTemp(nObjectiveSinks);
	std::vector<int> v_nTargetNodes(nObjectiveSinks); //nTargetNodes is the label dimensionality

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
	clmat randnums( 1, nSamples*nMaxNodeSize );

	//some extra intializations
	std::vector<clmat> thetas(0);

	char notestr[1000];
	DisplayGPUMemory(gpu_id);
	sprintf(notestr, "\n\n[%d] ConvNNoo training: *** Maxepochs: %d ***"
					 "\nnSamples:%d nValidBatches:%d nBatches:%d "
					 "nDataSources:%d nObjectiveSinks:%d nSourceStartEnd:{%d %d},"
					 "nSinkStartEnd:{%d %d} nNodeLayers:%d"
					 "\nnParams:%lld bRefill:%d nMaxNodeSize:%d CHECKGRAD:%d BEST:%d CallBack:%s\n",
					 gpu_id, nMaxEpochs, nSamples, nValidBatches, nBatches,
					 nDataSources, nObjectiveSinks, pnet->nSourceStart, pnet->nSourceEnd, pnet->nSinkStart, pnet->nSinkEnd,
					 nNodeLayers, pnet->nTotalParams, bRefill, nMaxNodeSize, CHECKGRAD, BEST,
					 BatchdataFetchCallbackName);
	clPrintf(notestr);

	////////////////////////////////////////////////////////////////////////////////////////////

	std::vector<myf> v_trainloss(nObjectiveSinks+1);
	int nMiniBatches = 0;
	float min_err = 1e80;
	myf fTrainLoss, fValidLoss;

	p_out[0] = p_out[1] = p_out[2] = p_out[3] = NULL;  //important

	//get data from matlab
	CallbackGetData<myf>( BatchdataFetchCallbackName, nSamples, nDataSources, nObjectiveSinks,
			v_nVisNodes, v_nTargetNodes, nBatches, nValidBatches, nMiniBatches);

	DisplayGPUMemory(gpu_id);
	///////////////////////////////////////////////////////////////////////////////////////////////

	cuTic();
	std::vector<int> batch_inds;
	randperm(nBatches, batch_inds, 1234567 );


	//float fms1, fms2, fms3, fms4;
	//fms1 = fms2 = fms3 = fms4 = 0;


	for (int epoch=0; epoch < nMaxEpochs; ++epoch){

		getLastCudaError("\n!last error!\n");

		myf f_rate = myf(p_rate[epoch]);
		myf f_momen = myf(p_momen[epoch]);
		myf f_noise = myf(p_noise[epoch]);

		if (nMiniBatches == 0 && !bRefill){
			nMiniBatches = nBatches; //no refill, we will recycle the obtained training data
			randperm(nBatches, batch_inds );

		}else if (nMiniBatches == 0){  //we want to refill
			//get data from matlab
			CallbackGetData<myf>( BatchdataFetchCallbackName, nSamples, nDataSources, nObjectiveSinks,
							v_nVisNodes, v_nTargetNodes, nBatches, nValidBatches, nMiniBatches);
		}

		int bb = (CHECKGRAD==1) ? 0 : batch_inds[nMiniBatches-1];  //randomly permute the order
		//int bb = nMiniBatches-1; 							  //always the same order

		for (int i = 0; i < v_nVisNodes.size(); ++i){
			cuda_clMatrixCpy(v_DataX[i], vp_batchdata[i]+nSamples*v_nVisNodes[i]*bb);
		}

		for (int i = 0; i < v_nTargetNodes.size(); ++i){
			cuda_clMatrixCpy(v_Targets[i], vp_batchdata_y[i]+nSamples*v_nTargetNodes[i]*bb);
		}
		nMiniBatches--;


		//add some noise
		if (f_noise > 0){

			for (int i = 0; i < v_nVisNodes.size(); ++i){
				clmat r;
				r.nI = nSamples;
				r.nJ = v_nVisNodes[i];
				r.pData = randnums.pData; //since randnums is always GEQ than visible layer nodes

				cr( cuda_clMatSetRandn( r, myf(0), myf(f_noise) ) )
				cr( EleWisefun( v_DataX[i], fctrPlus<myf>(), r, v_DataX[i]))
			}
		}

		//cuTic();
		pnet->fprop(v_DataX, randnums, true);
		//fms1 += cuToc();
		//cuTic();
		cr( pnet->getloss(v_Targets, v_TargetsTemp, true, v_trainloss) )
		//fms2 += cuToc();
		//cuTic();
		cr( pnet->bprop() )
		//fms3 += cuToc();
		//cuTic();
		cr( pnet->dEdw() )
		//fms4 += cuToc();

		fTrainLoss = std::accumulate(v_trainloss.begin(), v_trainloss.end(), myf(0)); //important for myf()

		if (CHECKGRAD==1)
			break;

		///////////////////////////////////////////////////////////////////////////////////////////
		//ADAGRAD, change dW
		if ( p_adagrad[epoch]==1 ){

			//update the cumulative variance of gradients
			cr(EleWisefun( fctrSq<myf>(), dW, dW_temp))
			cr( EleWisefun(myf(1), dW_temp, fctrAlphaPlusBeta<myf>(), myf(1), dW_var, dW_var ) )
			//can also use exponetial averaging above

			dW_temp.CopyFrom(dW_var);

			// maybe divide by number of epochs
			//cr(EleWisefun(dW_temp, fctrDiv<myf>(), myf(epoch), dW_temp))

			cr(EleWisefun( fctrSqrt<myf>(), dW_temp, dW_temp))
			cr(EleWisefun(dW_temp, fctrPlus<myf>(), myf(1e-6), dW_temp))

			cr( EleWisefun( dW, fctrDiv<myf>(), dW_temp, dW))
		}


		//my old
		cr( EleWisefun(f_rate, dW, fctrAlphaPlusBeta<myf>(), f_momen, dWinc, dWinc))
		/////////////// update the weights
		cr( EleWisefun( dWinc, fctrPlus<myf>(), W, W))

		//additional post processing such as weight constraints
		if (CHECKGRAD != 1){
			cr( pnet->PostProcess() ) //this will destroy dW
		}

		if ( false &&epoch % int(nBatches/5)==0)
		{
			float f_ms_since = cuToc();
			sprintf( notestr, "\nnEpoch:%05d r:%.5f mn:%.2f noise:%.3f, ada:%d mb:%05d time sec:%.5f",
					epoch, f_rate, f_momen, f_noise, int(p_adagrad[epoch]),	bb, f_ms_since/1e3 );
			clPrintf( notestr );
			cuTic();
		}

		//debug outputs
		//without nMaxEpochs >1, this below crashes when CHECKGRAD=1 for some reason!
		//bool bLookAtValid = (nMaxEpochs > 1 && (epoch < 100 || epoch % int(nMaxEpochs/30) == 0));

		bool bLookAtValid = (nMaxEpochs > 1 && epoch % nEpochsLookValid == 0);
		if (bLookAtValid){

			float f_ms_since;
			f_ms_since = cuToc();


			std::vector<myf> v_validloss(nObjectiveSinks+1, 0.0);

			cr( evaluate_loss( v_nVisNodes, v_DataX, v_nTargetNodes, v_Targets, v_TargetsTemp,
						vp_validdata, vp_validdata_y, nSamples, nValidBatches, randnums, pnet, true,
						v_validloss, fValidLoss, -1, NULL) )

			sprintf( notestr, "\nnEpoch:%05d r:%.5f mn:%.2f noise:%.3f, ada:%d "
							  "ValidN:%d mb:%05d sec:%.3f ",
					epoch, f_rate, f_momen, f_noise, int(p_adagrad[epoch]),	nValidBatches*nSamples,
					bb, f_ms_since/1000);
			clPrintf( notestr );

			sprintf( notestr, "TrnErr:"); clPrintf(notestr);
			for (int uu = 0; uu < v_trainloss.size();++uu){
				sprintf( notestr, " %.5f", v_trainloss[uu]); clPrintf(notestr);
			}

			sprintf( notestr, " ValErr:"); clPrintf(notestr);
			for (int uu = 0; uu < v_validloss.size()-1; ++uu){
				sprintf( notestr, " %.6f", v_validloss[uu]); clPrintf(notestr);
			}

			//sprintf( notestr, " timing: %f %f %f %f", fms1/(epoch+1),  fms2/(epoch+1), fms3/(epoch+1), fms4/(epoch+1)); clPrintf(notestr);


			if (BEST){
				if (fValidLoss < min_err){
					min_err = fValidLoss;

					W2.CopyFrom(W);

					if (thetas.size() == 0){
						clmat* pw = new clmat(W2.nI, W2.nJ);
						pw->CopyFrom(W2);
						thetas.push_back(*pw);
					}else{
						thetas[0].CopyFrom(W2);
					}
					clPrintf( " $best$" );
				}
			}

			if (isnan(fTrainLoss+fValidLoss) || isinf(fTrainLoss+fValidLoss))
				epoch = nMaxEpochs;


			cuTic();

		}// bLookAtValid


		//save the weights
		if (!BEST && epoch == save_wts_epochs[thetas.size()] ){

			W2.CopyFrom(W);

			clmat* pw = new clmat(W2.nI, W2.nJ);
			pw->CopyFrom(W2);
			thetas.push_back(*pw);
			clPrintf( "$" );
		}


		//check if user Ctrl-C
		 if (utIsInterruptPending()) {
            clPrintf("\nUser Ctrl-C Detected. END\n");
            epoch = nMaxEpochs;
        }

	} // epoch < nMaxEpochs


	//copy to output
	if( CHECKGRAD == 0 && nLhs == 2 ){
		OUT_f = mxArrayOutputFrom( fTrainLoss );
		OUT_W = mxCreateCellMatrix(1, thetas.size() );

		for (int i = 0; i < thetas.size(); ++i){
			mxSetCell( OUT_W, i, mxArrayOutputFrom(thetas[i]) );
		}
	}else if (CHECKGRAD == 1){
		OUT_f = mxArrayOutputFrom( fTrainLoss );
		cr(EleWisefun(dW, fctrMul<myf>(), myf(-1), dW))
		OUT_dW = mxArrayOutputFrom( dW );
	}else{
		clASSERT(false, "CHECKGRAD and nargout error!");
	}


	delete pnet;

	GpuRandDestroy();
	cublas_cr( cublasDestroy( cbh ) );

	}
	catch (int){
		//do nothing here
	}

	cudaThreadExit();

	if (p_out[0] != NULL && p_out[1] != NULL && p_out[2] != NULL && p_out[3] !=NULL){
		mxDestroyArray(p_out[0]);
		mxDestroyArray(p_out[1]);
		mxDestroyArray(p_out[2]);
		mxDestroyArray(p_out[3]);
	}
}

