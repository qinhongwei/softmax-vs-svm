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
#ifndef _CONVNNOO_COMMON_CU_
#define _CONVNNOO_COMMON_CU_

#ifdef DOUBLE_PRECISION
	typedef double myf;
#else
	typedef float myf;
#endif

typedef clMatrix<myf> clmat;
#define IN
#define OUT

#define FCMULT_MSZ 3

//#define CONV_CPU_MODE // to use CPU instead of GPU

#include "../conv/cu_jitters.h"
#include "../conv/cu_conv.h"

//global cpu data
std::vector<myf* > vp_batchdata;
std::vector<myf* > vp_batchdata_y;
std::vector<myf* > vp_validdata;
std::vector<myf* > vp_validdata_y;

static mxArray* p_out[4];


template <typename T>
int CallbackGetData (const char* cb,  int nSamples, int nDataSources, int nObjectiveSinks,
		std::vector<int>& v_nVisNodes,  std::vector<int>& v_nTargetNodes,
		int nBatches, int nValidBatches, OUT int& nMiniBatches)
{

	clASSERT( v_nVisNodes.size() == nDataSources, "v_nVisNodes.size() == nDataSources" );
	clASSERT( v_nTargetNodes.size() == nObjectiveSinks, "v_nTargetNodes.size() == nObjectiveSinks" );

	vp_batchdata.resize(nDataSources);
	vp_batchdata_y.resize(nObjectiveSinks);
	vp_validdata.resize(nDataSources);
	vp_validdata_y.resize(nObjectiveSinks);

	////////////////////////////////////////////////////////////////////////////////////////////
	if (p_out[0] != NULL && p_out[1] != NULL && p_out[2] != NULL && p_out[3] !=NULL){
		mxDestroyArray(p_out[0]);
		mxDestroyArray(p_out[1]);
		mxDestroyArray(p_out[2]);
		mxDestroyArray(p_out[3]);
	}

	mexCallMATLAB( 4, p_out, 0, NULL, cb );

	////////////////////////////////////////////////////////////////////////////////////////////////
	//Get Training Data
	clASSERT( mxIsCellEntryValid(p_out[0]), " p_out[0] is == NULL!");
	clASSERT( mxIsCell(p_out[0]), " failed mxIsCell(p_out[0]) ");
	clASSERT(nDataSources ==  mxGetNumberOfElements(p_out[0]), "CallbackGetData: nDataSource mismatch");

	clASSERT( mxIsCellEntryValid(p_out[2]), " p_out[2] is == NULL!");
	clASSERT( mxIsCell(p_out[2]), " failed mxIsCell(p_out[2]) ");
	clASSERT(nDataSources ==  mxGetNumberOfElements(p_out[2]), "CallbackGetData: nDataSource mismatch");


	for (int nn = 0; nn < nDataSources; ++nn)
	{
		mxArray* p_bd_nn 		= mxGetCell( p_out[0], nn);
		mxArray* p_valid_nn		= mxGetCell( p_out[2], nn);

		if (nBatches == 1){

			clCheckMatrixFormatT<T>(  p_bd_nn, -1, -1, "conv Data: callback's Data must be single/double!");
			clCheckMatrixFormatT<T>(  p_valid_nn, -1, -1, "conv ValidData: callback's Data must be single/double!");

			mxASSERT( mxGetNumberOfDimensions( p_bd_nn ) == 2
					&& size(p_bd_nn, 0) == nSamples
					&& size(p_bd_nn, 1) == v_nVisNodes[nn], "conv batchdata dimension mismatch! X nBatches==1");
			//valid
			mxASSERT( mxGetNumberOfDimensions( p_valid_nn ) == 2
					&& size(p_valid_nn, 0) == nSamples
					&& size(p_valid_nn, 1) == v_nVisNodes[nn], "conv batchdata dimension mismatch! ValidX nBatches==1");

		}else{

			clCheckMatrixFormatT<T>(  p_bd_nn, -1, -1, -1, "conv Data: callback's Data must be single/double!");
			clCheckMatrixFormatT<T>(  p_valid_nn, -1, -1, -1, "conv Data: callback's Data must be single/double!");

			mxASSERT( mxGetNumberOfDimensions( p_bd_nn ) == 3
					&& size(p_bd_nn, 0) == nSamples
					&& size(p_bd_nn, 1) == v_nVisNodes[nn]
					&& size(p_bd_nn, 2) == nBatches, "conv batchdata dimension mismatch! X");

			mxASSERT( mxGetNumberOfDimensions( p_valid_nn ) == 3
					&& size(p_valid_nn, 0) == nSamples
					&& size(p_valid_nn, 1) == v_nVisNodes[nn]
					&& size(p_valid_nn, 2) == nValidBatches, "conv batchdata dimension mismatch! ValidX");
		}

		vp_batchdata[nn] 	= (T*) mxGetData(p_bd_nn);
		vp_validdata[nn] 	= (T*) mxGetData(p_valid_nn);

	}// nn




	////////////////////////////////////////////////////////////////////////////////////////////////
	// target nodes

	clASSERT( mxIsCellEntryValid(p_out[1]), " p_out[1] is == NULL!");
	clASSERT( mxIsCell(p_out[1]), " failed mxIsCell(p_out[1]) ");
	clASSERT(nObjectiveSinks ==  mxGetNumberOfElements(p_out[1]), "CallbackGetData: nObjectiveSinks mismatch");

	clASSERT( mxIsCellEntryValid(p_out[3]), " p_out[3] is == NULL!");
	clASSERT( mxIsCell(p_out[3]), " failed mxIsCell(p_out[3]) ");
	clASSERT(nObjectiveSinks ==  mxGetNumberOfElements(p_out[3]), "CallbackGetData: nObjectiveSinks mismatch");


	for (int nn = 0; nn < nObjectiveSinks; ++nn)
	{
		mxArray* p_bd_y_nn 		= mxGetCell( p_out[1], nn);
		mxArray* p_valid_y_nn	= mxGetCell( p_out[3], nn);

		if (nBatches == 1){
			clCheckMatrixFormatT<T>(  p_bd_y_nn, -1, -1,  "conv Target: callback's Target must be single/double!");
			clCheckMatrixFormatT<T>(  p_valid_y_nn, -1, -1,  "conv ValidY: callback's Target must be single/double!");

			mxASSERT( mxGetNumberOfDimensions( p_bd_y_nn ) == 2
					&& size(p_bd_y_nn, 0) == nSamples
					&& size(p_bd_y_nn, 1) == v_nTargetNodes[nn], "conv batchdata dimension mismatch! Y nBatches==1");

			mxASSERT( mxGetNumberOfDimensions( p_valid_y_nn ) == 2
					&& size(p_valid_y_nn, 0) == nSamples
					&& size(p_valid_y_nn, 1) == v_nTargetNodes[nn], "conv batchdata dimension mismatch! ValidY nBatches==1");

		}else{

			clCheckMatrixFormatT<T>( p_bd_y_nn, -1, -1, -1,  "conv Target: callback's Target must be single/double!");
			clCheckMatrixFormatT<T>( p_valid_y_nn, -1, -1, -1, "conv ValidY: callback's Target must be single/double!");


			mxASSERT( mxGetNumberOfDimensions( p_bd_y_nn ) == 3
					&& size(p_bd_y_nn, 0) == nSamples
					&& size(p_bd_y_nn, 1) == v_nTargetNodes[nn]
					&& size(p_bd_y_nn, 2) == nBatches, "conv batchdata dimension mismatch! Y");

			mxASSERT( mxGetNumberOfDimensions(p_valid_y_nn ) == 3
					&& size(p_valid_y_nn, 0) == nSamples
					&& size(p_valid_y_nn, 1) == v_nTargetNodes[nn]
					&& size(p_valid_y_nn, 2) == nValidBatches, "conv batchdata dimension mismatch! ValidY");
		}
		vp_batchdata_y[nn] 	= (T*) mxGetData(p_bd_y_nn);
		vp_validdata_y[nn] 	= (T*) mxGetData(p_valid_y_nn);

	}// nn

	nMiniBatches = nBatches; //refill successful
	clPrintf("$");
	////////////////////////////////////////////////////////////////////////////////////////////
	return 0;
}

/*
 * nl is a pointer to params.net_layers
 *
 *
 */
clNet<myf>* create_net_from_matlab(IN const mxArray* nl, IN const mxArray* nws,
									IN clmat& W, IN clmat& dW,
								   IN int nDataSources, IN int nObjectiveSinks, IN int nSamples,
								   IN int nVerbose,
								   OUT int& nNodeLayers, int& nMaxNodeSize )
{
    clASSERT( nDataSources >= 1, " nDataSources >= 1 ");
    clASSERT( nObjectiveSinks >= 1, " nObjectiveSinks >= 1 ");

	clASSERT( mxIsCellEntryValid(nl), " params.net_layers is == NULL!");
	clASSERT( mxIsCell(nl), " failed mxIsCell(nl) ");

	nNodeLayers = mxGetNumberOfElements(nl);
	clASSERT( nNodeLayers >= nDataSources+nObjectiveSinks, "nNodeLayers >= nDataSources+nObjectiveSinks" );

	clNet<myf>* pnet = new clNet<myf>(nDataSources, nObjectiveSinks);
	pnet->nSourceStart = 0;
	pnet->nSourceEnd = pnet->nDataSources-1;
	pnet->nSinkStart = nNodeLayers-pnet->nObjectiveSinks;
	pnet->nSinkEnd = nNodeLayers-1;

	nMaxNodeSize = -9999;
	char notestr[1000];

	for (int ll = 0; ll < nNodeLayers; ++ll){

		//sprintf(notestr, "\nMemory before layer{%d}: ", ll);
		//clPrintf(notestr);
		//DisplayGPUMemory(gpu_id);

		mxArray* p_layer = mxGetCell( nl, ll);
		//some error checking
		clASSERT( mxIsCellEntryValid(p_layer), "nl{ll} is not valid");
		clASSERT( clIsStruct (p_layer), "nl{ll} is not a struct");

		const mxArray* pField = mxGetField(p_layer, 0, "type");
		clASSERT(pField != NULL, "nl{ll}.type field missing!!");

		char LayerTypeString[100];
		clASSERT( mxIsChar(pField), "nl{ll}.type must be a string");
		clCheckErr(mxGetString( pField, LayerTypeString, 100));
		std::string layertype_str = LayerTypeString;

		if (ll <= pnet->nSourceEnd){
			clASSERT( layertype_str.compare("fcdata") ==0 ||
					  layertype_str.compare("convdata") ==0 , "create_net_from_matlab: first layers much be data layers");
		}else{
			clASSERT( layertype_str.compare("fcdata") != 0 &&
					  layertype_str.compare("convdata") !=0 , "create_net_from_matlab: higher layers much NOT be data layers");
		}

		std::string symbol = (ll <= pnet->nSourceEnd) ? " {Input} " : (ll >= pnet->nSinkStart) ? " {Loss} " : "{Hidn}";

		//data layers
		if (layertype_str.compare("fcdata") == 0){//*****************************************

			float f_dropout = (float) clGetMatlabStructField(p_layer, "f_dropout");
			int nV = (int) clGetMatlabStructField(p_layer, "nV");

			std::vector<clLayer<myf>* > vppl(0);
			pnet->layers.push_back( 		new clLayerData<myf>( nSamples, nV, true, f_dropout, vppl));

			sprintf(notestr, "\n%s(%d)-->layer{%d} FCdata: \t size: %d,  dropout:%.3f ", symbol.c_str(), -1, ll,  nV, f_dropout);


		}else if (layertype_str.compare("convdata") == 0){//*****************************************

			float f_dropout = (float) clGetMatlabStructField(p_layer, "f_dropout");
			int nFilters = (int) clGetMatlabStructField(p_layer, "nFilters");
			int nI_grid = (int) clGetMatlabStructField(p_layer, "nI_grid");
			int nJ_grid = (int) clGetMatlabStructField(p_layer, "nJ_grid");

			std::vector<clLayer<myf>* > vppl(0);

			pnet->layers.push_back( new clLayerConvData<myf>(
					nSamples, nFilters, nI_grid, nJ_grid, true, f_dropout, vppl));

			sprintf(notestr, "\n%s(%d)-->layer{%d} Convdata: \t nFilters:%d nIJ_grid:%d %d, dropout:%.3f ",
					symbol.c_str(), -1, ll, nFilters, nI_grid, nJ_grid, f_dropout);


		}else if (layertype_str.compare("fc") == 0){//*****************************************

			int 	nV = (int) clGetMatlabStructField(p_layer, "nV");
			int 	nH = (int) clGetMatlabStructField(p_layer, "nH");
			int 	nNeuronType = (int) clGetMatlabStructField(p_layer, "nNeuronType");
			float 	f_dropout = (float) clGetMatlabStructField(p_layer, "f_dropout");
			float 	f_wtcost = (float) clGetMatlabStructField(p_layer, "f_wtcost");
			float 	f_wt_cons_val = (float) clGetMatlabStructField(p_layer, "f_wt_cons_val");

			int 	nPrevLayerID = (int) clGetMatlabStructField(p_layer, "nPrevLayerID");//1-based
			std::vector<clLayer<myf>* > vppl(1);
			clASSERT(nPrevLayerID-1 < ll && nPrevLayerID-1 >= 0, "PrevLayerID error!");
			vppl[0] = pnet->layers[nPrevLayerID-1];

			pnet->layers.push_back( new clLayerFC<myf>(nSamples, nV, nH, nNeuronType, true, f_wtcost, f_dropout, vppl, f_wt_cons_val));

			if (nNeuronType == 63){

				float 	fC = (float) clGetMatlabStructField(p_layer, "fC");
				float 	fC2 = (float) clGetMatlabStructField(p_layer, "fC2");

				clLayerFC<myf> * fcl = static_cast< clLayerFC<myf> * >( pnet->layers[pnet->layers.size()-1] );
				fcl->fC = fC;
				fcl->fC2 = fC2;

				sprintf(notestr, "\n%s(%d)-->layer{%d} FC:    \t size: %d %d, hiddentype:%d, wtcost:%.6f, wt_cons_val:%.3f dropout:%.3f  l2 tsvm's fC:%f %f",
					symbol.c_str(), nPrevLayerID-1, ll,  nV, nH, nNeuronType, f_wtcost, f_wt_cons_val, f_dropout, fcl->fC, fcl->fC2 );

			}else if( get_neuron_class(nNeuronType) == 6 ){
				float 	fC = (float) clGetMatlabStructField(p_layer, "fC");

				clLayerFC<myf> * fcl = static_cast< clLayerFC<myf> * >( pnet->layers[pnet->layers.size()-1] );
				fcl->fC = fC;

				sprintf(notestr, "\n%s(%d)-->layer{%d} FC:    \t size: %d %d, hiddentype:%d, wtcost:%.6f, wt_cons_val:%.3f dropout:%.3f  svm's fC:%f",
					symbol.c_str(), nPrevLayerID-1, ll,  nV, nH, nNeuronType, f_wtcost, f_wt_cons_val, f_dropout, fcl->fC);

			}else{
				sprintf(notestr, "\n%s(%d)-->layer{%d} FC:   \t size: %d %d, hiddentype:%d, wtcost:%.6f, wt_cons_val:%.3f dropout:%.3f",
					symbol.c_str(), nPrevLayerID-1, ll, nV, nH, nNeuronType, f_wtcost, f_wt_cons_val, f_dropout);
			}


		}else if (layertype_str.compare("convc") == 0){//*****************************************

			int 	nVisChannels = (int) clGetMatlabStructField(p_layer, "nVisChannels");
			int 	nVisI = (int) clGetMatlabStructField(p_layer, "nVisI");
			int 	nVisJ = (int) clGetMatlabStructField(p_layer, "nVisJ");
			int 	nFilters = (int) clGetMatlabStructField(p_layer, "nFilters");
			int 	nI_filt = (int) clGetMatlabStructField(p_layer, "nI_filt");
			int 	nJ_filt = (int) clGetMatlabStructField(p_layer, "nJ_filt");
			int 	nNeuronType = (int) clGetMatlabStructField(p_layer, "nNeuronType");
			float 	f_dropout = (float) clGetMatlabStructField(p_layer, "f_dropout");
			float 	f_wtcost = (float) clGetMatlabStructField(p_layer, "f_wtcost");
			int 	nI_grid = (int) clGetMatlabStructField(p_layer, "nI_grid");
			int 	nJ_grid = (int) clGetMatlabStructField(p_layer, "nJ_grid");
			int 	nPaddingStart = (int) clGetMatlabStructField(p_layer, "nPaddingStart");
			int 	nStride = (int) clGetMatlabStructField(p_layer, "nStride");
			int 	nPartialSum = (int) clGetMatlabStructField(p_layer, "nPartialSum");
			int 	nCPUMode = (int) clGetMatlabStructField(p_layer, "nCPUMode");


#ifndef CONV_CPU_MODE
			bool bCpuMode = (nCPUMode == 1);
#else
			bool bCpuMode = true;
#endif

			int 	nPrevLayerID = (int) clGetMatlabStructField(p_layer, "nPrevLayerID");//1-based
			std::vector<clLayer<myf>* > vppl(1);
			clASSERT(nPrevLayerID-1 < ll && nPrevLayerID-1 >= 0, "PrevLayerID error!");
			vppl[0] = pnet->layers[nPrevLayerID-1];

			pnet->layers.push_back( new clLayerConvC<myf>( nSamples, nVisChannels, nVisI, nVisJ,
					nI_filt, nJ_filt, nI_grid, nJ_grid, nFilters, nNeuronType,
					f_wtcost, f_dropout, bCpuMode /*_bCpuMode*/, nPaddingStart, nStride, nPartialSum, vppl ));

			long long nParams = pnet->layers[pnet->layers.size()-1]->nParams;
			sprintf(notestr, "\n%s(%d)-->layer{%d} ConvC: nParams:%Ld nVisCh:%d nVisIJ:[%d %d],  nIJ_filt:[%d %d], nFilters:%d "
					"nIJ_grid:[%d %d], cpumode:%d hiddentype:%d, \n \t\t\t\t\t wtcost:%.6f, dropout:%.3f "
					" nPaddingStart:%d nStride:%d nPartialSum:%d",
					symbol.c_str(), nPrevLayerID-1, ll, nParams, nVisChannels, nVisI, nVisJ, nI_filt, nJ_filt, nFilters, nI_grid, nJ_grid,
					bCpuMode, nNeuronType, f_wtcost, f_dropout, nPaddingStart, nStride, nPartialSum);


		}else if (layertype_str.compare("deconvc") == 0){//*****************************************

			int 	nVisChannels = (int) clGetMatlabStructField(p_layer, "nVisChannels");
			int 	nVisI = (int) clGetMatlabStructField(p_layer, "nVisI");
			int 	nVisJ = (int) clGetMatlabStructField(p_layer, "nVisJ");
			int 	nFilters = (int) clGetMatlabStructField(p_layer, "nFilters");
			int 	nI_filt = (int) clGetMatlabStructField(p_layer, "nI_filt");
			int 	nJ_filt = (int) clGetMatlabStructField(p_layer, "nJ_filt");
			int 	nNeuronType = (int) clGetMatlabStructField(p_layer, "nNeuronType");
			float 	f_dropout = (float) clGetMatlabStructField(p_layer, "f_dropout");
			float 	f_wtcost = (float) clGetMatlabStructField(p_layer, "f_wtcost");
			int 	nI_grid = (int) clGetMatlabStructField(p_layer, "nI_grid");
			int 	nJ_grid = (int) clGetMatlabStructField(p_layer, "nJ_grid");
			int 	nPaddingStart = (int) clGetMatlabStructField(p_layer, "nPaddingStart");
			int 	nStride = (int) clGetMatlabStructField(p_layer, "nStride");
			int 	nPartialSum = (int) clGetMatlabStructField(p_layer, "nPartialSum");
			int 	nCPUMode = (int) clGetMatlabStructField(p_layer, "nCPUMode");


#ifndef CONV_CPU_MODE
			bool bCpuMode = (nCPUMode == 1);
#else
			bool bCpuMode = true;
#endif

			int 	nPrevLayerID = (int) clGetMatlabStructField(p_layer, "nPrevLayerID");//1-based
			std::vector<clLayer<myf>* > vppl(1);
			clASSERT(nPrevLayerID-1 < ll && nPrevLayerID-1 >= 0, "PrevLayerID error!");
			vppl[0] = pnet->layers[nPrevLayerID-1];

			pnet->layers.push_back( new clLayerDeConvC<myf>( nSamples, nVisChannels, nVisI, nVisJ,
					nI_filt, nJ_filt, nI_grid, nJ_grid, nFilters, nNeuronType,
					f_wtcost, f_dropout, bCpuMode /*_bCpuMode*/, nPaddingStart, nStride, nPartialSum, vppl ));

			long long nParams = pnet->layers[pnet->layers.size()-1]->nParams;
			sprintf(notestr, "\n%s(%d)-->layer{%d} DeConvC:  nParams:%Ld, nVisCh:%d nVisIJ:[%d %d], nIJ_filt:[%d %d], nFilters:%d "
					"nIJ_grid:[%d %d], cpumode:%d hiddentype:%d, \n \t\t\t\t\t wtcost:%.6f, dropout:%.3f "
					" nPaddingStart:%d nStride:%d nPartialSum:%d",
					symbol.c_str(), nPrevLayerID-1, ll, nParams, nVisChannels, nVisI, nVisJ, nI_filt, nJ_filt, nFilters, nI_grid, nJ_grid,
					bCpuMode, nNeuronType, f_wtcost, f_dropout, nPaddingStart, nStride, nPartialSum);



		}else if (layertype_str.compare("convs") == 0){//*******************************************

			int 	nFilters = (int) clGetMatlabStructField(p_layer, "nFilters");
			int 	nI_sgrid = (int) clGetMatlabStructField(p_layer, "nI_sgrid");
			int 	nJ_sgrid = (int) clGetMatlabStructField(p_layer, "nJ_sgrid");
			int 	nNeuronType = (int) clGetMatlabStructField(p_layer, "nNeuronType");
			int 	nPoolingType = (int) clGetMatlabStructField(p_layer, "nPoolingType");
			int 	nPoolAKMode = (int) clGetMatlabStructField(p_layer, "nPoolAKMode");
			int 	nSizeX = (int) clGetMatlabStructField(p_layer, "nSizeX");
			int 	nStride = (int) clGetMatlabStructField(p_layer, "nStride");
			float 	f_dropout = (float) clGetMatlabStructField(p_layer, "f_dropout");

			int 	nPrevLayerID = (int) clGetMatlabStructField(p_layer, "nPrevLayerID");//1-based
			std::vector<clLayer<myf>* > vppl(1);
			clASSERT(nPrevLayerID-1 < ll && nPrevLayerID-1 >= 0, "PrevLayerID error!");
			vppl[0] = pnet->layers[nPrevLayerID-1];

			pnet->layers.push_back( new clLayerConvS<myf>(nSamples, nFilters, nI_sgrid, nJ_sgrid,
					nNeuronType, nPoolingType, nPoolAKMode, nSizeX, nStride, f_dropout, vppl));

			sprintf(notestr, "\n%s (%d)-->layer{%d}  ConvS:    nFilters:%d "
					"nIJ_sgrid:[%d %d], hiddentype:%d, nPoolingType:%d akmode:%d nSizeX:%d nStride:%d dropout:%.3f",
					symbol.c_str(), nPrevLayerID-1, ll, nFilters, nI_sgrid, nJ_sgrid, nNeuronType, nPoolingType, nPoolAKMode,
					nSizeX, nStride, f_dropout, nPrevLayerID);


		}else if (layertype_str.compare("convrn") == 0){//*****************************************

			int 	nFilters = (int) clGetMatlabStructField(p_layer, "nFilters");
			int 	nI_sgrid = (int) clGetMatlabStructField(p_layer, "nI_sgrid");
			int 	nJ_sgrid = (int) clGetMatlabStructField(p_layer, "nJ_sgrid");
			int 	nRNSize = (int) clGetMatlabStructField(p_layer, "nRNSize");
			float 	fRNScale = (float) clGetMatlabStructField(p_layer, "fRNScale");
			float 	fRNPower = (float) clGetMatlabStructField(p_layer, "fRNPower");


			int 	nPrevLayerID = (int) clGetMatlabStructField(p_layer, "nPrevLayerID");//1-based
			std::vector<clLayer<myf>* > vppl(1);
			clASSERT(nPrevLayerID-1 < ll && nPrevLayerID-1 >= 0, "PrevLayerID error!");
			vppl[0] = pnet->layers[nPrevLayerID-1];

			pnet->layers.push_back( new clLayerConvRN<myf>(nSamples, nFilters, nI_sgrid, nJ_sgrid,
					nRNSize, fRNScale, fRNPower, vppl));

			sprintf(notestr, "\nlayer{%d}<--(%d)%s:  ConvRN \t nFilters:%d "
					"nIJ_sgrid:[%d %d], nRNSize:%d fScale:%f fPower:%f",
					ll, nPrevLayerID-1, symbol.c_str(), nFilters, nI_sgrid, nJ_sgrid, nRNSize, fRNScale, fRNPower);

		}else if (layertype_str.compare("convjitter") == 0){//*******************************************

			int 	nVisChannels = (int) clGetMatlabStructField(p_layer, "nVisChannels");
			int 	nVisI = (int) clGetMatlabStructField(p_layer, "nVisI");
			int 	nVisJ = (int) clGetMatlabStructField(p_layer, "nVisJ");
			int 	nVisI2 = (int) clGetMatlabStructField(p_layer, "nVisI2");
			int 	nVisJ2 = (int) clGetMatlabStructField(p_layer, "nVisJ2");

			mxArray * pfield = mxGetField(p_layer, 0, "datamean");
			mxArray * pfield2 = mxGetField(p_layer, 0, "datastd");

			clMatrix<myf> datamean, datastd;
			load_mxMatObj( datamean, 1, nVisChannels*nVisJ2*nVisI2, pfield, "convjitter datamean error!");
			load_mxMatObj( datastd, 1, nVisChannels*nVisJ2*nVisI2, pfield2, "convjitter datastd error!");

			int 	nPrevLayerID = (int) clGetMatlabStructField(p_layer, "nPrevLayerID");//1-based
			std::vector<clLayer<myf>* > vppl(1);
			clASSERT(nPrevLayerID-1 < ll && nPrevLayerID-1 >= 0, "PrevLayerID error!");
			vppl[0] = pnet->layers[nPrevLayerID-1];

			pnet->layers.push_back( new clLayerConvJitter<myf>(nSamples, nVisChannels, nVisI, nVisJ,
					nVisI2, nVisJ2, datamean, datastd, vppl));

			sprintf(notestr, "\n%s (%d)-->layer{%d}  ConvJitter:    nVisChannels:%d "
					"nVisIJ:[%d %d], nVisIJ2:[%d %d]",
					symbol.c_str(), nPrevLayerID-1, ll, nVisChannels, nVisI, nVisJ, nVisI2, nVisJ2);

		}else if (layertype_str.compare("convxyrs") == 0){//****************************************

			int 	nVisChannels = (int) clGetMatlabStructField(p_layer, "nVisChannels");
			int 	nVisI = (int) clGetMatlabStructField(p_layer, "nVisI");
			int 	nVisJ = (int) clGetMatlabStructField(p_layer, "nVisJ");
			int 	nVisI2 = (int) clGetMatlabStructField(p_layer, "nVisI2");
			int 	nVisJ2 = (int) clGetMatlabStructField(p_layer, "nVisJ2");

			mxArray * pfield = mxGetField(p_layer, 0, "trns_low");
			mxArray * pfield2 = mxGetField(p_layer, 0, "trns_high");

			clMatrix<myf> trns_low, trns_high;
			load_mxMatObj( trns_low, 1, 4, pfield, "convxyrs trns_low error!");
			load_mxMatObj( trns_high, 1, 4, pfield2, "convxyrs trns_high error!");

			int 	nPrevLayerID = (int) clGetMatlabStructField(p_layer, "nPrevLayerID");//1-based
			std::vector<clLayer<myf>* > vppl(1);
			clASSERT(nPrevLayerID-1 < ll && nPrevLayerID-1 >= 0, "PrevLayerID error!");
			vppl[0] = pnet->layers[nPrevLayerID-1];

			pnet->layers.push_back( new clLayerConvXYRS<myf>(nSamples, nVisChannels, nVisI, nVisJ,
					nVisI2, nVisJ2, trns_low, trns_high, vppl));

			sprintf(notestr, "\n%s (%d)-->layer{%d}  clLayerConvXYRS:    nVisChannels:%d "
					"nVisIJ:[%d %d], nVisIJ2:[%d %d] low: %.2f %.2f %.2f %.2f  high:%.2f %.2f %.2f %.2f",
					symbol.c_str(), nPrevLayerID-1, ll, nVisChannels, nVisI, nVisJ, nVisI2, nVisJ2,
					trns_low.GetElem(0,0),trns_low.GetElem(0,1),trns_low.GetElem(0,2),trns_low.GetElem(0,3),
					trns_high.GetElem(0,0),trns_high.GetElem(0,1),trns_high.GetElem(0,2),trns_high.GetElem(0,3));

		}else if (layertype_str.compare("imagemirror") == 0){//*******************************************

			int 	nVisChannels = (int) clGetMatlabStructField(p_layer, "nVisChannels");
			int 	nVisI = (int) clGetMatlabStructField(p_layer, "nVisI");
			int 	nVisJ = (int) clGetMatlabStructField(p_layer, "nVisJ");

			int 	nPrevLayerID = (int) clGetMatlabStructField(p_layer, "nPrevLayerID");//1-based
			std::vector<clLayer<myf>* > vppl(1);
			clASSERT(nPrevLayerID-1 < ll && nPrevLayerID-1 >= 0, "PrevLayerID error!");
			vppl[0] = pnet->layers[nPrevLayerID-1];

			pnet->layers.push_back( new clLayerImageMirror<myf>(nSamples, nVisChannels, nVisI, nVisJ, vppl));

			sprintf(notestr, "\n%s (%d)-->layer{%d}  ImageMirror:    nVisChannels:%d "
					"nVisIJ:[%d %d],",
					symbol.c_str(), nPrevLayerID-1, ll, nVisChannels, nVisI, nVisJ);
		}else{
			sprintf(notestr, "\nlayer{%d} type:%s not recognized!! \n",ll, layertype_str.c_str() );
			clASSERT(false, notestr);
		}

		if (nVerbose==1){
			clPrintf(notestr);
		}
		nMaxNodeSize = MAX(nMaxNodeSize, pnet->layers[pnet->layers.size()-1]->nHidNodes );

	}//ll

	cr( pnet->init(W, dW) ) //must be before wtsharing

	///// -----------------------------------------------------------------------------------------
	//wts sharing
	if (nws != NULL){
		clASSERT( mxIsCell(nws), " failed mxIsCell(nws) ");
		int num_ws = mxGetNumberOfElements(nws);

		sprintf(notestr, "\nThis Net shares weights. Number of sets:%d", num_ws);
		clPrintf(notestr);

		//should be after pnet->init()
		for (int i = 0; i < num_ws; ++i){

			mxArray* p_set = mxGetCell( nws, i);
			std::vector<float> wtshare;

			load_mxSingleVec( wtshare, p_set, "wtshare p_set missing or not single or dim wrong!");
			clASSERT(wtshare.size() > 0, "wtshare.size() > 0");

			sprintf(notestr, "\nset:%d     num layers shared:%lu. \tlayers: ", i, wtshare.size());
			clPrintf(notestr);

			clWtsShare<myf> set;
			for (int j = 0; j < wtshare.size(); ++j){

				int nid = wtshare[j]-1; //in matlab it is one-based
				clASSERT(nid >= nDataSources && nid < pnet->layers.size(),
						"wtshare[j] >= nDataSources");

				clASSERT(pnet->layers[nid]->name.compare("fc")==0, "wtshare must be fc layers");

				sprintf(notestr, "%d ", nid);
				clPrintf(notestr);

				clLayerFC<myf> * pl = static_cast< clLayerFC<myf> * >( pnet->layers[nid] );
				set.Add( &pl->wts_vw,  &pl->d_wts_vw );
			}
			pnet->v_shares.push_back(set);
		}

		////
		for (unsigned int ui = 0; ui < pnet->v_shares.size(); ++ui){
			cr( pnet->v_shares[ui].validate_equal() )
		}
		clPrintf("\nvalidate_equal done.\n");
	}//nws


	return pnet;
}


//this function evalutes the loss on a dataset
/*
 *
 *
 *  vp_output is a vector of myf* to the hidden activations
 *  set vector size to 0 means we don't want to retrieve them
 *  else, they have to have the same size as nObjectiveSinks
 *	nLayerEst set to -1 is ok too, means we won't copy activations
 */
int evaluate_loss(IN const std::vector<int>& v_nVisNodes, IN std::vector<clmat>& v_DataX,
				  IN const std::vector<int>& v_nTargetNodes, IN std::vector<clmat>& v_Targets,
				  IN std::vector<clmat>& v_TargetsTemp,
				  IN const std::vector<myf*>& vp_vd, IN const std::vector<myf*>& vp_vd_y,
				  IN int nSamples, IN int nValidBatches, IN clmat& randnums, IN clNet<myf>* pnet,
				  IN bool bGetLoss /*false if just want last layer activation*/,
				  OUT std::vector<myf>& v_validloss, OUT myf& fValidLoss,
				  OUT int nLayerEst, OUT myf* h_est  )
{


	int nObjectiveSinks = v_nTargetNodes.size();

	if (v_Targets.size() != nObjectiveSinks)
		return -1;

	if (v_validloss.size() != nObjectiveSinks+1)
		return -2;

	if (v_nVisNodes.size() != v_DataX.size() || v_nVisNodes.size() < 1)
		return -3;

	if (v_Targets.size() != v_TargetsTemp.size())
		return -4;

	if (v_nVisNodes.size() != vp_vd.size() || v_nTargetNodes.size() != vp_vd_y.size())
		return -5;

	//if (vp_output.size() != 0 && vp_output.size() != v_nTargetNodes.size())
		//return -6;
	if (nLayerEst > -1 && h_est == NULL)
		return -6;

	if (( nLayerEst > -1) && (nLayerEst <= pnet->nSourceEnd || nLayerEst > pnet->nSinkEnd))
		return -7;



	for (int uu = 0; uu < v_validloss.size(); ++uu){
		v_validloss[uu] = 0;
	}
	std::vector<myf> v_validloss_batch(nObjectiveSinks+1);

	char notestr[100];

	for (int kk = 0; kk < nValidBatches; ++kk ){

		if ( fmod( double(kk)/nValidBatches, 0.05 ) == 0)
		{
			sprintf(notestr, "$");
			clPrintf(notestr);
		}

		for (int i = 0; i < v_nVisNodes.size(); ++i){
			cuda_clMatrixCpy(v_DataX[i], vp_vd[i]+nSamples*v_nVisNodes[i]*kk);
		}
		for (int i = 0; i < v_nTargetNodes.size(); ++i){
			cuda_clMatrixCpy(v_Targets[i], vp_vd_y[i]+nSamples*v_nTargetNodes[i]*kk);
		}

		cr( pnet->fprop(v_DataX, randnums, false) )

		if (nLayerEst > -1){ //copy activation
			int nInJ = pnet->layers[nLayerEst]->nodes.nI*pnet->layers[nLayerEst]->nodes.nJ;
			cuda_clMatrixCpy(h_est+nInJ*kk, pnet->layers[nLayerEst]->nodes);
		}

		if (bGetLoss){
			cr( pnet->getloss(v_Targets, v_TargetsTemp, false, v_validloss_batch) )
		}

		for (int uu = 0; uu < v_validloss.size();++uu){
			v_validloss[uu] += v_validloss_batch[uu]/nValidBatches;
		}
	}
	fValidLoss = std::accumulate(v_validloss.begin(), v_validloss.end(), myf(0));

	return 0;
}



//this function evalutes the loss on a dataset
/*  different from evaluate_loss in that here we want to get the training objective out
 *  , not just the number of cases missed
 *
 *
 *  vp_output is a vector of myf* to the hidden activations
 *  set vector size to 0 means we don't want to retrieve them
 *  else, they have to have the same size as nObjectiveSinks
 *	nLayerEst set to -1 is ok too, means we won't copy activations
 */
int evaluate_loss_training_err(IN const std::vector<int>& v_nVisNodes, IN std::vector<clmat>& v_DataX,
				  IN const std::vector<int>& v_nTargetNodes, IN std::vector<clmat>& v_Targets,
				  IN std::vector<clmat>& v_TargetsTemp,
				  IN const std::vector<myf*>& vp_vd, IN const std::vector<myf*>& vp_vd_y,
				  IN int nSamples, IN int nValidBatches, IN clmat& randnums, IN clNet<myf>* pnet,
				  OUT std::vector<myf>& v_validloss, OUT myf& fValidLoss,
				  OUT int nLayerEst, OUT myf* h_est  )
{


	int nObjectiveSinks = v_nTargetNodes.size();

	if (v_Targets.size() != nObjectiveSinks)
		return -1;

	if (v_validloss.size() != nObjectiveSinks+1)
		return -2;

	if (v_nVisNodes.size() != v_DataX.size() || v_nVisNodes.size() < 1)
		return -3;

	if (v_Targets.size() != v_TargetsTemp.size())
		return -4;

	if (v_nVisNodes.size() != vp_vd.size() || v_nTargetNodes.size() != vp_vd_y.size())
		return -5;

	//if (vp_output.size() != 0 && vp_output.size() != v_nTargetNodes.size())
		//return -6;
	if (nLayerEst > -1 && h_est == NULL)
		return -6;

	if (( nLayerEst > -1) && (nLayerEst <= pnet->nSourceEnd || nLayerEst > pnet->nSinkEnd))
		return -7;



	for (int uu = 0; uu < v_validloss.size(); ++uu){
		v_validloss[uu] = 0;
	}
	std::vector<myf> v_validloss_batch(nObjectiveSinks+1);

	char notestr[100];
	for (int kk = 0; kk < nValidBatches; ++kk ){

		if ( fmod( double(kk)/nValidBatches, 0.05 ) == 0)
		{
			sprintf(notestr, "$");
			clPrintf(notestr);
		}

		for (int i = 0; i < v_nVisNodes.size(); ++i){
			cuda_clMatrixCpy(v_DataX[i], vp_vd[i]+nSamples*v_nVisNodes[i]*kk);
		}
		for (int i = 0; i < v_nTargetNodes.size(); ++i){
			cuda_clMatrixCpy(v_Targets[i], vp_vd_y[i]+nSamples*v_nTargetNodes[i]*kk);
		}

		cr( pnet->fprop(v_DataX, randnums, false) ) //this should be false so we don't do dropout

		if (nLayerEst > -1){ //copy activation
			int nInJ = pnet->layers[nLayerEst]->nodes.nI*pnet->layers[nLayerEst]->nodes.nJ;
			cuda_clMatrixCpy(h_est+nInJ*kk, pnet->layers[nLayerEst]->nodes);
		}

		if ( true ){
			cr( pnet->getloss(v_Targets, v_TargetsTemp, true, v_validloss_batch) )
		}

		for (int uu = 0; uu < v_validloss.size();++uu){
			v_validloss[uu] += v_validloss_batch[uu]/nValidBatches;
		}
	}
	fValidLoss = std::accumulate(v_validloss.begin(), v_validloss.end(), myf(0));

	return 0;
}

#endif
