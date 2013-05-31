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

#ifndef _CU_CONV_H_
#define _CU_CONV_H_

#include "cu_util.h"
#include "cu_clmatrix.h"
#include "cu_cltensor.h"
#include "mlp.h"



#define IN
#define OUT

enum CONVMODE { VALID, SAME, FULL };


template <typename T>
int filter2d_cpu(	T *pSource, int nVisChannels, int nVisI, int nVisJ,
				 	T *pKernel, int nI_filt, int nJ_filt, int nFilters,
				 	T *pDest, int nI_grid, int nJ_grid, CONVMODE mode,
				 	bool bConv = false, bool bZeroDest = true);

template <typename T>
int filter2d_cpu_akmaj(	T *pSource, int nVisChannels, int nVisI, int nVisJ, int nSamples,
							T *pKernel, int nI_filt, int nJ_filt, int nFilters,
							T *pDest, int nI_grid, int nJ_grid, CONVMODE mode,
							bool bConv = false, bool bZeroDest = true);


template< typename T>
int ConvPooling(IN const clMatrix<T>& Prev, int nFilters, int nI, int nJ, int nSamples,
				int nPoolingType, OUT clMatrix<T>& Out);

template< typename T>
int ConvPoolingBackProp(IN const clMatrix<T>& Prev, IN const  clMatrix<T>& Nodes,
		IN const  clMatrix<T>& dEdaNodes, int nFilters, int nI, int nJ, int nSamples,
				int nPoolingType, T gamma, OUT clMatrix<T>& dEdaPrev );



template <typename T>
class clLayerConvData: public clLayer<T>{
	typedef  clMatrix<T> 	clmat;
public:
	//Data connected layer, no storage, only views
	int nFilters, nI_grid, nJ_grid;

	//constructor
	clLayerConvData(int _nSamples, int _nFilters, int _nI_grid, int _nJ_grid,
			bool _bSamplesLeadDim, float _f_dropout, std::vector<clLayer<T>* >& _vppl)
	: clLayer<T>(0/*nNeuronType*/, 0/*nParms*/, _nFilters*_nI_grid*_nJ_grid /*_nHidNodes*/,
			_nSamples, _bSamplesLeadDim, _f_dropout, "convdata", _vppl )
	{
		clASSERT(_bSamplesLeadDim, "error: _bSamplesLeadDim == false");

		nFilters = _nFilters;
		nI_grid = _nI_grid;
		nJ_grid = _nJ_grid;

		this->nodes.nI = this->nSamples;
		this->nodes.nJ = this->nHidNodes;
		clASSERT( this->vppl.size()==0, "vppl.size()==0");
	};

	int validate(){
		//should not be called
		return -1;
	};
	////////////////////////////////////////////////////////////////////////////////////////////////
	//activate forward
	////////////////////////////////////////////////////////////////////////////////////////////////
	int forward(clMatrix<T>& randnums, bool bLearning ){
		cr( forward_nonlinearity( this->nodes, this->nNeuronType, this->f_dropout, randnums, bLearning) )
		return 0;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////
	//backpropagate the partial derivatives
	////////////////////////////////////////////////////////////////////////////////////////////////
	int backward( bool b=true){
		//should not be called
		return -1;
	}

	int dEdw(){
		return -1;
	}
	int assign_ptr( T* ptr, bool bDerive ){
		return -1;
	}

	T getloss_wtcost(){
		return 0;
	}

	int post_process(){
		return 0;
	}
};



///////////////////////////////////////////////////////////////////////////////////////////////////
//data layout is  [nSamples; nJ_sgrid*nI_sgrid*nFilters]
template <typename T>
class clLayerConvS: public clLayer<T>{  //a pooling layer
	typedef  clMatrix<T> 	clmat;
public:

	int nFilters;
	int nI_sgrid, nJ_sgrid;
	int nPoolingType;		//0 - sum, 1 - max
	bool bUseAKMode;
	int nSizeX;				//pooling size width and height
	int nStride;			//stride we need

	//constructor
	clLayerConvS(int _nSamples, int _nFilters, int _nI_sgrid, int _nJ_sgrid,
				 int _nNeuronType, int _nPoolingType, bool _bUseAKMode, int _nSizeX, int _nStride,
				 float _f_dropout, std::vector<clLayer<T>* >& _vppl)
	: clLayer<T>(_nNeuronType, 0 /*params*/, _nFilters*_nI_sgrid*_nJ_sgrid /*_nHidNodes*/, _nSamples,
			true /*_bSamplesLeadDim*/, _f_dropout /*_f_dropout*/, "convs", _vppl )
	{

		nFilters = _nFilters;
		nI_sgrid = _nI_sgrid;
		nJ_sgrid = _nJ_sgrid;
		nPoolingType = _nPoolingType;
		bUseAKMode = _bUseAKMode;
		nSizeX = _nSizeX;
		nStride = _nStride;

		this->nodes.CreateData( this->nSamples, this->nHidNodes);
		this->dEda_nodes.CreateData( this->nSamples, this->nHidNodes );

		if (bUseAKMode){
			clASSERT( nI_sgrid == nJ_sgrid, "nI_sgrid == nJ_sgrid");
		}else{
			clASSERT( nSizeX == 2,  "nSizeX == 2" );
			clASSERT( nStride == 2,  "nStride == 2" );
			clASSERT( _nNeuronType ==0, "nNeuronType must be 0, else bug in backprop");
		}
	};

	//defined later (end of code)
	int validate();
	int backward(bool bNonlin = true);

	////////////////////////////////////////////////////////////////////////////////////////////////
	//activate forward
	////////////////////////////////////////////////////////////////////////////////////////////////
	int forward(clMatrix<T>& randnums, bool bLearning );

	// calculate the partial derivative w.r.t the weights
	// will corrupt the activations dEda_nodes
	int dEdw(){		return 0;	}

	int assign_ptr( T* ptr, bool bDerive ){		return 0;	}

	//uses d_wts_vw as buffer
	T getloss_wtcost(){		return 0;	}
	int post_process(){		return 0;	}
};




///////////////////////////////////////////////////////////////////////////////////////////////////
//data layout is  [nSamples; nJ_sgrid*nI_sgrid*nFilters]
template <typename T>
class clLayerConvRN: public clLayer<T>{  //a pooling layer
	typedef  clMatrix<T> 	clmat;
public:

	int nFilters, nSize;
    T fScale, fPower;
    int nI_sgrid, nJ_sgrid;

    clmat denoms;
    clmat temp;

	//constructor
	clLayerConvRN(int _nSamples, int _nFilters, int _nI_sgrid, int _nJ_sgrid, int _nSize,
				 T _fScale, T _fPower, std::vector<clLayer<T>* >& _vppl)
	: clLayer<T>( 0/*_nNeuronType*/, 0 /*params*/, _nFilters*_nI_sgrid*_nJ_sgrid /*_nHidNodes*/, _nSamples,
			true /*_bSamplesLeadDim*/, 0.0 /*_f_dropout*/, "convrn", _vppl)
	{

		nFilters = _nFilters;
		nI_sgrid = _nI_sgrid;
		nJ_sgrid = _nJ_sgrid;
		fScale = _fScale;
		fPower = _fPower;
		nSize = _nSize;

		this->nodes.CreateData( this->nSamples, this->nHidNodes);
		this->dEda_nodes.CreateData( this->nSamples, this->nHidNodes );
		denoms.CreateData( this->nSamples, this->nHidNodes);
		temp.CreateData( this->nSamples, this->nHidNodes);
	};

	int validate();
	////////////////////////////////////////////////////////////////////////////////////////////////
	//activate forward
	////////////////////////////////////////////////////////////////////////////////////////////////
	int forward( clMatrix<T>& randnums, bool bLearning );
	int backward( bool bNonlin =true );

	// calculate the partial derivative w.r.t the weights
	// will corrupt the activations dEda_nodes
	int dEdw(){
		return 0;
	}

	int assign_ptr( T* ptr, bool bDerive ){
		return 0;
	}

	//uses d_wts_vw as buffer
	T getloss_wtcost(){
		return 0;
	}

	int post_process(){
		return 0;
	}
};


///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class clLayerConvC: public clLayer<T>{ //Convolution C layer
	typedef  clMatrix<T> 	clmat;
public:
	clmat wts_vw, bias_vw, d_wts_vw, d_bias_vw;  //these are views
	//wts is nFilters by nJ_filt*nI_filt*nVisChannels
	//bias must be nFilters by 1
	clTensor<T> T_wts, T_wts_perm; //permutated Tensor wts, needed for bprop
	clTensor<T> T_nodes_perm, T_prev_nodes_perm;

	//these are for cpu debugging purposes
	T* nodes_cpu, *dEda_nodes_cpu, *wts_cpu, *prev_nodes_cpu, *prev_dEda_nodes_cpu, *wtsbuffer;

	int nVisChannels;
	int nVisI, nVisJ;
	int nI_filt, nJ_filt;
	int nI_grid, nJ_grid;
	int nFilters;
	int nPaddingStart;
	int nStride;
	int nPartialSum;

	bool bCpuMode;       //do conv operations on cpu

	T f_wtcost;

	int* dev_perm_buf;

	clmat wts_partialsum_temp;

	//constructor
	clLayerConvC(int _nSamples, int _nVisChannels, int _nVisI, int _nVisJ, int _nI_filt, int _nJ_filt,
			int _nI_grid, int _nJ_grid, int _nFilters, 	int _nNeuronType, T _f_wtcost,
			float _f_dropout, bool _bCpuMode, int _nPaddingStart, int _nStride, int _nPartialSum,
			std::vector<clLayer<T>* >& _vppl)

	: clLayer<T>(_nNeuronType, (_nVisChannels*_nI_filt*_nJ_filt+1)*_nFilters /*nParms*/,
			_nFilters*_nI_grid*_nJ_grid /*nHidNodes*/, _nSamples,
			true /*_bSamplesLeadDim*/, _f_dropout, "convc", _vppl)
	{
		nVisChannels 	= _nVisChannels;
		nVisI 			= _nVisI;
		nVisJ 			= _nVisJ;
		nI_filt 		= _nI_filt;
		nJ_filt 		= _nJ_filt;
		nI_grid 		= _nI_grid;
		nJ_grid 		= _nJ_grid;
		nFilters 		= _nFilters;
		f_wtcost 		= _f_wtcost;
		bCpuMode 		= _bCpuMode;
		nPaddingStart 	= _nPaddingStart;  //e.g. -2.   default: 0
		nStride			= _nStride;		   //default 1
		nPartialSum		= _nPartialSum;    //default is 0

		if (nPartialSum > 0){

			clASSERT( nI_grid*nJ_grid % nPartialSum == 0, "nPartialSum invalid!");

			wts_partialsum_temp.CreateData(	nFilters,
					nJ_filt*nI_filt*nVisChannels*(nI_grid*nJ_grid/nPartialSum) );
		}

		this->nodes.CreateData(  this->nSamples, nFilters*nI_grid*nJ_grid );
		this->dEda_nodes.CreateData(this->nodes.nI, this->nodes.nJ);

		if (bCpuMode){
			nodes_cpu 		= new T[nFilters*nI_grid*nJ_grid*this->nSamples];
			dEda_nodes_cpu 	= new T[nFilters*nI_grid*nJ_grid*this->nSamples];
			wts_cpu 		= new T[nVisChannels*nI_filt*nJ_filt*nFilters];
			prev_nodes_cpu 	= new T[nVisChannels*nVisI*nVisJ*this->nSamples];
			prev_dEda_nodes_cpu 	= new T[nVisChannels*nVisI*nVisJ*this->nSamples];
			wtsbuffer 		= new T[nI_filt*nJ_filt];

			checkCudaErrors( cudaMalloc((void**) &(dev_perm_buf), sizeof(int)*4 ));

			std::vector<int> dimvec(4);
			dimvec[0] = nFilters;
			dimvec[1] = nJ_filt;
			dimvec[2] = nI_filt;
			dimvec[3] = nVisChannels;

			T_wts.CreateData(dimvec);

			dimvec[0] = nVisChannels;
			dimvec[3] = nFilters;
			T_wts_perm.CreateData(dimvec);

			dimvec[0] = nFilters;
			dimvec[1] = nJ_grid;
			dimvec[2] = nI_grid;
			dimvec[3] = this->nSamples;
			T_nodes_perm.CreateData(dimvec);

			dimvec[0] = nVisChannels;
			dimvec[1] = nVisJ;
			dimvec[2] = nVisI;
			dimvec[3] = this->nSamples;
			T_prev_nodes_perm.CreateData(dimvec);

			clASSERT(nPartialSum==0 && nStride==1 && nPaddingStart == 0, "old conditions");
		}
	};

	~clLayerConvC(){

		if (bCpuMode){
			delete [] nodes_cpu;
			delete [] dEda_nodes_cpu;
			delete [] wts_cpu;
			delete [] prev_nodes_cpu;
			delete [] prev_dEda_nodes_cpu;
			delete [] wtsbuffer;

			cudaFree(dev_perm_buf);
		}


	}

	//assigns the wts view to pointer of parameters already allocated
	int assign_ptr( T* ptr, bool bDerive );
	int validate();

	////////////////////////////////////////////////////////////////////////////////////////////////
	//activate forward
	////////////////////////////////////////////////////////////////////////////////////////////////
	int forward( clMatrix<T>& randnums, bool bLearning );

	////////////////////////////////////////////////////////////////////////////////////////////////
	//backpropagate the partial derivatives
	////////////////////////////////////////////////////////////////////////////////////////////////
	// x=nonlinearity(a), updates prevlayer's dEdx, put it in ---> p_prevlayer->dEdEda_nodes
	int backward(bool bNonlin = true);

	// calculate the partial derivative w.r.t the weights
	// will corrupt the activations dEda_nodes
	int dEdw();

	//uses d_wts_vw as buffer
	T getloss_wtcost(){
		if ( f_wtcost > 0){
			cr (EleWisefun(fctrSq<T>(), wts_vw, d_wts_vw ))
			return f_wtcost*0.5*Sum2DInplace(d_wts_vw);
		}else if (f_wtcost == 0){
			return 0;
		}else{
			clASSERT(false, "f_wtcost < 0 !");
			return 0;
		}
	}

	int post_process(){
		return 0;
	}
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// deconvolution, similar to conv, but going back down-wards
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class clLayerDeConvC: public clLayer<T>{ // De-convolution C layer
	typedef  clMatrix<T> 	clmat;
public:
	clmat wts_vw, bias_vw, d_wts_vw, d_bias_vw;  //these are views
	//wts is nFilters by nJ_filt*nI_filt*nVisChannels
	//bias must be nFilters by 1
	clTensor<T> T_wts, T_wts_perm; //permutated Tensor wts, needed for bprop
	clTensor<T> T_nodes_perm, T_prev_nodes_perm;

	//these are for cpu debugging purposes
	T* nodes_cpu, *dEda_nodes_cpu, *wts_cpu, *prev_nodes_cpu, *prev_dEda_nodes_cpu, *wtsbuffer;

	int nVisChannels;
	int nVisI, nVisJ;
	int nI_filt, nJ_filt;
	int nI_grid, nJ_grid;
	int nFilters;
	int nPaddingStart;
	int nStride;
	int nPartialSum;

	bool bCpuMode;       //do conv operations on cpu

	T f_wtcost;

	int* dev_perm_buf;

	clmat wts_partialsum_temp;

	//constructor
	clLayerDeConvC(int _nSamples, int _nVisChannels, int _nVisI, int _nVisJ, int _nI_filt, int _nJ_filt,
			int _nI_grid, int _nJ_grid, int _nFilters, 	int _nNeuronType, T _f_wtcost,
			float _f_dropout, bool _bCpuMode, int _nPaddingStart, int _nStride, int _nPartialSum,
			std::vector<clLayer<T>* >& _vppl)

	: clLayer<T>(_nNeuronType, (_nFilters*_nI_filt*_nJ_filt+1)*_nVisChannels /*nParms*/,
			_nVisChannels*_nVisJ*_nVisI /*nHidNodes*/, _nSamples,
			true /*_bSamplesLeadDim*/, _f_dropout, "deconvc", _vppl)
	{
		nVisChannels 	= _nVisChannels;
		nVisI 			= _nVisI;
		nVisJ 			= _nVisJ;
		nI_filt 		= _nI_filt;
		nJ_filt 		= _nJ_filt;
		nI_grid 		= _nI_grid;
		nJ_grid 		= _nJ_grid;
		nFilters 		= _nFilters;
		f_wtcost 		= _f_wtcost;
		bCpuMode 		= _bCpuMode;
		nPaddingStart 	= _nPaddingStart;  //e.g. -2.   default: 0
		nStride			= _nStride;		   //default 1
		nPartialSum		= _nPartialSum;    //default is 0

		if (nPartialSum > 0){

			clASSERT( nI_grid*nJ_grid % nPartialSum == 0, "nPartialSum invalid!");

			wts_partialsum_temp.CreateData(	nFilters,
					nJ_filt*nI_filt*nVisChannels*(nI_grid*nJ_grid/nPartialSum) );
		}

		this->nodes.CreateData(  this->nSamples, nVisChannels*nVisJ*nVisI );
		this->dEda_nodes.CreateData(this->nodes.nI, this->nodes.nJ);

		if (bCpuMode){
			nodes_cpu 		= new T[nVisChannels*nVisJ*nVisI*this->nSamples];
			dEda_nodes_cpu 	= new T[nVisChannels*nVisJ*nVisI*this->nSamples];
			wts_cpu 		= new T[nVisChannels*nJ_filt*nI_filt*nFilters];
			prev_nodes_cpu 	= new T[nFilters*nJ_filt*nI_filt*this->nSamples];
			prev_dEda_nodes_cpu 	= new T[nFilters*nJ_filt*nI_filt*this->nSamples];
			wtsbuffer 		= new T[nJ_filt*nI_filt];

			checkCudaErrors( cudaMalloc((void**) &(dev_perm_buf), sizeof(int)*4 ));

			std::vector<int> dimvec(4);
			dimvec[0] = nFilters;
			dimvec[1] = nJ_filt;
			dimvec[2] = nI_filt;
			dimvec[3] = nVisChannels;

			T_wts.CreateData(dimvec);

			dimvec[0] = nVisChannels;
			dimvec[3] = nFilters;
			T_wts_perm.CreateData(dimvec);

			dimvec[0] = nVisChannels;
			dimvec[1] = nVisJ;
			dimvec[2] = nVisI;
			dimvec[3] = this->nSamples;
			T_nodes_perm.CreateData(dimvec);

			dimvec[0] = nFilters;
			dimvec[1] = nJ_grid;
			dimvec[2] = nI_grid;
			dimvec[3] = this->nSamples;
			T_prev_nodes_perm.CreateData(dimvec);

			clASSERT(nPartialSum==0 && nStride==1 && nPaddingStart == 0, "old conditions");
		}
	};

	~clLayerDeConvC(){

		if (bCpuMode){
			delete [] nodes_cpu;
			delete [] dEda_nodes_cpu;
			delete [] wts_cpu;
			delete [] prev_nodes_cpu;
			delete [] prev_dEda_nodes_cpu;
			delete [] wtsbuffer;

			cudaFree(dev_perm_buf);
		}


	}

	//assigns the wts view to pointer of parameters already allocated
	int assign_ptr( T* ptr, bool bDerive );

	int validate();

	////////////////////////////////////////////////////////////////////////////////////////////////
	//activate forward
	////////////////////////////////////////////////////////////////////////////////////////////////
	int forward( clMatrix<T>& randnums, bool bLearning );

	////////////////////////////////////////////////////////////////////////////////////////////////
	//backpropagate the partial derivatives
	////////////////////////////////////////////////////////////////////////////////////////////////
	// x=nonlinearity(a), updates prevlayer's dEdx, put it in ---> p_prevlayer->dEdEda_nodes
	int backward(bool);

	// calculate the partial derivative w.r.t the weights
	// will corrupt the activations dEda_nodes
	int dEdw();

	//uses d_wts_vw as buffer
	T getloss_wtcost(){
		if ( f_wtcost > 0){
			cr (EleWisefun(fctrSq<T>(), wts_vw, d_wts_vw ))
			return f_wtcost*0.5*Sum2DInplace(d_wts_vw);
		}else if (f_wtcost == 0){
			return 0;
		}else{
			clASSERT(false, "f_wtcost < 0 !");
			return 0;
		}
	}

	int post_process(){
		return 0;
	}
};



#endif /* _CU_CONV_H_ */
