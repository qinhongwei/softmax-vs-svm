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

#ifndef _MLP_H_
#define _MLP_H_

//this file contains function useful for multilayer neuralnets:  mlp

#include <vector>

#include "cu_clmatrix.h"
#include "cu_matrix_ops.h"
#include "elewise.h"
#include "bsxfun.h"
#include "cu_dnn_ops.h"
#include "cu_gpu_rand.h"


#define IN
#define OUT


inline int get_neuron_class(int nNeuronType){
	if (nNeuronType <= 10){
		return nNeuronType;
	}else{
		return nNeuronType/10;
	}
}


template <typename T>
int wts_vector2cell_device(IN const clMatrix<T> & X, IN const std::vector<int>& ll,
		 OUT std::vector<clMatrix<T> > & wts, OUT std::vector<clMatrix<T> > & biases );

template <typename T>
int forward_nonlinearity( clMatrix<T>& nodes, int nNeuronType, float f_dropout,
		clMatrix<T>& randnums, bool bLearning = false);

template <typename T>
int backward_nonlinearity(const clMatrix<T>& nodes, clMatrix<T>& dEda_nodes, int nNodeType );


///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class clLayer{
	typedef  clMatrix<T> 	clmat;
public:
	int nNeuronType;
	int nParams;  //total number of params

	clmat nodes, dEda_nodes; //pd_nodes is dE_da, where a is before the nonlinearity

	int nHidNodes;  //number of output nodes for this layer
	int nSamples; //number of training cases
	bool bSamplesLeadDim; //num samples is the leading dimension?? (true: a N by d matrix)
	float f_dropout;
	std::string name;

	std::vector<clLayer<T>* > vppl;	//vector pointers to prev layers

	virtual int 	validate() = 0; //make sure dimensions match
	virtual int 	forward(clMatrix<T>&, bool b) = 0;
	virtual int 	backward(bool b=true) = 0;
	virtual int 	dEdw() = 0;
	virtual int 	assign_ptr( T* ptr, bool bDerive ) = 0;
	virtual T  		getloss_wtcost() = 0;
	virtual int  	post_process() = 0;
	virtual			~clLayer(){};

	clLayer(){
		nNeuronType = -1;
		nParams = 0;
		vppl.resize(0);
	};

	clLayer(int nNeuronType, int nParams, int nHidNodes, int nSamples, bool bSamplesLeadDim,
			float f_dropout, std::string name, std::vector<clLayer<T>* >& _vppl) :
		nNeuronType(nNeuronType),
		nParams(nParams),
		nHidNodes(nHidNodes),
		nSamples(nSamples),
		bSamplesLeadDim(bSamplesLeadDim),
		f_dropout(f_dropout),
		name(name),
		vppl(_vppl)
	{};
};



template <typename T>
class clLayerData: public clLayer<T>{
	typedef  clMatrix<T> 	clmat;
public:
	//Data connected layer, no storage, only views

	//constructor
	clLayerData(int _nSamples, int _nHidNodes, bool _bSamplesLeadDim, float _f_dropout,
			std::vector<clLayer<T>* >& _vppl)
	: clLayer<T>(0, 0, _nHidNodes, _nSamples, _bSamplesLeadDim, _f_dropout, "fcdata", _vppl)
	{
		clASSERT(_bSamplesLeadDim, "_bSamplesLeadDim must = true");
		if (this->bSamplesLeadDim){
			this->nodes.nI = this->nSamples;
			this->nodes.nJ = this->nHidNodes;
		}else{
			this->nodes.nI = this->nHidNodes;
			this->nodes.nJ = this->nSamples;
		}

		clASSERT(this->vppl.size()==0, "vppl.size()==0");
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
	int backward(bool b=true){
		//should not be called
		return -1;
	}
	// calculate the partial derivative w.r.t the weights
	// will corrupt the activations dEda_nodes
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
template <typename T>
class clLayerFC: public clLayer<T>{  //fully connected layer
	typedef  clMatrix<T> 	clmat;
public:
	clmat wts_vw, bias_vw, d_wts_vw, d_bias_vw;  //these are views
	T f_wtcost;
	T fC, fC2;		//for l2_svm
	int nVisNodes;
	T wt_cons_val; //weight constraint value

	//constructor
	clLayerFC(int _nSamples, int _nVisNodes, int _nHidNodes,
			int _nNeuronType, bool _bSamplesLeadDim, T _f_wtcost, float _f_dropout,
			std::vector<clLayer<T>* >& _vppl, T _wt_cons_val=0 )
	: clLayer<T>(_nNeuronType, (_nVisNodes+1)*_nHidNodes, _nHidNodes, _nSamples,
			_bSamplesLeadDim, _f_dropout, "fc", _vppl )
	{
		nVisNodes = _nVisNodes;
		if (this->bSamplesLeadDim){
			this->nodes.CreateData(this->nSamples, this->nHidNodes);
			this->dEda_nodes.CreateData(this->nSamples, this->nHidNodes);
		}else{
			this->nodes.CreateData(this->nHidNodes, this->nSamples);
			this->dEda_nodes.CreateData(this->nHidNodes, this->nSamples);
		}

		f_wtcost = _f_wtcost;
		wt_cons_val = _wt_cons_val;
		fC = 0;
		fC2 = 0;
	};

	int assign_ptr( T* ptr, bool bDerive );
	int validate();

	////////////////////////////////////////////////////////////////////////////////////////////////
	//activate forward
	////////////////////////////////////////////////////////////////////////////////////////////////
	int forward( clMatrix<T>& randnums, bool bLearning );

	////////////////////////////////////////////////////////////////////////////////////////////////
	//backpropagate the partial derivatives
	////////////////////////////////////////////////////////////////////////////////////////////////
	// x=nonlinearity(a), updates prevlayer's dEdx ---> p_prevlayer->dEdEda_nodes
	int backward(bool bNonlin = true);

	// calculate the partial derivative w.r.t the weights
	// will corrupt the activations dEda_nodes
	int dEdw();

	//uses d_wts_vw as buffer
	T getloss_wtcost();

	//truncate gradients which are too big
	//this should be called after all of the dEdw function has been called
	int gradient_trunc();	//not finished implementation

	//make sure input weights to a hidden unit has a specific norm or smaller
	//this should be called after a weight update
	int post_process();
};




////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class clWtsShare{
	typedef  clMatrix<T> clmat;
	public:
	std::vector<clmat*>	vp_wts;
	std::vector<clmat*>	vp_d_wts;
	std::vector<bool> v_transposed; //w.r.t the first element

	clWtsShare(){
		vp_wts.resize(0);
		vp_d_wts.resize(0);
		v_transposed.resize(0);
	};

	void Add(clmat* p_wts, clmat* p_d_wts ){

		clASSERT(p_wts!= NULL && p_d_wts!=NULL, "p_wts or p_d_wts is NULL");
		clASSERT(p_wts->nI == p_d_wts->nI && p_wts->nJ == p_d_wts->nJ,
				"clWtsShare()::Add p_wts or p_d_wts are not of same size");

		if (vp_wts.size()==0){
			clASSERT(p_wts->nI*p_wts->nJ > 0,	"clWtsShare()::Add error1!");
			clASSERT(p_d_wts->nI*p_d_wts->nJ > 0,	"clWtsShare()::Add error2!");

			vp_wts.push_back(p_wts);
			vp_d_wts.push_back(p_d_wts);
			v_transposed.push_back(false);
		}else{
			if (p_wts->nI==vp_wts[0]->nI && p_wts->nJ==vp_wts[0]->nJ ){

				vp_wts.push_back(p_wts);
				vp_d_wts.push_back(p_d_wts);
				v_transposed.push_back(false);

			}else if(p_wts->nI==vp_wts[0]->nJ && p_wts->nJ==vp_wts[0]->nI){

				vp_wts.push_back(p_wts);
				vp_d_wts.push_back(p_d_wts);
				v_transposed.push_back(true);

			}else{
				clASSERT(false,	"clWtsShare()::Add error3");
			}
		}
	}

	//validate all wts in this set has equal values
	//expensive operation
	int validate_equal(){

		clASSERT( vp_wts.size() > 1, "vp_wts.size() > 1");
		clASSERT( vp_wts.size() == v_transposed.size(), "vp_wts.size() == v_transposed.size()");
		clASSERT( vp_wts.size() == vp_d_wts.size(), "vp_wts.size() == vp_d_wts.size()");

		clmat temp(vp_wts[0]->nI, vp_wts[0]->nJ);

		for (unsigned int i = 1; i < vp_wts.size(); ++i)
		{
			temp.CopyFrom(*vp_wts[0]);
			if (v_transposed[i]){
				EleWisefun2D( T(1), temp, fctrAlphaMinusBeta<T>(), T(1), 't', *vp_wts[i], temp);
			}else{
				EleWisefun( temp, fctrMinus<T>(), *vp_wts[i], temp);
			}
			T tot = Sum2DInplace(temp);

			if ( tot != 0){
				char notestr[100];
				sprintf(notestr, "\nclWtsShare::validate_equal() error! %lf\n", tot);
				clPrintf(notestr);
				return -i;
			}
		}

		return 0;
	}

};



/* example on how to set up a net
	see mexcuConvNNoo.cu
*/

template <typename T>
class clNet{

	typedef  clMatrix<T> 		clmat;

public:

	std::vector< clLayer<T>* >				layers;  //e.g. [784 500 500 10]
	int nDataSources; 		//how many independent input data do we have. e.g. 1
	int nObjectiveSinks;

	int nSourceStart, nSourceEnd, nSinkStart, nSinkEnd;

	long long nTotalParams; // number of total parameters in the net

	std::vector< clWtsShare<T> > v_shares;

	//constructor
	clNet(){
		clASSERT(false, "clNet() constructor");
	}

	clNet(int _nDataSources, int _nObjectiveSinks){

		nDataSources = _nDataSources;
		nObjectiveSinks = _nObjectiveSinks;

		clASSERT( nDataSources > 0 && nObjectiveSinks > 0, "nDataSources > 0 && nObjectiveSinks > 0");

		nSourceStart=nSourceEnd=nSinkStart=nSinkEnd = 0;
		nTotalParams =0;
		v_shares.resize(0);
	}

	~clNet(){

		for (unsigned int k = 0; k < layers.size(); ++k){
			delete layers[k];
		}
	}


	//DataX can potentially get corrupted by dropout at the first layer!
	int fprop( std::vector<clmat>& v_DataX, clmat& randnums, bool bLearning){

		clASSERT(v_DataX.size() == nDataSources, "v_DataX.size() == nDataSources");
		for (int i = 0; i < nDataSources; ++i){
			if ( !clMatrixSizeEqual( layers[i]->nodes, v_DataX[i] ))
				return -1-i;
		}

		for (int i = 0; i < nDataSources; ++i){
			layers[i]->nodes.pData = v_DataX[i].pData;
			layers[i]->forward(randnums, bLearning ); //only for dropout
		}

		for (int k = nDataSources; k < layers.size(); ++k){
			layers[k]->forward( randnums, bLearning );
		}
		return 0;
	}

	int bprop(){

		if (nSinkEnd < nSinkStart)
			return -1;


		//assume layers[nSinkStart:nSinkEnd]->dEda_nodes is already set, we update
		//we need to set everthing to zero because some layers wants to accumulate dEda (if it has
		//multiple outgoing connections)
		for (int k = nSinkStart-1; k > nSourceEnd; --k){
			layers[k]->dEda_nodes.SetVal(0.0);
		}

		//get layers[sinks]->dEda_nodes 's values with bNonlin = false;
		for (int k = nSinkStart; k <= nSinkEnd; ++k){
			cr( layers[k]->backward( false) )
		}

		for (int k = nSinkStart-1; k > nSourceEnd; --k){
			layers[k]->backward();
		}
		return 0;
	}



	//get loss and find dEda for last layer
	//temp must be same size as DataY (except for loss type 6)
	//input: layers[end]->nodes, DataY
	//outputs: f and layers[end]->dEda_nodes must be set
	//v_f has the vector of costs, last entry must be the weight cost
	int getloss( std::vector<clmat>& v_DataY, std::vector<clmat>& v_temp, bool bLearning, std::vector<T>& v_f ){

		if (v_DataY.size() != v_temp.size() || v_DataY.size() == 0 )
			return -1;
		if (nSinkStart < nDataSources || nSinkStart==0 || nSinkEnd < nSinkStart || nSinkEnd >= layers.size() )
			return -2;
		if (v_DataY.size() != v_f.size()-1)
			return -3;

		clLayerFC<T> * fcl;
		for (int kk = nSinkStart; kk <= nSinkEnd; ++kk){

			int kkk = kk-nSinkStart;
			if ( layers[kk]->nNeuronType == 5 && v_DataY[kkk].nJ != 1) //classification
				return -11;
			if ( get_neuron_class(layers[kk]->nNeuronType) == 6 && v_DataY[kkk].nJ != 1) //classification
				return -11;
			if (v_DataY[kkk].nI != layers[kk]->nSamples )
				return -12;
			if ( layers[kk]->nNeuronType != 5 && get_neuron_class(layers[kk]->nNeuronType) != 6
				 && v_DataY[kkk].nJ != layers[kk]->nodes.nJ )
				return -13;
			if ( get_neuron_class(layers[kk]->nNeuronType) != 6 && !clMatrixSizeEqual(v_temp[kkk], v_DataY[kkk]))
				return -14;

			if (layers[kk]->nNeuronType == 63){

				if (v_temp[kkk].nI != layers[kk]->nodes.nI &&  v_temp[kkk].nJ != layers[kk]->nodes.nJ*3 ){
					return -15;
				}

			}else if ( get_neuron_class(layers[kk]->nNeuronType) == 6 && !clMatrixSizeEqual(v_temp[kkk], layers[kk]->nodes ))
			{
				return -16;
			}


			int nTargetOutputs = v_DataY[kkk].nJ;
			int nSamples = v_DataY[kkk].nI;

			//note the grad computed below is negative of the error loss
			switch(layers[kk]->nNeuronType){
			case 0: //MSE
				//calculate Err MSE
				cr( EleWisefun( v_DataY[kkk], fctrMinus<T>(), layers[kk]->nodes, v_temp[kkk]))
				cr( EleWisefun( fctrSq<T>(),  v_temp[kkk],  v_temp[kkk]))
				v_f[kkk] = 0.5/(nSamples)*Sum2DInplace( v_temp[kkk] );

				cr( EleWisefun( v_DataY[kkk], fctrMinus<T>(), layers[kk]->nodes,  v_temp[kkk] ))
				cr( EleWisefun(  v_temp[kkk], fctrDiv<T>(), T(nSamples), layers[kk]->dEda_nodes ))

				break;
			case 1: //bernoulli CE
				cr( ce_f_Ix_logistic( layers[kk]->nodes, v_DataY[kkk], layers[kk]->dEda_nodes, &v_f[kkk] ))
				break;
			case 2:
				break;
			case 3:
				break;
			case 4:
				break;
			case 5: //softmax
				if (bLearning){
					cr( ce_f_Ix(  layers[kk]->nodes, v_DataY[kkk], layers[kk]->dEda_nodes, v_temp[kkk], &v_f[kkk] ))
				}else{
					cr( LogProbTestErrs( layers[kk]->nodes, v_temp[kkk], v_DataY[kkk],  &v_f[kkk] ))  //number of errors
						v_f[kkk] = 1-v_f[kkk];
				}
				break;
			case 6: //l2svm

				if (bLearning){
					fcl = static_cast< clLayerFC<T> * >( layers[kk]  );
					cr( loss_l2svm( layers[kk]->nodes, v_DataY[kkk], fcl->fC, layers[kk]->dEda_nodes, v_temp[kkk], v_f[kkk] ))
				}else{
					clmat temp = v_temp[kkk].ColView(0);
					cr( LogProbTestErrs( layers[kk]->nodes, temp, v_DataY[kkk],  &v_f[kkk] ))  //number of errors
					v_f[kkk] = 1-v_f[kkk];
				}
				break;
			case 61: //l1svm
				if (bLearning){
					fcl = static_cast< clLayerFC<T> * >( layers[kk]  );
					cr( loss_l1svm( layers[kk]->nodes, v_DataY[kkk], fcl->fC, layers[kk]->dEda_nodes, v_temp[kkk], v_f[kkk] ))
				}else{
					clmat temp = v_temp[kkk].ColView(0);
					cr( LogProbTestErrs( layers[kk]->nodes, temp, v_DataY[kkk],  &v_f[kkk] ))  //number of errors
					v_f[kkk] = 1-v_f[kkk];
				}
				break;

			case 63: //l2 transductive svm

				if (bLearning){
					fcl = static_cast< clLayerFC<T> * >( layers[kk]  );
					cr( loss_l2_tsvm( layers[kk]->nodes, v_DataY[kkk], fcl->fC, fcl->fC2, layers[kk]->dEda_nodes, v_temp[kkk], v_f[kkk] ))
				}else{
					clmat temp = v_temp[kkk].ColView(0);
					cr( LogProbTestErrs( layers[kk]->nodes, temp, v_DataY[kkk],  &v_f[kkk] ))  //number of errors
					v_f[kkk] = 1-v_f[kkk];
				}
				break;

			default:
				clASSERT(false, "layers[kk]->nNeuronType is invalid!");
				break;
			}
		}// kk

		v_f[ v_f.size()-1] = 0;
		if (bLearning){ //L2 wtcost
			for (int k = 0; k < layers.size(); ++k){
				v_f[v_f.size()-1] += layers[k]->getloss_wtcost();
			}
		}

		return 0;
	}



	int dEdw(){

		clASSERT(layers.size() >= nDataSources+nObjectiveSinks, "layer.size() >= nDataSources+nObjectiveSinks");

		for (int k = nSourceEnd+1; k < layers.size(); ++k){
			layers[k]->dEdw();
		}

		///////////////////////////////////////////////////////////////////////////////////////////
		//take care of weight sharings in the net
		for (unsigned int ui = 0; ui < v_shares.size(); ++ui){

			clASSERT( v_shares[ui].vp_wts.size() > 1, "v_shares[ui].vp_wts.size()>1");
			clASSERT( v_shares[ui].vp_wts.size() == v_shares[ui].vp_d_wts.size(),
					"v_shares[ui].vp_wts.size()==vp_d_wts.size() ");

			//first aggregate
			for (unsigned int uj = 1; uj < v_shares[ui].vp_d_wts.size(); ++uj){

				if (v_shares[ui].v_transposed[uj]){
					EleWisefun2D(T(1), *v_shares[ui].vp_d_wts[0], fctrAlphaPlusBeta<T>(),
								 T(1), 't', *v_shares[ui].vp_d_wts[uj],
								 *v_shares[ui].vp_d_wts[0]);
				}else{
					EleWisefun( *v_shares[ui].vp_d_wts[0], fctrPlus<T>(),
							    *v_shares[ui].vp_d_wts[uj], *v_shares[ui].vp_d_wts[0]);
				}
			}

			//now copy to all other copies of the weight
			for (unsigned int uj = 1; uj < v_shares[ui].vp_d_wts.size(); ++uj){

				if (v_shares[ui].v_transposed[uj]){
					EleWisefun2D( T(0), *v_shares[ui].vp_d_wts[uj], fctrAlphaPlusBeta<T>(),
								  T(1), 't', *v_shares[ui].vp_d_wts[0],
								  *v_shares[ui].vp_d_wts[uj]);
				}else{
					v_shares[ui].vp_d_wts[uj]->CopyFrom(*v_shares[ui].vp_d_wts[0]);
				}
			}

		}

		return 0;
	}


	int init( const clmat& w, const clmat& dw ){

		clASSERT(layers.size() >= nDataSources+nObjectiveSinks, "layer.size()  >= nDataSources+nObjectiveSinks");

		for (int kk = nSinkStart; kk <= nSinkEnd; ++kk){
			clASSERT( layers[kk]->bSamplesLeadDim == true, "Loss layers, samples should be leading");
		}

		nTotalParams = 0;
		for (int k = 0; k < layers.size(); ++k){
			nTotalParams += layers[k]->nParams;
		}
		clASSERT( nTotalParams > 0, "nTotalParams > 0");

		if (w.nI != nTotalParams || w.nJ != 1){
			char notestr[100];
			sprintf(notestr, "\nMLP.init() Error! nTotalParams:%lld, w.nI:%d\n", nTotalParams, w.nI);
			clASSERT(false, notestr);
		}
		clASSERT( w.nI == nTotalParams && w.nJ == 1, "clNet::init() w not correct dimensions");
		clASSERT( dw.nI == nTotalParams && dw.nJ == 1, "clNet::init() dw not correct dimensions");

		long long counter = 0;
		for (int k = 0; k < layers.size(); ++k){
			layers[k]->assign_ptr( w.pData + counter, false );
			layers[k]->assign_ptr( dw.pData + counter, true);
			counter += layers[k]->nParams;

			if (k >= nDataSources){
				int err = layers[k]->validate();
				if (err != 0 ){
					char notestr[100];
					sprintf(notestr, "\nMLP.init() Error! layer:%d %s   validate() err:%d\n",
							k, layers[k]->name.c_str(), err);
					clASSERT(false, notestr);
				}
			}
		}
		clASSERT( counter == nTotalParams, " counter == nTotalParams ");

		return 0;
	}

	int PostProcess(){
		for (int k = nSourceEnd+1; k < layers.size(); ++k){
			layers[k]->post_process();
		}
		return 0;
	}
};


#endif
