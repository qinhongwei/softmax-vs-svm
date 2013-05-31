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

#include "mlp.h"

/***********************************************************************************************************
 * @brief: convert a weight vector on device to a *shallow* vectors
 * 			X must be already on the device!
 * @param[in]:  clMat X is a dim by 1 column vector of serialized weights
 * 				ll - is layer size, including visibles, e.g. [784 500 500 10] for MNIST classification
 * @param[out]: wts - vector of clMat with pointers to somewhere in X, does not make deep copy
 * 			   biases - same dimension as wts, bias for first hidden layer is biases[0]
 * @topology:
 * @note:
 * 		the serialization and deserialization of the weights must be row major,
		meaning that each row of the weight matrix W (nVis by nHid)
		The augmented Weight matrix with the biases on the last row is (nVis+1 by nHid)
		So to serialize in matlab, we must take the transpose of it and then use (:)
		after transpose, the bias row become the last column.

		OR, simply serialize W' using matlab's (:), W' is nHid by nVis+1
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
int wts_vector2cell_device(IN const clMatrix<T> & X, IN const std::vector<int>& ll,
		OUT std::vector<clMatrix<T> > & wts, OUT std::vector<clMatrix<T> > & biases ){

	if (X.nI <= 0 || X.nJ != 1)
		return -1;

	int nWtsDim = X.nI;
	int nWtLayers = ll.size()-1;
	wts.resize(nWtLayers);
	biases.resize(nWtLayers);

	long long counter = 0;
	for (int i = 0; i < nWtLayers; ++i){

		int nVis = ll[i];
		int nHid = ll[i+1];

		wts[i].nI = nHid;  //transposed from the usual nVis by nHid.
		wts[i].nJ = nVis;
		wts[i].pData = X.pData+counter;
		counter += nHid*nVis;

		biases[i].nI = 1;
		biases[i].nJ = nHid;
		biases[i].pData = X.pData+counter;
		counter += nHid;
	}
	clASSERT( counter == nWtsDim, "wts_vector2cell_device: counter neq nWtsDim!");
	return 0;
}



/*
 * nNeuronType     - 0 - linear
					 1 - sigmoid
					 2 - tanh
					 3 - relu
					 31 - soft relu
					 32 - 2side relu
					 4 - softsign
					 5 - softmax
					 6 - l2svm- 1 vs rest
 */
template <typename T>
int forward_nonlinearity( clMatrix<T>& nodes, int nNeuronType, float f_dropout,
		clMatrix<T>& randnums, bool bLearning){

	const float BIRELU_CONS = 0.5f;
	//const float	SLINEAR_CONS = 0.5f;
	//const float	SLINEAR_SLOPE = 2.0f;

	switch(nNeuronType){
	case 0:
		break;
	case 1:
		cr(EleWisefun( fctrSigm<T>(), nodes, nodes) )
		break;
	case 2:
		cr(EleWisefun( fctrTanh<T>(), nodes, nodes) )
		break;
	case 3://relu
		cr( EleWisefun( nodes, fctrMax<T>(), T(0.0), nodes ) )
		break;
	case 31: //softrelu
		cr( EleWisefun( fctrLog1pExp<T>(), nodes, nodes) )
		break;
	case 32: //bi-relu activation
		cr( EleWisefun( nodes, fctrBiReluActivation<T>(), T(BIRELU_CONS), nodes ) )
		break;
	case 33: //s-linear activation
		//cr( EleWisefun( fctrSLinearActivation<T>(), nodes, nodes ) )
		cr( EleWisefun( nodes, fctrMax<T>(), T(0.0), nodes ) )
		cr( EleWisefun( fctrSq<T>(), nodes,  nodes ) )
		break;
	case 4:
		break;
	case 5:
		//softmax
		clASSERT( nodes.nJ <= MAX_THREADS, "MAX_THREADS (probably 512) too small for output dimension");
		cr( SoftmaxProbSafe( nodes, nodes) ) //compute the softmax
		break;
	case 6:
		break;
	case 61:
		break;
	case 62:
		break;
	case 63:
		break;
	default:
		clASSERT(false, "HIDDENTYPE is invalid!");
		break;
	}

	if (f_dropout > 0){
		if ( bLearning ){
			cr( Dropout( nodes, randnums, f_dropout) )
		}else{
			cr(EleWisefun(nodes, fctrMul<T>(), T( 1-f_dropout), nodes))
		}
	}

	return 0;
}


//note, need becareful with dropout and backprop, if a node is zeroed out,
// ***Backprop() must make sure the dE/da must also be zero!
template <typename T>
int backward_nonlinearity(const clMatrix<T>& nodes, clMatrix<T>& dEda_nodes, int nNodeType ){

	if (!clMatrixSizeEqual(nodes, dEda_nodes))
		return -1;

	switch ( nNodeType ){
	case 0:
		break;
	case 1:
		cr ( SigmoidBackprop( nodes, dEda_nodes) )
		break;
	case 2:
		cr ( TanhBackprop( nodes, dEda_nodes) )
		break;
	case 3:
		cr ( ReluBackprop( nodes, dEda_nodes) )
		break;
	case 31:
		cr ( SoftReluBackprop( nodes, dEda_nodes) )
		break;
	case 32:
		cr ( ReluBackprop( nodes, dEda_nodes) )
		break;
	case 33:
		//cr( EleWisefun( nodes, fctrSLinearActivationDeriv<T>(), dEda_nodes, dEda_nodes ) )
		cr ( ReluQuadBackprop( nodes, dEda_nodes) )
		break;
	case 4:
		break;
	default:
		clASSERT(false, "backprop HIDDENTYPE is invalid!");
		break;
	}
	return 0;
}





template <typename T>
int clLayerFC<T>::validate(){

	clASSERT(this->vppl.size() == 1, "vppl.size() == 1");
	clLayer<T>* p_prevlayer = this->vppl[0];

	if (!(p_prevlayer->nHidNodes == nVisNodes))
		return -1;
	if (!(p_prevlayer->nSamples == this->nSamples ))
		return -2;
	if (!(wts_vw.nI == this->nHidNodes && d_wts_vw.nI == this->nHidNodes))
		return -3;
	if (!(wts_vw.nJ == nVisNodes && d_wts_vw.nJ == nVisNodes))
		return -4;
	if (!(bias_vw.nI == 1 && d_bias_vw.nI == 1))
		return -5;
	if (!(bias_vw.nJ == this->nHidNodes && d_bias_vw.nJ == this->nHidNodes))
		return -6;
	if (!(wts_vw.nI*wts_vw.nJ > 0 && d_wts_vw.nI*d_wts_vw.nJ > 0))
		return -7;
	if (!(bias_vw.nI*bias_vw.nJ > 0 && d_bias_vw.nI*d_bias_vw.nJ > 0))
		return -8;
	if (!(this->nParams == (nVisNodes+1)*this->nHidNodes))
		return -9;
	if (!(clMatrixSizeEqual(this->nodes, this->dEda_nodes) ))
		return -10;
	if (!(this->nodes.nI*this->nodes.nJ > 0))
		return -11;
	if (!(this->f_dropout >= 0 && this->f_dropout <= 1))
		return -12;

	return 0;


}


template <typename T>
int clLayerFC<T>::forward( clMatrix<T>& randnums, bool bLearning ){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if ( this->nNeuronType == 5 && !this->bSamplesLeadDim)
		return -1;
	if ( get_neuron_class(this->nNeuronType) == 6 && !this->bSamplesLeadDim)
		return -1;

	if ( this->bSamplesLeadDim == true && p_prevlayer->bSamplesLeadDim == true){
		cr( ABeqC( T(1.0), p_prevlayer->nodes, 'n', wts_vw, 't', this->nodes) )
					cr( Bsxfun( this->nodes, fctrPlus<T>(), bias_vw, this->nodes) )
	}else if ( this->bSamplesLeadDim == true && p_prevlayer->bSamplesLeadDim == false){
		cr( ABeqC( T(1.0), p_prevlayer->nodes, 't', wts_vw, 't', this->nodes) )
					cr( Bsxfun( this->nodes, fctrPlus<T>(), bias_vw, this->nodes) )
	}else if ( this->bSamplesLeadDim == false && p_prevlayer->bSamplesLeadDim == true){
		cr( ABeqC( T(1.0), wts_vw, 'n',  p_prevlayer->nodes, 't', this->nodes) )
					cr( Bsxfun( this->nodes, fctrPlus<T>(), bias_vw.TrView(), this->nodes) )
	}else{
		cr( ABeqC( T(1.0), wts_vw, 'n',  p_prevlayer->nodes, 'n', this->nodes) )
					cr( Bsxfun( this->nodes, fctrPlus<T>(), bias_vw.TrView(), this->nodes) )
	}

	cr( forward_nonlinearity( this->nodes, this->nNeuronType, this->f_dropout, randnums, bLearning) )
	return 0;



}


template <typename T>
int clLayerFC<T>::backward(bool bNonlin ){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if (bNonlin){
		//initially, dEda is dEdy, need to pass thru nonlinearity (unless the last layer)
		cr( backward_nonlinearity(this->nodes, this->dEda_nodes, this->nNeuronType ) )
	}

	//the check below is critical for the bottom data layer
	if (p_prevlayer->dEda_nodes.nI*p_prevlayer->dEda_nodes.nJ > 0){

		//note that ABpCeqC is to allow addition of dEda from multiple next layers
		if ( p_prevlayer->bSamplesLeadDim == true && this->bSamplesLeadDim == true){
			cr( ABpCeqC( T(1.0), this->dEda_nodes, 'n', wts_vw, 'n', p_prevlayer->dEda_nodes) )
		}else if ( p_prevlayer->bSamplesLeadDim == true && this->bSamplesLeadDim == false){
			cr( ABpCeqC( T(1.0), this->dEda_nodes, 't', wts_vw, 'n', p_prevlayer->dEda_nodes) )
		}else if ( p_prevlayer->bSamplesLeadDim == false && this->bSamplesLeadDim == true){
			cr( ABpCeqC( T(1.0), wts_vw, 't', this->dEda_nodes, 't', p_prevlayer->dEda_nodes) )
		}else{
			cr( ABpCeqC( T(1.0), wts_vw, 't', this->dEda_nodes, 'n', p_prevlayer->dEda_nodes) )
		}

	}
	return 0;

}


template <typename T>
int clLayerFC<T>::dEdw(){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if ( this->bSamplesLeadDim == true && p_prevlayer->bSamplesLeadDim == true){
		cr( ABeqC( T(1.0), this->dEda_nodes, 't',  p_prevlayer->nodes, 'n', d_wts_vw ))
	}else if ( this->bSamplesLeadDim == true && p_prevlayer->bSamplesLeadDim == false){
		cr( ABeqC( T(1.0), this->dEda_nodes, 't', p_prevlayer->nodes, 't', d_wts_vw) )
	}else if ( this->bSamplesLeadDim == false && p_prevlayer->bSamplesLeadDim == true){
		cr( ABeqC( T(1.0), this->dEda_nodes, 'n', p_prevlayer->nodes, 'n', d_wts_vw) )
	}else{
		cr( ABeqC( T(1.0), this->dEda_nodes, 'n', p_prevlayer->nodes, 't', d_wts_vw) )
	}

	if ( f_wtcost > 0){
		cr( EleWisefun( -f_wtcost, wts_vw, fctrAlphaPlusBeta<T>(), T(1), d_wts_vw, d_wts_vw) )
	}

	if (this->bSamplesLeadDim){
		cr( SumInplace( this->dEda_nodes, 1) )
					cr( RowEleWisefun( T(1), this->dEda_nodes, 0, fctrAlphaPlusBeta<T>(),
							T(0), d_bias_vw, 0, d_bias_vw, 0 ))
	}else{
		cr( SumInplace( this->dEda_nodes, 2) )
					d_bias_vw.CopyFrom( this->dEda_nodes.ColView(0).TrView() );
	}
	return 0;

}


template <typename T>
T clLayerFC<T>::getloss_wtcost(){
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


template <typename T>
int clLayerFC<T>::post_process(){

	if (wt_cons_val > 0){
		cr ( EleWisefun(fctrSq<T>(), wts_vw, d_wts_vw ))
					cr ( SumInplace(d_wts_vw, 2) )
					clmat hh = d_wts_vw.ColView(0);
		cr (EleWisefun( hh, fctrDiv<T>(), wt_cons_val, hh ))
		cr (EleWisefun( hh, fctrMax<T>(), T(1.0f), hh ))
		cr (Bsxfun( wts_vw, fctrDiv<T>(), hh, wts_vw ))
	}

	return 0;

}



template <typename T>
int clLayerFC<T>::assign_ptr( T* ptr, bool bDerive ){

	if (bDerive){
		d_wts_vw.nI = this->nHidNodes;
		d_wts_vw.nJ = this->nVisNodes;
		d_wts_vw.pData = ptr;
		d_bias_vw.nI = 1;
		d_bias_vw.nJ =this->nHidNodes;
		d_bias_vw.pData = ptr + this->nHidNodes*this->nVisNodes;
	}else{
		wts_vw.nI = this->nHidNodes;
		wts_vw.nJ = this->nVisNodes;
		wts_vw.pData = ptr;
		bias_vw.nI = 1;
		bias_vw.nJ = this->nHidNodes;
		bias_vw.pData = ptr + this->nHidNodes*this->nVisNodes;
	}
	return 0;
}



//explicit instantiation
template int
wts_vector2cell_device<float>(IN const clMatrix<float> & X, IN const std::vector<int>& ll,
		OUT std::vector<clMatrix<float> > & wts, OUT std::vector<clMatrix<float> > & biases );
template int
wts_vector2cell_device<double>(IN const clMatrix<double> & X, IN const std::vector<int>& ll,
		OUT std::vector<clMatrix<double> > & wts, OUT std::vector<clMatrix<double> > & biases );

template int
forward_nonlinearity<float>( clMatrix<float>& nodes, int nNeuronType, float f_dropout,
		clMatrix<float>& randnums, bool bLearning );
template int
forward_nonlinearity<double>( clMatrix<double>& nodes, int nNeuronType, float f_dropout,
		clMatrix<double>& randnums, bool bLearning );

template int
backward_nonlinearity<float>(const clMatrix<float>& nodes, clMatrix<float>& dEda_nodes, int nNodeType );

template int
backward_nonlinearity<double>(const clMatrix<double>& nodes, clMatrix<double>& dEda_nodes, int nNodeType );


template class clLayerFC<float>;
template class clLayerFC<double>;


