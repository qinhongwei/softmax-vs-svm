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

/*
 * cu_jitters.h
 *
 *  Created on: May 8, 2013
 *      Author: tang
 */

#ifndef _CU_JITTERS_H_
#define _CU_JITTERS_H_



#include "cu_util.h"
#include "cu_clmatrix.h"
#include "cu_cltensor.h"
#include "mlp.h"


#define IN
#define OUT


///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class clLayerConvJitter: public clLayer<T>{ //Convolution C jitter layer
	typedef  clMatrix<T> 	clmat;

public:
	int nVisChannels;
	int nVisI, nVisJ, nVisI2, nVisJ2;

	clmat shift_i, shift_j;

	clmat datamean, datastd;
	clmat nodes_temp;
	int nCounter;

	//constructor
	clLayerConvJitter(int _nSamples, int _nVisChannels, int _nVisI, int _nVisJ, int _nVisI2, int _nVisJ2,
			clmat& _datamean, clmat& _datastd,
			std::vector<clLayer<T>* >& _vppl)

	: clLayer<T>( 0 /*_nNeuronType*/, 0 /*nParms*/,
			_nVisChannels*_nVisJ2*_nVisI2 /*nHidNodes*/, _nSamples,
			true /*_bSamplesLeadDim*/, 0/*_f_dropout*/, "convjitter", _vppl)
	{
		nVisChannels 	= _nVisChannels;
		nVisI 			= _nVisI;
		nVisJ 			= _nVisJ;
		nVisI2 			= _nVisI2;
		nVisJ2 			= _nVisJ2;

		this->nodes.CreateData(  this->nSamples, this->nHidNodes);
		shift_i.CreateData(this->nSamples,1);
		shift_j.CreateData(this->nSamples,1);

		datamean.CreateData( 1, this->nHidNodes );
		datastd.CreateData( 1, this->nHidNodes);
		datamean.CopyFrom(_datamean);
		datastd.CopyFrom(_datastd);

//		datastd.SetVal(T(0.1));
//		nodes_temp.CreateData(  this->nSamples, this->nHidNodes);
//		nCounter = 0;
	};

	~clLayerConvJitter(){
	}

	//assigns the wts view to pointer of parameters already allocated
	int assign_ptr( T* ptr, bool bDerive ){
		return 0;
	}

	int validate();

	////////////////////////////////////////////////////////////////////////////////////////////////
	//activate forward
	////////////////////////////////////////////////////////////////////////////////////////////////
	int forward( clMatrix<T>& randnums, bool bLearning );

	////////////////////////////////////////////////////////////////////////////////////////////////
	//backpropagate the partial derivatives
	////////////////////////////////////////////////////////////////////////////////////////////////
	// x=nonlinearity(a), updates prevlayer's dEdx, put it in ---> p_prevlayer->dEdEda_nodes
	int backward(bool bNonlin = true){
		return 0;
	}

	// calculate the partial derivative w.r.t the weights
	// will corrupt the activations dEda_nodes
	int dEdw(){
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



/*
 * x, y, are xy locations on image
 * r is -PI to PI, rotation
 * s is the scale
 */
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class clLayerConvXYRS: public clLayer<T>{ //Convolution C xyrs transformation layer
	typedef  clMatrix<T> 	clmat;
public:
	int nVisChannels;
	int nVisI, nVisJ, nVisI2, nVisJ2;

	clMatrix<float> Coords;     	//nSamples by nVisJ2*nVisI2*2
	clMatrix<float> Transforms; 	//nSamples by 4 - x y r s
	clMatrix<float> transform_range; 	//1 by 4 - x y r s
	clMatrix<float> transform_low; 		//1 by 4 - x y r s

	//constructor
	clLayerConvXYRS(int _nSamples, int _nVisChannels, int _nVisI, int _nVisJ, int _nVisI2, int _nVisJ2,
			clMatrix<float>& _trns_low, clMatrix<float>& _trns_high, std::vector<clLayer<T>* >& _vppl)

	: clLayer<T>( 0 /*_nNeuronType*/, 0/*nParms*/,
			_nVisChannels*_nVisJ2*_nVisI2 /*nHidNodes*/, _nSamples,
			true /*_bSamplesLeadDim*/, 0/*_f_dropout*/, "convxyrs", _vppl)
	{
		nVisChannels 	= _nVisChannels;
		nVisI 			= _nVisI;
		nVisJ 			= _nVisJ;
		nVisI2 			= _nVisI2;
		nVisJ2 			= _nVisJ2;

		this->nodes.CreateData(  this->nSamples, this->nHidNodes);

		Coords.CreateData( this->nSamples, nVisJ2*nVisI2*2 );
		Transforms.CreateData( this->nSamples, 4 );
		transform_range.CreateData(1, 4);
		transform_low.CreateData(1, 4);

		//assertion
		cr(EleWisefun(_trns_high, fctrGreaterOrEqualThan<T>(), _trns_low, transform_range))
		clASSERT(Sum2DInplace(transform_range) == 4, "trns_high not always >= trns_low");

		transform_low.CopyFrom(_trns_low);
		cr(EleWisefun(_trns_high, fctrMinus<T>(), transform_low, transform_range ))

		//orientation should be -M_PI to M_PI
		clASSERT( _trns_low.GetElem(0,2) >= -M_PI, "rot low > -M_PI" );
		clASSERT( _trns_high.GetElem(0,2) <= M_PI, "rot low < M_PI" );

	};

	~clLayerConvXYRS(){
	}

	//assigns the wts view to pointer of parameters already allocated
	int assign_ptr( T* ptr, bool bDerive ){
		return 0;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////
	//activate forward
	////////////////////////////////////////////////////////////////////////////////////////////////
	int forward( clMatrix<T>& randnums, bool bLearning );

	////////////////////////////////////////////////////////////////////////////////////////////////
	//backpropagate the partial derivatives
	////////////////////////////////////////////////////////////////////////////////////////////////
	// x=nonlinearity(a), updates prevlayer's dEdx, put it in ---> p_prevlayer->dEdEda_nodes
	int backward(bool bNonlin = true){
		return 0;
	}

	// calculate the partial derivative w.r.t the weights
	// will corrupt the activations dEda_nodes
	int dEdw(){
		return 0;
	}

	//uses d_wts_vw as buffer
	T getloss_wtcost(){
		return 0;
	}

	int post_process(){
		return 0;
	}

	int validate();
};



///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class clLayerImageMirror: public clLayer<T>{
	typedef  clMatrix<T> 	clmat;

public:
	int nVisChannels;
	int nVisI, nVisJ;

	clmat flip;

	//constructor
	clLayerImageMirror(int _nSamples, int _nVisChannels, int _nVisI, int _nVisJ,
			std::vector<clLayer<T>* >& _vppl)

	: clLayer<T>( 0 /*_nNeuronType*/, 0 /*nParms*/,
			_nVisChannels*_nVisJ*_nVisI /*nHidNodes*/, _nSamples,
			true /*_bSamplesLeadDim*/, 0/*_f_dropout*/, "imagemirror", _vppl)
	{
		nVisChannels 	= _nVisChannels;
		nVisI 			= _nVisI;
		nVisJ 			= _nVisJ;


		this->nodes.CreateData(  this->nSamples, this->nHidNodes);
		flip.CreateData(this->nSamples,1);
	};

	~clLayerImageMirror(){
	}

	//assigns the wts view to pointer of parameters already allocated
	int assign_ptr( T* ptr, bool bDerive ){
		return 0;
	}

	int validate();

	////////////////////////////////////////////////////////////////////////////////////////////////
	//activate forward
	////////////////////////////////////////////////////////////////////////////////////////////////
	int forward( clMatrix<T>& randnums, bool bLearning );

	////////////////////////////////////////////////////////////////////////////////////////////////
	//backpropagate the partial derivatives
	////////////////////////////////////////////////////////////////////////////////////////////////
	// x=nonlinearity(a), updates prevlayer's dEdx, put it in ---> p_prevlayer->dEdEda_nodes
	int backward(bool bNonlin = true){
		return 0;
	}

	// calculate the partial derivative w.r.t the weights
	// will corrupt the activations dEda_nodes
	int dEdw(){
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

#endif /* CU_JITTERS_H_ */
