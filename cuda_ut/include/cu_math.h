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

#ifndef _CU_MATH_H_
#define _CU_MATH_H_

//this header have some common definitions needed for math stuff

enum NodesSampleEnum{
	BIN_SAMPLE,  //apply sigmoid and then binary sample it
	ACTI_PROB,   //apply sigmoid to get probability of activation
	ACTI_SAMPLE, // given the activation probabilities, sample binary units
	GAUSS_SAMPLE, // sample from a gaussian with mean specified already with fsqrt_v_var
	TANH,
	SOFT_SIGN     // x/(1+abs(x))
};


enum LogicalOpEnum{
	GREATER,
	LESS,
	EQ,
	GREATER_OR_EQ,
	LESS_OR_EQ,
	NOT_EQ
};

/////////////////////////////////////////////////////////////////
// functors takes 1 parameter, T should be float or double
/////////////////////////////////////////////////////////////////


template <typename T>
class fctrInv{	public:
	__device__ T operator() (T a ) const {
		return 1/a;
	} };

template <typename T>
class fctrAbs{	public:
	__device__ T operator() (T a ) const {
		return fabsf(a);
	} };

template <>
class fctrAbs<double>{	public:
	__device__ double operator() (double a ) const {
		return fabs(a);
	} };

template <typename T>
class fctrLog{
	public:
	__device__ T operator() (T a) const {
		return logf(a);
	}
};

template <>
class fctrLog<double>{
	public:
	__device__ double operator() (double a ) const {
		return log(a);
	}
};

template <typename T>
class fctrExp{
	public:
	__device__ T operator() (T a ) const {
		return expf( a );
	}
};

template <>
class fctrExp<double> {
	public:
	__device__ double operator() ( double a ) const {
		return exp( a );
	}
};

template <typename T>
class fctrSq{
	public:
	__device__ T operator() (T a ) const {
		return a*a;
	}
};


template <typename T>
class fctrSqrt{
	public:
	__device__ T operator() (T a ) const {
		return sqrtf(a);
	}
};
template <>
class fctrSqrt<double>{
	public:
	__device__ double operator() ( double a ) const {
		return sqrt(a);
	}
};

template <typename T>
class fctrRsqrt{
	public:
	__device__ T operator() (T a ) const {
		return rsqrtf(a);
	}
};
template <>
class fctrRsqrt<double>{
	public:
	__device__ double operator() ( double a ) const {
		return rsqrt(a);
	}
};


template <typename T>
class fctrRint{
	public:
	__device__ T operator() (T a ) const {
		return rintf(a);
	}
};
template <>
class fctrRint<double>{
	public:
	__device__ double operator() ( double a ) const {
		return rint(a);
	}
};

template <typename T>
class fctrSigm{
	public:
	__device__ T operator() (T a ) const {
		 //i think we can divide by inf to equal zero, so we don't need any extra checking
		//return (a > -FLOAT_EXP_SATURATE) ? 1/(1+expf(-a)) : 0.0f;
		return 1/(1+expf(-a));
	}
};
template <>
class fctrSigm<double>{
	public:
	__device__ double operator() (double a ) const {
		return 1/( 1 + exp(-a) );
	}
};

template <typename T>
class fctrTanh{
	public:
	__device__ T operator() ( T a ) const {
		return 1.7159f*tanhf(0.66667f*a);
	}
};
template <>
class fctrTanh<double>{
	public:
	__device__ double operator() (double a ) const {
		return 1.7159*tanh(0.66667*a);
	}
};

template <typename T>
class fctrSoftSign{
	public:
	__device__ T operator() (T a ) const {
		return a/(1+fabsf(a));
	}
};
template <>
class fctrSoftSign<double>{
	public:
	__device__ double operator() (double a ) const {
		return a/(1+fabs(a));
	}
};

template <typename T>
class fctrBiReluActivation{
	public:
	__device__ T operator() (T a, T b) const {

		if (a > b ){
			return a-b;
		}else if(a < -b){
			return a+b;
		}else{
			return 0;
		}
	}
};

//needed for partial derivative calculation
template <typename T>
class fctrBiReluActivationDeriv{
	public:
	__device__ T operator() (T a, T b) const {

		if (a > b || a < -b)
			return 1;
		else
			return 0;
	}
};


template <typename T>
class fctrSLinearActivation{
	public:
	__device__ T operator() (T a) const {

		const T thresh = 0.5;
		const T slope = 0.4;

		if (a > thresh ){
			return (a-thresh)+thresh*slope;
		}else if(a < 0){
			return 0;
		}else{
			return a*slope;
		}
	}
};

//needed for partial derivative calculation
template <typename T>
class fctrSLinearActivationDeriv{
	public:
	__device__ T operator() (T y, T partial_derive) const {

		const T thresh = 0.5;
		const T slope = 0.4;

		if (y <= thresh*slope && y > 0 ){
			return partial_derive*slope;
		}else if (y > thresh*slope){
			return partial_derive;
		}else {
			return 0;
		}
	}
};


template <typename T>
class fctrLog1pExp{	public:
	__device__ T operator() (T a ) const {

		if (a < MAX_FLOAT_EXP)
			return log1pf( expf( a ) );
		else //for x > 80, log(1+exp(a)) == a
			return a;
	} };
template <>
class fctrLog1pExp<double>{	public:
	__device__ double operator() (double a ) const {
		if (a < MAX_DOUBLE_EXP)
			return log1p( exp( a ) );
		else //for x > 80, log(1+exp(a)) == a
			return a;
	} };

/* from nts
//note that a is the output (after the log1pexp nonlinearlity)
template <typename T>
class fctrLog1pExpDeriv{	public:
	__device__ T operator() (T a ) const {
		return 1-expf(-a);
	} };
template <>
class fctrLog1pExpDeriv<double>{	public:
	__device__ double operator() (double a ) const {
		return 1-exp(-a);
	} };
*/


/////////////////////////////////////////////////////////////////
// functors takes 2 parameter
/////////////////////////////////////////////////////////////////
template <typename T>
class fctrPlus{
	public:
	__device__ T operator() (T a, T b) const {
		return a+b;
	}
};

template <typename T>
class fctrMinus{
	public:
	__device__ T operator() (T a, T b) const {
		return a-b;
	}
};

template <typename T>
class fctrMul{
	public:
	__device__ T operator() (T a, T b) const {
		return a*b;
	}
};

template <typename T>
class fctrDiv{
	public:
	__device__ T operator() (T a, T b) const {
		return a/b;
	}
};

template <typename T>
class fctrMulSq{
	public:
	__device__ T operator() (T a, T b) const {
		return a*b*b;
	}
};

template <typename T>
class fctrSigmBinSample{
	public:
	__device__ T operator() (T a, T randnum ) const {
		float activ = 1/(1+expf(-a));
		return (activ > randnum)? 1.0f : 0.0f;
	}
};
template <>
class fctrSigmBinSample<double>{
	public:
	__device__ double operator() (double a, double randnum ) const {
		double activ = 1/(1+exp(-a));
		return (activ > randnum)? 1.0f : 0.0f;
	}
};

template <typename T>
class fctrSampleBernoulli{
	public:
	__device__ T operator() (T a, T randnum ) const {
		return (a > randnum)? 1.0f : 0.0f;
	}
};

template <typename T>
class fctrDiffSq{	public:
	__device__ T operator() (T a, T b ) const {
		return (a-b)*(a-b);
	} };

template <typename T>
class fctrMax{	public:
	__device__ T operator() (T a, T b ) const {
		return fmaxf(a,b);
	} };
template <>
class fctrMax<double>{	public:
	__device__ double operator() (double a, double b ) const {
		return fmax(a,b);
	} };

template <typename T>
class fctrMin{	public:
	__device__ T operator() (T a, T b ) const {
		return fminf(a,b);
	} };
template <>
class fctrMin<double>{	public:
	__device__ double operator() (double a, double b ) const {
		return fmin(a,b);
	} };

// returns 1 if a == b
template <typename T>
class fctrEquals{
	public:
	__device__ T operator() (T a, T b) const {
		return T(a==b);
	}
};

// returns 1 if a > b
template <typename T>
class fctrGreaterThan{
	public:
	__device__ T operator() (T a, T b) const {
		return T(a > b);
	}
};

// returns 1 if a > b
template <typename T>
class fctrGreaterOrEqualThan{
	public:
	__device__ T operator() (T a, T b) const {
		return T(a >= b);
	}
};

// returns 1 if a < b
template <typename T>
class fctrLessThan{
	public:
	__device__ T operator() (T a, T b) const {
		return T(a < b);
	}
};

// returns 1 if a < b
template <typename T>
class fctrLessOrEqualThan{
	public:
	__device__ T operator() (T a, T b) const {
		return T(a <= b);
	}
};




/////////////////////////////////////////////////////////////////
// functors takes 3 parameter
/////////////////////////////////////////////////////////////////
//Note: try to phase this out, use fctrAlphaOpBeta instead!

/*
//a*x + y
class fctrAlphaPlus{ public:
	__device__ float operator() (float a, float alpha, float b) const {
		//return __fadd_rn(__fmul_rn(alpha,a), b); more accurate, but could be slower on comput 1.x
		return alpha*a+b;
	}	};
//a*x - y
class fctrAlphaMinus{ public:
	__device__ float operator() (float a, float alpha, float b) const {
		return alpha*a-b;
	}	};
//a*x * y
class fctrAlphaMul{ public:
	__device__ float operator() (float a, float alpha, float b ) const {
		return alpha*a*b;
	}	};
//a*x / y
class fctrAlphaDiv{ public:
	__device__ float operator() (float a, float alpha, float b ) const {
		return alpha*a/b;
	}	};
*/

/////////////////////////////////////////////////////////////////
// functors takes 4 parameter
/////////////////////////////////////////////////////////////////

//a*x + b*y
template <typename T>
class fctrAlphaPlusBeta{ public:
	__device__ T operator() (T a, T alpha, T b, T beta ) const {
		return alpha*a+beta*b;
	}	};

//a*x - b*y
template <typename T>
class fctrAlphaMinusBeta{ public:
	__device__ T operator() (T a, T alpha, T b, T beta ) const {
		return (alpha*a)-(beta*b);
	}	};

//a*x * b*y
template <typename T>
class fctrAlphaMulBeta{ public:
	__device__ T operator() (T a, T alpha, T b, T beta ) const {
		return alpha*a*beta*b;
	}	};

//a*x / b*y
template <typename T>
class fctrAlphaDivBeta{ public:
	__device__ T operator() (T a, T alpha, T b, T beta ) const {
		return (alpha*a)/(beta*b);
	}	};




#endif
