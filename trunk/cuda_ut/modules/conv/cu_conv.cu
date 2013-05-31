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

#include <helper_cuda.h>
#include "cu_conv.h"
#include "cu_jitters.h"

#ifndef CONV_CPU_MODE
#include "cudaconv2/filter_acts_cl.cu"
#include "cudaconv2/img_acts_cl.cu"
#include "cudaconv2/weight_acts_cl.cu"
#endif

#include "cudaconv2/conv_util_cl.cu"
#include "cudaconv2/normalization_cl.cu"





// pSource must be size nVisChannels*nVisI*nVisJ by 1
// pKernel must be size nVisChannels*nI_filt*nJ_filt by nFilters
// pDest must be size nFilters*nI_grid*nJ_grid by 1
// everything is column major
// if bConv==true, we are doing convolution instead of filtering
template <typename T>
int filter2d_cpu(	T *pSource, int nVisChannels, int nVisI, int nVisJ,
		T *pKernel, int nI_filt, int nJ_filt, int nFilters,
		T *pDest, int nI_grid, int nJ_grid, CONVMODE mode,
		bool bConv, bool bZeroDest)
{
	if (nVisChannels*nFilters <= 0 )
		return -2;

	int nKernelHalfHeight = nI_filt/2;
	int nKernelHalfWidth  = nJ_filt/2;

	int nLeftDx   = nKernelHalfWidth;
	int nRightDx  = nJ_filt - nKernelHalfWidth - 1;
	int nTopDy    = nKernelHalfHeight;
	int nBottomDy = nI_filt - nKernelHalfHeight - 1;

	//char notestr[1000];
	//sprintf(notestr, "%d %d %d %d \n",nLeftDx,nRightDx, nTopDy, nBottomDy);
	//clPrintf(notestr);

	// Do convolution
	switch (mode){
	case VALID:
		//clPrintf("valid\n");
		if ( nVisI < nI_filt ||  nVisJ < nJ_filt )
			return -1;
		if (nI_grid != nVisI-nI_filt+1 || nJ_grid != nVisJ-nJ_filt+1 )
			return -3;

		for (int k = 0; k < nFilters; ++k){
			//clPrintf("k\n");
			for (int ii = 0; ii < nI_grid; ++ii){
				for (int jj = 0; jj < nJ_grid; ++jj) {

					T* pp_dest = pDest + (jj*nI_grid+ii)*nFilters+k;
					if (bZeroDest){
						*pp_dest = 0;
					}

					for (int c = 0; c < nVisChannels; ++c){

						int kernel_ind, kernel_offset;
						if (!bConv){
							kernel_ind = ((k*nJ_filt+0)*nI_filt+0)*nVisChannels+c;
							kernel_offset = nVisChannels;
						}else{
							kernel_ind = ((k*nJ_filt+nJ_filt-1)*nI_filt+nI_filt-1)*nVisChannels+c;
							kernel_offset = -nVisChannels;
						}

						int src_ind = (jj*nVisI+ii)*nVisChannels+c;
						int src_offset = nVisChannels;

						for (int n = -nLeftDx; n <= nRightDx; ++n) {
							for (int m = -nTopDy; m <= nBottomDy; ++m) {
								*pp_dest += pKernel[kernel_ind]*pSource[src_ind];

								kernel_ind += kernel_offset;
								src_ind += src_offset;
							}
							src_ind += (nVisI-nI_filt)*nVisChannels; //skip to the next column
						}
					}//c
				} //jj
			} //ii
		}//k

		break;

	case SAME:
		//clPrintf("same\n");
		if (nI_grid != nVisI || nJ_grid != nVisJ )
			return -4;

		for (int k = 0; k < nFilters; ++k){
			for (int ii = 0; ii < nI_grid; ++ii){
				for (int jj = 0; jj < nJ_grid; ++jj) {

					T* pp_dest = pDest + (jj*nI_grid+ii)*nFilters+k;
					if (bZeroDest){
						*pp_dest = 0;
					}

					for (int c = 0; c < nVisChannels; ++c){

						int kernel_ind, kernel_offset;
						if (!bConv){
							kernel_ind = ((k*nJ_filt+0)*nI_filt+0)*nVisChannels+c;
							kernel_offset = nVisChannels;
						}else{
							kernel_ind = ((k*nJ_filt+nJ_filt-1)*nI_filt+nI_filt-1)*nVisChannels+c;
							kernel_offset = -nVisChannels;
						}

						for (int n = -nLeftDx; n <= nRightDx; ++n) {

							if (n+jj >= 0 && n+jj < nVisJ){  //if in bounds horizontal

								for (int m = -nTopDy; m <= nBottomDy; ++m) {

									if ( m+ii >= 0 && m+ii < nVisI ){ //if in bounds vertical

										const int src_ind = ( (n+jj)*nVisI+ (m+ii))*nVisChannels+c;
										*pp_dest += pKernel[kernel_ind]*pSource[src_ind];
									}
									kernel_ind += kernel_offset;
								}
							}else{
								kernel_ind += nI_filt*kernel_offset;
							}
						}
					}//c
				} //jj
			} //ii
		}//k

		break;

	case FULL:

		if (nI_grid != nVisI+nI_filt-1 || nJ_grid != nVisJ+nJ_filt-1)
			return -5;

		for (int k = 0; k < nFilters; ++k){
			for (int ii = 0; ii < nI_grid; ++ii){
				for (int jj = 0; jj < nJ_grid; ++jj) {

					T* pp_dest = pDest + (jj*nI_grid+ii)*nFilters+k;
					if (bZeroDest){
						*pp_dest = 0;
					}

					for (int c = 0; c < nVisChannels; ++c){

						int kernel_ind, kernel_offset;
						if (!bConv){
							kernel_ind = ((k*nJ_filt+0)*nI_filt+0)*nVisChannels+c;
							kernel_offset = nVisChannels;
						}else{
							kernel_ind = ((k*nJ_filt+nJ_filt-1)*nI_filt+nI_filt-1)*nVisChannels+c;
							kernel_offset = -nVisChannels;
						}

						for (int n = -nLeftDx; n <= nRightDx; ++n) {

							int image_j = jj-nJ_filt+1+nLeftDx+n; //wouldbe location on the image coordinates

							if ( image_j >= 0 && image_j < nVisJ){  //if in bounds horizontal

								for (int m = -nTopDy; m <= nBottomDy; ++m) {

									int image_i = ii-nI_filt+1+nTopDy+m; //wouldbe location on the image coordinates

									if ( image_i >= 0 && image_i < nVisI ){ //if in bounds vertical

										const int src_ind = ( image_j*nVisI+ image_i)*nVisChannels+c;
										*pp_dest += pKernel[kernel_ind]*pSource[src_ind];
									}
									kernel_ind += kernel_offset;
								}
							}else{
								kernel_ind += nI_filt*kernel_offset;
							}
						}
					}//c
				} //jj
			} //ii
		}//k
		break;
	default:
		return -6;
	}

	return 0;
}

template int filter2d_cpu<float>(float *pSource, int nVisChannels, int nVisI, int nVisJ,
		float *pKernel, int nI_filt, int nJ_filt, int nFilters,
		float *pDest, int nI_grid, int nJ_grid, CONVMODE mode,
		bool bConv, bool bZeroDest);
template int filter2d_cpu<double>(double *pSource, int nVisChannels, int nVisI, int nVisJ,
		double *pKernel, int nI_filt, int nJ_filt, int nFilters,
		double *pDest, int nI_grid, int nJ_grid, CONVMODE mode,
		bool bConv, bool bZeroDest);


// according to AK's data ordering
// note: this works for all nSample cases
// pSource must be size: nSamples by nVisJ*nVisI*nVisChannels
// pKernel must be size: nFilters by nJ_filt*nI_filt*nVisChannels
// pDest must be size:	nSamples by nJ_grid*nI_grid*nFilters
// if bConv==true, we are doing convolution instead of filtering
template <typename T>
int filter2d_cpu_akmaj(	T *pSource, int nVisChannels, int nVisI, int nVisJ, int nSamples,
		T *pKernel, int nI_filt, int nJ_filt, int nFilters,
		T *pDest, int nI_grid, int nJ_grid, CONVMODE mode,
		bool bConv, bool bZeroDest)
{
	if (nVisChannels*nFilters <= 0 )
		return -2;

	int nKernelHalfHeight = nI_filt/2;
	int nKernelHalfWidth  = nJ_filt/2;

	int nLeftDx   = nKernelHalfWidth;
	int nRightDx  = nJ_filt - nKernelHalfWidth - 1;
	int nTopDy    = nKernelHalfHeight;
	int nBottomDy = nI_filt - nKernelHalfHeight - 1;

	//char notestr[1000];
	//sprintf(notestr, "%d %d %d %d \n",nLeftDx,nRightDx, nTopDy, nBottomDy);
	//clPrintf(notestr);

	// Do convolution
	switch (mode){
	case VALID:
		if ( nVisI < nI_filt ||  nVisJ < nJ_filt )
			return -1;
		if (nI_grid != nVisI-nI_filt+1 || nJ_grid != nVisJ-nJ_filt+1 )
			return -3;

		for (int nn = 0; nn < nSamples; ++nn){

			for (int k = 0; k < nFilters; ++k){
				for (int ii = 0; ii < nI_grid; ++ii){
					for (int jj = 0; jj < nJ_grid; ++jj) {

						T* pp_dest = pDest + nn + nSamples*(jj+nJ_grid*(ii + nI_grid*k));
						if (bZeroDest){
							*pp_dest = 0;
						}

						for (int c = 0; c < nVisChannels; ++c){

							int kernel_ind, kernel_offset;
							if (!bConv){
								kernel_ind = ((c*nI_filt+0)*nJ_filt+0)*nFilters+k;
								kernel_offset = nFilters;
							}else{
								kernel_ind = ((c*nI_filt+nI_filt-1)*nJ_filt+nJ_filt-1)*nFilters+k;
								kernel_offset = -nFilters;
							}

							int src_ind = ((c*nVisI+ii)*nVisJ+jj)*nSamples+nn;
							int src_offset = nSamples;

							for (int m = -nTopDy; m <= nBottomDy; ++m) {
								for (int n = -nLeftDx; n <= nRightDx; ++n) {
									*pp_dest += pKernel[kernel_ind]*pSource[src_ind];

									kernel_ind 	+= kernel_offset;
									src_ind 	+= src_offset;
								}
								src_ind += (nVisJ-nJ_filt)*nSamples; //skip to the next row
							}
						}//c
					} //jj
				} //ii
			}//k

		}//nn

		break;

	case SAME:
		//clPrintf("same\n");
		if (nI_grid != nVisI || nJ_grid != nVisJ )
			return -4;

		for (int nn = 0; nn < nSamples; ++nn){
			for (int k = 0; k < nFilters; ++k){
				for (int ii = 0; ii < nI_grid; ++ii){
					for (int jj = 0; jj < nJ_grid; ++jj) {

						T* pp_dest = pDest + ((k*nI_grid+ii)*nJ_grid+jj)*nSamples + nn;
						if (bZeroDest){
							*pp_dest = 0;
						}

						for (int c = 0; c < nVisChannels; ++c){

							int kernel_ind, kernel_offset;
							if (!bConv){
								kernel_ind = ((c*nI_filt+0)*nJ_filt+0)*nFilters+k;
								kernel_offset = nFilters;
							}else{
								kernel_ind = ((c*nI_filt+nI_filt-1)*nJ_filt+nJ_filt-1)*nFilters+k;
								kernel_offset = -nFilters;
							}

							for (int m = -nTopDy; m <= nBottomDy; ++m) {
								if ( m+ii >= 0 && m+ii < nVisI ){ //if in bounds vertical

									for (int n = -nLeftDx; n <= nRightDx; ++n) {
										if (n+jj >= 0 && n+jj < nVisJ){  //if in bounds horizontal

											const int src_ind = ((c*nVisI+(m+ii))*nVisJ+(n+jj))*nSamples+nn;
											*pp_dest += pKernel[kernel_ind]*pSource[src_ind];
										}
										kernel_ind += kernel_offset;
									}
								}else{
									kernel_ind += nJ_filt*kernel_offset;
								}
							}
						}//c
					} //jj
				} //ii
			}//k
		}//nn

		break;

	case FULL:

		if (nI_grid != nVisI+nI_filt-1 || nJ_grid != nVisJ+nJ_filt-1)
			return -5;

		for (int nn = 0; nn < nSamples; ++nn){
			for (int k = 0; k < nFilters; ++k){
				for (int ii = 0; ii < nI_grid; ++ii){
					for (int jj = 0; jj < nJ_grid; ++jj) {

						T* pp_dest = pDest + ((k*nI_grid+ii)*nJ_grid+jj)*nSamples + nn;
						if (bZeroDest){
							*pp_dest = 0;
						}

						for (int c = 0; c < nVisChannels; ++c){

							int kernel_ind, kernel_offset;
							if (!bConv){
								kernel_ind = ((c*nI_filt+0)*nJ_filt+0)*nFilters+k;
								kernel_offset = nFilters;
							}else{
								kernel_ind = ((c*nI_filt+nI_filt-1)*nJ_filt+nJ_filt-1)*nFilters+k;
								kernel_offset = -nFilters;
							}

							for (int m = -nTopDy; m <= nBottomDy; ++m){

								const int image_i = ii-nI_filt+1+nTopDy+m; //wouldbe location on the image coordinates
								if ( image_i >= 0 && image_i < nVisI ){ //if in bounds vertical

									for (int n = -nLeftDx; n <= nRightDx; ++n) {

										const int image_j = jj-nJ_filt+1+nLeftDx+n; //wouldbe location on the image coordinates
										if ( image_j >= 0 && image_j < nVisJ){  //if in bounds horizontal

											const int src_ind = ((c*nVisI+image_i)*nVisJ+image_j)*nSamples + nn;
											*pp_dest += pKernel[kernel_ind]*pSource[src_ind];
										}
										kernel_ind += kernel_offset;
									}
								}else{
									kernel_ind += nJ_filt*kernel_offset;
								}
							}
						}//c
					} //jj
				} //ii
			}//k
		}//nn
		break;
	default:
		return -6;
	}

	return 0;
}
template int filter2d_cpu_akmaj<float>(	float *pSource, int nVisChannels, int nVisI, int nVisJ, int nSamples,
		float *pKernel, int nI_filt, int nJ_filt, int nFilters,
		float *pDest, int nI_grid, int nJ_grid, CONVMODE mode,
		bool bConv, bool bZeroDest);
template int filter2d_cpu_akmaj<double>(	double *pSource, int nVisChannels, int nVisI, int nVisJ, int nSamples,
		double *pKernel, int nI_filt, int nJ_filt, int nFilters,
		double *pDest, int nI_grid, int nJ_grid, CONVMODE mode,
		bool bConv, bool bZeroDest);





/***********************************************************************************************************
 * @brief: function to do max pooling
 *
 * @param[in]:	pPrev - pointer to previous conv layer, nFilters*(nI*2)*(nJ*2)*nSamples
 * 				nPoolingType - 0-avg; 1-max;
 * 				pOut - pointer to the pooled layer, nFilters*nI*nJ*nSamples
 * 				dim - total dimensions of the pOut
 *
 * @param[out]:
 * @topology:
 * @note:
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template<typename T>
__device__ inline T pooling_max_op(T t1, T t2, T t3, T t4)
{
	return fmaxf(t1, fmaxf(t2, fmaxf(t3, t4)));
}

template<>
__device__ inline double pooling_max_op<double>(double t1, double t2, double t3, double t4)
{
	return fmax(t1, fmax(t2, fmax(t3, t4)));
}


template<typename T>
__device__ inline T pooling_sum_op(T t1, T t2, T t3, T t4)
{
	return t1+t2+t3+t4;
}

//fprop
//layout is: [nSamples; nJ_sgrid*nI_sgrid*nFilters]
template<typename T>
__global__ void conv_pooling_kernel(
		IN const T * pPrev, int nSamples, int nI, int nJ, int nPoolingType,
		OUT T * pOut, int dim)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;


	const int nnn = nJ*nI*4*nSamples;
	const int nn = nJ*2*nSamples;

	for (int i = ind; i < dim; i += totalThreads){

		const int ss = i % nSamples;
		int rest = i / nSamples;
		const int jj = rest % nJ;
		rest = i / (nSamples*nJ);
		const int ii = rest % nI;
		const int ff = i / (nSamples*nJ*nI);

		const int topleft 	= ff*nnn+(ii*2)*nn+(jj*2)*nSamples+ss;
		const int topright 	= ff*nnn+(ii*2)*nn+(jj*2+1)*nSamples+ss;
		const int botleft 	= ff*nnn+(ii*2+1)*nn+(jj*2)*nSamples+ss;
		const int botright	= ff*nnn+(ii*2+1)*nn+(jj*2+1)*nSamples+ss;

		if (nPoolingType == 0){
			pOut[i] = pooling_sum_op( pPrev[botright], pPrev[botleft], pPrev[topleft], pPrev[topright]);
		}else if (nPoolingType == 1){
			pOut[i] = pooling_max_op( pPrev[botright], pPrev[botleft], pPrev[topleft], pPrev[topright]);
		}
	}
}

//bprop
/*
 * p_prev is the larger conv previous layer
 * p_nodes is the smaller pooled nodes
 * p_dEda is partial derivative of the pooled nodes
 * p_dEda_prev is the larger conv partial derivatives
 *
 * layout is: [nSamples; nJ_sgrid*nI_sgrid*nFilters]
 */
template<typename T>
__global__ void conv_pooling_bprop_kernel(
		IN const T* p_prev, IN const T* p_nodes, IN const T * p_dEda, int nSamples,
		int nI, int nJ, int nPoolingType, T gamma, OUT T * p_dEda_prev, int dim)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;


	const int nnn = nJ*nI*4*nSamples;
	const int nn = nJ*2*nSamples;

	for (int i = ind; i < dim; i += totalThreads){

		const int ss = i % nSamples;
		int rest = i / nSamples;
		const int jj = rest % nJ;
		rest = i / (nSamples*nJ);
		const int ii = rest % nI;
		const int ff = i / (nSamples*nJ*nI);

		const int topleft 	= ff*nnn+(ii*2)*nn+(jj*2)*nSamples+ss;
		const int topright 	= ff*nnn+(ii*2)*nn+(jj*2+1)*nSamples+ss;
		const int botleft 	= ff*nnn+(ii*2+1)*nn+(jj*2)*nSamples+ss;
		const int botright	= ff*nnn+(ii*2+1)*nn+(jj*2+1)*nSamples+ss;

		if (gamma != 1){
			p_dEda_prev[topleft]  *= gamma;
			p_dEda_prev[topright] *= gamma;
			p_dEda_prev[botleft]  *= gamma;
			p_dEda_prev[botright] *= gamma;
		}

		if (nPoolingType == 0){

			p_dEda_prev[topleft] += p_dEda[i];
			p_dEda_prev[topright] += p_dEda[i];
			p_dEda_prev[botleft]  += p_dEda[i];
			p_dEda_prev[botright] += p_dEda[i];

		}else if (nPoolingType == 1){

			const T maxval = p_nodes[i];
			//the ladder will help break a tie among the p_prev
			if (	 p_prev[botright] == maxval){
				p_dEda_prev[botright] += p_dEda[i];

			}else if(p_prev[botleft] == maxval){
				p_dEda_prev[botleft] += p_dEda[i];

			}else if(p_prev[topright] == maxval){
				p_dEda_prev[topright] += p_dEda[i];

			}else if(p_prev[topleft] == maxval){
				p_dEda_prev[topleft] += p_dEda[i];
			}else{
				//error scenario
				p_dEda_prev[topleft]=p_dEda_prev[topright]=p_dEda_prev[botleft]=p_dEda_prev[botright]=9999;
			}
		}
	}
}

/*
 * nI, nJ is the dimension of Out, not Prev
 */
template< typename T>
int ConvPooling(IN const clMatrix<T>& Prev, int nFilters, int nI, int nJ, int nSamples,
		int nPoolingType, OUT clMatrix<T>& Out){

	if (Prev.nI != nSamples || Out.nI != nSamples)
		return -1;
	if (Prev.nJ != nFilters*nI*nJ*4 || Out.nJ != nFilters*nI*nJ)
		return -2;
	if (nPoolingType <0 || nPoolingType > 1)
		return -3;

	const unsigned int datadim = Out.nI*Out.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	conv_pooling_kernel<<<dim_grid, dim_block>>>( Prev.pData, nSamples, nI, nJ, nPoolingType,
			Out.pData, datadim );

	return 0;
}

template int ConvPooling<float>(IN const clMatrix<float>& Prev, int nFilters, int nI, int nJ, int nSamples,
		int nPoolingType, OUT clMatrix<float>& Out);
template int ConvPooling<double>(IN const clMatrix<double>& Prev, int nFilters, int nI, int nJ, int nSamples,
		int nPoolingType, OUT clMatrix<double>& Out);


/*
 * nI, nJ is the dimension of Out
 */
template< typename T>
int ConvPoolingBackProp(IN const clMatrix<T>& Prev, IN const  clMatrix<T>& Nodes,
		IN const  clMatrix<T>& dEdaNodes, int nFilters, int nI, int nJ, int nSamples,
		int nPoolingType, T gamma, OUT clMatrix<T>& dEdaPrev )
{

	if (!(clMatrixSizeEqual(Prev, dEdaPrev) ))
		return -1;
	if (!(clMatrixSizeEqual(Nodes, dEdaNodes) ))
		return -2;

	if (Prev.nI != nSamples || Nodes.nI != nSamples)
		return -1;
	if (Prev.nJ != nFilters*nI*nJ*4 || Nodes.nJ != nFilters*nI*nJ)
		return -2;
	if (nPoolingType <0 || nPoolingType > 1)
		return -3;

	const unsigned int datadim = Nodes.nI*Nodes.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	conv_pooling_bprop_kernel<<<dim_grid, dim_block>>>( Prev.pData, Nodes.pData, dEdaNodes.pData,
			nSamples, nI, nJ, nPoolingType, gamma, dEdaPrev.pData, datadim );

	return 0;
}

template int ConvPoolingBackProp<float>(IN const clMatrix<float>& Prev, IN const  clMatrix<float>& Nodes,
		IN const  clMatrix<float>& dEdaNodes, int nFilters, int nI, int nJ, int nSamples,
		int nPoolingType, float gamma, OUT clMatrix<float>& dEdaPrev );
template int ConvPoolingBackProp<double>(IN const clMatrix<double>& Prev, IN const  clMatrix<double>& Nodes,
		IN const  clMatrix<double>& dEdaNodes, int nFilters, int nI, int nJ, int nSamples,
		int nPoolingType, double gamma, OUT clMatrix<double>& dEdaPrev );

template<typename T>
int clLayerConvS<T>::validate(){

	clASSERT(this->vppl.size()==1, "vppl.size()==1");
	clLayer<T>* p_prevlayer = this->vppl[0];

	if (p_prevlayer->name.compare("convc") != 0 )
		return -1;

	const clLayerConvC<T> * prevlayer2 = static_cast< const clLayerConvC<T>* >( p_prevlayer );

	if (!(p_prevlayer->nSamples == this->nSamples ))
		return -2;
	if ( !p_prevlayer->bSamplesLeadDim )
		return -3;
	if (!(this->nParams == 0))
		return -9;
	if (!(clMatrixSizeEqual(this->nodes, this->dEda_nodes) ))
		return -10;
	if (!(this->nodes.nI*this->nodes.nJ > 0))
		return -11;
	if (!(this->f_dropout >= 0 && this->f_dropout <= 1))
		return -12;
	if (!(prevlayer2->nFilters == nFilters))
		return -15;

	if (bUseAKMode){

		int diff = (nI_sgrid-1)*nStride+nSizeX - prevlayer2->nI_grid;
		if (!( diff >= 0 && diff < nSizeX))
			return -13;

		diff = (nJ_sgrid-1)*nStride+nSizeX - prevlayer2->nJ_grid;
		if (!( diff >= 0 && diff < nSizeX))
			return -14;

		if (nI_sgrid != nJ_sgrid)
			return -15;

	}else{

		if (!(prevlayer2->nI_grid == nI_sgrid*2))
			return -13;
		if (!(prevlayer2->nJ_grid == nJ_sgrid*2))
			return -14;
		if (!(p_prevlayer->nHidNodes == nFilters*nI_sgrid*nJ_sgrid*4 ))
			return -21;
	}

	return 0;
};

template<typename T>
int clLayerConvS<T>::forward(clMatrix<T>& randnums, bool bLearning ){

	clLayer<T>* p_prevlayer = this->vppl[0];
	if (bUseAKMode){
		if (nPoolingType == 0){
			convLocalPool( p_prevlayer->nodes, this->nodes, nFilters, nSizeX /*_sizeX*/, 0 /*_start*/,
					nStride /*_stride*/, nI_sgrid /*_outputsX*/, AvgPooler<T>());
		}else{
			convLocalPool(p_prevlayer->nodes, this->nodes, nFilters, nSizeX /*_sizeX*/, 0 /*_start*/,
					nStride /*_stride*/, nI_sgrid /*_outputsX*/, MaxPooler<T>());
		}


	}else{ //my own version
		cr( ConvPooling( p_prevlayer->nodes, nFilters, nI_sgrid, nJ_sgrid, this->nSamples,
				nPoolingType, this->nodes ) )
	}

	cr( forward_nonlinearity( this->nodes, this->nNeuronType, this->f_dropout, randnums, bLearning) )
	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//backpropagate the partial derivatives
////////////////////////////////////////////////////////////////////////////////////////////////
// x=nonlinearity(a), updates prevlayer's dEdx ---> p_prevlayer->dEdEda_nodes
// Backprob Has a bug when the convs layer have nonlinearity!!!
template<typename T>
int clLayerConvS<T>::backward( bool bNonlin){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if (p_prevlayer->name.compare("convc") != 0 )
		return -1;

	if (bNonlin){
		//initially, dEda is dEdy, need to pass thru nonlinearity (unless the last layer of a net)
		cr( backward_nonlinearity(this->nodes, this->dEda_nodes, this->nNeuronType ) )
	}

	if (p_prevlayer->dEda_nodes.nI*p_prevlayer->dEda_nodes.nJ > 0){

		const clLayerConvC<T> * prevlayer2 = static_cast< const clLayerConvC<T>* >( p_prevlayer );

		if (bUseAKMode){

			clASSERT( prevlayer2->nI_grid == prevlayer2->nJ_grid, "prevlayer2.nI_grid == prevlayer2.nJ_grid");

			if (nPoolingType == 0){
				convLocalAvgUndo<T>(this->dEda_nodes, p_prevlayer->dEda_nodes,
						nSizeX /*_sizeX*/, 0 /*_start*/, nStride /*_stride*/, nI_sgrid /*_outputsX*/,
						prevlayer2->nI_grid, 1 /*scaleTargets*/, 1);
			}else{
				convLocalMaxUndo<T>(p_prevlayer->nodes, this->dEda_nodes, this->nodes, p_prevlayer->dEda_nodes,
						nSizeX /*_sizeX*/, 0 /*_start*/, nStride /*_stride*/, nI_sgrid /*_outputsX*/, 1 /*scaleTargets*/, 1);
			}

		}else{
			cr( ConvPoolingBackProp( p_prevlayer->nodes, this->nodes, this->dEda_nodes,
					nFilters, nI_sgrid, nJ_sgrid, this->nSamples, nPoolingType, T(1)/*gamma*/, p_prevlayer->dEda_nodes ) )
		}
	}
	return 0;
};

template class clLayerConvS<float>;
template class clLayerConvS<double>;


template<typename T>
int clLayerConvRN<T>::forward(clMatrix<T>& randnums, bool bLearning ){
	clLayer<T>* p_prevlayer = this->vppl[0];
	convResponseNorm<T>(p_prevlayer->nodes, denoms, this->nodes, nFilters, nSize, fScale, fPower);
	return 0;
}


template<typename T>
int clLayerConvRN<T>::backward(  bool bNonlin ){

	clLayer<T>* p_prevlayer = this->vppl[0];
	if (p_prevlayer->name.compare("convs") != 0 )
		return -1;

	if (p_prevlayer->dEda_nodes.nI*p_prevlayer->dEda_nodes.nJ > 0){
		temp.CopyFrom(this->nodes);
		convResponseNormUndo<T>(this->dEda_nodes, denoms, p_prevlayer->nodes, this->nodes,
				p_prevlayer->dEda_nodes, nFilters, nSize, fScale, fPower, 1.0 /*scaleTargets*/, 1);

		this->nodes.CopyFrom(temp); //since above function ruins this->nodes
	}
	return 0;
}

template <typename T>
int clLayerConvRN<T>::validate(){

	clASSERT( this->vppl.size()==1, "vppl.size()==1");
	clLayer<T>* p_prevlayer = this->vppl[0];

	if ( p_prevlayer->name.compare("convs") != 0 )
		return -1;
	if (!(p_prevlayer->nSamples == this->nSamples ))
		return -2;
	if ( !p_prevlayer->bSamplesLeadDim )
		return -3;
	if (!(this->nParams == 0))
		return -9;
	if (!(clMatrixSizeEqual(this->nodes, this->dEda_nodes) ))
		return -10;
	if (!(clMatrixSizeEqual(this->nodes, denoms) ))
		return -10;
	if (!(this->nodes.nI*this->nodes.nJ > 0))
		return -11;
	if (!(this->f_dropout >= 0 && this->f_dropout <= 1))
		return -12;

	const clLayerConvS<T> * prevlayer2 = static_cast< const clLayerConvS<T> * >( p_prevlayer );

	if (prevlayer2->nFilters != nFilters)
		return -113;
	if (prevlayer2->nI_sgrid != nI_sgrid || prevlayer2->nJ_sgrid != nJ_sgrid)
		return -114;

	return 0;

}

template class clLayerConvRN<float>;
template class clLayerConvRN<double>;



template <typename T>
int clLayerConvC<T>::assign_ptr( T* ptr, bool bDerive )
{

	if (bDerive){
		d_wts_vw.nI = nFilters;
		d_wts_vw.nJ = nVisChannels*nI_filt*nJ_filt;
		d_wts_vw.pData = ptr;
		d_bias_vw.nI = 1;
		d_bias_vw.nJ = nFilters;
		d_bias_vw.pData = ptr + d_wts_vw.nI*d_wts_vw.nJ;

		clASSERT( d_wts_vw.nI*d_wts_vw.nJ+d_bias_vw.nI*d_bias_vw.nJ == this->nParams,
				"clLayerConvC::assign_ptr nParams error!");
	}else{
		wts_vw.nI = nFilters;
		wts_vw.nJ = nVisChannels*nI_filt*nJ_filt;
		wts_vw.pData = ptr;
		bias_vw.nI = 1;
		bias_vw.nJ = nFilters;
		bias_vw.pData = ptr + wts_vw.nI*wts_vw.nJ;

		clASSERT( wts_vw.nI*wts_vw.nJ+bias_vw.nI*bias_vw.nJ == this->nParams,
				"clLayerConvC::assign_ptr nParams error!");
	}

	return 0;
}



template <typename T>
int clLayerConvC<T>::validate()
{
	clASSERT( this->vppl.size()==1, "vppl.size()==1");
	clLayer<T>* p_prevlayer = this->vppl[0];

	if (!(p_prevlayer->nHidNodes == nVisChannels*nVisI*nVisJ))
		return -1;
	if ( !p_prevlayer->bSamplesLeadDim )
		return -11;
	if (!(p_prevlayer->nSamples == this->nSamples ))
		return -2;
	if (!(wts_vw.nJ == nVisChannels*nI_filt*nJ_filt && d_wts_vw.nJ == wts_vw.nJ))
		return -3;
	if (!(wts_vw.nI == nFilters && d_wts_vw.nI == nFilters))
		return -4;
	if (!(bias_vw.nI == 1 && d_bias_vw.nI == 1))
		return -5;
	if (!(bias_vw.nJ == nFilters && d_bias_vw.nJ == nFilters))
		return -6;
	if (!(wts_vw.nI*wts_vw.nJ > 0 && d_wts_vw.nI*d_wts_vw.nJ > 0))
		return -7;
	if (!(bias_vw.nI*bias_vw.nJ > 0 && d_bias_vw.nI*d_bias_vw.nJ > 0))
		return -8;
	if (!(this->nParams == (nVisChannels*nI_filt*nJ_filt+1)*nFilters))
		return -9;
	if (!(clMatrixSizeEqual(this->nodes, this->dEda_nodes) ))
		return -10;
	if (!(this->nodes.nI*this->nodes.nJ > 0))
		return -11;
	if (!(this->f_dropout >= 0 && this->f_dropout <= 1))
		return -12;
	if (nI_filt > nVisI || nJ_filt > nVisJ)
		return -15;
	if (  (nI_grid-1)*nStride+nI_filt != nVisI+(-nPaddingStart)*2
			||(nJ_grid-1)*nStride+nJ_filt != nVisJ+(-nPaddingStart)*2 )
		return -16;
	if (nPaddingStart > 0)
		return -17;
	if (nStride < 1 || nStride > MIN(nI_filt, nJ_filt))
		return -18;

	if (p_prevlayer->name.compare("convdata") == 0){

		const clLayerConvData<T> *prevlayer2 = static_cast< const clLayerConvData<T> * >( p_prevlayer );

		if (prevlayer2->nFilters != nVisChannels)
			return -103;
		if (prevlayer2->nI_grid != nVisI || prevlayer2->nJ_grid != nVisJ)
			return -104;

	}else if (p_prevlayer->name.compare("convs") == 0){
		const clLayerConvS<T> * prevlayer2 = static_cast< const clLayerConvS<T> * >( p_prevlayer );

		if (prevlayer2->nFilters != nVisChannels)
			return -113;
		if (prevlayer2->nI_sgrid != nVisI || prevlayer2->nJ_sgrid != nVisJ)
			return -114;

	}else if (p_prevlayer->name.compare("convrn") == 0){
		const clLayerConvRN<T> * prevlayer2 = static_cast< const clLayerConvRN<T> * >( p_prevlayer );

		if (prevlayer2->nFilters != nVisChannels)
			return -113;
		if (prevlayer2->nI_sgrid != nVisI || prevlayer2->nJ_sgrid != nVisJ)
			return -114;

	}else if (p_prevlayer->name.compare("convc") == 0){

		const clLayerConvC<T> * prevlayer2 = static_cast< const clLayerConvC<T> * >( p_prevlayer );

		if (prevlayer2->nFilters != nVisChannels)
			return -113;
		if (prevlayer2->nI_grid != nVisI || prevlayer2->nJ_grid != nVisJ)
			return -114;

	}else if (p_prevlayer->name.compare("convjitter") == 0){

		const clLayerConvJitter<T> * prevlayer2 = static_cast< const clLayerConvJitter<T> * >( p_prevlayer );

		if (prevlayer2->nVisChannels != nVisChannels)
			return -113;
		if (prevlayer2->nVisI2 != nVisI || prevlayer2->nVisJ2 != nVisJ)
			return -114;

	}else if (p_prevlayer->name.compare("convxyrs") == 0){

		const clLayerConvXYRS<T> * prevlayer2 = static_cast< const clLayerConvXYRS<T> * >( p_prevlayer );

		if (prevlayer2->nVisChannels != nVisChannels)
			return -113;
		if (prevlayer2->nVisI2 != nVisI || prevlayer2->nVisJ2 != nVisJ)
			return -114;

	}else if (p_prevlayer->name.compare("elewisemul") == 0){
		//const clLayerEleWiseMul<T> * prevlayer2 = static_cast< const clLayerEleWiseMul<T> * >( p_prevlayer );
	}else{
		return -115;
	}

	return 0;
}

template <typename T>
int clLayerConvC<T>::forward( clMatrix<T>& randnums, bool bLearning ){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if ( this->nNeuronType == 5 && !this->bSamplesLeadDim)
		return -1;
	if (!p_prevlayer->bSamplesLeadDim)
		return -2;

	if ( !bCpuMode ){
#ifndef CONV_CPU_MODE
		_filterActs<T>( p_prevlayer->nodes, this->nSamples /*imgStride*/, wts_vw, this->nodes,
				nVisI, nI_grid, nJ_grid, nPaddingStart /*paddingStart*/, nStride /*moduleStride*/,
				nVisChannels, 1 /*numGroups*/,
				0.0 /*scaleTargets*/, 1.0 /*scaleOutput*/, true /*conv should be always true */);
#endif
	}else{

		cuda_clMatrixCpy( (void*)prev_nodes_cpu, p_prevlayer->nodes);
		cuda_clMatrixCpy( (void*)wts_cpu, wts_vw);


		cr( filter2d_cpu_akmaj<T>(prev_nodes_cpu, nVisChannels, nVisI, nVisJ, this->nSamples,
				wts_cpu, nI_filt, nJ_filt, nFilters,
				nodes_cpu, nI_grid, nJ_grid, VALID, false, true /*bZeroDest*/ ) )

		cuda_clMatrixCpy(this->nodes, (void*) nodes_cpu );
	}


	clmat tempview = this->nodes.ReshapeView(nI_grid*nJ_grid*this->nSamples, nFilters);
	cr(Bsxfun(tempview, fctrPlus<T>(), bias_vw, tempview))

	cr( forward_nonlinearity( this->nodes, this->nNeuronType, this->f_dropout, randnums, bLearning) )
	return 0;
}


template <typename T>
int clLayerConvC<T>::backward(bool bNonlin){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if (!p_prevlayer->bSamplesLeadDim)
		return -1;

	if (bNonlin){
		//initially, dEda is dEdy, need to pass thru nonlinearity (unless the last layer of a net)
		cr( backward_nonlinearity(this->nodes, this->dEda_nodes, this->nNeuronType ) )
	}

	if (p_prevlayer->dEda_nodes.nI*p_prevlayer->dEda_nodes.nJ > 0){

		if ( !bCpuMode ){
#ifndef CONV_CPU_MODE

			_imgActs<T>( this->dEda_nodes, wts_vw, p_prevlayer->dEda_nodes, nVisI, nVisJ, nI_grid,
					nPaddingStart /*paddingStart*/, nStride /*moduleStride*/, nVisChannels, 1 /*numGroups*/,
					1.0 /*scaleTargets*/, 1.0 /*scaleOutput*/, true /*conv should be always true */);
#endif
		}else{
			//we need to permute the wts first
			clmat wtsT_vw = T_wts.ReshapeMatrixView( wts_vw.nI, wts_vw.nJ);
			wtsT_vw.CopyFrom(wts_vw);

			clmat wtsTperm_vw = T_wts_perm.ReshapeMatrixView( wts_vw.nI, wts_vw.nJ);

			std::vector<int> perminds(4);
			perminds[0] = 3;
			perminds[1] = 1;
			perminds[2] = 2;
			perminds[3] = 0;

			cr(TensorPermute(T_wts, perminds, dev_perm_buf, T_wts_perm) )

			//copy stuff to the cpu
			cuda_clMatrixCpy( (void*)wts_cpu, wtsTperm_vw);
			cuda_clMatrixCpy( (void*)dEda_nodes_cpu, this->dEda_nodes);
			cuda_clMatrixCpy( (void*)prev_dEda_nodes_cpu, p_prevlayer->dEda_nodes);  //so we can accumulate the partial derivatives

			cr( filter2d_cpu_akmaj<T>(dEda_nodes_cpu, nFilters, nI_grid, nJ_grid, this->nSamples,
					wts_cpu, nI_filt, nJ_filt, nVisChannels,
					prev_dEda_nodes_cpu, nVisI, nVisJ, FULL, true, false/*bZeroDest*/) )

			cuda_clMatrixCpy( p_prevlayer->dEda_nodes, (void*) prev_dEda_nodes_cpu );
		}
	}

	return 0;
}


template <typename T>
int clLayerConvC<T>::dEdw(){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if (!p_prevlayer->bSamplesLeadDim)
		return -1;
	if (nI_filt != nJ_filt)
		return -2;

	if ( !bCpuMode ){
#ifndef CONV_CPU_MODE

		if (nPartialSum > 0){

			// wts_partialsum_temp.SetVal(0); //not needed

			_weightActs<T>( p_prevlayer->nodes, this->nSamples, this->dEda_nodes, wts_partialsum_temp, nVisI,
					nI_grid, nJ_grid, nI_filt, nPaddingStart /* paddingStart*/, nStride/*moduleStride*/,
					nVisChannels, 1 /*numGroups*/, nPartialSum /*partialSum*/, 0.0 /*scaleTargets*/, 1.0 /*scaleOutput*/);

			clmat temp1 = wts_partialsum_temp.ReshapeView(nFilters*nJ_filt*nI_filt*nVisChannels,
					nI_grid*nJ_grid/nPartialSum );
			//				cuTic();
			cr( SumInplace( temp1, 2) )
			clmat temp2 = d_wts_vw.ReshapeView(nFilters*nJ_filt*nI_filt*nVisChannels,1);
			temp2.CopyFrom( temp1.ColView(0) );
			//				float fms = cuToc();
			//
			//				char notestr[100];
			//				sprintf( notestr, " conv:dEdw: %.3f", fms);
			//				clPrintf(notestr);

		}
		else{
			_weightActs<T>( p_prevlayer->nodes, this->nSamples, this->dEda_nodes, d_wts_vw, nVisI,
					nI_grid, nJ_grid, nI_filt, nPaddingStart /* paddingStart*/, nStride/*moduleStride*/,
					nVisChannels, 1 /*numGroups*/, nPartialSum /*partialSum*/, 0.0 /*scaleTargets*/, 1.0 /*scaleOutput*/);

		}
#endif

	}else{

		std::vector<int> dimvec(4);
		dimvec[0] = this->nSamples;
		dimvec[1] = nJ_grid;
		dimvec[2] = nI_grid;
		dimvec[3] = nFilters;

		clTensor<T> T_dEda_nodes = clTensor<T>(dimvec, this->dEda_nodes);

		dimvec[0] = this->nSamples;
		dimvec[1] = nVisJ;
		dimvec[2] = nVisI;
		dimvec[3] = nVisChannels;
		clTensor<T> T_prev_nodes = clTensor<T>(dimvec, p_prevlayer->nodes);

		std::vector<int> perminds(4);
		perminds[0] = 3;
		perminds[1] = 1;
		perminds[2] = 2;
		perminds[3] = 0;

		cr(TensorPermute(T_dEda_nodes, perminds, dev_perm_buf, T_nodes_perm ) )
		cr(TensorPermute(T_prev_nodes, perminds, dev_perm_buf, T_prev_nodes_perm ) )

		//copy permuted version to cpu
		cuda_clMatrixCpy( (void*) dEda_nodes_cpu, T_nodes_perm.VecView());
		cuda_clMatrixCpy( (void*) prev_nodes_cpu, T_prev_nodes_perm.VecView());

		cr( filter2d_cpu_akmaj<T>(prev_nodes_cpu, this->nSamples, nVisI, nVisJ, nVisChannels,
				dEda_nodes_cpu, nI_grid, nJ_grid, nFilters,
				wts_cpu, nI_filt, nJ_filt, VALID, false ) )
		//copy back to gpu
		clMatrix<T> T_wts_perm_view = T_wts_perm.VecView();
		cuda_clMatrixCpy( T_wts_perm_view, (void*) wts_cpu);
		cr(TensorPermute( T_wts_perm, perminds, dev_perm_buf, T_wts ) )

		d_wts_vw.CopyFrom(T_wts.ReshapeMatrixView( d_wts_vw.nI, d_wts_vw.nJ));

	}


	if ( f_wtcost > 0){
		cr( EleWisefun( -f_wtcost, wts_vw, fctrAlphaPlusBeta<T>(), T(1), d_wts_vw, d_wts_vw) )
	}

	if (this->bSamplesLeadDim){
		clMatrix<T> dEda_nodes_reshaped = this->dEda_nodes.ReshapeView(nJ_grid*nI_grid*this->nSamples, nFilters);
		cr( SumInplace( dEda_nodes_reshaped, 1) )

		cr( RowEleWisefun(T(1), dEda_nodes_reshaped, 0,
				fctrAlphaPlusBeta<T>(), T(0), d_bias_vw, 0, d_bias_vw, 0) )

	}else{
		clASSERT(false, "this->bSamplesLeadDim == false");
	}
	return 0;
}

template class clLayerConvC<float>;
template class clLayerConvC<double>;


template <typename T>
int clLayerDeConvC<T>::assign_ptr( T* ptr, bool bDerive ){

	if (bDerive){
		d_wts_vw.nI = nFilters;
		d_wts_vw.nJ = nVisChannels*nJ_filt*nI_filt;
		d_wts_vw.pData = ptr;
		d_bias_vw.nI = 1;
		d_bias_vw.nJ = nVisChannels;
		d_bias_vw.pData = ptr + d_wts_vw.nI*d_wts_vw.nJ;

		clASSERT( d_wts_vw.nI*d_wts_vw.nJ+d_bias_vw.nI*d_bias_vw.nJ == this->nParams,
				"clLayerConvC::assign_ptr nParams error!");
	}else{
		wts_vw.nI = nFilters;
		wts_vw.nJ = nVisChannels*nJ_filt*nI_filt;
		wts_vw.pData = ptr;
		bias_vw.nI = 1;
		bias_vw.nJ = nVisChannels;
		bias_vw.pData = ptr + wts_vw.nI*wts_vw.nJ;

		clASSERT( wts_vw.nI*wts_vw.nJ+bias_vw.nI*bias_vw.nJ == this->nParams,
				"clLayerConvC::assign_ptr nParams error!");
	}

	return 0;
}


template <typename T>
int clLayerDeConvC<T>::validate(){


	clASSERT( this->vppl.size()==1, "vppl.size()==1");
	clLayer<T>* p_prevlayer = this->vppl[0];

	if (!(p_prevlayer->nHidNodes == nFilters*nJ_grid*nI_grid))
		return -1;
	if ( !p_prevlayer->bSamplesLeadDim )
		return -11;
	if (!(p_prevlayer->nSamples == this->nSamples ))
		return -2;
	if (!(wts_vw.nJ == nVisChannels*nI_filt*nJ_filt && d_wts_vw.nJ == wts_vw.nJ))
		return -3;
	if (!(wts_vw.nI == nFilters && d_wts_vw.nI == nFilters))
		return -4;
	if (!(bias_vw.nI == 1 && d_bias_vw.nI == 1))
		return -5;
	if (!(bias_vw.nJ == nVisChannels && d_bias_vw.nJ == nVisChannels))
		return -6;
	if (!(wts_vw.nI*wts_vw.nJ > 0 && d_wts_vw.nI*d_wts_vw.nJ > 0))
		return -7;
	if (!(bias_vw.nI*bias_vw.nJ > 0 && d_bias_vw.nI*d_bias_vw.nJ > 0))
		return -8;
	if (!(this->nParams == (nFilters*nI_filt*nJ_filt+1)*nVisChannels))
		return -9;
	if (!(clMatrixSizeEqual(this->nodes, this->dEda_nodes) ))
		return -10;
	if (!(this->nodes.nI*this->nodes.nJ > 0))
		return -11;
	if (!(this->f_dropout >= 0 && this->f_dropout <= 1))
		return -12;
	if (nI_filt > nVisI || nJ_filt > nVisJ)
		return -15;
	if (  (nI_grid-1)*nStride+nI_filt != nVisI+(-nPaddingStart)*2
			||(nJ_grid-1)*nStride+nJ_filt != nVisJ+(-nPaddingStart)*2 )
		return -16;
	if (nPaddingStart > 0)
		return -17;
	if (nStride < 1 || nStride > MIN(nI_filt, nJ_filt))
		return -18;

	if (p_prevlayer->name.compare("fc") == 0){

		const clLayerFC<T> *prevlayer2 = static_cast< const clLayerFC<T> * >( p_prevlayer );

		if (prevlayer2->nHidNodes != nFilters*nJ_grid*nI_grid)
			return -103;
		if (prevlayer2->nSamples != this->nSamples)
			return -104;

	}else if (p_prevlayer->name.compare("convs") == 0){
		const clLayerConvS<T> * prevlayer2 = static_cast< const clLayerConvS<T> * >( p_prevlayer );

		if (prevlayer2->nFilters != nFilters)
			return -113;
		if (prevlayer2->nI_sgrid != nI_grid || prevlayer2->nJ_sgrid != nJ_grid)
			return -114;

	}else if (p_prevlayer->name.compare("convrn") == 0){
		const clLayerConvRN<T> * prevlayer2 = static_cast< const clLayerConvRN<T> * >( p_prevlayer );

		if (prevlayer2->nFilters != nFilters)
			return -113;
		if (prevlayer2->nI_sgrid != nI_grid || prevlayer2->nJ_sgrid != nJ_grid)
			return -114;

	}else if (p_prevlayer->name.compare("convc") == 0){

		const clLayerConvC<T> * prevlayer2 = static_cast< const clLayerConvC<T> * >( p_prevlayer );

		if (prevlayer2->nFilters != nFilters)
			return -113;
		if (prevlayer2->nI_grid != nI_grid || prevlayer2->nJ_grid != nJ_grid)
			return -114;

	}else if (p_prevlayer->name.compare("deconvc") == 0){

		const clLayerDeConvC<T> * prevlayer2 = static_cast< const clLayerDeConvC<T> * >( p_prevlayer );

		if (prevlayer2->nVisChannels != nFilters)
			return -113;
		if (prevlayer2->nVisI != nI_grid || prevlayer2->nVisJ != nJ_grid)
			return -114;

	}else if (p_prevlayer->name.compare("elewisemul") == 0){
		//const clLayerEleWiseMul<T> * prevlayer2 = static_cast< const clLayerEleWiseMul<T> * >( p_prevlayer );
	}else{
		return -115;
	}

	return 0;
}


template <typename T>
int clLayerDeConvC<T>::forward( clMatrix<T>& randnums, bool bLearning ){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if (!p_prevlayer->bSamplesLeadDim)
		return -1;

	if ( !bCpuMode ){
#ifndef CONV_CPU_MODE
		_imgActs<T>(  p_prevlayer->nodes, wts_vw, this->nodes, nVisI, nVisJ, nI_grid,
				nPaddingStart /*paddingStart*/, nStride /*moduleStride*/, nVisChannels, 1 /*numGroups*/,
				0.0 /*scaleTargets*/, 1.0 /*scaleOutput*/, true /*conv should be always true */);
#endif
	}else{
		//we need to permute the wts first
		clmat wtsT_vw = T_wts.ReshapeMatrixView( wts_vw.nI, wts_vw.nJ);
		wtsT_vw.CopyFrom(wts_vw);

		clmat wtsTperm_vw = T_wts_perm.ReshapeMatrixView( wts_vw.nI, wts_vw.nJ);

		std::vector<int> perminds(4);
		perminds[0] = 3;
		perminds[1] = 1;
		perminds[2] = 2;
		perminds[3] = 0;

		cr(TensorPermute(T_wts, perminds, dev_perm_buf, T_wts_perm) )

		//copy stuff to the cpu
		cuda_clMatrixCpy( (void*)wts_cpu, wtsTperm_vw);
		cuda_clMatrixCpy( (void*)nodes_cpu, this->nodes);
		cuda_clMatrixCpy( (void*)prev_nodes_cpu, p_prevlayer->nodes);  //so we can accumulate the partial derivatives

		cr( filter2d_cpu_akmaj<T>(prev_nodes_cpu, nFilters, nI_grid, nJ_grid, this->nSamples,
				wts_cpu, nI_filt, nJ_filt, nVisChannels,
				nodes_cpu, nVisI, nVisJ, FULL, true, true/*bZeroDest*/) )

		cuda_clMatrixCpy( this->nodes, (void*) nodes_cpu );
	}

	clmat tempview = this->nodes.ReshapeView(nVisJ*nVisI*this->nSamples, nVisChannels);
	cr(Bsxfun(tempview, fctrPlus<T>(), bias_vw, tempview))

	cr( forward_nonlinearity( this->nodes, this->nNeuronType, this->f_dropout, randnums, bLearning) )
	return 0;
}


template <typename T>
int clLayerDeConvC<T>::backward(bool bNonlin = true){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if (!p_prevlayer->bSamplesLeadDim)
		return -1;

	if (bNonlin){
		//initially, dEda is dEdy, need to pass thru nonlinearity (unless the last layer of a net)
		cr( backward_nonlinearity(this->nodes, this->dEda_nodes, this->nNeuronType ) )
	}

	if (p_prevlayer->dEda_nodes.nI*p_prevlayer->dEda_nodes.nJ > 0){

		if ( !bCpuMode ){
#ifndef CONV_CPU_MODE

			_filterActs<T>(this->dEda_nodes, this->nSamples /*imgStride*/, wts_vw, p_prevlayer->dEda_nodes,
					nVisI, nI_grid, nJ_grid, nPaddingStart /*paddingStart*/, nStride /*moduleStride*/,
					nVisChannels, 1 /*numGroups*/,
					1.0 /*scaleTargets*/, 1.0 /*scaleOutput*/, true /*conv should be always true */);
#endif
		}else{

			cuda_clMatrixCpy( (void*)dEda_nodes_cpu, p_prevlayer->dEda_nodes);
			cuda_clMatrixCpy( (void*)wts_cpu, wts_vw);

			cr( filter2d_cpu_akmaj<T>(dEda_nodes_cpu, nVisChannels, nVisI, nVisJ, this->nSamples,
					wts_cpu, nI_filt, nJ_filt, nFilters,
					prev_dEda_nodes_cpu, nI_grid, nJ_grid, VALID, false, false /*bZeroDest*/ ) )

			cuda_clMatrixCpy(p_prevlayer->dEda_nodes, (void*) prev_dEda_nodes_cpu );
		}
	}


	return 0;
}


template <typename T>
int clLayerDeConvC<T>::dEdw(){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if (!p_prevlayer->bSamplesLeadDim)
		return -1;
	if (nI_filt != nJ_filt)
		return -2;

	if ( !bCpuMode ){
#ifndef CONV_CPU_MODE

		if (nPartialSum > 0){

			// wts_partialsum_temp.SetVal(0); //not needed

			_weightActs<T>(  this->dEda_nodes, this->nSamples, p_prevlayer->nodes, wts_partialsum_temp, nVisI,
					nI_grid, nJ_grid, nI_filt, nPaddingStart /* paddingStart*/, nStride/*moduleStride*/,
					nVisChannels, 1 /*numGroups*/, nPartialSum /*partialSum*/, 0.0 /*scaleTargets*/, 1.0 /*scaleOutput*/);

			clmat temp1 = wts_partialsum_temp.ReshapeView(nFilters*nJ_filt*nI_filt*nVisChannels,
					nI_grid*nJ_grid/nPartialSum );
			cr( SumInplace( temp1, 2) )

			clmat temp2 = d_wts_vw.ReshapeView(nFilters*nJ_filt*nI_filt*nVisChannels,1);
			temp2.CopyFrom( temp1.ColView(0) );
		}
		else if (nPartialSum==0){
			_weightActs<T>( this->dEda_nodes, this->nSamples, p_prevlayer->nodes, d_wts_vw, nVisI,
					nI_grid, nJ_grid, nI_filt, nPaddingStart /* paddingStart*/, nStride/*moduleStride*/,
					nVisChannels, 1 /*numGroups*/, nPartialSum /*partialSum*/, 0.0 /*scaleTargets*/, 1.0 /*scaleOutput*/);
		}
		else{
			return -3;
		}
#endif

	}else{

		std::vector<int> dimvec(4);
		dimvec[0] = this->nSamples;
		dimvec[1] = nVisJ;
		dimvec[2] = nVisI;
		dimvec[3] = nVisChannels;

		clTensor<T> T_dEda_nodes = clTensor<T>(dimvec, this->dEda_nodes);

		dimvec[0] = this->nSamples;
		dimvec[1] = nJ_grid;
		dimvec[2] = nI_grid;
		dimvec[3] = nFilters;
		clTensor<T> T_prev_nodes = clTensor<T>(dimvec, p_prevlayer->nodes);

		std::vector<int> perminds(4);
		perminds[0] = 3;
		perminds[1] = 1;
		perminds[2] = 2;
		perminds[3] = 0;

		cr(TensorPermute(T_dEda_nodes, perminds, dev_perm_buf, T_nodes_perm ) )
		cr(TensorPermute(T_prev_nodes, perminds, dev_perm_buf, T_prev_nodes_perm ) )

		//copy permuted version to cpu
		cuda_clMatrixCpy( (void*) dEda_nodes_cpu, T_nodes_perm.VecView());
		cuda_clMatrixCpy( (void*) prev_nodes_cpu, T_prev_nodes_perm.VecView());

		cr( filter2d_cpu_akmaj<T>(dEda_nodes_cpu, this->nSamples, nVisI, nVisJ, nVisChannels,
				prev_nodes_cpu, nI_grid, nJ_grid, nFilters,
				wts_cpu, nI_filt, nJ_filt, VALID, false ) )
		//copy back to gpu
		clMatrix<T> T_wts_perm_view = T_wts_perm.VecView();
		cuda_clMatrixCpy( T_wts_perm_view, (void*) wts_cpu);
		cr(TensorPermute( T_wts_perm, perminds, dev_perm_buf, T_wts ) )

		d_wts_vw.CopyFrom(T_wts.ReshapeMatrixView( d_wts_vw.nI, d_wts_vw.nJ));

	}


	if ( f_wtcost > 0){
		cr( EleWisefun( -f_wtcost, wts_vw, fctrAlphaPlusBeta<T>(), T(1), d_wts_vw, d_wts_vw) )
	}

	if (this->bSamplesLeadDim){
		clMatrix<T> dEda_nodes_reshaped = this->dEda_nodes.ReshapeView(nVisJ*nVisI*this->nSamples, nVisChannels);
		cr( SumInplace( dEda_nodes_reshaped, 1) )

		cr( RowEleWisefun(T(1), dEda_nodes_reshaped, 0,
				fctrAlphaPlusBeta<T>(), T(0), d_bias_vw, 0, d_bias_vw, 0) )

	}else{
		clASSERT(false, "this->bSamplesLeadDim == false");
	}
	return 0;
}

template class clLayerDeConvC<float>;
template class clLayerDeConvC<double>;





