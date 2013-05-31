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

#include "cu_dnn_ops.h"

#define IN
#define OUT


//Host code

/***********************************************************************************************************
 * @brief:   Samples a set of nodes on gpu (using new functor elewise functions)
 * 			 this should replace nodes_sample_gpu
 *
 * @param[in]:	nodes: nSamples by ndims, clMat object
 * 				randums: a clMat big enough to contain random numbers
 * 				std_vec: vector of standard deviation, 1 by ndimensions
 *
 * @param[out]: nodes
 *
 * @topology:   n/a
 * @note:		unless type is ACTI_PROB, TANH, randnums.pData can't be NULL
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
int nodes_sample( clMatrix<T>& nodes, clMatrix<T>& randnums, NodesSampleEnum type,clMatrix<T>& std_vec)
{
	const int nI = nodes.nI;
	const int nJ = nodes.nJ;

	T mean = T(0.0);
	T stddev = T(1.0);

	clMatrix<T> shell;

	switch(type){
	case BIN_SAMPLE:

		if (randnums.nI*randnums.nJ < nodes.nI*nodes.nJ)
			return -1;

		shell.pData = randnums.pData;
		shell.nI = nI;
		shell.nJ = nJ;
		cr( cuda_clMatSetRand( shell ) ) //randnums dimension may be bigger than nodes
		cr(EleWisefun( nodes, fctrSigmBinSample<T>(), shell, nodes) )
		break;
	case ACTI_PROB:
		cr(EleWisefun( fctrSigm<T>(), nodes, nodes) )
		break;
	case ACTI_SAMPLE:

		if (randnums.nI*randnums.nJ < nodes.nI*nodes.nJ)
			return -1;

		shell.pData = randnums.pData;
		shell.nI = nI;
		shell.nJ = nJ;

		cr( cuda_clMatSetRand( shell ) ) //randnums dimension may be bigger than nodes
		cr(EleWisefun( nodes, fctrSampleBernoulli<T>(), shell, nodes) )
		break;
	case TANH:
		cr(EleWisefun( fctrTanh<T>(), nodes, nodes) )
		break;
	case GAUSS_SAMPLE:

		if (nodes.nJ != std_vec.nJ || 1 != std_vec.nI || randnums.nI*randnums.nJ < nodes.nI*nodes.nJ)
			return -1;

		shell.pData = randnums.pData;
		shell.nI = nI;
		shell.nJ = nJ;

		cr( cuda_clMatSetRandn( shell, mean, stddev ) )
		cr( Bsxfun( shell, fctrMul<T>(), std_vec, shell ) )
		cr( EleWisefun( nodes, fctrPlus<T>(), shell, nodes) )
		break;
	default:
		return -2;
	}
	return 0;
}

template int
nodes_sample<float>( clMatrix<float>& nodes, clMatrix<float>& randnums, NodesSampleEnum type,clMatrix<float>& std_vec);
template int
nodes_sample<double>( clMatrix<double>& nodes, clMatrix<double>& randnums, NodesSampleEnum type,clMatrix<double>& std_vec);






/***********************************************************************************************************
 * @brief: performs softmax function by selecting one value from each row
 * 			this method subtracts off so to avoid some numerical problems
 * @param[in]:  pMat is a dim1 x dim2 matrix of values to take softmax, dim2 must be less than or eq 512
 * @param[out]: pOut is the output probability matrix where each element within a row must be
 * 				replaced with exp(.)/sum(exp(row));
 * @topology:	blockDim.x must be exponent of 2, grid should only be 1 in (x or j) direction
 * @note:		remember to allocate shared memory
 * 				11111111111111111111111111111 (grid id)
 * 				11111111111111111111111111111
 * 				22222222222222222222222222222
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
//single precision version
template <typename T>
__global__ void _softmax_safe(const T* pMat, int dim1, int dim2, T* pOut)
{
	extern __shared__ T sfdata[];
	T * sdata2 = &(sfdata[blockDim.x*blockDim.y]); //stores the correct exp values

	int j = threadIdx.x;
	int i = blockIdx.y*blockDim.y+threadIdx.y;

	///////////////////////////////////////
	//max for each row

	if( i < dim1 && j < dim2) {
		sfdata[j*blockDim.y+threadIdx.y] = pMat[j*dim1+i];  //copy to shared memory
	}
	__syncthreads();

	//need to do max( sfdata,[], 2)
	unsigned int s = blockDim.x/2;
	unsigned int loc1 = j*blockDim.y+threadIdx.y;
	unsigned int loc2 = (j+s)*blockDim.y+threadIdx.y;

	///////////////////////////////////////
	//max for each row
	if (i < dim1 && j<s && j+s < dim2 ){
		sfdata[loc1] = fmaxf( sfdata[loc2], sfdata[loc1] );
	}
	//if j > s, don't need to sum anything
	__syncthreads();

	for (s = blockDim.x/4; s>0; s >>= 1)
	{
		loc2 = (j+s)*blockDim.y+threadIdx.y; //loc2 changes with s
		if (i < dim1 && j < s && j+s < dim2){
			sfdata[loc1] = fmaxf( sfdata[loc2], sfdata[loc1]);
		}
		__syncthreads();
	}

	//after finding the max, we get alpha
	T alpha = 0;
	if (i < dim1){
		alpha = sfdata[threadIdx.y] - MAX_FLOAT_EXP / 2.0f; //this contains "alpha"
	}
	__syncthreads();

	// calculate exp( of pMat(i,j) - alpha )
	//save in both shared memory sfdata and sdata2
	if( i < dim1 && j < dim2) {
		sfdata[j*blockDim.y+threadIdx.y] = expf( pMat[j*dim1+i] - alpha ); //prevent overflow
		sdata2[j*blockDim.y+threadIdx.y] = sfdata[j*blockDim.y+threadIdx.y];
	}
	__syncthreads();

	//need to do sum exp of all data
	s = blockDim.x/2;
	loc1 = j*blockDim.y+threadIdx.y;
	loc2 = (j+s)*blockDim.y+threadIdx.y;

	//max for each row
	if (i < dim1 && j<s && j+s < dim2 ){
		sfdata[loc1] = __fadd_rn( sfdata[loc2], sfdata[loc1] );
	}
	//if j > s, don't need to sum anything
	__syncthreads();

	for (s = blockDim.x/4; s>0; s >>= 1)
	{
		loc2 = (j+s)*blockDim.y+threadIdx.y; //loc2 changes with s
		if (i < dim1 && j < s && j+s < dim2){
			sfdata[loc1] = __fadd_rn( sfdata[loc2], sfdata[loc1] );
		}
		__syncthreads();
	}

	if( i < dim1 && j < dim2) {
		 pOut[j*dim1+i] = sdata2[loc1]/sfdata[threadIdx.y];
		 //threadIdx.y is (threadIdx.y,0), which is the sum
	}
}


//double version
template <>
__global__ void _softmax_safe(const double* pMat, int dim1, int dim2, double* pOut)
{
	extern __shared__ double sfdata_dbl[];
	double * sdata2 = &(sfdata_dbl[blockDim.x*blockDim.y]); //stores the correct exp values

	int j = threadIdx.x;
	int i = blockIdx.y*blockDim.y+threadIdx.y;

	///////////////////////////////////////
	//max for each row

	if( i < dim1 && j < dim2) {
		sfdata_dbl[j*blockDim.y+threadIdx.y] = pMat[j*dim1+i];  //copy to shared memory
	}
	__syncthreads();

	//need to do max( sfdata_dbl,[], 2)
	unsigned int s = blockDim.x/2;
	unsigned int loc1 = j*blockDim.y+threadIdx.y;
	unsigned int loc2 = (j+s)*blockDim.y+threadIdx.y;

	///////////////////////////////////////
	//max for each row
	if (i < dim1 && j<s && j+s < dim2 ){
		sfdata_dbl[loc1] = fmax( sfdata_dbl[loc2], sfdata_dbl[loc1] );
	}
	//if j > s, don't need to sum anything
	__syncthreads();

	for (s = blockDim.x/4; s>0; s >>= 1)
	{
		loc2 = (j+s)*blockDim.y+threadIdx.y; //loc2 changes with s
		if (i < dim1 && j < s && j+s < dim2){
			sfdata_dbl[loc1] = fmax( sfdata_dbl[loc2], sfdata_dbl[loc1]);
		}
		__syncthreads();
	}

	//after finding the max, we get alpha
	double alpha = 0;
	if (i < dim1){
		alpha = sfdata_dbl[threadIdx.y] - MAX_DOUBLE_EXP / 2.0; //this contains "alpha"
	}
	__syncthreads();

	// calculate exp( of pMat(i,j) - alpha )
	//save in both shared memory sfdata and sdata2
	if( i < dim1 && j < dim2) {
		sfdata_dbl[j*blockDim.y+threadIdx.y] = exp( pMat[j*dim1+i] - alpha ); //prevent overflow
		sdata2[j*blockDim.y+threadIdx.y] = sfdata_dbl[j*blockDim.y+threadIdx.y];
	}
	__syncthreads();

	//need to do sum exp of all data
	s = blockDim.x/2;
	loc1 = j*blockDim.y+threadIdx.y;
	loc2 = (j+s)*blockDim.y+threadIdx.y;

	//max for each row
	if (i < dim1 && j<s && j+s < dim2 ){
		sfdata_dbl[loc1] = ( sfdata_dbl[loc2]+ sfdata_dbl[loc1] );
	}
	//if j > s, don't need to sum anything
	__syncthreads();

	for (s = blockDim.x/4; s>0; s >>= 1)
	{
		loc2 = (j+s)*blockDim.y+threadIdx.y; //loc2 changes with s
		if (i < dim1 && j < s && j+s < dim2){
			sfdata_dbl[loc1] = ( sfdata_dbl[loc2] + sfdata_dbl[loc1] );
		}
		__syncthreads();
	}

	if( i < dim1 && j < dim2) {
		 pOut[j*dim1+i] = sdata2[loc1]/sfdata_dbl[threadIdx.y];
		 //threadIdx.y is (threadIdx.y,0), which is the sum
	}
}



//Tested!!, this shouldn't have any numerical problems and should always be used
//compared to Softmax
//note that mat and outmat can be the same pointer, in this case it will be an inplace transformation
template <typename T>
int SoftmaxProbSafe(IN const clMatrix<T> & mat, OUT clMatrix<T> & outmat){

	//find the largest number that is an exponent of 2
	unsigned int u = mat.nJ;
	if ((u & (u - 1)) != 0){	//if u is not an exponent of 2 already
		unsigned char counter = 0;
		while( u != 0 ){
			u = u >> 1;
			counter++;
		}
		u = 1 << counter;
	}

	if (u > MAX_THREADS || mat.nI != outmat.nI || mat.nJ != outmat.nJ)
		return -1;

	dim3 dim_block_softmax(u, MAX_THREADS/u);
	dim3 dim_grid_softmax(1, (mat.nI+dim_block_softmax.y-1)/dim_block_softmax.y );
	unsigned int nsharedbytes = dim_block_softmax.x*dim_block_softmax.y*sizeof(T)*2;
	_softmax_safe<<< dim_grid_softmax, dim_block_softmax, nsharedbytes>>>(mat.pData, mat.nI, mat.nJ, outmat.pData);

	return 0;
}

template int SoftmaxProbSafe<float>(IN const clMatrix<float> & mat, OUT clMatrix<float> & outmat);
template int SoftmaxProbSafe<double>(IN const clMatrix<double> & mat, OUT clMatrix<double> & outmat);


/***********************************************************************************************************
 * @brief:   gpu version of updating the label units ( softmax multinomial group)
 * 			 we over write within the matrix
 * @param[in]:	clMat v - nSamples by nDim
 * 				n_label_ind_zerobased - start of the label units
 * 				The chunk of label should be from n_label_ind_zerobased to v.nJ-1 inclusive *
 * 				float * pRandNums
 * 				clMat rand_nums  for random numbers
 * 				res_mat is nSamples by nLabels to take in samples
 *
 * @topology:
 * @note:
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
//template <typename T>
//int softmax_sample( clMatrix<T>& v, int lab_st_zero_based, clMatrix<T>& resmat, clMatrix<T>& randnums)
//{
//
//	if (lab_st_zero_based >= v.nJ || v.nI != resmat.nI || v.nJ-lab_st_zero_based != resmat.nJ)
//		return -1;
//
//	clMatrix<T> v_label; //sub matrix of v
//	v_label.nI = v.nI;
//	v_label.nJ = v.nJ-lab_st_zero_based;
//	v_label.pData = v.pData + lab_st_zero_based*v.nI;
//
//	cr( SoftmaxProbSafe( v_label, v_label )) //inplace softmax calculations
//	cr( mnrnd_gpu( v_label, 1, randnums, resmat) )
//
//	cuda_clMatrixCpy( v_label, resmat); //copy the result back to the submatrix
//	return 0;
//}
//
//template int softmax_sample<float>( clMatrix<float>& v, int lab_st_zero_based, clMatrix<float>& resmat, clMatrix<float>& randnums);
//template int softmax_sample<double>( clMatrix<double>& v, int lab_st_zero_based, clMatrix<double>& resmat, clMatrix<double>& randnums);
//

/***********************************************************************************************************
 * @brief:  calculates the unnormalized probability of a RBM
 * @param[in]:	devpX - this should be v*vh_W+h_biases, but we provide as input since it may be calculated already
 * 				devpX is nI x nJ
 * 				devp_v - visible activations
 * 				devp_v_biases - the visible layer biases
 * @param[out]: pOut - the out put vector to put the result in
 * 				nOutInd - the index in pOut to put the result in
 * @topology:   1D or 2D thread block and 1D or 2D grid
 * @note:
 * @change:
 * @tested:
 * @to_do:		look at a smart way of delegating the block and grid dimensions!!!
 * 				fix sum_inplace so it works correctly, not hack
 ***********************************************************************************************************
 */
/*
void unnorm_pr(IN float * devpX, int nI, int nJ, IN float* devp_v, IN float * devp_v_biases, int nVisNodes, float* pOut, int nOutInd)
{
	float temp[1];

	dim3 block_dim(256, 2);
	dim3 grid_dim( (nJ+block_dim.x-1)/block_dim.x,
			       (nI+block_dim.y-1)/block_dim.y);

	log_1plus_exp<<< grid_dim, block_dim >>>(devpX, nI, nJ);

	int err = _sum_inplace(devpX, nJ, 1, 1); //since devpX is 1 dimension for now, we "transpose it"
	if (err != 0){
		//printf("\n unnorm_pr error!!");
	}

	temp[0] = cublasSdot (nVisNodes, devp_v, 1, devp_v_biases, 1);
	cutilSafeCall(cudaMemcpy( pOut+nOutInd, temp , sizeof(float)*1, cudaMemcpyHostToDevice));
	cublasSaxpy(1, 1.0f, devpX, 1, pOut+nOutInd, 1);
}


//this is same as above, except we already know vvb;
void unnorm_pr(IN float * devpX, int nI, int nJ, IN float vvb, float* pOut, int nOutInd)
{
	float temp[1];

	dim3 block_dim(256, 2);
	dim3 grid_dim( (nJ+block_dim.x-1)/block_dim.x,
			       (nI+block_dim.y-1)/block_dim.y);

	log_1plus_exp<<< grid_dim, block_dim >>>(devpX, nI, nJ);

	int err = _sum_inplace(devpX, nJ, 1, 1); //since devpX is 1 dimension for now, we "transpose it"
	if (err != 0){
		//printf("\n unnorm_pr error!!");
	}

	temp[0] = vvb;
	cutilSafeCall(cudaMemcpy( pOut+nOutInd, temp , sizeof(float)*1, cudaMemcpyHostToDevice));
	cublasSaxpy(1, 1.0f, devpX, 1, pOut+nOutInd, 1);
}
*/



/***********************************************************************************************************
 * @brief: update_q updates the q or pr hid active using exponential decay - q = priorweight*qprev+(1-priorweight)*qcurrent
 * 		   qcurrent = mean( hid_probs)
 * @param[in]: pSumTemp, contains in its zeroth row the partial sum
 * 			   nI and nJ are the dimensions of pSumTemp
 * 			   assuming nI is the number of samples
 * @param[out]:
 * @note:      assume 1D layout
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
__global__ void update_q(IN const float * pSumTemp, int nI, int nJ, IN OUT float* pQ, float priorweight)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if (j < nJ){
		pQ[j] = (1-priorweight)/float(nI)*pSumTemp[j*nI]+priorweight*pQ[j];
	}
}


/***********************************************************************************************************
 * @brief: The transposed version of update_q function
 * 			pSumTemp is d by nsamples, and its zero-th column contains the sums
 * 			update_q updates the q or pr hid active using exponential decay - q = priorweight*qprev+(1-priorweight)*qcurrent
 * 		   qcurrent = mean( hid_probs)
 * @param[in]: pSumTemp, contains in its zeroth row the partial sum
 * 			   nI and nJ are the dimensions of pSumTemp
 * 			   assuming nI is the number of dimensions
 * @param[out]:
 * @note:      assume 1D layout, covers 1 column of pSumTemp
 * 			   blockDim.x = 1, blockDim.y = 512
 *
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
__global__ void update_q_tr(IN const float * pSumTemp, int nI, int nJ, IN OUT float* pQ, float priorweight)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	if (i < nI){
		pQ[i] = (1-priorweight)/float(nJ)*pSumTemp[i]+priorweight*pQ[i];
	}
}


/***********************************************************************************************************
 * @brief: get dye E by dye a, w.r.t. sparse error criterion from 3D obj paper by Vinod
 * @param[in]:	pHidProbs is a nSample by nHidNodes probability matrix
 * 				pQ is 1 by (nJ or nHidNodes) overall hidden layer activation
 * 				p is desired activation [0 to 1.0]
 * @param[out]: pOut is nI by nJ of the results
 * @note:      assume 1D layout, similar to bsxfun kernels with row vector
 * @change:
 * @tested:
 * @to_do:     shared memory access of q may speed up?
 ***********************************************************************************************************
 */
template <typename T>
__global__ void get_dE_da_1dkernel(IN const T * pHidProbs, int nI, int nJ,
								  IN const T* pQ, T p, T * pOut)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (int i = ind; i < nI*nJ; i += totalThreads)
	{
		T q = pQ[i/nI];
		pOut[i] = pHidProbs[i]*(1-pHidProbs[i])*(p-q)/(q*(1-q));
	}
}

/***********************************************************************************************************
 * This is deprecated, should use get_dE_da_1dkernel instead
 *
 * @brief: get dye E by dye a, w.r.t. sparse error criterion from 3D obj paper by Vinod
 * @param[in]:	pHidProbs is a nSample by nHidNodes probability matrix
 * 				pQ is 1 by nJ or nHidNodes overall hidden layer activation
 * 				p is desired activation [0 1.0]
 * @param[out]: pOut is nI by nJ of the results
 * @note:      assume 2D layout
 *
 * @change:
 * @tested:
 * @to_do:     shared memory access of q may speed up?
 ***********************************************************************************************************
 */
__global__ void deprecated_get_dE_da(IN const float * pHidProbs, int nI, int nJ, IN OUT float* pQ, float p, float * pOut)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if (j < nJ && i < nI){
		float q = pQ[j];
		pOut[j*nI+i] = pHidProbs[j*nI+i]*(1-pHidProbs[j*nI+i])*(p-q)/(q*(1-q));
	}
}


/***********************************************************************************************************
 * @brief: get dye E by dye a, w.r.t. sparse error criterion from 3D obj paper by Vinod
 * 			This is transposed version of get_dE_da, where pHidProbs is dim by nsamples instead
 * @param[in]:	pHidProbs is a nSample by nHidNodes probability matrix
 * 				pQ is 1 by nI or nHidNodes overall hidden layer activation
 * 				p is desired activation [0 1.0]
 * @param[out]: pOut is nI by nJ of the results
 * @note:      assume 2D layout
 * @change:
 * @tested:
 * @to_do:     shared memory access of q may speed up?
 ***********************************************************************************************************
 */
__global__ void get_dE_da_tr(IN const float * pHidProbs, int nI, int nJ, IN OUT float* pQ, float p, float * pOut)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if (j < nJ && i < nI){
		float q = pQ[i];
		pOut[j*nI+i] = pHidProbs[j*nI+i]*(1-pHidProbs[j*nI+i])*(p-q)/(q*(1-q));
	}
}

/***********************************************************************************************************
 *	@brief: get dye E by dye a, w.r.t. sparse error criterion of sparisty within a population
 *			different from get_dE_da (which is sparsity across time)
 * @param[in]:	pHidProbs is a nSample by nHidNodes probability matrix
 * 				pQ is nI by 1 of average activation over a population to 1 visible vector.
 * 				p is desired activation [0 1.0]
 * @param[out]: pOut is nI by nJ of the results
 * @note:      assume 1D layout like bsxfun with column vector
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
__global__ void get_dE_da_popu_1dkernel(IN const T* pHidProbs, int nI, int nJ,
										IN T* pQ, T p, T* pOut)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (int k = ind; k < nI*nJ; k += totalThreads)
	{
		T q = pQ[k % nI];
		pOut[k] = pHidProbs[k]*(1-pHidProbs[k])*(p-q)/(q*(1-q));
	}
}



/***********************************************************************************************************
 * This is deprecated, should use get_dE_da_popu_1dkernel instead
 *	@brief: get dye E by dye a, w.r.t. sparse error criterion of sparisty within a population
 *			different from get_dE_da (which is sparsity across time)
 * @param[in]:	pHidProbs is a nSample by nHidNodes probability matrix
 * 				pQ is nI by 1 of average activation over a population to 1 visible vector.
 * 				p is desired activation [0 1.0]
 * @param[out]: pOut is nI by nJ of the results
 * @note:      assume 2D layout
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
__global__ void deprecated_get_dE_da_popu(IN const float* pHidProbs, int nI, int nJ, IN float* pQ, float p, float* pOut)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if (j < nJ && i < nI){
		float q = pQ[i]; //note this is i, whereas dE_da is j.
		pOut[j*nI+i] = pHidProbs[j*nI+i]*(1-pHidProbs[j*nI+i])*(p-q)/(q*(1-q));
	}
}




/***********************************************************************************************************
 * @brief: for cross entropy finetuning, get the error f and partial derivative
 * 			of the output nodes
 * @param[in]:	pNodes - dim1 by dim2 contains [0 1.0] activations
 * 				pTargetData - dim1 by 1, contains [0 9] of the target or correct answer
 * @param[out]:	pIxNodes - dim1 by dim2, output buffer for partial derivatives
 * 				pTargetLogProb - dim1 by 1 - contains the log of the probabilities of the indicated
 * 								output activation of the k-th target node
 * @note:      assume 1D vertical layout, blockDim.x == 1, gridDim.x == 1
 * 			   pIxNodesData must be equal to pNodes at the start!
 * 			   dim1 is number of samples
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
__global__ void ce_get_f_Ixout(IN const T* pNodes, IN const T* pTargetData, OUT T* pIxNodesData,
							   OUT T* pTargetLogProb, int dim1)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if  (i < dim1){
		int nDigInd = pTargetData[i]*dim1+i; //convert to a index
		pTargetLogProb[i] = logf( fmaxf( pNodes[nDigInd], 1e-40 ) );
		pIxNodesData[ nDigInd ] -= 1.0f;
	}
}

//double version
template <>
__global__ void ce_get_f_Ixout(IN const double* pNodes, IN const double* pTargetData, OUT double* pIxNodesData,
							   OUT double* pTargetLogProb, int dim1)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if  (i < dim1){
		int nDigInd = pTargetData[i]*dim1+i; //convert to a index
		pTargetLogProb[i] = log( fmax( pNodes[nDigInd], 1e-80 ) );
		pIxNodesData[ nDigInd ] -= 1.0;
	}
}



/***********************************************************************************************************
* @brief: wrapper to calculate the partial derivative of multinomial cross entropy functional
* @param[in]:	NodesOut - nSamples by 10 the output softmax variables
* 				Target - nSamples by 1, with 0 to 9 (for example) indicator
*
* @param[out]:	pTargetLogProb - dim1 by 1 - contains the log of the probabilities of the indicated
 * 								output activation of the k-th target node
 *				IxOut - partial derivative of the output variable, i.e. before the softmax
 *				IxOut should be d (log p) / d out, which is negative of the grad of ce error
* 				pf - pointer to the float error
* @topology:
* @note:
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template <typename T>
int ce_f_Ix(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, OUT clMatrix<T>& IxOut,
		OUT clMatrix<T>& TargetLogProb, OUT T* pf){

	if (NodesOut.nI != Target.nI ||
		NodesOut.nI != IxOut.nI ||
		NodesOut.nI != TargetLogProb.nI)
		return -1;
	if (NodesOut.nJ != IxOut.nJ ||
		Target.nJ != 1 ||TargetLogProb.nJ != 1)
		return -2;

	if (NodesOut.nI > MAX_GRIDS)
		return -3;

	int numSamples = NodesOut.nI;
	IxOut.CopyFrom(NodesOut); //copy to IxOut, it's needed by the kernel!!

	dim3 dim_block_y(1, MAX_THREADS);
	dim3 dim_grid(1, int(numSamples+dim_block_y.y-1)/dim_block_y.y );
	ce_get_f_Ixout<<< dim_grid, dim_block_y>>>(NodesOut.pData, Target.pData, IxOut.pData, TargetLogProb.pData, numSamples);

	cr (SumInplace( TargetLogProb, 1))
	pf[0] = -TargetLogProb.GetElem(0,0)/T(numSamples);
    cr ( cl_cublas_scal( T(-1.0/double(numSamples)), IxOut ))  //this is to make IxOut == - dE/d out
    return 0;
}

template int ce_f_Ix<float>(const clMatrix<float>& NodesOut, const clMatrix<float>& Target,
		OUT clMatrix<float>& IxOut, OUT clMatrix<float>& TargetLogProb, OUT float* pf);

template int ce_f_Ix<double>(const clMatrix<double>& NodesOut, const clMatrix<double>& Target,
		OUT clMatrix<double>& IxOut, OUT clMatrix<double>& TargetLogProb, OUT double* pf);


//difference is that we want TargetLogProb to not be sum over all nSamples
template <typename T>
int ce_f_Ix2(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, OUT clMatrix<T>& IxOut,
		OUT clMatrix<T>& TargetLogProb){

	if (NodesOut.nI != Target.nI ||
		NodesOut.nI != IxOut.nI ||
		NodesOut.nI != TargetLogProb.nI)
		return -1;
	if (NodesOut.nJ != IxOut.nJ ||
		Target.nJ != 1 ||TargetLogProb.nJ != 1)
		return -2;

	int numSamples = NodesOut.nI;
	IxOut.CopyFrom(NodesOut); //copy to IxOut, it's needed by the kernel!!

	dim3 dim_block_y(1, MAX_THREADS);
	dim3 dim_grid(1, int(numSamples+dim_block_y.y-1)/dim_block_y.y );
	ce_get_f_Ixout<<< dim_grid, dim_block_y>>>(NodesOut.pData, Target.pData, IxOut.pData, TargetLogProb.pData, numSamples);
    cr ( cl_cublas_scal( T(-1.0/double(numSamples)), IxOut )) //this is to make IxOut == - dE/d out
    return 0;
}

template int ce_f_Ix2<float>(const clMatrix<float>& NodesOut, const clMatrix<float>& Target,
		OUT clMatrix<float>& IxOut, OUT clMatrix<float>& TargetLogProb);

template int ce_f_Ix2<double>(const clMatrix<double>& NodesOut, const clMatrix<double>& Target,
		OUT clMatrix<double>& IxOut, OUT clMatrix<double>& TargetLogProb);


/***********************************************************************************************************
 * @brief: for cross entropy finetuning, get the error f and partial derivative
 * 			of the output nodes
 * 			for logistic outputs!
 *
 * @param[in]:	pNodes - dim1 by dim2 contains [0 1.0] activations
 * 				pTargetData - dim1 by dim2, contains [0 or 1] of the target or correct answer
 *
 * @param[out]:	pIxNodes - dim1 by dim2, output buffer for partial derivatives
 * 				pTargetLogProb - dim1 by dim2 - contains the log of the probabilities of the indicated
 * 								output activation of the k-th target node
 * @note:
 *
 *
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
__global__ void ce_get_f_Ixout_logistic(IN const T* pNodes, IN const T* pTargetData,
							   OUT T* pTargetLogProb, uint64_t dim )
{
	const uint64_t ind = blockIdx.x*blockDim.x + threadIdx.x;
	const uint64_t totalThreads = blockDim.x*gridDim.x;

	for (uint64_t i = ind; i < dim; i += totalThreads){
		T prob = pNodes[i];
		pTargetLogProb[i] = pTargetData[i]*logf( fmaxf(prob, 1e-7) )+ (1-pTargetData[i])*logf( fmaxf(1-prob, 1e-7) );
	}
}

template <>
__global__ void ce_get_f_Ixout_logistic(IN const double* pNodes, IN const double* pTargetData,
							   OUT double* pTargetLogProb, uint64_t dim )
{
	const uint64_t ind = blockIdx.x*blockDim.x + threadIdx.x;
	const uint64_t totalThreads = blockDim.x*gridDim.x;

	for (uint64_t i = ind; i < dim; i += totalThreads){
		double prob = pNodes[i];
		pTargetLogProb[i] = pTargetData[i]*log( fmax(prob, 1e-50) )+ (1-pTargetData[i])*log( fmax(1-prob, 1e-50) );
	}
}



/***********************************************************************************************************
* @brief: wrapper to calculate the partial derivative of a cross entropy functional
* 		 for logistic outputs!
*
* @param[in]:	NodesOut - nSamples by J, where J is the number of output variables, each is a logistic/bernoulli
* 				Target - nSamples by J, where each row is a vector of 0 or 1's
*
* @param[out]:	TargetLogProb - dim1 by J - contains the log of the probabilities of the indicated
 * 								output activation of the target
 * 								this will be used by the kernel
 * 				it serves as a buffer
 *
 *				IxOut - partial derivative of the output variable, i.e. before the sigmoid
* 				pf - pointer to the float error
* @topology:
* @note:  here the CE is not divided by N, where N is the number of samples
*
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template <typename T>
int ce_f_Ix_logistic(const clMatrix<T>& NodesOut, const clMatrix<T>& Target,
				OUT clMatrix<T>& IxOut, OUT T* pf)
{
	if (NodesOut.nI != Target.nI ||
		NodesOut.nI != IxOut.nI )
		return -1;
	if (NodesOut.nJ != IxOut.nJ ||
		NodesOut.nJ != Target.nJ)
		return -2;

	const unsigned int datadim = NodesOut.nI*NodesOut.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	ce_get_f_Ixout_logistic<<< dim_grid, dim_block>>>(NodesOut.pData, Target.pData, IxOut.pData, datadim );
	*pf =  -1/T(NodesOut.nI)*Sum2DInplace( IxOut );

	cr( EleWisefun( Target, fctrMinus<T>(), NodesOut, IxOut))
	cr( EleWisefun( IxOut, fctrDiv<T>(), T(NodesOut.nI), IxOut))


    return 0;
}

template int ce_f_Ix_logistic<float>(const clMatrix<float>& NodesOut, const clMatrix<float>& Target,
				OUT clMatrix<float>& IxOut, OUT float* pf);
template int ce_f_Ix_logistic<double>(const clMatrix<double>& NodesOut, const clMatrix<double>& Target,
				OUT clMatrix<double>& IxOut, OUT double* pf);

//difference is that we want a matrix of log p(y|x) in IxOut
template <typename T>
int ce_f_Ix_logistic2(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, OUT clMatrix<T>& IxOut)
{
	if (NodesOut.nI != Target.nI ||
		NodesOut.nI != IxOut.nI )
		return -1;
	if (NodesOut.nJ != IxOut.nJ ||
		NodesOut.nJ != Target.nJ)
		return -2;

	const unsigned int datadim = NodesOut.nI*NodesOut.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	ce_get_f_Ixout_logistic<<< dim_grid, dim_block>>>(NodesOut.pData, Target.pData, IxOut.pData, datadim );
    return 0;
}

template int ce_f_Ix_logistic2<float>(const clMatrix<float>& NodesOut, const clMatrix<float>& Target,
				OUT clMatrix<float>& IxOut);
template int ce_f_Ix_logistic2<double>(const clMatrix<double>& NodesOut, const clMatrix<double>& Target,
				OUT clMatrix<double>& IxOut);




//kernel for LogProbTestErrs
//assume 1D layout y layout
//pMat is nI by nJ
//pLabel and pOut is nI by 1
template <typename T>
__global__ void logprob_test_errs(const T* pMat, const T* pLabel,
								  T * pOut, int nI, int nJ){

	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if (i < nI){
		if (pLabel[i] >= 0){
			pOut[i] = pMat[ int(pLabel[i])*nI+i ];
		}else{
			pOut[i] = T(9876.54321);
		}
	}
}


//this function computes the test errors when given a matrix mat: N by D dimensional choise
//a Label N by 1 zero based target index, where the max (mat(0,:)) should be equal to Label(0)
//will destroy mat
//samples with label == -1 will be ignored
template <typename T>
int LogProbTestErrs( clMatrix<T>& mat, clMatrix<T>& buffer, const clMatrix<T>& label, T* pcnt_correct)
{
	if (mat.nI != label.nI || mat.nI != buffer.nI || buffer.nJ != 1 || label.nJ != 1)
		return -1;

	dim3 dim_block(1, BLOCK_DIM);
	dim3 dim_grid( 1, (mat.nI+dim_block.y-1)/dim_block.y);
	logprob_test_errs<<< dim_grid, dim_block>>>( mat.pData, label.pData, buffer.pData, mat.nI, mat.nJ);
	cr( ReduceInplace(fctrMax<T>(), mat, 2) )
	clMatrix<T> shell = mat.ColView(0);
	cr(EleWisefun(shell, fctrEquals<T>(), buffer, buffer))
	*pcnt_correct = Sum2DInplace(buffer);

	cr(EleWisefun( label, fctrGreaterThan<T>(), T(-1), buffer ))
	T fNumLabeled = Sum2DInplace(buffer);
	*pcnt_correct /= fNumLabeled;

	return 0;
}

template int LogProbTestErrs<float>( clMatrix<float>& mat, clMatrix<float>& buffer,
		const clMatrix<float>& label, float* pcnt_correct);
template int LogProbTestErrs<double>( clMatrix<double>& mat, clMatrix<double>& buffer,
		const clMatrix<double>& label, double* pcnt_correct);



/***********************************************************************************************************
* @brief: kernel function for AddL1WtCost
* @param[in]: float* pData of vhW and vhWInc
* 			  f_rate is the learning rate
* 			  f_l1_wtcost is the l1 constant
* @param[out]:
* @topology: 2D covering the matrix
* @note:
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/

template <typename T>
__global__ void add_l1_wtcost_kernel(const T* p_vhW, T* p_vhWInc, int nI, int nJ, T f_rate, T f_l1_wtcost ){

	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < nI && j < nJ){
		if (p_vhW[j*nI+i] > T(0.0)){
			p_vhWInc[j*nI+i] -= f_rate*f_l1_wtcost;
		}else if (p_vhW[j*nI+i] < T(0.0) ){
			p_vhWInc[j*nI+i] += f_rate*f_l1_wtcost;
		}
	}
}

/***********************************************************************************************************
* @brief: this function addes the gradient of the L1 penalty cost to the weight increment matrix
* @param[in]: f_rate is learning rate, f_l1_wtcost is the weight cost of L1 penalty
* @param[out]:
* @topology:
* @note:
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template <typename T>
int AddL1WtCost(const  clMatrix<T>& vhW,  clMatrix<T>& vhWInc, T f_rate, T f_l1_wtcost){

	if (vhW.nI != vhWInc.nI || vhW.nJ != vhWInc.nJ)
		return -1;

	dim3 dim_block(BLOCK_DIM, BLOCK_DIM);
	dim3 dim_grid( (vhW.nJ+dim_block.x-1)/dim_block.x,
				   (vhW.nI+dim_block.y-1)/dim_block.y);

	add_l1_wtcost_kernel<<< dim_grid, dim_block >>>( vhW.pData, vhWInc.pData, vhW.nI, vhW.nJ, f_rate, f_l1_wtcost);

	return 0;
}

template int AddL1WtCost<float>(const  clMatrix<float>& vhW,  clMatrix<float>& vhWInc, float f_rate, float f_l1_wtcost);
template int AddL1WtCost<double>(const  clMatrix<double>& vhW,  clMatrix<double>& vhWInc, double f_rate, double f_l1_wtcost);

/***********************************************************************************************************
 * @brief: matrix multiplication inplace to get the partial derivative after a sigmoid
 * 		   pNodes = pNodes.*(1-pAfterSigmoid).*(pAfterSigmoid)
 * @param[in]:	pNodes is to be multiplied inplace by above equation
 * @param[out]: pOut is nI by nJ of the results
 * @note:      assume 2D layout
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
__global__ void matmul_derive_sigm(IN OUT T* pNodes, IN const T * pAfterSigmoid, int nI, int nJ, unsigned int nInJ)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (unsigned int i = ind; i < nInJ; i += totalThreads){
		pNodes[i] *= (pAfterSigmoid[i]*(1-pAfterSigmoid[i]));
	}
}


//2D matrix, element-wise multiplies pd_p by pd_p.*p.*(1-p)
template <typename T>
int SigmoidBackprop( const clMatrix<T>&  p, clMatrix<T>&  pd_p){

	if (p.nI != pd_p.nI || p.nJ != pd_p.nJ)
		return -1;

	const unsigned int datadim = p.nI*p.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	matmul_derive_sigm<<< dim_grid, dim_block>>>( pd_p.pData, p.pData, pd_p.nI, pd_p.nJ, datadim);
	return 0;

}

template
int SigmoidBackprop<float>( const clMatrix<float>&  p, clMatrix<float>&  pd_p);
template
int SigmoidBackprop<double>( const clMatrix<double>&  p, clMatrix<double>&  pd_p);


/***********************************************************************************************************
 * @brief: matrix multiplication inplace to get the partial derivative after a tanh
 * 		   pNodes = pNodes.*(1.7159*0.66667-0.66667/1.7159*p.^2)
 * @param[in]:	pNodes is to be multiplied inplace by above equation
 * @param[out]:
 * @note:      assume 2D layout
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
__global__ void matmul_derive_tanh(IN OUT T* pNodes, IN const T * pAfterTanh, int nI, int nJ,  unsigned int nInJ)
{
	const T cons1 = 1.7159;
	const T cons2 = 0.66667;

	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (unsigned int i = ind; i < nInJ; i += totalThreads){
		if (pAfterTanh[i] == 0){  //no way that by chance we get exactly 0, must be dropout
			pNodes[i] = 0;			//if dropout, the partial derivative will be set to 0
		}
		else{
			pNodes[i] *= (cons1*cons2-cons2/cons1*(pAfterTanh[i]*pAfterTanh[i]));
		}
	}
}

//2D matrix, element-wise multiplies pd_p by d tanh / d input
template <typename T>
int TanhBackprop( const clMatrix<T>&  p, clMatrix<T>&  pd_p){

	if (p.nI != pd_p.nI || p.nJ != pd_p.nJ)
		return -1;

	const unsigned int datadim = p.nI*p.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	matmul_derive_tanh<<< dim_grid, dim_block>>>( pd_p.pData, p.pData,
												  pd_p.nI, pd_p.nJ, datadim);
	return 0;

}


template
int TanhBackprop<float>( const clMatrix<float>&  p, clMatrix<float>&  pd_p);
template
int TanhBackprop<double>( const clMatrix<double>&  p, clMatrix<double>&  pd_p);

/***********************************************************************************************************
 * @brief: kernel for backprogagation in Relu hidden units
 * @param[in]:
 * @param[out]:
 * @note:
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
__global__ void derive_relu_kernel(IN OUT T* p_dEda, IN const T * pNodes, int nI, int nJ, unsigned int nInJ)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (unsigned int i = ind; i < nInJ; i += totalThreads){
		/*if (pNodes[i] == 0){
			p_dEda[i] = 0;
		}
		*/
		p_dEda[i] *= (pNodes[i] > 0); //equivalent to above
	}
}

//2D matrix, element-wise multiplies pd_p by pd_p.*( p < 0 set to 0)
template <typename T>
int ReluBackprop( const clMatrix<T>&  p, clMatrix<T>&  pd_p){

	if (p.nI != pd_p.nI || p.nJ != pd_p.nJ)
		return -1;

	const unsigned int datadim = p.nI*p.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	derive_relu_kernel<<< dim_grid, dim_block>>>( pd_p.pData, p.pData, pd_p.nI, pd_p.nJ, datadim);
	return 0;
}


template int ReluBackprop<float>( const clMatrix<float>&  p, clMatrix<float>&  pd_p);
template int ReluBackprop<double>( const clMatrix<double>&  p, clMatrix<double>&  pd_p);


/***********************************************************************************************************
 * @brief: kernel for backprogagation in Soft Relu hidden units
 * @param[in]:
 * @param[out]:
 * @note:
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
__global__ void derive_softrelu_kernel(IN OUT T* p_dEda, IN const T * pNodes, int nI, int nJ, unsigned int nInJ)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (unsigned int i = ind; i < nInJ; i += totalThreads){
		if (pNodes[i] == 0){
			p_dEda[i] = 0;
		}
		else{
			p_dEda[i] *= 1-expf(-pNodes[i]);
		}
	}
}

template <>
__global__ void derive_softrelu_kernel<double>(IN OUT double* p_dEda, IN const double * pNodes, int nI, int nJ, unsigned int nInJ)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (unsigned int i = ind; i < nInJ; i += totalThreads){
		if (pNodes[i] == 0){
			p_dEda[i] = 0;
		}
		else{
			p_dEda[i] *= 1-exp(-pNodes[i]);
		}
	}
}

//2D matrix, element-wise multiplies pd_p by pd_p.*( 1-exp(-p) )
template <typename T>
int SoftReluBackprop( const clMatrix<T>&  p, clMatrix<T>&  pd_p){

	if (p.nI != pd_p.nI || p.nJ != pd_p.nJ)
		return -1;

	const unsigned int datadim = p.nI*p.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	derive_softrelu_kernel<<< dim_grid, dim_block>>>( pd_p.pData, p.pData, pd_p.nI, pd_p.nJ, datadim);
	return 0;
}

template int SoftReluBackprop<float>( const clMatrix<float>&  p, clMatrix<float>&  pd_p);
template int SoftReluBackprop<double>( const clMatrix<double>&  p, clMatrix<double>&  pd_p);



/***********************************************************************************************************
 * @brief: kernel for backprogagation in Relu hidden units
 * @param[in]:
 * @param[out]:
 * @note:
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
__global__ void derive_relu_quad_kernel(IN OUT T* p_dEda, IN const T * pNodes, int nI, int nJ, unsigned int nInJ)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (unsigned int i = ind; i < nInJ; i += totalThreads){
		if (pNodes[i] == 0){
			p_dEda[i] = 0;
		}else{
			p_dEda[i] *= 2*sqrtf(pNodes[i]);
		}
	}
}

template <>
__global__ void derive_relu_quad_kernel<double>(IN OUT double* p_dEda, IN const double * pNodes, int nI, int nJ, unsigned int nInJ)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int totalThreads = blockDim.x*gridDim.x;

	for (unsigned int i = ind; i < nInJ; i += totalThreads){
		if (pNodes[i] == 0){
			p_dEda[i] = 0;
		}else{
			p_dEda[i] *= 2*sqrtf(pNodes[i]);
		}
	}
}

//qudratic relu backprop
template <typename T>
int ReluQuadBackprop( const clMatrix<T>&  p, clMatrix<T>&  pd_p){

	if (p.nI != pd_p.nI || p.nJ != pd_p.nJ)
		return -1;

	const unsigned int datadim = p.nI*p.nJ;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (datadim + dim_block.x-1)/dim_block.x) );

	derive_relu_quad_kernel<<< dim_grid, dim_block>>>( pd_p.pData, p.pData, pd_p.nI, pd_p.nJ, datadim);
	return 0;
}

template int ReluQuadBackprop<float>( const clMatrix<float>&  p, clMatrix<float>&  pd_p);
template int ReluQuadBackprop<double>( const clMatrix<double>&  p, clMatrix<double>&  pd_p);







/***************************************************************************************************
* @brief: calculates the logsumexp(x) in a numerically stable way
* @param[in]: mat
* 			 dim - 1 or 2, which dimension do we want to do logsumexp on ?
* 			 out - a buffer matrix same size as mat
* @param[out]: logz nI by 1 vector of answer
* @topology:
* @note:    currently only support dim==2
*			currently only supports this float version
* @change:
* @tested:
* @to_do:
****************************************************************************************************
*/
template <typename T>
int LogSumExp(IN const clMatrix<T> & mat, clMatrix<T> & out, int dim, OUT clMatrix<T>& logz ){

	if ( mat.nI != out.nI || mat.nJ != out.nJ )
		return -1;
	if (dim != 2 ) //currently only support dim ==2
		return -4;
	if ( logz.nI != out.nI || logz.nJ != 1) //currently only support dim ==2
		return -5;
	if ( logz.pData == out.pData )         //can't be the same matrix
		return -6;

	out.CopyFrom(mat);
	cr( ReduceInplace( fctrMax<T>(), out, dim) );

	clMatrix<T> shell;
	shell.nI = out.nI;
	shell.nJ = 1;
	shell.pData = out.pData;

	cr( EleWisefun( shell, fctrMinus<T>(), T(MAX_FLOAT_EXP / 2.0f), logz) )  //not best for double precision!
	//logz is now 'alpha', or the shift

	cr( Bsxfun(  mat, fctrMinus<T>(), logz, out))
	cr( EleWisefun( fctrExp<T>(), out, out ))
	cr( ReduceInplace( fctrPlus<T>(), out, dim) )

	cr (EleWisefun (fctrLog<T>(), shell, shell) )
	cr( EleWisefun(shell, fctrPlus<T>(), logz, logz)) //we shift back, and put the result in logz

	return 0;
}

template <>
int LogSumExp<double>(IN const clMatrix<double> & mat, clMatrix<double> & out, int dim,
		OUT clMatrix<double>& logz ){

	if ( mat.nI != out.nI || mat.nJ != out.nJ )
		return -1;
	if (dim != 2 ) //currently only support dim ==2
		return -4;
	if ( logz.nI != out.nI || logz.nJ != 1) //currently only support dim ==2
		return -5;
	if ( logz.pData == out.pData )         //can't be the same matrix
		return -6;

	out.CopyFrom(mat);
	cr( ReduceInplace( fctrMax<double>(), out, dim) );

	clMatrix<double> shell;
	shell.nI = out.nI;
	shell.nJ = 1;
	shell.pData = out.pData;

	cr( EleWisefun( shell, fctrMinus<double>(), MAX_DOUBLE_EXP / 2.0, logz) )
	//logz is now 'alpha', or the shift

	cr( Bsxfun(  mat, fctrMinus<double>(), logz, out))
	cr( EleWisefun( fctrExp<double>(), out, out ))
	cr( ReduceInplace( fctrPlus<double>(), out, dim) )

	cr (EleWisefun (fctrLog<double>(), shell, shell) )
	cr( EleWisefun(shell, fctrPlus<double>(), logz, logz)) //we shift back, and put the result in logz

	return 0;
}

template
int LogSumExp<float>(IN const clMatrix<float> & mat, clMatrix<float> & out, int dim, OUT clMatrix<float>& logz );
template
int LogSumExp<double>(IN const clMatrix<double> & mat, clMatrix<double> & out, int dim, OUT clMatrix<double>& logz );





//apply dropout with probiability of f_dropout to nodes, Inplace
template <typename T>
int Dropout( clMatrix<T>& nodes, clMatrix<T>& randnums, float f_dropout)
{
	const int nI = nodes.nI;
	const int nJ = nodes.nJ;

	if (nI*nJ <= 0)
		return -1;
	if (randnums.nI*randnums.nJ < nodes.nI*nodes.nJ)
		return -2;
	if (f_dropout < 0 || f_dropout > 1)
		return -3;

	clMatrix<T> shell;
	shell.pData = randnums.pData;
	shell.nI = nI;
	shell.nJ = nJ;

	cr( cuda_clMatSetRand( shell ) ) //randnums dimension may be bigger than nodes

	cr( EleWisefun( shell, fctrGreaterThan<T>(), T(f_dropout), shell ) )
	cr( EleWisefun( shell, fctrMul<T>(), nodes, nodes ) )

	return 0;
}

template int Dropout<float>( clMatrix<float>& nodes, clMatrix<float>& randnums, float f_dropout);
template int Dropout<double>( clMatrix<double>& nodes, clMatrix<double>& randnums, float f_dropout);





/***********************************************************************************************************
 * @brief: 	for l2svm multiclass, kernel need to set the T matrix (nSamples by nClasses)
 * 			with 1's and -1's.
 *
 * @param[in]:	pY is nSamples by 1. its value (0 to nClasses-1) must be integer
 *
 * @param[out]:	pT is nSamples by nClasses
 * @note:      assume 1D vertical layout, blockDim.x == 1, gridDim.x == 1
 *			   pT must be set to -1's to start
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
__global__ void l2svm_set_T( IN const T* pY, int nSamples, OUT T* pT)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if  (i < nSamples && pY[i] >= 0){
		int nInd = pY[i]*nSamples+i;  //index in pT
		pT[nInd] = T(1.0);
	}
}

/***********************************************************************************************************
* @brief: calculate the loss and the derivatives of the l2_svm
* @param[in]:	NodesOut - nSamples by 10 (e.g) the output softmax variables (NodesOut == X*W+bias)
* 				Target  -  nSamples by 1, with 0 to 9 (e.g.) indicator
*
* @param[out]:	pTargetLogProb - dim1 by 1 - contains the log of the probabilities of the indicated
 * 								output activation of the k-th target node
 *				IxOut - partial derivative of loss w.r.t NodesOut
*
* @topology:
* @note:
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template <typename T>
int loss_l2svm(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, T fC,
				OUT clMatrix<T>& IxOut, clMatrix<T>& NodesOutTemp, OUT T& fLoss )
{

	if (NodesOut.nI != Target.nI ||
		NodesOut.nI != IxOut.nI ||
		NodesOut.nI != NodesOutTemp.nI)
		return -1;

	if (NodesOut.nJ != IxOut.nJ ||
		Target.nJ != 1 ||NodesOutTemp.nJ != NodesOut.nJ)
		return -2;

	int nSamples = NodesOut.nI;
	IxOut.SetVal(-1);

	dim3 dim_block_y(1, MAX_THREADS);
	dim3 dim_grid(1, int(nSamples+dim_block_y.y-1)/dim_block_y.y );
	l2svm_set_T<<< dim_grid, dim_block_y>>>(Target.pData, nSamples, IxOut.pData);  //IxOut is set to T

	cr(EleWisefun(T(-1.0), NodesOut, fctrAlphaMulBeta<T>(), T(1), IxOut, NodesOutTemp))
	cr(EleWisefun( NodesOutTemp, fctrPlus<T>(), T(1), NodesOutTemp))
	cr(EleWisefun( NodesOutTemp, fctrMax<T>(), T(0), NodesOutTemp))  //it is equal to `margin' now: max(0,1-a*t)

	clMatrix<T> tempcol(nSamples, 1);   //allocation, Slow!!!!!
	cr(EleWisefun( Target, fctrGreaterThan<T>(), T(-1), tempcol ))
	cr (Bsxfun( NodesOutTemp, fctrMul<T>(), tempcol, NodesOutTemp ))  //zero out unlabeled portion


	cr(EleWisefun( NodesOutTemp, fctrMul<T>(), IxOut, IxOut))

	T fNumLabeled = Sum2DInplace(tempcol);
	cr(EleWisefun( IxOut, fctrMul<T>(), T(+2*fC/fNumLabeled), IxOut))  //should be -2*fC/nSamples, but mlp needs -ve gradient w.r.t error

	cr(EleWisefun( fctrSq<T>(), NodesOutTemp, NodesOutTemp))
	fLoss = Sum2DInplace(NodesOutTemp)*fC/fNumLabeled;

//	PrintfInMatlab(Target);
//	PrintfInMatlab(tempcol);
//	PrintfInMatlab(NodesOutTemp);
	//PrintfInMatlab(IxOut);

    return 0;
}

template int loss_l2svm<float>(const clMatrix<float>& NodesOut, const clMatrix<float>& Target, float fC,
				OUT clMatrix<float>& IxOut, clMatrix<float>& NodesOutTemp, OUT float& fLoss );
template int loss_l2svm<double>(const clMatrix<double>& NodesOut, const clMatrix<double>& Target, double fC,
				OUT clMatrix<double>& IxOut, clMatrix<double>& NodesOutTemp, OUT double& fLoss );

/***********************************************************************************************************
* @brief: calculate the loss and the derivatives of the l1_svm
* @param[in]:	NodesOut - nSamples by 10 (e.g) the output softmax variables (NodesOut == X*W+bias)
* 				Target  -  nSamples by 1, with 0 to 9 (e.g.) indicator
*
* @param[out]:	pTargetLogProb - dim1 by 1 - contains the log of the probabilities of the indicated
 * 								output activation of the k-th target node
 *				IxOut - partial derivative of loss w.r.t NodesOut
*
* @topology:
* @note:		does not work with unlabeled data
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template <typename T>
int loss_l1svm(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, T fC,
				OUT clMatrix<T>& IxOut, clMatrix<T>& NodesOutTemp, OUT T& fLoss )
{

	if (NodesOut.nI != Target.nI ||
		NodesOut.nI != IxOut.nI ||
		NodesOut.nI != NodesOutTemp.nI)
		return -1;

	if (NodesOut.nJ != IxOut.nJ ||
		Target.nJ != 1 ||NodesOutTemp.nJ != NodesOut.nJ)
		return -2;

	int nSamples = NodesOut.nI;
	IxOut.SetVal(-1);

	dim3 dim_block_y(1, MAX_THREADS);
	dim3 dim_grid(1, int(nSamples+dim_block_y.y-1)/dim_block_y.y );
	l2svm_set_T<<< dim_grid, dim_block_y>>>(Target.pData, nSamples, IxOut.pData);  //IxOut is set to T

	cr(EleWisefun(T(-1.0), NodesOut, fctrAlphaMulBeta<T>(), T(1), IxOut, NodesOutTemp))
	cr(EleWisefun( NodesOutTemp, fctrPlus<T>(), T(1), NodesOutTemp))
	cr(EleWisefun( NodesOutTemp, fctrMax<T>(), T(0), NodesOutTemp))  //it is equal to `margin' now: max(0,1-a*t)
	fLoss = Sum2DInplace(NodesOutTemp)*fC/nSamples;

	cr(EleWisefun(T(-1.0), NodesOut, fctrAlphaMulBeta<T>(), T(1), IxOut, NodesOutTemp))
	cr(EleWisefun( NodesOutTemp, fctrPlus<T>(), T(1), NodesOutTemp)) // 1-a*t
	cr(EleWisefun( NodesOutTemp, fctrGreaterThan<T>(), T(0), NodesOutTemp))

	cr(EleWisefun( T(+fC/nSamples), IxOut, fctrAlphaMulBeta<T>(), T(1), NodesOutTemp, IxOut))
	//should be -fC/nSamples, but mlp needs -ve gradient w.r.t error

    return 0;
}

template int loss_l1svm<float>(const clMatrix<float>& NodesOut, const clMatrix<float>& Target, float fC,
				OUT clMatrix<float>& IxOut, clMatrix<float>& NodesOutTemp, OUT float& fLoss );

template int loss_l1svm<double>(const clMatrix<double>& NodesOut, const clMatrix<double>& Target, double fC,
				OUT clMatrix<double>& IxOut, clMatrix<double>& NodesOutTemp, OUT double& fLoss );

/***********************************************************************************************************
 * @brief: 	for l2 tsvm multiclass, kernel need to set the T matrix (nSamples by nClasses)
 * 			with 1's and -1's.
 *
 * @param[in]:	pY is nSamples by 1. its value (0 to nClasses-1) must be integer OR -1 for unlabeled
 *
 * @param[out]:	pT is nSamples by nClasses
 * @note:      assume 1D vertical layout, blockDim.x == 1, gridDim.x == 1
 *			   pT must be set to -1's to start
 * @change:
 * @tested:
 * @to_do:
 ***********************************************************************************************************
 */
template <typename T>
__global__ void l2_tsvm_set_T( IN const T* pY, int nSamples, OUT T* pT)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if  (i < nSamples && pY[i] >= 0 ){
		int nInd = pY[i]*nSamples+i;  //index in pT
		pT[nInd] = T(1.0);
	}
}

/***********************************************************************************************************
* @brief: calculate the loss and the derivatives of the l2_svm
* @param[in]:	NodesOut - nSamples by 10 (e.g) the output softmax variables (NodesOut == X*W+bias)
* 				Target  -  nSamples by 1, with 0 to 9 (e.g.) indicator
*
* @param[out]:	pTargetLogProb - dim1 by 1 - contains the log of the probabilities of the indicated
 * 								output activation of the k-th target node
 *				IxOut - partial derivative of loss w.r.t NodesOut
*
* @topology:
* @note:		NodesOutTemp must have 3 times the columns as NodesOut
* @change:
* @tested:
* @to_do:
***********************************************************************************************************
*/
template <typename T>
int loss_l2_tsvm(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, T fC1, T fC2,
				OUT clMatrix<T>& IxOut, clMatrix<T>& NodesOutTemp, OUT T& fLoss )
{

	if (NodesOut.nI != Target.nI ||
		NodesOut.nI != IxOut.nI ||
		NodesOut.nI != NodesOutTemp.nI)
		return -1;

	if (NodesOut.nJ != IxOut.nJ ||
		Target.nJ != 1 ||NodesOutTemp.nJ != 3*NodesOut.nJ)
		return -2;

	int nSamples = NodesOut.nI;

	clMatrix<T> Temp1, Temp2, Temp3;
	Temp1.nI = Temp2.nI = Temp3.nI = nSamples;
	Temp1.nJ = Temp2.nJ = Temp3.nJ = NodesOut.nJ;
	Temp1.pData = NodesOutTemp.pData;
	Temp2.pData = NodesOutTemp.pData+NodesOut.nI*NodesOut.nJ;
	Temp3.pData = NodesOutTemp.pData+2*NodesOut.nI*NodesOut.nJ;

	IxOut.SetVal(-1);

	dim3 dim_block_y(1, MAX_THREADS);
	dim3 dim_grid(1, int(nSamples+dim_block_y.y-1)/dim_block_y.y );
	l2_tsvm_set_T<<< dim_grid, dim_block_y>>>(Target.pData, nSamples, IxOut.pData);  //IxOut is set to T

	cr(EleWisefun(T(-1.0), NodesOut, fctrAlphaMulBeta<T>(), T(1), IxOut, Temp1))
	cr(EleWisefun( Temp1, fctrPlus<T>(), T(1), Temp1))
	cr(EleWisefun( Temp1, fctrMax<T>(), T(0), Temp1))  //it is equal to `margin' now: max(0,1-a*t)

	clMatrix<T> tempcol = Temp3.ColView(0);
	cr(EleWisefun( Target, fctrGreaterThan<T>(), T(-1), tempcol ))
	cr (Bsxfun( Temp1, fctrMul<T>(), tempcol, Temp1 ))  //zero out unlabeled portion

	T fNumLabeled = Sum2DInplace(tempcol);
	T fNumUnLabeled = nSamples-fNumLabeled;

	cr(EleWisefun( Temp1, fctrMul<T>(), IxOut, IxOut))
	cr(EleWisefun( IxOut, fctrMul<T>(), T(+2*fC1/fNumLabeled), IxOut))  //should be -2*fC/nSamples, but mlp needs -ve gradient w.r.t error

	cr(EleWisefun( fctrSq<T>(), Temp1, Temp1))
	T fLoss1 = Sum2DInplace(Temp1)*fC1/fNumLabeled;

	//unlabeled objective
	if (fNumUnLabeled > 0){

		cr(EleWisefun(fctrAbs<T>(), NodesOut, Temp1))
		cr(EleWisefun( Temp1, fctrMul<T>(), T(-1), Temp1))
		cr(EleWisefun( Temp1, fctrPlus<T>(), T(1), Temp1))
		cr(EleWisefun( Temp1, fctrGreaterThan<T>(), T(0), Temp2 ))  //step function
		cr(EleWisefun( Temp1, fctrMax<T>(), T(0), Temp1))  // max(0, 1 - abs(a))

		cr(EleWisefun( Target, fctrLessThan<T>(), T(0), tempcol ))
		cr (Bsxfun( Temp1, fctrMul<T>(), tempcol, Temp1 ))  //zero out labeled portion
		T fLoss2 = Sum2DInplace(Temp1)*fC2/fNumUnLabeled;

		fLoss = fLoss1+fLoss2;

		cr(EleWisefun( NodesOut, fctrGreaterThan<T>(), T(0), Temp1 ))
		cr(EleWisefun( Temp1, fctrMul<T>(), T(2), Temp1))
		cr(EleWisefun( Temp1, fctrMinus<T>(), T(1), Temp1))  //-1 or +1

		cr(EleWisefun( Temp1, fctrMul<T>(), Temp2, Temp1))
		cr (Bsxfun( Temp1, fctrMul<T>(), tempcol, Temp1 ))  //zero out labeled portion

		cr(EleWisefun( T(1), IxOut, fctrAlphaPlusBeta<T>(),  T(fC2/fNumUnLabeled), Temp1, IxOut))
	}else{
		fLoss = fLoss1;
	}
    return 0;
}


template int loss_l2_tsvm<float>(const clMatrix<float>& NodesOut, const clMatrix<float>& Target,
		float fC1, float fC2, OUT clMatrix<float>& IxOut, clMatrix<float>& NodesOutTemp, OUT float& fLoss );
template int loss_l2_tsvm<double>(const clMatrix<double>& NodesOut, const clMatrix<double>& Target,
		double fC1, double fC2, OUT clMatrix<double>& IxOut, clMatrix<double>& NodesOutTemp, OUT double& fLoss );





