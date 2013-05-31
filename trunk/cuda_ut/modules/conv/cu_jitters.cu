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

#include "cu_jitters.h"
#include "cu_conv.h"

/*
 * nVisJ2 and nVisI2 is smaller than or equal nVisJ and nVisI
 pData is: [ nSamples; nVisJ*nVisI*nVisChs ]

 layout is 1D in x, covering nSamples, nVisJ2, nVisI2, nVisChs;

 dI - nSamples by 1  [0 to nVisI-nVisI2]
 dJ - nSamples by 1  [0 to nVisJ-nVisJ2]
 pJitter - [ nSamples; nVisJ2*nVisI2*nVischs ]

 */
template<typename T>
__global__ void image_jitter_kernel(  IN const T * pData, int nSamples, int nVisJ, int nVisI,
		int nVisJ2, int nVisI2, int nVisChs, IN const T* dJ, IN const T* dI, OUT T * pJitter )
{
	const uint64_t deployed_threads = blockDim.x*gridDim.x;
	const uint64_t start = blockIdx.x*blockDim.x + threadIdx.x;

#define jitter_at_big(ptr,a,b,c,d) ( *(ptr+(a)+(b)*nSamples+(c)*nSamples*nVisJ+(d)*nSamples*nVisJ*nVisI) )
#define jitter_at_small(ptr,a,b,c,d) ( *(ptr+(a)+(b)*nSamples+(c)*nSamples*nVisJ2+(d)*nSamples*nVisJ2*nVisI2) )

	for (uint64_t ind = start; ind < nSamples*nVisJ2*nVisI2*nVisChs; ind += deployed_threads)
	{
		const int small_i = (ind / (nSamples*nVisJ2)) % nVisI2;
		const int small_j = (ind / nSamples) % nVisJ2;
		const int n = ind % nSamples;
		const int c = (ind / (nSamples*nVisJ2*nVisI2)) % nVisChs;
		const int big_i = small_i + dI[n];
		const int big_j = small_j + dJ[n];

		if (big_i >= 0 && big_i < nVisI && big_j >= 0 && big_j < nVisJ){
			jitter_at_small(pJitter, n, small_j, small_i, c) =
					jitter_at_big(pData, n, big_j, big_i, c);
		}

	} // for
}







/*
 * nVisJ2 and nVisI2 is smaller than or equal nVisJ and nVisI
 pData is: [ nSamples; nVisJ*nVisI*nVisChs ]

 layout is 1D in x, covering nSamples, nVisJ2, nVisI2, nVisChs;

 pCoords - nSamples by nVisJ2 by nVisI2 by 2: coordinate points (j,i)
 pJitter - [ nSamples; nVisJ2*nVisI2*nVischs ]

 */
template<typename T>
__global__ void image_bilinear_sample_kernel(  IN const T * pData, int nSamples,
		int nVisJ, int nVisI, int nVisJ2, int nVisI2, int nVisChs,
		IN const float* pCoords, T val_oob, OUT T * pJitter )
{
	const uint64_t total_sites =  nSamples*nVisJ2*nVisI2*nVisChs;
	const uint64_t deployed_threads = blockDim.x*gridDim.x;
	const uint64_t start = blockIdx.x*blockDim.x + threadIdx.x;

#define jitter_at_big(ptr,a,b,c,d) ( *(ptr+(a)+(b)*nSamples+(c)*nSamples*nVisJ+(d)*nSamples*nVisJ*nVisI) )
#define jitter_at_small(ptr,a,b,c,d) ( *(ptr+(a)+(b)*nSamples+(c)*nSamples*nVisJ2+(d)*nSamples*nVisJ2*nVisI2) )

	for (uint64_t ind = start; ind < total_sites; ind += deployed_threads)
	{
		const int small_i = (ind / (nSamples*nVisJ2)) % nVisI2;
		const int small_j = (ind / nSamples) % nVisJ2;
		const int n = ind % nSamples;
		const int c = (ind / (nSamples*nVisJ2*nVisI2)) % nVisChs;

		const float x = pCoords[n+small_j*nSamples+small_i*nSamples*nVisJ2];
		const float y = pCoords[n+small_j*nSamples+small_i*nSamples*nVisJ2+1*nSamples*nVisJ2*nVisI2];

		const int nWidth = nVisJ;
		const int nHeight = nVisI;
		T val;

		//if out of bound, return specified values
		if ( x >= nWidth || y >= nHeight || x < 0 || y < 0){
			val = val_oob;
		}else{

			//now we know that x,y is inside the image, maybe on the borders
			//now find the topleft of the four points
			float x_shifted, y_shifted;
			x_shifted = x - 0.5f;
			y_shifted = y - 0.5f;

			int j, i;
			j = int(floorf(x_shifted));
			i = int(floorf(y_shifted));
			//below is wrong, as a x_shifted == -0.3 will get j == 0 instead of -1
			//j = (int) (x_shifted);	//truncate and find top left index
			//i = (int) (y_shifted);	//truncate

			//KEYPOINT: the subtraction of 0.5 is critical, since the center of each "pixel/data point"
			//is shifted to (0.5,0.5), (0.5,1.5) ...

			//now check to see if our keypoint is on the border
			//on the corners, can't interpolate
			if ( (j < 0 || j == nWidth-1 ) && (i < 0 || i == nHeight-1)){

				//return the value NOT shifted by 0.5
				val = jitter_at_big(pData, n, int(x), int(y), c);

			}else if ( j < 0 || j == nWidth - 1)
			{
				//means the top value is along the two sides; so interpolate updown
				int i0 = int(y_shifted);
				int i1 = i0+1;
				//assert(int(x) < nWidth);
				//int(x) here because 0<=x<0.5 || nWidth-1+0.5<=x <= nWidth
				T i0_val = jitter_at_big(pData, n, int(x), i0, c);
				T i1_val = jitter_at_big(pData, n, int(x), i1, c);

				val = ((y_shifted) - int(y_shifted))*(i1_val - i0_val) + i0_val;

			}else if( i < 0 || i == nHeight - 1 )
			{
				//means the top value is along the two sides; so interpolate leftright
				int j0 = int(x_shifted);
				int j1 = j0+1;
				//int(y) here because 0<=y<0.5 || nHeight-1+0.5<=y <= nHeight
				T j0_val = jitter_at_big(pData, n, j0, int(y), c);
				T j1_val = jitter_at_big(pData, n, j1, int(y), c);
				val = ((x_shifted) - int(x_shifted))*(j1_val - j0_val) + j0_val;

			}else{

				//now we can interpolate worry free since we know that x and y will have
				//all 4 nearest neighbors to interpolate from
				int i0, i1, j0, j1;

				i0 = int(floor(y_shifted));
				j0 = int(floor(x_shifted));
				i0 = int( y_shifted );   //floor not needed as we assume y_shifted is positive
				j0 = int( x_shifted );   //floor not needed as we assume x_shifted is positive
				i1 = i0 + 1;
				j1 = j0 + 1;

				//JUST TO DOUBLE CHECK FOR DEBUGGING PURPOSES
				//				assert( i0 >= 0 && i0 < nHeight &&
				//						i1 >= 0 && i1 < nHeight &&
				//						j0 >= 0 && j0 < nWidth &&
				//						j1 >= 0 && j1 < nWidth);

				T a00, a01, a10, a11, a0001, a1011;
				a00 = jitter_at_big(pData, n, j0, i0, c);
				a01 = jitter_at_big(pData, n, j1, i0, c);
				a10 = jitter_at_big(pData, n, j0, i1, c);
				a11 = jitter_at_big(pData, n, j1, i1, c);

				a0001 = (x_shifted - j0)*(a01-a00) + a00;
				a1011 = (x_shifted - j0)*(a11-a10) + a10;

				val = (y_shifted - i0)*(a1011 - a0001) + a0001;
			}
		}// if

		jitter_at_small(pJitter, n, small_j, small_i, c) = val;

	} // for
}

/*
 * pTransform - nSamples by 4
 * pCoord - nSamples by nVisJ2*nVisI2*2
 *
 * layout is 1D - nSamples by nVisJ2 by nVisI2
 */
__global__ void transform_to_coords_kernel(  IN const float * pTransform, int nSamples,
		int nVisJ2, int nVisI2, OUT float* pCoords )
{
	const uint64_t total_sites = nSamples*nVisJ2*nVisI2;
	const uint64_t deployed_threads = blockDim.x*gridDim.x;
	const uint64_t start = blockIdx.x*blockDim.x + threadIdx.x;

#define at_coords(ptr,a,b,c,d) ( *(ptr+(a)+(b)*nSamples+(c)*nSamples*nVisJ2+(d)*nSamples*nVisJ2*nVisI2) )

	float half_j = float(nVisJ2)/2-0.5f;
	float half_i = float(nVisI2)/2-0.5f;

	for (uint64_t ind = start; ind < total_sites; ind += deployed_threads)
	{
		const int small_i = (ind / (nSamples*nVisJ2)) % nVisI2;
		const int small_j = (ind / nSamples) % nVisJ2;
		const int n = ind % nSamples;

		float dx = 			pTransform[n+0*nSamples];
		float dy = 			pTransform[n+1*nSamples];
		float orientation = pTransform[n+2*nSamples];
		float s = 			pTransform[n+3*nSamples];

		float pgridx = float(small_j) - half_j;
		float pgridy = float(small_i) - half_i;

		//first we rotate the Grid location,
		//then we add the offset
		float x_location = cosf(orientation)*pgridx - sinf(orientation)*pgridy*-1;
		float y_location = -(sinf(orientation)*pgridx + cosf(orientation)*pgridy*-1);

		//note that the top is actually negative y,because our image is in the 4th quadrant
		//however this has no effect as DataInterp2D is also sampled in the 4th quadrant

		at_coords(pCoords, n, small_j, small_i, 0) = s*x_location+dx;
		at_coords(pCoords, n, small_j, small_i, 1) = s*y_location+dy;

		//hack to make it same as jitters
		//at_coords(pCoords, n, small_j, small_i, 0) = round(s*x_location+dx)-0.5;
		//at_coords(pCoords, n, small_j, small_i, 1) = round(s*y_location+dy)-0.5;
	}

}


template <typename T>
int clLayerConvJitter<T>::validate(){

	clASSERT( this->vppl.size()==1, "vppl.size()==1");
	clLayer<T>* p_prevlayer = this->vppl[0];

	if (!(p_prevlayer->nHidNodes == nVisChannels*nVisI*nVisJ))
		return -1;
	if ( !p_prevlayer->bSamplesLeadDim )
		return -12;
	if (!(p_prevlayer->nSamples == this->nSamples ))
		return -3;
	if (!(this->nParams == 0))
		return -9;
	if (nVisI2 > nVisI || nVisJ2 > nVisJ)
		return -10;

	if (!(this->nodes.nI*this->nodes.nJ > 0))
		return -11;

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

	}else{
		return -115;
	}

	return 0;
}


template <typename T>
int clLayerConvJitter<T>::forward( clMatrix<T>& randnums, bool bLearning ){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if ( this->nNeuronType == 5 && !this->bSamplesLeadDim)
		return -1;
	if (!p_prevlayer->bSamplesLeadDim)
		return -2;

	cr( cuda_clMatSetRand( shift_j ) )
	cr( cuda_clMatSetRand( shift_i ) )
	cr(EleWisefun(shift_j, fctrMul<T>(), T(nVisJ-nVisJ2+1-1e-5), shift_j ))
	cr(EleWisefun(shift_i, fctrMul<T>(), T(nVisI-nVisI2+1-1e-5), shift_i ))

	const uint64_t total_sites = this->nSamples*nVisJ2*nVisI2*nVisChannels;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (total_sites+dim_block.x-1)/dim_block.x ) );

	image_jitter_kernel<<<dim_grid, dim_block>>>( p_prevlayer->nodes.pData, this->nSamples,
			nVisJ, nVisI, nVisJ2, nVisI2, nVisChannels, shift_j.pData, shift_i.pData, this->nodes.pData);

	cr (Bsxfun( this->nodes, fctrMinus<T>(), datamean, this->nodes ))
	cr (Bsxfun( this->nodes, fctrDiv<T>(), datastd, this->nodes ))

	//PrintfInMatlab(this->nodes);

	return 0;
}



template class clLayerConvJitter<float>;
template class clLayerConvJitter<double>;


template <typename T>
int clLayerConvXYRS<T>::forward( clMatrix<T>& randnums, bool bLearning ){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if ( this->nNeuronType == 5 && !this->bSamplesLeadDim)
		return -1;
	if (!p_prevlayer->bSamplesLeadDim)
		return -2;

	//set random transformations
	cr( cuda_clMatSetRand( Transforms ) )
	cr( Bsxfun( Transforms, fctrMul<T>(), transform_range, Transforms ))
	cr( Bsxfun( Transforms, fctrPlus<T>(), transform_low, Transforms ))


	//set the coordinates
	{
		const uint64_t total_sites = this->nSamples*nVisJ2*nVisI2;
		dim3 dim_block( MEDIUM_NUM_THREADS );
		dim3 dim_grid( MIN( MAX_GRIDS, (total_sites+dim_block.x-1)/dim_block.x ) );
		transform_to_coords_kernel<<<dim_grid, dim_block>>>(Transforms.pData, this->nSamples,
				nVisJ2, nVisI2, Coords.pData);
	}

	//PrintfInMatlab(Coords);

	{
		const uint64_t total_sites = this->nSamples*nVisJ2*nVisI2*nVisChannels;
		dim3 dim_block( MEDIUM_NUM_THREADS );
		dim3 dim_grid( MIN( MAX_GRIDS, (total_sites+dim_block.x-1)/dim_block.x ) );

		image_bilinear_sample_kernel<<<dim_grid, dim_block>>>( p_prevlayer->nodes.pData,
				this->nSamples,	nVisJ, nVisI, nVisJ2, nVisI2, nVisChannels,
				Coords.pData, T(0) /*oob*/, this->nodes.pData);
	}

	//PrintfInMatlab(this->nodes);
	return 0;
}


template <typename T>
int clLayerConvXYRS<T>::validate(){

	clASSERT( this->vppl.size()==1, "vppl.size()==1");
	clLayer<T>* p_prevlayer = this->vppl[0];

	if (!(p_prevlayer->nHidNodes == nVisChannels*nVisI*nVisJ))
		return -1;
	if ( !p_prevlayer->bSamplesLeadDim )
		return -12;
	if (!(p_prevlayer->nSamples == this->nSamples ))
		return -3;
	if (!(this->nParams == 0))
		return -9;
	if (nVisI2 > nVisI || nVisJ2 > nVisJ)
		return -10;

	if (!(this->nodes.nI*this->nodes.nJ > 0))
		return -11;

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
	}
	else if (p_prevlayer->name.compare("imagemirror") == 0){

		const clLayerImageMirror<T> * prevlayer2 = static_cast< const clLayerImageMirror<T> * >( p_prevlayer );

		if (prevlayer2->nVisChannels != nVisChannels)
			return -113;
		if (prevlayer2->nVisI != nVisI || prevlayer2->nVisJ != nVisJ)
			return -114;

	}else{
		return -115;
	}

	return 0;
}


template class clLayerConvXYRS<float>;
template class clLayerConvXYRS<double>;




/*

 pData is: [ nSamples; nVisJ*nVisI*nVisChs ]

 layout is 1D in x, covering nSamples, nVisJ, nVisI, nVisChs;

 flip - nSamples by 1  [0 or 1]

 pJitter - [ nSamples; nVisJ*nVisI*nVischs ]

 */
template<typename T>
__global__ void image_mirror_kernel(  IN const T * pData, int nSamples, int nVisJ, int nVisI,
		int nVisChs, IN const T* flip, OUT T * pJitter )
{
	const uint64_t total_sites =  nSamples*nVisJ*nVisI*nVisChs;
	const uint64_t deployed_threads = blockDim.x*gridDim.x;
	const uint64_t start = blockIdx.x*blockDim.x + threadIdx.x;

#define jitter_at_big(ptr,a,b,c,d) ( *(ptr+(a)+(b)*nSamples+(c)*nSamples*nVisJ+(d)*nSamples*nVisJ*nVisI) )

	for (uint64_t ind = start; ind < total_sites; ind += deployed_threads)
	{
		const int i = (ind / (nSamples*nVisJ)) % nVisI;
		const int j = (ind / nSamples) % nVisJ;
		const int n = ind % nSamples;
		const int c = (ind / (nSamples*nVisJ*nVisI)) % nVisChs;

		if ( flip[n] == 0){
			jitter_at_big(pJitter, n, j, i, c) = jitter_at_big(pData, n, j, i, c);
		}else{
			jitter_at_big(pJitter, n, j, i, c) = jitter_at_big(pData, n, nVisJ-1-j, i, c);
		}

	} // for
}



template <typename T>
int clLayerImageMirror<T>::validate(){

	clASSERT( this->vppl.size()==1, "vppl.size()==1");
	clLayer<T>* p_prevlayer = this->vppl[0];

	if (!(p_prevlayer->nHidNodes == nVisChannels*nVisI*nVisJ))
		return -1;
	if ( !p_prevlayer->bSamplesLeadDim )
		return -12;
	if (!(p_prevlayer->nSamples == this->nSamples ))
		return -3;
	if (!(this->nParams == 0))
		return -9;

	if (!(this->nodes.nI*this->nodes.nJ > 0))
		return -11;

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

	}else{
		return -115;
	}

	return 0;
}


template <typename T>
int clLayerImageMirror<T>::forward( clMatrix<T>& randnums, bool bLearning ){

	clLayer<T>* p_prevlayer = this->vppl[0];

	if ( this->nNeuronType == 5 && !this->bSamplesLeadDim)
		return -1;
	if (!p_prevlayer->bSamplesLeadDim)
		return -2;

	cr( cuda_clMatSetRand( flip ) )
	cr(EleWisefun(flip, fctrGreaterOrEqualThan<T>(), T(0.5), flip ))

	const uint64_t total_sites = this->nSamples*nVisJ*nVisI*nVisChannels;
	dim3 dim_block( MEDIUM_NUM_THREADS );
	dim3 dim_grid( MIN( MAX_GRIDS, (total_sites+dim_block.x-1)/dim_block.x ) );

	image_mirror_kernel<<<dim_grid, dim_block>>>( p_prevlayer->nodes.pData, this->nSamples,
			nVisJ, nVisI, nVisChannels, flip.pData, this->nodes.pData);

	//PrintfInMatlab(this->nodes);

	return 0;
}



template class clLayerImageMirror<float>;
template class clLayerImageMirror<double>;

