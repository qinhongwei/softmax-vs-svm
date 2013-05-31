/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//#include <cutil_inline.h>
//#include <nvmatrix.cuh>
//#include <cudaconv2.cuh>


#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of module and B_Y * filtersPerThread
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numColors, filterPixels, numFilters) if conv
 *              (numModules, numColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * B_Y one of 4, 8, 16; 		filtersPerThread one of 1, 2, 4, 8
 * B_X one of 16, 32; 			imgsPerThread one of 1, 2, 4
 *
 *
 *
 * Number of filters per module should be divisible by B_Y * filtersPerThread
 * checkImgBounds indicates whether number of images is divisible by B_X * imgsPerThread
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors,
          bool scale, bool checkImgBounds, typename T>
__global__ void filterActs_YxX_color(T* images, T* filters, T* targets,
                                   const int numImages, const int numFilters,
                                   const int imgSizeY, const int imgSizeX, const int filterSize,
                                   const int paddingStart, const int moduleStride,
                                   const int numModulesY, const int numModulesX, const int imgStride,
                                   const T scaleTargets, const T scaleOutputs,
                                   const bool conv)
{
    __shared__ T shFilters[B_Y*numColors][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ T shImages[B_Y*numColors][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;

    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);  //how many blocks are dedicated to each module(grid point)
    const int moduleIdx = blockIdx.y / blocksPerModule;			//which module (grid point)
    const int blockFilterIdx = blockIdx.y % blocksPerModule;	//which block within a module (which can have multiple blocks)

    const int tidx = threadIdx.y * B_X + threadIdx.x;	//row-major scan within each block

    const int imgLoadModPosY = (moduleIdx / numModulesX) * moduleStride; //row-major scan of modules
    const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);			//weird re-use of threads on shared memory loading (max # of threads is 128)
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);			//maybe useful if # of threads > B_Y*filtersPerThread
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
    images += myImgIdx;
    filters += filtersPerThread * B_Y * blockFilterIdx + shFilterLoadY * numFilters + shFilterLoadX;

    if (!conv) {
        filters += moduleIdx * numColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y) * numImages * numModulesY * numModulesX
            + myImgIdx;


    T prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }

    for (int p = 0; p < filterPixels; p += B_Y) {

        //Load B_Y pixels from B_Y*filtersPerThread filters
        if (shFilterLoadY < B_Y) {
            #pragma unroll
            for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                if (p + p2 + shFilterLoadY < filterPixels) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[(c * filterPixels + p + p2) * numFilters];
                    }
                } else {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
                    }
                }
            }
        }

        // Load B_Y pixels from B_X*imgsPerThread images
        const int pixIdx = p + threadIdx.y;
        if (pixIdx < filterPixels) {
            const int x = paddingStart + imgLoadModPosX + pixIdx % filterSize;
            const int y = paddingStart + imgLoadModPosY + pixIdx / filterSize;
            if (y >= 0 && y< imgSizeY && x >= 0 && x < imgSizeX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = images[imgStride * (c * imgPixels + y * imgSizeX + x) + i * B_X];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            } else { // Padding
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < B_Y*numColors; i++) {
            #pragma unroll
            for(int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for(int g = 0; g < imgsPerThread; g++) {
                    prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                }
            }

        }
        __syncthreads();
    }
    
    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] = scaleOutputs * prod[f][g];
                }
            }
        }
    }
}





/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of B_Y * filtersPerThread
 *
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 * colorCache: how many colors to put into shmem
 *
 * numFilters should be divisible by B_Y * filtersPerThread
 * numImages be divisible by B_X * imgsPerThread
 * numFilterColors should be divisible by colorCache.
 * numImgColors must be even.
 * numFilters must be divisible by numGroups.
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds, typename T>
__global__ void filterActs_YxX_sparse(T* images, T* filters, T* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups,
                                       const T scaleTargets, const T scaleOutputs,
                                       const bool conv) {
    __shared__ T shFilters[B_Y*colorCache][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ T shImages[B_Y*colorCache][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += blockColorIdx * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y) * numImages * numModules
            + myImgIdx;

    T prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }
//    __shared__ int imgPos[]
    for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
        for (int p = 0; p < filterPixels; p += B_Y) {
            /*
             * Load B_Y pixels from B_Y*filtersPerThread filters
             */
            if (shFilterLoadY < B_Y) {
                #pragma unroll
                for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                    if (p + p2 + shFilterLoadY < filterPixels) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[((oc+c) * filterPixels + p + p2) * numFilters];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
                        }
                    }
                }
            }

            /*
             * Load B_Y pixels from B_X*imgsPerThread images
             */
            const int pixIdx = p + threadIdx.y;
            if (pixIdx < filterPixels) {
                const int x = imgLoadModPosX + pixIdx % filterSize;
                const int y = imgLoadModPosY + pixIdx / filterSize;
                if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                    T* m = &images[imgStride * (oc * imgPixels + y * imgSizeX + x)];
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < B_Y*colorCache; i++) {
                #pragma unroll
                for(int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                    }
                }

            }
            __syncthreads();
        }
    }

    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
                }
            }
        }
    }
}




/*
 * images:      (numImgColors, imgSizeY, imgSizeX; numImages) with stride given
 * 	// images does not need to be a contiguous matrix
 *
 * filters:     (numFilterColors, filterPixels; numFilters)             if conv
 *              (numModules, numFilterColors, filterPixels; numFilters) otherwise
 *
 * targets:     (numFilters, numModules; numImages)
 * 
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128. 
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast. 
 *
 * scaleTargets, scaleOutput:
 * targets = scaleTargets*(old targets) + scaleOutput*(images conv filter)
 * conv: true for conv wt sharing, false for local filters
 */
template<typename T>
 void _filterActs( IN const clMatrix<T>& images, int imgStride, IN const clMatrix<T>& filters, OUT clMatrix<T>& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   T scaleTargets, T scaleOutput, bool conv){

    int numFilterColors = numImgColors / numGroups;
    int numFilters = filters.nI;
    int numModules = numModulesY * numModulesX;
    int numImages = images.nI;
    int imgPixels = images.nJ / numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    int filterModuleMult = conv ? 1 : numModules;
    
    clASSERT(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)),"err1001");
    clASSERT(numGroups == 1 || numFilterColors % 2 == 0,"err1002");
    clASSERT(numFilters % (16 * numGroups) == 0,"err1003");
    clASSERT(numImgColors % numGroups == 0,"err1004");
    clASSERT(images.nJ == imgPixels * numImgColors,"err1005");
    clASSERT(imgSizeY * imgSizeX == imgPixels,"err1006");
    int numFiltersPerGroup = numFilters / numGroups;

    int filterPixels = filters.nJ / (filterModuleMult * numFilterColors);
    int filterSize = int(sqrt(filterPixels));
    clASSERT(filterSize * filterSize == filterPixels,"err1007");
    clASSERT(filters.nJ == filterModuleMult * numFilterColors * filterPixels,"err1008");

    // These routines don't handle the case when only part of the image is visited in the convolution
    clASSERT(paddingStart <= 0,"err1009");
    clASSERT(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX,"err1010");
    clASSERT(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY,"err1011");
    clASSERT(moduleStride <= filterSize,"err1012");
    
    //assert(!images.isTrans());
    //assert(!filters.isTrans());
    //assert(!targets.isTrans());

    //assert(filters.isContiguous());
    //assert(targets.isContiguous());

    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    dim3 blocks = numFiltersPerGroup % 32 == 0 ? dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 8))
                                               : dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 4));
    dim3 threads(32, 4);
    bool checkImgBounds = numImages % (32*imgsPerThread) != 0;
    if (scaleTargets == 0) {
        //targets.resize(numFilters * numModules, numImages);
    	clASSERT(targets.nJ == numFilters * numModules, "err1013");
        clASSERT(targets.nI == numImages, "err1014");
    } else {
        clASSERT(targets.nJ == numFilters * numModules, "err1013");
        clASSERT(targets.nI == numImages, "err1014");
    }
    
    if (imgsPerThread == 4) {
        if (numImgColors <= 3) {
            clASSERT(numGroups == 1, "err1015"); // It has to be based on above definitions, but just to be sure.
            if (scaleTargets == 0) { // don't scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 1, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 1, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 1, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 1, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                         if (numFilters % 32 == 0) {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, false, true, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 4, 8, 3, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, false, true, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 4, 4, 3, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    } else {
                         if (numFilters % 32 == 0) {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, false, false, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 4, 8, 3, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, false, false, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 4, 4, 3, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    }
                }
            } else { // do scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 1, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 1, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 1, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 1, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 3, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 3, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 3, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 3, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }
            }
        } else {
            if (scaleTargets == 0) { // don't scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 8, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 4, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 8, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 4, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            } else { // do scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 8, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 4, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 8, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 4, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
    } else if (imgsPerThread == 2) {
        if (numImgColors <= 3) {
            clASSERT(numGroups == 1, "err1016"); // It has to be based on above definitions, but just to be sure.
            if (scaleTargets == 0) { // don't scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 1, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 1, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 1, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 1, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 1, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 1, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 1, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 1, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 2, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 2, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 2, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 2, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                         if (numFilters % 32 == 0) {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 3, false, true, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 2, 8, 3, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 3, false, true, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 2, 4, 3, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    } else {
                         if (numFilters % 32 == 0) {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 3, false, false, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 2, 8, 3, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 3, false, false, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 2, 4, 3, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    }
                }
            } else { // do scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 1, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 1, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 1, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 1, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 1, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 1, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 1, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 1, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 2, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 2, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 2, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 2, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 3, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 3, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 3, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 3, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 3, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 3, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 3, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 3, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }
            }
        } else {
            if (scaleTargets == 0) { // don't scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 8, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 8, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 4, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 4, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 8, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 8, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 4, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 4, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            } else { // do scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 8, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 8, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 4, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 4, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 8, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 8, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 4, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 4, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }    
    } else {
        if (numImgColors <= 3) {
            clASSERT(numGroups == 1, "err1017"); // It has to be based on above definitions, but just to be sure.
            if (scaleTargets == 0) { // don't scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 1, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 1, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 1, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 1, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 1, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 1, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 1, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 1, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 2, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 2, false, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 2, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 2, false, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                         if (numFilters % 32 == 0) {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 3, false, true, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 1, 8, 3, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 3, false, true, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 1, 4, 3, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    } else {
                         if (numFilters % 32 == 0) {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 3, false, false, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 1, 8, 3, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 3, false, false, T>, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 1, 4, 3, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    }
                }
            } else { // do scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 1, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 1, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 1, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 1, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 1, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 1, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 1, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 1, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 2, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 2, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 2, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 2, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 3, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 3, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 3, true, true, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 3, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 3, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 3, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 3, true, false, T>, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 3, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }
            }
        } else {
            if (scaleTargets == 0) { // don't scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 8, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 8, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 4, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 4, 2, false, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 8, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 8, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 4, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 4, 2, false, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            } else { // do scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 8, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 8, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 4, 2, false, true, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 4, 2, true, true, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 8, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 8, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 4, 2, false, false, T>, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 4, 2, true, false, T> <<<blocks, threads>>>(images.pData, filters.pData, targets.pData,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
    }
    
    getLastCudaError("filterActs: kernel execution failed");
}
