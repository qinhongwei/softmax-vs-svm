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

#ifndef _CONV_UTIL_CL_CU_
#define _CONV_UTIL_CL_CU_

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif



template <typename T>
class AvgPooler {
public:
    __device__ inline T operator()(const T a, const T b) const {
        return a + b;
    }
    __device__ inline T getBaseValue() const {
        return 0;
    }
    __device__ inline T output(const T a, const int regionSize) const {
        return a / regionSize;
    }
};

template <typename T>
class MaxPooler {
public:
    __device__ inline T operator()(const T a, const T b) const {
        return fmaxf(a, b);
    }
    __device__ inline T getBaseValue() const {
        return -2e38;
    }
    __device__ inline T output(const T a, const int regionSize) const {
        return a;
    }
};

template <>
class MaxPooler<double>{
public:
    __device__ inline double operator()(const double a, const double b) const {
        return fmax(a, b);
    }
    __device__ inline double getBaseValue() const {
        return -2e38;
    }
    __device__ inline double output(const double a, const int regionSize) const {
        return a;
    }
};

template <typename T>
class MaxAbsPooler {
public:
    __device__ inline T operator()(const T a, const T b) const {
        return fabsf(a) > fabsf(b) ? a : b;
    }
    __device__ inline T getBaseValue() const {
        return 0.0;
    }
    __device__ inline T output(const T a, const int regionSize) const {
        return a;
    }
};

template <>
class MaxAbsPooler<double> {
public:
    __device__ inline double operator()(const double a, const double b) const {
        return fabs(a) > fabs(b) ? a : b;
    }
    __device__ inline double getBaseValue() const {
        return 0.0;
    }
    __device__ inline double output(const double a, const int regionSize) const {
        return a;
    }
};


/*
 * Block size B_YxB_X
 * blockIdx.x determines output.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines output.y, filter idx in batches of B_Y*filtersPerThread
 *
 * So each block does one output for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 */

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds, typename T>
__global__ void kLocalPool(T* imgs, T* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX, const int strideX,
                           const int outputsX, Agg agg) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = DIVUP(numFilters, B_Y*filtersPerThread);
    const int outputIdxX = blockIdx.x / numImgBlocks;
    const int outputIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
    const int myFilterIdx = (blockFilterIdx + threadIdx.y*filtersPerThread);
    if (myFilterIdx >= numFilters) {
        return;
    }

    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;

    imgs += myFilterIdx * imgPixels * numImages + imgIdx;
    target += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;

    T prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue();
        }
    }

    const int loopStartY = MAX(0, startImgPxY);
    const int loopStartX = MAX(0, startImgPxX);
    const int loopEndY = MIN(imgSize, startImgPxY + subsX);
    const int loopEndX = MIN(imgSize, startImgPxX + subsX);
    const int regionSize = (loopEndY - loopStartY) * (loopEndX - loopStartX);
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        prod[f][i] = agg(prod[f][i], imgs[(f * imgPixels + imgPx) * numImages + i * B_X]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize);
            }
        }
    }
}


/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 *
 * So each block does a 4x4 region for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 *
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 *
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 *
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 *
 * To be used when the stride is 1 and the pooling region is fairly large.
 */
template<class Agg, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds, typename T>
__global__ void kLocalPool2(T* imgs, T* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX,
                           const int outputsX, Agg agg) {
    __shared__ T shImgs[filtersPerThread][B_X*imgsPerThread];
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockOutputX = 4*(blockIdx.x / numImgBlocks);
    const int blockOutputY = 4*(blockIdx.y / numFilterBlocks);
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;

//    const int blockOutputIdx = blockOutputY * outputsX + blockOutputX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    const int myX = threadIdx.y % 4;
    const int myY = threadIdx.y / 4;

    const int myOutputIdxY = blockOutputY + myY;
    const int myOutputIdxX = blockOutputX + myX;
    const int myOutputIdx = myOutputIdxY * outputsX + myOutputIdxX;

    const int startImgPxX = startX + blockOutputX;
    const int startImgPxY = startX + blockOutputY;
    const int endImgPxX = startImgPxX + subsX;
    const int endImgPxY = startImgPxY + subsX;

    const int myStartImgPxY = startImgPxY + myY;
    const int myStartImgPxX = startImgPxX + myX;
    const int myEndImgPxY = endImgPxY + myY;
    const int myEndImgPxX = endImgPxX + myX;

    const int loopStartY = MAX(startImgPxY, 0);
    const int loopStartX = MAX(startImgPxX, 0);
    const int loopEndY = MIN(imgSize, endImgPxY + 3);
    const int loopEndX = MIN(imgSize, endImgPxX + 3);

    const int imgIdx = blockImgIdx + threadIdx.x;

    imgs += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    target += (blockFilterIdx * numOutputs + myOutputIdx) * numImages + imgIdx;

    T prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue();
        }
    }
    int regionSize = 0;
    for (int y = loopStartY; y < loopEndY; y++) {
        const bool isInY = y >= myStartImgPxY && y < myEndImgPxY ;
        for (int x = loopStartX; x < loopEndX; x++) {
            // Load a pixel
            const int px = y * imgSize + x;
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shImgs[ly + loadY][lx + loadX] = imgs[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();

            // Is this pixel in my region?
            if (isInY && x >= myStartImgPxX && x < myEndImgPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] = agg(prod[f][i], shImgs[f][threadIdx.x + i * B_X]);
                        }
                    }
                }
                ++regionSize;
            }
            __syncthreads();

        }
    }
    if (myOutputIdxY < outputsX && myOutputIdxX < outputsX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize);
                }
            }
        }
    }
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, outputs, numImages)
 */
template<class Pooler, typename T>
void convLocalPool(const clMatrix<T>& images, clMatrix<T>& target, int numFilters,
                   int subsX, int startX, int strideX, int outputsX, Pooler pooler) {

    int numImages = images.nI;
    int imgPixels = images.nJ / numFilters;
    clASSERT(images.nJ == numFilters * imgPixels, "convLocalPool error1");
    int imgSize = int(sqrt(imgPixels));
    clASSERT(imgSize * imgSize == imgPixels, "convLocalPool error2");

    //clASSERT(!images.isTrans());
    //clASSERT(!target.isTrans());
    //clASSERT(images.isContiguous());
//    clASSERT(numFilters % 4 == 0);
//    clASSERT(numImages % 128 == 0);

    int outputs = outputsX * outputsX;
    //target.resize(numFilters*outputs, numImages);
    clASSERT(target.nI == numImages && target.nJ == numFilters*outputs, "target size3");

    if (strideX == 1 && subsX >= 6) {
        int imgsPerThread = numImages % 128 == 0 ? 8 : 4;
        int filtersPerThread = numFilters % 4 == 0 ? 4 : numFilters % 3 == 0 ? 3 : numFilters % 2 == 0 ? 2 : 1;
        int bx = 8;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        clASSERT((imgsPerThread * bx) % 32 == 0, "convLocalPool error4");
        clASSERT(numFilters % filtersPerThread == 0, "convLocalPool error5");
        dim3 threads(bx, 16);
        dim3 blocks(DIVUP(outputsX, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(outputsX, 4) * numFilters / filtersPerThread);
        if (imgsPerThread == 8) {
            if (filtersPerThread == 1) {
                 if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, true, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 1, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, false, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 1, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, true, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 2, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, false, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 2, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, true, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 3, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, false, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 3, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, true, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 4, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, false, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 4, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            }
        } else if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
                 if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, true, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 1, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, false, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 1, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, true, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 2, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, false, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 2, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, true, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 3, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, false, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 3, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, true, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 4, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, false, T>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 4, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            }
        }
    } else {

        int filtersPerThread = numFilters % 8 == 0 ? 2 : 1;
        int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
        bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
        dim3 threads(32, 4);
        dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numFilters, 4 * filtersPerThread) * outputsX);
        if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, true, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 1, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, false, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 1, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 2, true, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 2, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 2, false, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 2, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        } else if (imgsPerThread == 2) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, true, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 1, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, false, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 1, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 2, true, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 2, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 2, false, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 2, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        } else {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, true, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 1, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, false, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 1, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 2, true, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 2, true, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 2, false, T>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 2, false, T><<<blocks, threads>>>(images.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        }

    }

    getLastCudaError("convLocalPool: kernel execution failed");
}





/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 *
 * So each block does one output pixel for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 *
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds, typename T>
__global__ void kLocalAvgUndo(T* avgGrads, T* target, const int imgSize, const int numFilters,
                              const int numImages, const int subsX, const int startX, const int strideX, const int outputsX,
                              const T scaleTargets, const T scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));

    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;

    const int blockPx = blockPxY * imgSize + blockPxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);

    const int imgIdx = blockImgIdx + threadIdx.x;

    avgGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages + imgIdx;
    target += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;

    T prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }

    if (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX
            && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX) {

        for (int my = startOutputY; my < endOutputY; my++) {
            const T regionStartY = fmaxf(0, startX + my * strideX);
            const T regionEndY = fminf(imgSize, startX + my * strideX + subsX);
            const T regionSizeY = regionEndY - regionStartY;
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                const T regionStartX = fmaxf(0, startX + mx * strideX);
                const T regionEndX = fminf(imgSize, startX + mx * strideX + subsX);
                const T regionSizeX = regionEndX - regionStartX;
                // It's important to do the division here, because pushing division into the below
                // loops makes the code 4x slower.
                const T regionSizeInv = 1.0f / (regionSizeX * regionSizeY);
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] += avgGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X] * regionSizeInv;
                        }
                    }
                }
            }
        }
    }

    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
                }
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 *
 * So each block does one output pixel for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * maxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 *
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds, typename T>
__global__ void kLocalMaxUndo(T* imgs, T* maxGrads, T* maxActs, T* target, const int imgSize, const int numFilters,
                              const int numImages, const int subsX, const int startX, const int strideX, const int outputsX,
                              const T scaleTargets, const T scaleOutputs) {
    __shared__ T shImgs[B_Y*filtersPerThread][B_X*imgsPerThread];
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));

    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;

    const int blockPx = blockPxY * imgSize + blockPxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);

    const int imgIdx = blockImgIdx + threadIdx.x;

    imgs += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    maxGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages
            + imgIdx;
    maxActs += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages
            + imgIdx;

    target += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;

    T prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }

    if  (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX
         && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i] = imgs[f * B_Y * imgPixels * numImages + i * B_X];
                }
            }
        }
        for (int my = startOutputY; my < endOutputY; my++) {
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            const T ma = maxActs[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                            const T mg = maxGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                            const T img = shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i];

                            prod[f][i] += (img == ma) * mg;
                        }
                    }
                }
            }
        }
    }
    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
                }
            }
        }
    }
}


/*
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * MaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
template<typename T>
void convLocalMaxUndo(clMatrix<T>& images, clMatrix<T>& maxGrads, clMatrix<T>& maxActs, clMatrix<T>& target,
                      int subsX, int startX, int strideX, int outputsX, T scaleTargets, T scaleOutput) {
    int outputs = outputsX * outputsX;
    int numImages = images.nI;
    int numFilters = maxGrads.nJ / outputs;
    int imgPixels = images.nJ / numFilters;
    clASSERT(images.nJ == numFilters * imgPixels, "convLocalMaxUndo error1");
    int imgSize = int(sqrt(imgPixels));

    clASSERT(imgSize * imgSize == imgPixels, "convLocalMaxUndo error2");
    clASSERT(maxGrads.nJ == numFilters * outputs, "convLocalMaxUndo error3");
    clASSERT(maxGrads.nI == numImages, "convLocalMaxUndo error4");
    //clASSERT(!images.isTrans());
    //clASSERT(!target.isTrans());
    //clASSERT(!maxGrads.isTrans());
    //clASSERT(!maxActs.isTrans());
    //clASSERT(images.isContiguous());
    //clASSERT(maxGrads.isContiguous());
    //clASSERT(maxActs.isContiguous());
    clASSERT(maxGrads.nI == maxActs.nI && maxGrads.nJ == maxActs.nJ, "convLocalMaxUndo error5");
    clASSERT(numFilters % 16 == 0, "convLocalAvgUndo error6");
//    clASSERT(numImages % 128 == 0);

    clASSERT(strideX <= subsX, "convLocalMaxUndo error7");

    //target.resize(images);
    clASSERT(target.nJ == images.nJ && target.nI == images.nI, "convLocalAvgUndo error8");
    //clASSERT(target.isContiguous());
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    int checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * imgSize, (numFilters / (4 * 2)) * imgSize);

    if (imgsPerThread == 4) {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 4, 2, false, true, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 4, 2, true, true, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 4, 2, false, false, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 4, 2, true, false, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        }
    } else if (imgsPerThread == 2) {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 2, 2, false, true, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 2, 2, true, true, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 2, 2, false, false, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 2, 2, true, false, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        }
    } else {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 1, 2, false, true, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 1, 2, true, true, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 1, 2, false, false, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 1, 2, true, false, T><<<blocks, threads>>>(images.pData, maxGrads.pData, maxActs.pData, target.pData,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        }
    }

    getLastCudaError("convLocalMaxUndo: kernel execution failed");
}

/*
 * avgGrads:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
template<typename T>
void convLocalAvgUndo(clMatrix<T>& avgGrads, clMatrix<T>& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize,
                      T scaleTargets, T scaleOutput) {
    int numImages = avgGrads.nI;

    int outputs = outputsX * outputsX;
    int imgPixels = imgSize * imgSize;
    int numFilters = avgGrads.nJ / outputs;
    clASSERT(avgGrads.nJ == numFilters * outputs, "convLocalAvgUndo error1");

   // clASSERT(!target.isTrans());
    //clASSERT(!avgGrads.isTrans());
    //clASSERT(avgGrads.isContiguous());
    clASSERT(numFilters % 16 == 0, "convLocalAvgUndo error2");
//    clASSERT(numImages % 128 == 0);

    clASSERT(strideX <= subsX, "convLocalAvgUndo error3");

    //target.resize(numFilters * imgPixels, numImages);
    clASSERT(target.nI == numImages && target.nJ == numFilters*imgPixels, "convLocalAvgUndo error4");
    //clASSERT(target.isContiguous());
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    int checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * imgSize, (numFilters / (4 * 4)) * imgSize);

    if (imgsPerThread == 4) {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 4, 4, false, true, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 4, 4, true, true, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 4, 4, false, false, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 4, 4, true, false, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        }
    } else if (imgsPerThread == 2) {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 2, 4, false, true, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 2, 4, true, true, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 2, 4, false, false, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 2, 4, true, false, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        }
    } else {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 1, 4, false, true, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 1, 4, true, true, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 1, 4, false, false, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 1, 4, true, false, T><<<blocks, threads>>>(avgGrads.pData, target.pData,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        }
    }

    getLastCudaError("convLocalAvgUndo: kernel execution failed");
}



#endif
