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

#ifndef _NORMALIZATION_CL_CU_
#define _NORMALIZATION_CL_CU_

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif


__device__ inline float square(const float a) {
    return a * a;
}


/*
 * Block size 1x128
 * blockIdx.x determines pixel.x, image idx in batches of 128*imgsPerThread
 * blockIdx.y determines pixel.y
 *
 * So each block does one output for some number of images and all the fliters.
 *
 * threadIdx.x determines img idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int imgsPerThread, int numFilters, bool checkCaseBounds, typename T>
__global__ void kCNorm_fewfilter(T* imgs, T* meanDiffs, T* denoms, T* target, const int imgSize,
                                  const int numImages, const int sizeX, const T addScale, const T powScale) {

    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, 128*imgsPerThread);
    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * 128 * imgsPerThread;

    const int pxIdx = pxIdxY * imgSize + pxIdxX;

    const int startPxX = -sizeX/2 + pxIdxX;
    const int startPxY = -sizeX/2 + pxIdxY;
    const int imgIdx = blockImgIdx + threadIdx.x;

    imgs += pxIdx * numImages + imgIdx;
    denoms += pxIdx * numImages + imgIdx;
    meanDiffs  += imgIdx;
    target += pxIdx * numImages + imgIdx;

    T prod[numFilters][imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
            #pragma unroll
            for (int f = 0; f < numFilters; f++) {
                prod[f][i] = 0;
            }
        }
    }
    const int loopStartY = MAX(0, startPxY);
    const int loopStartX = MAX(0, startPxX);
    const int loopEndY = MIN(imgSize, startPxY + sizeX);
    const int loopEndX = MIN(imgSize, startPxX + sizeX);

    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
                    #pragma unroll
                    for (int f = 0; f < numFilters; f++) {
                        prod[f][i] += square(meanDiffs[(f * imgPixels + imgPx) * numImages + i * 128]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
            #pragma unroll
            for (int f = 0; f < numFilters; f++) {
                prod[f][i] = 1 + addScale * prod[f][i];
                denoms[f * imgPixels * numImages + i * 128] = prod[f][i];
                target[f * imgPixels * numImages + i * 128] = imgs[f * imgPixels * numImages + i * 128] * __powf(prod[f][i], -powScale);
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 *
 * So each block does one pixel for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * means:       (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds, typename T>
__global__ void kCNorm_manyfilter(T* imgs, T* meanDiffs, T* denoms, T* target, const int imgSize,
                                  const int numFilters, const int numImages, const int sizeX,
                                  const T addScale, const T powScale) {
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(B_Y*filtersPerThread);
    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;

    const int pxIdx = pxIdxY * imgSize + pxIdxX;

    const int startPxX = -sizeX/2 + pxIdxX;
    const int startPxY = -sizeX/2 + pxIdxY;
    const int imgIdx = blockImgIdx + threadIdx.x;

    imgs += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
    meanDiffs += (blockFilterIdx + threadIdx.y) * imgPixels * numImages + imgIdx;
    denoms += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
    target += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;

    T prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                prod[f][i] = 0;
            }
        }
    }

    const int loopStartY = MAX(0, startPxY);
    const int loopStartX = MAX(0, startPxX);
    const int loopEndY = MIN(imgSize, startPxY + sizeX);
    const int loopEndX = MIN(imgSize, startPxX + sizeX);

    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        prod[f][i] += square(meanDiffs[(f * B_Y * imgPixels + imgPx) * numImages + i * B_X]);
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
                prod[f][i] = 1 + addScale * prod[f][i];
                denoms[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                target[f * B_Y * imgPixels * numImages + i * B_X] = imgs[f * B_Y * imgPixels * numImages + i * B_X] * __powf(prod[f][i], -powScale);
            }
        }
    }
}


/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 *
 * So each block does 4x4 region of pixels for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * means:       (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 *
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 *
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by filtersPerThread
 *
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 */
template<int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds, typename T>
__global__ void kCNorm2(T* imgs, T* meanDiffs, T* denoms, T* target, const int imgSize,
                         const int numFilters, const int numImages, const int sizeX, const T addScale, const T powScale) {
    __shared__ T shDiffs[filtersPerThread][B_X*imgsPerThread];
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockPxX = 4*(blockIdx.x / numImgBlocks);
    const int blockPxY = 4*(blockIdx.y / numFilterBlocks);
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;

    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    const int startPxX = MAX(0, -sizeX/2 + blockPxX);
    const int startPxY = MAX(0, -sizeX/2 + blockPxY);
    const int endPxX = MIN(imgSize, blockPxX + DIVUP(sizeX, 2) + 3);
    const int endPxY = MIN(imgSize, blockPxY + DIVUP(sizeX, 2) + 3);

    const int myPxX = blockPxX + threadIdx.y % 4;
    const int myPxY = blockPxY + threadIdx.y / 4;
    const int myPxIdx = myPxY * imgSize + myPxX;
//    const bool doWork = myPxX < imgSize && myPxY < imgSize;
    const int myStartPxY = -sizeX/2 + myPxY;
    const int myStartPxX = -sizeX/2 + myPxX;
    const int myEndPxY = myPxY + DIVUP(sizeX, 2);
    const int myEndPxX = myPxX + DIVUP(sizeX, 2);

    const int imgIdx = blockImgIdx + threadIdx.x;

    imgs        += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    meanDiffs   += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    denoms      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    target      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;

    T prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                prod[f][i] = 0;
            }
        }
    }

    for (int y = startPxY; y < endPxY; y++) {
        const bool isInY = y >= myStartPxY && y < myEndPxY;
        for (int x = startPxX; x < endPxX; x++) {
            const int px = y * imgSize + x;
            // All the threads load a pixel from memory
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shDiffs[ly + loadY][lx + loadX] = meanDiffs[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();

            // Each row of threads decides if it's interested in this pixel
            if (isInY && x >= myStartPxX && x < myEndPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] += square(shDiffs[f][threadIdx.x + i * B_X]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
//    imgs -= (loadY * imgPixels - myPxIdx) * numImages + loadX;
//    imgs += threadIdx.x;
    if (myPxX < imgSize && myPxY < imgSize) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    prod[f][i] = 1 + addScale * prod[f][i];
                    denoms[f * imgPixels * numImages + i * B_X] = prod[f][i];
                    target[f * imgPixels * numImages + i * B_X] = imgs[f * imgPixels * numImages + i * B_X] * __powf(prod[f][i], -powScale);
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
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
 *
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 *
 * TODO: this isn't really ideal
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds, typename T>
__global__ void kRNormUndo(T* outGrads, T* denoms, T* inputs, T* acts, T* target, const int imgSize, const int numFilters,
                              const int numImages, const int sizeX, const T powScale, const T scaleTargets, const T scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(B_Y*filtersPerThread);

    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / numFilterBlocks;

    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;

    const int blockPx = blockPxY * imgSize + blockPxX;
    const int imgPixels = imgSize * imgSize;

    const int startY = MAX(0, blockPxY + sizeX/2 - sizeX + 1);
    const int startX = MAX(0, blockPxX + sizeX/2 - sizeX + 1);
    const int endY = MIN(imgSize, blockPxY + sizeX/2 + 1);
    const int endX = MIN(imgSize, blockPxX + sizeX/2 + 1);

    const int imgIdx = blockImgIdx + threadIdx.x;

    acts        += ((blockFilterIdx + threadIdx.y) * imgPixels) * numImages + imgIdx;
    inputs      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    denoms      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    outGrads    += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    target      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;

    T prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }

    for (int sy = startY; sy < endY; sy++) {
        for (int sx = startX; sx < endX; sx++) {
            const int outPx = sy * imgSize + sx;

            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        prod[f][i] += acts[(f * B_Y * imgPixels + outPx) * numImages + i * B_X];
                    }
                }
            }
        }
    }
//    outGrads += blockPx * numImages;
    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    const T inp = inputs[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const T out = outGrads[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const T den = denoms[(f * B_Y * imgPixels) * numImages + i * B_X];
                    prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
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
                    const T inp = inputs[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const T out = outGrads[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const T den = denoms[(f * B_Y * imgPixels) * numImages + i * B_X];
                    prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
                    target[f * B_Y * imgPixels * numImages + i * B_X] =
                                                scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X]
                                                + scaleOutputs * prod[f][i];
                }
            }
        }
    }
}


/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 *
 * So each block does 4x4 region for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 *
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
 *
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 *
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by filtersPerThread
 *
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 */
template<int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds, typename T>
__global__ void kRNormUndo2(T* outGrads, T* denoms, T* inputs, T* acts, T* target, const int imgSize, const int numFilters,
                            const int numImages, const int sizeX, const T powScale, const T scaleTargets, const T scaleOutputs) {
    __shared__ T shActs[filtersPerThread][B_X*imgsPerThread];
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockPxX = 4*(blockIdx.x / numImgBlocks);
    const int blockPxY = 4*(blockIdx.y / numFilterBlocks);
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;

    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;

    const int startPxX = MAX(0, -DIVUP(sizeX,2) + blockPxX + 1);
    const int startPxY = MAX(0, -DIVUP(sizeX,2) + blockPxY + 1);
    const int endPxX = MIN(imgSize, blockPxX + sizeX/2 + 4);
    const int endPxY = MIN(imgSize, blockPxY + sizeX/2 + 4);

    const int myPxX = blockPxX + threadIdx.y % 4;
    const int myPxY = blockPxY + threadIdx.y / 4;
    const int myPxIdx = myPxY * imgSize + myPxX;
//    const bool doWork = myPxX < imgSize && myPxY < imgSize;
    const int myStartPxY = -DIVUP(sizeX,2) + myPxY + 1;
    const int myStartPxX = -DIVUP(sizeX,2) + myPxX + 1;
    const int myEndPxY = myPxY + sizeX/2 + 1;
    const int myEndPxX = myPxX + sizeX/2 + 1;

    const int imgIdx = blockImgIdx + threadIdx.x;

    acts        += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    denoms      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    inputs      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    outGrads    += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    target      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;

    T prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }

    for (int y = startPxY; y < endPxY; y++) {
        const bool isInY = y >= myStartPxY && y < myEndPxY;
        for (int x = startPxX; x < endPxX; x++) {
            const int px = y * imgSize + x;
            // All the threads load a pixel from memory
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shActs[ly + loadY][lx + loadX] = acts[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();

            // Each row of threads decides if it's interested in this pixel
            if (isInY && x >= myStartPxX && x < myEndPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] += shActs[f][threadIdx.x + i * B_X];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
    acts -= (loadY * imgPixels - myPxIdx) * numImages + loadX;
    acts += threadIdx.x;
    if (myPxX < imgSize && myPxY < imgSize) {
        if (!add) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        const T out = outGrads[f * imgPixels * numImages + i * B_X];
                        const T den = denoms[f * imgPixels * numImages + i * B_X];
                        const T inp = inputs[f * imgPixels * numImages + i * B_X];
                        prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
                        target[f * imgPixels * numImages + i * B_X] = prod[f][i];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        const T out = outGrads[f * imgPixels * numImages + i * B_X];
                        const T den = denoms[f * imgPixels * numImages + i * B_X];
                        const T inp = inputs[f * imgPixels * numImages + i * B_X];
                        prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
                        target[f * imgPixels * numImages + i * B_X] = scaleTargets * target[f * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
                    }
                }
            }
        }

    }
}

/*
 * acts := -2 x scale x acts x outGrads / denoms
 */
template<int B_X, int eltsPerThread, typename T>
__global__ void kRNormUndoPrelims(T* acts, T* denoms, T* outGrads,
                                  const uint numElements, const T scale) {
    const uint e = B_X * blockIdx.x * eltsPerThread + threadIdx.x;
    const uint numThreads = B_X * gridDim.x;
    for (uint i = e; i < numElements; i += numThreads*eltsPerThread) {
        #pragma unroll
        for (uint k = 0; k < eltsPerThread; k++) {
            if (i + k * B_X < numElements) {
                acts[i + k * B_X] = __fdividef(scale*outGrads[i + k * B_X] * acts[i + k * B_X], denoms[i + k * B_X]);
            }
        }
    }
}




/*
 * images:      (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 */
template<typename T>
void convContrastNorm(const clMatrix<T>& images, const clMatrix<T>& meanDiffs, clMatrix<T>& denoms,
		clMatrix<T>& target, int numFilters, int sizeX, T addScale, T powScale){

    int numImages = images.nI;
    int imgPixels = images.nJ / numFilters;
    clASSERT(images.nJ == numFilters * imgPixels, "convContrastNorm err0");
    int imgSize = int(sqrt(imgPixels));
    clASSERT(imgSize * imgSize == imgPixels, "convContrastNorm err1");
    clASSERT(meanDiffs.nI==images.nI && meanDiffs.nJ == images.nJ, "convContrastNorm err2");

    //clASSERT(!meanDiffs.isTrans());
    //clASSERT(!images.isTrans());
    //clASSERT(images.isContiguous());
    //clASSERT(meanDiffs.isContiguous());

    clASSERT(numFilters % 16 == 0 || numFilters <= 8, "convContrastNorm err20");

    //target.resize(images);
    //denoms.resize(images);
    clASSERT(target.nI==images.nI && target.nJ == images.nJ, "convContrastNorm err3");
    clASSERT(denoms.nI==images.nI && denoms.nJ == images.nJ, "convContrastNorm err4");

    //clASSERT(target.isContiguous());
    if (sizeX >= 6 && numFilters % 4 == 0) {
        // This one is faster for large regions (my tests show regions >= 6...)
        int imgsPerThread = 8;
        int filtersPerThread = 4;
        int bx = 8;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        clASSERT((imgsPerThread * bx) % 32 == 0, "convContrastNorm err5");
        clASSERT(numFilters % filtersPerThread == 0, "convContrastNorm err6");
        dim3 threads(bx, 16);
        dim3 blocks(DIVUP(imgSize, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(imgSize, 4) * numFilters / filtersPerThread);

        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kCNorm2<8, 8, 4, true, T>, cudaFuncCachePreferL1); // L1 faster here
            kCNorm2<8, 8, 4, true, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                           imgSize, numFilters, numImages, sizeX, addScale, powScale);
        } else {
            cudaFuncSetCacheConfig(kCNorm2<8, 8, 4, false, T>, cudaFuncCachePreferL1); // L1 faster here
            kCNorm2<8, 8, 4, false, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                           imgSize, numFilters, numImages, sizeX, addScale, powScale);
        }
    } else {
        bool checkCaseBounds = numImages % 128 != 0;
        if (numFilters <= 8) {
            dim3 threads(128);
            dim3 blocks(DIVUP(numImages,128) * imgSize, imgSize);
            if (numFilters == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 1, true, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 1, true, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 1, false, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 1, false, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, true, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 2, true, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, false, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 2, false, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, true, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 3, true, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, false, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 3, false, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, true, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 4, true, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, false, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 4, false, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 5) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, true, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 5, true, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, false, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 5, false, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 6) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, true, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 6, true, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, false, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 6, false, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 7) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, true, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 7, true, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, false, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 7, false, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            } else  if (numFilters == 8) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, true, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 8, true, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, false, T>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 8, false, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                      imgSize, numImages, sizeX, addScale, powScale);
                }
            }
        } else {
            dim3 threads(32, 4);
            dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 2)) * imgSize);
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, true, T>, cudaFuncCachePreferL1);
                kCNorm_manyfilter<4, 32, 4, 2, true, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                  imgSize, numFilters, numImages, sizeX, addScale, powScale);
            } else {
                cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, false, T>, cudaFuncCachePreferL1);
                kCNorm_manyfilter<4, 32, 4, 2, false, T><<<blocks, threads>>>(images.pData, meanDiffs.pData, denoms.pData, target.pData,
                                                                  imgSize, numFilters, numImages, sizeX, addScale, powScale);
            }
        }
    }
    getLastCudaError("convResponseNorm: kernel execution failed");
}


template<typename T>
void convResponseNorm(const clMatrix<T>& images, clMatrix<T>& denoms, clMatrix<T>& target, int numFilters, int sizeX, T addScale, T powScale) {
    convContrastNorm<T>(images, images, denoms, target, numFilters, sizeX, addScale, powScale);
}



/*
 * outGrads:    (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages)
 * inputs:      (numFilters, imgPixels, numImages)
 * acts:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, imgPixels, numImages)
 *
 * THIS WILL OVERWRITE THE ACTS MATRIX.
 */
template<typename T>
void convResponseNormUndo(const clMatrix<T>& outGrads, const clMatrix<T>& denoms, const clMatrix<T>& inputs, const clMatrix<T>& acts, clMatrix<T>& target, int numFilters,
                         int sizeX, T addScale, T powScale, T scaleTargets, T scaleOutput) {
    int numImages = outGrads.nI;
    int imgPixels = outGrads.nJ / numFilters;

    int imgSize = int(sqrt(imgPixels));
    clASSERT(imgSize * imgSize == imgPixels, "convResponseNormUndo err1");

    clASSERT(outGrads.nJ == numFilters * imgPixels, "convResponseNormUndo err2");

    //clASSERT(denoms.isSameDims(outGrads));
    //clASSERT(acts.isSameDims(denoms));

    clASSERT(outGrads.nI==denoms.nI && outGrads.nJ == denoms.nJ, "convContrastNorm err3");
    clASSERT(acts.nI==denoms.nI && acts.nJ == denoms.nJ, "convContrastNorm err4");


    /*clASSERT(!denoms.isTrans());
    clASSERT(!outGrads.isTrans());
    clASSERT(!acts.isTrans());
    clASSERT(!target.isTrans());
    clASSERT(outGrads.isContiguous());
    */

    clASSERT(numFilters % 16 == 0, "convResponseNormUndo err5");

    clASSERT(target.nI==outGrads.nI && target.nJ == outGrads.nJ, "convContrastNorm err6");
    //clASSERT(target.isContiguous());


    // First do acts := -2 x scale x acts x outGrads / denoms
    // so that the main routine only has to do an addition in its inner loop.
    int prelimEltsPerThread = 4;
    dim3 threads(128);
    dim3 blocks(MIN(512, DIVUP(outGrads.nI*outGrads.nJ,(threads.x * prelimEltsPerThread))));
    kRNormUndoPrelims<128, 4, T><<<blocks, threads>>>(acts.pData, denoms.pData, outGrads.pData, outGrads.nI*outGrads.nJ, -2*addScale*powScale);

    // Now the main routine
    if (sizeX >= 6 && numFilters % 4 == 0) {
        // This one is faster for large regions (my tests show regions >= 6...)
        int imgsPerThread = numImages % 128 == 0 ? 8 : numImages % 64 == 0 ? 4 : 2;
        int filtersPerThread = 4;
        int bx = 16;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        clASSERT((imgsPerThread * bx) % 32 == 0, "convContrastNorm err7");

        threads = dim3(bx, 16);
        blocks = dim3(DIVUP(imgSize, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(imgSize, 4) * numFilters / filtersPerThread);
        if (imgsPerThread == 8) {
            if (checkCaseBounds) {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, true, true, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 8, 4, true, true, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, false, true, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 8, 4, false, true, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            } else {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, true, false, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 8, 4, true, false, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, false, false, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 8, 4, false, false, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            }
        } else if (imgsPerThread == 4) {
            if (checkCaseBounds) {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 4, 4, true, true, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 4, 4, true, true, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 4, 4, false, true, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 4, 4, false, true, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            } else {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 4, 4, true, false, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 4, 4, true, false, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 4, 4, false, false, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 4, 4, false, false, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            }
        } else {
            if (checkCaseBounds) {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 2, 4, true, true, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 2, 4, true, true, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 2, 4, false, true, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 2, 4, false, true, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            } else {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 2, 4, true, false, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 2, 4, true, false, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 2, 4, false, false, T>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 2, 4, false, false, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                                  target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            }
        }
    } else {
        int imgsPerThread = numImages % 64 == 0 ? 2 : 1;
        bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
        threads = dim3(32, 4);
        blocks = dim3(DIVUP(numImages,32*imgsPerThread) * imgSize, (numFilters / (4 * 2)) * imgSize);

        if (imgsPerThread == 2) {
            if (checkCaseBounds) {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, false, true, T>, cudaFuncCachePreferL1);
                    kRNormUndo<4, 32, 2, 2, false, true, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                              target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, true, true, T>, cudaFuncCachePreferL1);
                    kRNormUndo<4, 32, 2, 2, true, true, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                              target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
                }
            } else {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, false, false, T>, cudaFuncCachePreferL1);
                    kRNormUndo<4, 32, 2, 2, false, false, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                              target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, true, false, T>, cudaFuncCachePreferL1);
                    kRNormUndo<4, 32, 2, 2, true, false, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                              target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
                }
            }
        } else {
            if (checkCaseBounds) {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo<4, 32, 1, 2, false, true, T>, cudaFuncCachePreferL1);
                    kRNormUndo<4, 32, 1, 2, false, true, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                              target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo<4, 32, 1, 2, true, true, T>, cudaFuncCachePreferL1);
                    kRNormUndo<4, 32, 1, 2, true, true, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                              target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
                }
            } else {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo<4, 32, 1, 2, false, false, T>, cudaFuncCachePreferL1);
                    kRNormUndo<4, 32, 1, 2, false, false, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                              target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo<4, 32, 1, 2, true, false, T>, cudaFuncCachePreferL1);
                    kRNormUndo<4, 32, 1, 2, true, false, T><<<blocks, threads>>>(outGrads.pData, denoms.pData, inputs.pData, acts.pData,
                                                                              target.pData, imgSize, numFilters, numImages, sizeX, powScale,
                                                                              scaleTargets, scaleOutput);
                }
            }
        }
    }
    getLastCudaError("kRNormUndo: kernel execution failed");
}

template<typename T>
void convContrastNormUndo(clMatrix<T>& outGrads, clMatrix<T>& denoms, clMatrix<T>& meanDiffs, clMatrix<T>& acts, clMatrix<T>& target, int numFilters,
                         int sizeX, T addScale, T powScale, T scaleTargets, T scaleOutput) {
    convResponseNormUndo<T>(outGrads, denoms, meanDiffs, acts, target, numFilters, sizeX, addScale, powScale, scaleTargets, scaleOutput);
}




#endif
