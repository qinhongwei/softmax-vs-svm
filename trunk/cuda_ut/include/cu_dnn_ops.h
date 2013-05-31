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

#ifndef _CU_DNN_OPS_H_
#define _CU_DNN_OPS_H_

#include "cu_clmatrix.h"
#include "cu_matrix_ops.h"
#include "bsxfun.h"
#include "cu_gpu_rand.h"

template <typename T>
int nodes_sample( clMatrix<T>& nodes, clMatrix<T>& randnums,
					NodesSampleEnum type,clMatrix<T>& std_vec);
template <typename T>
int SoftmaxProbSafe(IN const clMatrix<T> & mat, OUT clMatrix<T> & outmat);

template <typename T>
int softmax_sample( clMatrix<T>& v, int lab_st_zero_based,
					clMatrix<T>& resmat, clMatrix<T>& randnums);

template <typename T>
int ce_f_Ix(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, OUT clMatrix<T>& IxOut,
		OUT clMatrix<T>& TargetLogProb, OUT T* pf);

template <typename T>
int ce_f_Ix2(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, OUT clMatrix<T>& IxOut,
		OUT clMatrix<T>& TargetLogProb);

template <typename T>
int ce_f_Ix_logistic(const clMatrix<T>& NodesOut, const clMatrix<T>& Target,
				OUT clMatrix<T>& IxOut, OUT T* pf);
template <typename T>
int ce_f_Ix_logistic2(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, OUT clMatrix<T>& IxOut);

template <typename T>
int LogProbTestErrs( clMatrix<T>& mat, clMatrix<T>& buffer, const clMatrix<T>& label, T* pcnt_correct);

template <typename T>
int AddL1WtCost(const  clMatrix<T>& vhW,  clMatrix<T>& vhWInc, T f_rate, T f_l1_wtcost);

template <typename T>
int SigmoidBackprop( const clMatrix<T>&  p, clMatrix<T>&  pd_p);

template <typename T>
int TanhBackprop( const clMatrix<T>&  p, clMatrix<T>&  pd_p);

template <typename T>
int ReluBackprop( const clMatrix<T>&  p, clMatrix<T>&  pd_p);

template <typename T>
int SoftReluBackprop( const clMatrix<T>&  p, clMatrix<T>&  pd_p);

template <typename T>
int ReluQuadBackprop( const clMatrix<T>&  p, clMatrix<T>&  pd_p);

template <typename T>
int LogSumExp(IN const clMatrix<T> & mat, clMatrix<T> & out, int dim, OUT clMatrix<T>& logz );

template <typename T>
int Dropout( clMatrix<T>& nodes, clMatrix<T>& randnums, float f_dropout);

template <typename T>
int loss_l2svm(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, T fC,
				OUT clMatrix<T>& IxOut, clMatrix<T>& NodesOutTemp, OUT T& fLoss );
template <typename T>
int loss_l1svm(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, T fC,
				OUT clMatrix<T>& IxOut, clMatrix<T>& NodesOutTemp, OUT T& fLoss );
template <typename T>
int loss_l2_tsvm(const clMatrix<T>& NodesOut, const clMatrix<T>& Target, T fC1, T fC2,
				OUT clMatrix<T>& IxOut, clMatrix<T>& NodesOutTemp, OUT T& fLoss );

#endif
