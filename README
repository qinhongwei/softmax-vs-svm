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


////////////////////////////////////////////////////////////////////
CT 5.30.2013
Note: I have only tested this on linux Ubuntu 12.04, with cuda 5.
it should work with previous cuda versions with minor tweaks to the 
build scripts

////////////////////////////////////////////////////////////////////
Compiling:
////////////////////////////////////////////////////////////////////
to make shared CUDA/C++ shared library:
0. install cuda 5
1. cd into cuda_ut folder
2. update variables 'CUDA_PATH' and 'CUDA_SAMPLES_PATH' in Makefile
3. make (this may take 10 mins, make sure that nvcc used is 
		 version 5 and it is on the PATH.)
4. cd modules
5. make mexf="./deep_nn/mexcuConvNNoo.mex ./deep_nn/mexcuConvNNooFF.mex"


////////////////////////////////////////////////////////////////////
Learning:
////////////////////////////////////////////////////////////////////
1. cd to matlab folder
2. download train.csv and test.csv
3. if using tcsh, setenv LD_PRELOAD /usr/lib/x86_64-linux-gnu/libstdc++.so.6
	and setenv LD_LIBRARY_PATH somewhere/face_exp/cuda_ut/lib
	(note that the path for libstdc++.so.6 may vary for different OS)
4. start matlab
5. run load_from_kaggle.m
6. run script_face_exp.m

////////////////////////////////////////////////////////////////////
Prediction:
////////////////////////////////////////////////////////////////////
1. run fe_pred.m





