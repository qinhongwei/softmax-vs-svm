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

//contains some global variables

#include "cu_util.h"



//global variables for timing
cudaEvent_t g_start, g_stop;


void cuTic(){
	cudaEventCreate(&g_start);
	cudaEventCreate(&g_stop);
	cudaEventRecord( g_start, 0 ); //start timing
}


//returns ms
float cuToc(){
	cudaEventRecord( g_stop, 0 );
	cudaEventSynchronize( g_stop );

	float time;
	cudaEventElapsedTime( &time, g_start, g_stop );

	//destroy timer
	cudaEventDestroy( g_start );
	cudaEventDestroy( g_stop );
	return time;
}

//this is the global handel to the cublas context, the main program must
//cublasCreate() it and cublasDestroy() it.
cublasHandle_t cbh;

// display how much memory is left on the gpu;
void DisplayGPUMemory(int gpuid){

	size_t debugfree, debugtotal;
	cudaMemGetInfo 	( 	&debugfree, &debugtotal);
	char str[100];
	sprintf(str, "\n[%d] Board Memory free:%.4f total:%.4f", gpuid,
			float(debugfree)/1048576, float(debugtotal)/1048576);
	clPrintf( str);
}



// random generator function:
ptrdiff_t randperm_myrandom (ptrdiff_t i)
{
	return rand()%i;
}

// returning from a random vector from 0 to n-1
void randperm(int n, std::vector<int>& inds, unsigned int seed) {

    //srand ( unsigned ( time (NULL) ) );
	if (seed != 0){
		srand ( seed );
	}

	inds.resize(n);
	for(size_t i = 0; i < inds.size(); i++){
		inds[i] = i;
	}

	// pointer object to it:
	ptrdiff_t (*p_randperm_myrandom)(ptrdiff_t) = randperm_myrandom;


	std::random_shuffle( inds.begin(), inds.end(), p_randperm_myrandom );
}


uint64_t prod(const std::vector<int>& inds ){
	uint64_t val = 1;
	for (unsigned int i = 0; i < inds.size(); ++i){
		val *= inds[i];
	}
	return val;
}


double prod(const std::vector<double>& inds ){
	double val = 1;
	for (unsigned int i = 0; i < inds.size(); ++i){
		val *= inds[i];
	}
	return val;
}






