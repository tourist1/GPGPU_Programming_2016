#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

/**********************************
		processing helper
**********************************/

__global__ void initialize(const char* text, int *pos, int *tmp, int text_size) {
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	if(tid < text_size) {
		pos[tid] = (text[tid] != '\n');
		tmp[tid] = pos[tid];
	}
}

__global__ void countPos(const char *text, int *pos, int *temp, int text_size, int round) {
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	if(tid < text_size) {
		if(tid > 0 && pos[tid] && pos[tid] == pos[tid-1]) { // avoiding consecutive 0 like 000
			temp[tid] += pos[tid-round];
		}
	}
}

__global__ void assign(int *pos, int *temp, int text_size) {
	int tid = blockDim.x*blockIdx.x+threadIdx.x;
	if(tid < text_size) {
		pos[tid] = temp[tid];
	}
}

/**********************************
		assignment part
**********************************/

void CountPosition(const char *text, int *pos, int text_size)
{
	int blocksize = 512;
	int nblock = CeilDiv(text_size, blocksize);
	int *reg;
	//bool done = true;
	cudaMalloc(&reg, sizeof(int)*text_size);
	initialize<<<nblock, blocksize>>>(text, pos, reg, text_size);
	
	for(int i = 0; i <= 9; ++i) { // assume that k == 500
		cudaDeviceSynchronize();
		countPos<<<nblock, blocksize>>>(text, pos, reg, text_size, (1<<i));
		cudaDeviceSynchronize();
		assign<<<nblock, blocksize>>>(pos, reg, text_size);
	}
	cudaFree(reg);
}

struct is_one {
	__host__ __device__
	bool operator()(const int x) {
		return x == 1;
	}
};

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;  //device ptr
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough for context and index buffer
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);
	//transform, sequence, inclusive_scan, sequence and cudaMemcpy
	// TODO
	
	//thrust::copy(pos_d, pos_d+text_size, flag_d);
	thrust::sequence(cumsum_d, cumsum_d+text_size);
	
	nhead = thrust::copy_if(cumsum_d, cumsum_d+text_size, pos_d, head_d, is_one()) - head_d;
	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
	char *tmp = (char*)malloc(text_size);
	int count[26] = {0};
	cudaMemcpy(tmp, text, text_size, cudaMemcpyDeviceToHost);
	for(int i = 0; i < text_size; ++i) {
		if(tmp[i] != '\n')
			count[tmp[i]-'a']++;
	}
	for(int i = 0; i < 26; ++i) {
		printf("%c %d\n", i+'a', count[i]);
	}
	free(tmp);
}

































