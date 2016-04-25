#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void CalculateFixed(
	const float *background, 
	const float *buf1, // target
	const float *mask, 
	float *output, //fixed
	const int wb, const int hb, const int wt, const int ht, 
	const int oy, const int ox
) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	
	const int yb = oy+yt, xb = ox+xt;
	const int curb = wb*yb+xb;
	
	int northt = wt*(yt-1)+xt, northb = wb*(yb-1)+xb;
	int southt = wt*(yt+1)+xt, southb = wb*(yb+1)+xb;
	int eastt = wt*yt+xt+1, eastb = wb*yb+xb+1;
	int westt = wt*yt+xt-1, westb = wb*yb+xb-1;
	
	int direction[8] = {northt*3, westt*3, eastt*3, southt*3,
		northb*3, westb*3, eastb*3, southb*3};
	bool condition[8] = {yt-1 >= 0, xt-1 >= 0, xt+1 < wt, yt+1 < ht, 
		yt-1 >= 0 && !mask[northt],
		xt-1 >= 0 && !mask[westt],
		xt+1 < wt && !mask[eastt],
		yt+1 < ht && !mask[southt]
	};
	if (yt < ht && xt < wt) {
		for(int j = 0; j < 3; ++j)
			output[curt*3+j] = 4*buf1[curt*3+j];
			
		for(int i = 0; i < 4; ++i) { // check if Nt, St, Wt or Et is inbound
			if(condition[i])
				for(int j = 0; j < 3; ++j)
					output[curt*3+j] -= buf1[direction[i]+j];
		}
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			for(int i = 4; i < 8; ++i) // check if Nb, Sb, Wb or Eb is fixed
				if(condition[i])
					for(int j = 0; j < 3; ++j)
						output[curt*3+j] += background[direction[i]+j];
				else if(!condition[i-4])
					for(int j = 0; j < 3; ++j)
						output[curt*3+j] = background[direction[i]+j];
		}
	}
}

__global__ void PoissonImageCloningIteration(
	const float *fixed, 
	const float *mask, 
	const float *buf1, 
	float *buf2, 
	const int wt, 
	const int ht
) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	
	int northt = wt*(yt-1)+xt;
	int southt = wt*(yt+1)+xt;
	int eastt = wt*yt+xt+1;
	int westt = wt*yt+xt-1;
	
	int direction[4] = {northt*3, westt*3, eastt*3, southt*3};
	bool condition[4] = {
		yt-1 >= 0 && mask[northt] > 127.0f,
		xt-1 >= 0 && mask[westt] > 127.0f,
		xt+1 < wt && mask[eastt] > 127.0f,
		yt+1 < ht && mask[southt] > 127.0f
	};
	
	for(int j = 0; j < 3; ++j)
		buf2[curt*3+j] = fixed[curt*3+j];
	if (yt < ht && xt < wt) {
		for(int i = 0; i < 4; ++i) {
			if(condition[i])
				for(int j = 0; j < 3; ++j)
					buf2[curt*3+j] += buf1[direction[i]+j];
		}
	}
	for(int j = 0; j < 3; ++j)
		buf2[curt*3+j] /= 4;
}
 
void PoissonImageCloning(
  const float *background,
  const float *target,
  const float *mask,
  float *output,
  const int wb, const int hb, const int wt, const int ht,
  const int oy, const int ox
) {
  // set up
  float *fixed, *buf1, *buf2;
  cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
  cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
  cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

  // initialize the iteration
  dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
  CalculateFixed<<<gdim, bdim>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
	);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

  // iterate
  for (int i = 0; i < 20000; ++i) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht
		);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht
		);
	}

	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<gdim, bdim>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);

	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}