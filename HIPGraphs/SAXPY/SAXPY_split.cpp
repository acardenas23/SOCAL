//HIP SAXPY 
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>
using namespace std;

__global__ void saxpy_add(int n, float  a, float *x, float *y)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int col = bx * blockDim.x + tx;

    for (int i = col; i < n; i += stride)
    {
        y[i] = x[i] + y[i];
    }
}

__global__ void saxpy_mult(int n, float  a, float *x, float *y)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int col = bx * blockDim.x + tx;

    for (int i = col; i < n; i += stride)
    {
        x[i] = a * x[i];
    }
}

void saxpy(int n, float a, float *x, float *y)
{
    unsigned blocks = 32;
    unsigned threadsPerBlock = 1024;

    hipLaunchKernelGGL(saxpy_mult,
                    dim3(blocks), dim3(threadsPerBlock),0,0,
                    n, a, x, y);
    hipLaunchKernelGGL(saxpy_add,
                    dim3(blocks), dim3(threadsPerBlock),0,0,
                    n, a, x, y);
    
}

int main (){
    int n = 10;
    float *x_d, *y_d;
    float *x_h, *y_h;
    // float x_h[n] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // float y_h[n] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float a = 1.0;

    // hipHostMalloc(&x_d, n * sizeof(float));
    // hipHostMalloc(&y_d, n * sizeof(float));

    hipMalloc(&x_d, n * sizeof(float));
    hipMalloc(&y_d, n * sizeof(float));
    x_h = (float*)malloc(n * sizeof(float));
    y_h = (float*)malloc(n * sizeof(float));
    hipMemcpy(x_d, x_h, (n * sizeof(float)), hipMemcpyHostToDevice);
    
    saxpy(n, a, x_d, y_d);
    hipDeviceSynchronize();

    cout << "hi" << endl;

    hipMemcpy(y_h, y_d, (n * sizeof(float)), hipMemcpyDeviceToHost);
    hipFree(x_d); hipFree(y_d);
    free(x_h); free(y_h);

    // for (int i = 0; i< 8; ++i){
    //     cout << y_d[i] << endl;
    // }

    return 0;
}


