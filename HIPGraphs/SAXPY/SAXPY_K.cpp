#include <hip/hip_runtime.h>
#include <iostream>
#define N 8

__global__ void saxpy(float a, float *x, float *y, float *z) {
        int i = threadIdx.x;
        if (i < N) {
                z[i] = a * x[i] + y[i];
        }
}

int main() {
        float *d_x, *d_y, *d_z;
        float a = 1;
        hipHostMalloc(&d_x, N*sizeof(float));
        hipHostMalloc(&d_y, N*sizeof(float));
        hipHostMalloc(&d_z, N*sizeof(float));
        for (int i = 0; i < N; i++) {
                d_x[i] = i + 1;
                d_y[i] = i + 1;
        }
        saxpy<<<1,N>>>(a, d_x, d_y, d_z);
        hipDeviceSynchronize();
        for (int i = 0; i < N; i++) {
                std::cout << a * d_x[i] + d_y[i] << " | " << d_z[i] << std::endl;
        }
        return 0;
}
