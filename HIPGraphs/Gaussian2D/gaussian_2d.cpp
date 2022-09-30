/*This program is to solve a linear system using Gaussian Elimination.
    This version does not use HIP Graphs and is based off rodinia.*/

#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>
#include <cstdio>
using namespace std;

#ifdef RD_WG_SIZE_0_0
#define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define MAXBLOCKSIZE RD_WG_SIZE
#else
#define MAXBLOCKSIZE 512
#endif

// 2D defines. Go from specific to general
#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_XY RD_WG_SIZE
#else
#define BLOCK_SIZE_XY 4
#endif

int Size;
float *m, *a, *b; 
float *finalVec;
FILE *fp;
//a is matrix that is entered
//m is multiplier matrix
//b is an array, a single row

__global__ void Fan1(float *m, float *a, int Size, int t);
__global__ void Fan2(float *m, float *a, float *b, int Size, int j1, int t);
void Gaussian2DGraph(float *m_hip, float *a_hip, float *b_hip, int t);
void InitMat(float *ary, int nrow, int ncol);
void create_matrix(float *m, int size);
void PrintMat(float *ary, int nrow, int ncol);
void PrintAry(float *ary, int ary_size);
void InitAry(float *ary, int ary_size);
void InitPerRun();
void BackSub();
void Gaussian2DGPU();

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void create_matrix(float *m, int size) {
    int i, j;
    float lamda = -0.01;
    float coe[2 * size - 1];
    float coe_i = 0.0;

    for (i = 0; i < size; i++) {
        coe_i = 10 * exp(lamda * i);
        j = size - 1 + i;
        coe[j] = coe_i;
        j = size - 1 - i;
        coe[j] = coe_i;
    }


    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            m[i * size + j] = coe[size - 1 - i + j];
        }
    }
}

int main(int argc, char *argv[]){
    char flag;

    //argv[1] points to the first command
    //Command: ./gaussian2d -f filename -s size
    for(int i = 1; i < argc; i++){
        if(argv[i][0] == '-'){ //flag
            flag = argv[i][1];
            if(flag == 's'){
                i++;
                Size = atoi(argv[i]);
                printf("Size is: %d\n", Size);
                cout << "\nCreate square matrix of size " << Size << endl;
                a = (float *)malloc(Size * Size * sizeof(float));
                create_matrix(a, Size);

                b = (float *)malloc(Size * sizeof(float));
                for (int j = 0; j < Size; j++)
                    b[j] = 1.0;

                m = (float *)malloc(Size * Size * sizeof(float));

            }else if (flag == 'f'){
                i++;
                cout << "Reading file from " << argv[i] << endl;
                fp = fopen(argv[i], "r");
                fscanf(fp, "%d", &Size);
                a = (float *) malloc(Size * Size * sizeof(float)); 
                b = (float *) malloc(Size * sizeof(float));
                m = (float *) malloc(Size * Size * sizeof(float));
                InitMat(a, Size, Size);
                InitAry(b, Size);

            }
        }
    }
    
    InitPerRun();

    Gaussian2DGPU(); //mem stuff + kernel call

    printf("Matrix m is: \n");
    PrintMat(m, Size, Size);

    printf("Matrix a is: \n");
    PrintMat(a, Size, Size);

    printf("Array b is : \n");
    PrintAry(b, Size);

    BackSub();

    printf("final solution is: \n");
    PrintAry(finalVec, Size);

    free(m); free(a); free(b);

    return 0;
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub() {
    // create a new vector to hold the final answer
    finalVec = (float *)malloc(Size * sizeof(float));
    // solve "bottom up"
    int i, j;
    for (i = 0; i < Size; i++) {
        finalVec[Size - i - 1] = b[Size - i - 1];
        for (j = 0; j < i; j++) {
            finalVec[Size - i - 1] -=
                *(a + Size * (Size - i - 1) + (Size - j - 1)) *
                finalVec[Size - j - 1];
        }
        finalVec[Size - i - 1] = finalVec[Size - i - 1] /
                                 *(a + Size * (Size - i - 1) + (Size - i - 1));
    }
}

//Fan1 Calculates multiplier matrix
__global__ void Fan1(float *m_hip, float *a_hip, int Size, int t) {
    //printf("hi from Fan1() \n");

    if (threadIdx.x + blockIdx.x * blockDim.x >= Size - 1 - t)
        return;
    *(m_hip + Size * (blockDim.x * blockIdx.x + threadIdx.x + t + 1) + t) =
        *(a_hip + Size * (blockDim.x * blockIdx.x + threadIdx.x + t + 1) + t) /
        *(a_hip + Size * t + t);
}
//Fan2 modifies the matrix into LUD
__global__ void Fan2(float *m_hip, float *a_hip, float *b_hip, int Size,
                     int j1, int t) {
    //printf("hi from Fan2() \n");
    if (threadIdx.x + blockIdx.x * blockDim.x >= Size - 1 - t)
        return;
    if (threadIdx.y + blockIdx.y * blockDim.y >= Size - t)
        return;

    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);

    a_hip[Size * (xidx + 1 + t) + (yidx + t)] -=
        m_hip[Size * (xidx + 1 + t) + t] * a_hip[Size * t + (yidx + t)];
    if (yidx == 0) {
        // printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
        // printf("xidx:%d,yidx:%d\n",xidx,yidx);
        b_hip[xidx + 1 + t] -=
            m_hip[Size * (xidx + 1 + t) + (yidx + t)] * b_hip[t];
    }
}

void Gaussian2DGPU(){
    int t = 0;
    float *m_hip, *a_hip, *b_hip;

    //allocate mem on GPU
    hipMalloc((void**)&m_hip, Size * Size * sizeof(float));
    hipMalloc((void**)&a_hip, Size * Size * sizeof(float));
    hipMalloc((void**)&b_hip, Size * sizeof(float));
    
    //copy mem to GPU
    hipMemcpy(m_hip, m, Size * Size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(a_hip, a, Size * Size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(b_hip, b, Size * sizeof(float), hipMemcpyHostToDevice);

    int block_size, grid_size;
    block_size = MAXBLOCKSIZE;
    grid_size = (Size / block_size) + (!(Size % block_size) ? 0 : 1);

    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);

    int blockSize2d, gridSize2d;
    blockSize2d = BLOCK_SIZE_XY;
    gridSize2d = (Size / blockSize2d) + (!(Size % blockSize2d ? 0 : 1));

    dim3 dimBlockXY(blockSize2d, blockSize2d);
    dim3 dimGridXY(gridSize2d, gridSize2d);

    for (t = 0; t < (Size-1); t++){
        hipLaunchKernelGGL(Fan1,
                            dimGrid, dimBlock,0,0,
                            m_hip, a_hip, Size, t);
        //hipGraphAddKernelNode(&nodes, graph, nullptr, o, &node_params)
        hipLaunchKernelGGL(Fan2,
                            dimGridXY, dimBlockXY,0,0,
                            m_hip, a_hip, b_hip, Size, Size-t, t);
        hipDeviceSynchronize(); //cudaThreadSynchronize()
    }

    hipMemcpy(m, m_hip, Size * Size * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(a, a_hip, Size * Size * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(b, b_hip, Size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(m_hip); hipFree(a_hip); hipFree(b_hip);

}



/*
    InitPerRun() -- Initializes contents of multiplier matrix **m
*/
void InitPerRun() {
    int i;
    for (i = 0; i < Size * Size; i++)
        *(m + i) = 0.0;
        printf("hi from InitPerRun() \n");
}

void InitMat(float *ary, int nrow, int ncol) {
    int i, j;

    for (i = 0; i < nrow; i++) {
        for (j = 0; j < ncol; j++) {
            fscanf(fp, "%f", ary + Size * i + j);
        }
    }

}

void InitAry(float *ary, int ary_size) {
    int i;
    //printf("Initiating array\n");
    for (i = 0; i < ary_size; i++) {
        fscanf(fp, "%f", &ary[i]);
    }
}

void PrintMat(float *ary, int nrow, int ncol) {
    int i, j;
    printf("Printing Matrix\n");
    for (i = 0; i < nrow; i++) {
        for (j = 0; j < ncol; j++) {
            printf("%8.2f ", *(ary + Size * i + j));
        }
        printf("\n");
    }
    printf("\n");
}

void PrintAry(float *ary, int ary_size) {
    int i;
    for (i = 0; i < ary_size; i++) {
        printf("%.2f ", ary[i]);
    }
    printf("\n\n");
}