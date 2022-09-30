#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>
using namespace std;

__global__ void saxpy_add(int n, float  a, float *x, float *y)
{
    int stride = blockDim.x * gridDim.x;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = col; i < n; i += stride)
        y[i] = x[i] + y[i];
}

__global__ void saxpy_mult(int n, float  a, float *x, float *y)
{
    int stride = blockDim.x * gridDim.x;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = col; i < n; i += stride)
        x[i] = a * x[i];
}

void saxpyGraph(int n, float a, float *x, float *y)
{
    unsigned blocks = 32;
    unsigned threadsPerBlock = 1024;
    bool graphCreated = false;
    hipStream_t stream;
    hipGraph_t graph; //structure and content of graph
    hipGraphExec_t graph_exec = NULL; //executable graph
    hipGraphNode_t *nodes = NULL;
    size_t numNodes = 0;

    hipStreamCreate(&stream);
    if(!graphCreated)
    {
        hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
        hipLaunchKernelGGL(saxpy_mult,
                            dim3(blocks), dim3(threadsPerBlock),0,stream,
                            n, a, x, y);
        hipStreamSynchronize(stream);//make sure previous kernel is complete
        hipLaunchKernelGGL(saxpy_add,
                            dim3(blocks), dim3(threadsPerBlock),0,stream,
                            n, a, x, y);
        hipStreamSynchronize(stream);
        hipStreamEndCapture(stream, &graph);
        hipGraphGetNodes(graph,nodes,&numNodes);
        printf("\nNum of nodes in the graph = %zu\n", numNodes);
        hipGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
        graphCreated = true;
    }
    hipGraphLaunch(graph_exec, stream);  
}

int main ()
{
    int n = 10; //num iterations
    float *x_d, *y_d;
    float *x_h, *y_h;
    float a = 1.0;

    hipMalloc(&x_d, n * sizeof(float));
    hipMalloc(&y_d, n * sizeof(float));
    x_h = (float*)malloc(n * sizeof(float));
    y_h = (float*)malloc(n * sizeof(float));
    hipMemcpy(x_d, x_h, (n * sizeof(float)), hipMemcpyHostToDevice);
    
    saxpyGraph(n, a, x_d, y_d);

    
    hipDeviceSynchronize();

    hipMemcpy(y_h, y_d, (n * sizeof(float)), hipMemcpyDeviceToHost);
    hipFree(x_d); hipFree(y_d);
    free(x_h); free(y_h);

    return 0;
}

