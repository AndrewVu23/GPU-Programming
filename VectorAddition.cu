%%cuda
#include <iostream>
#include <stdio.h>
#include <math.h>

// N: Vector Length
#define N 1024

// CUDA function (kernel) decleration
// This is a Vector Addition Kernel
__global__ void vecAdd(float *A_d, float *B_d, float *C_d) {
    
    // Calculate the number of indexes (or number of iterations in a
    // conventional for loop)
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop parallelism: Each thread is an iteration of the
    // conventional for loop
    if (i < N) {
        C_d[i] = A_d[i] + B_d[i];
    }
}

int main(void) {
    
    // _h: host's pointers
    // _d: device's pointers
    float *A_h = (float *)malloc(N * sizeof(float));
    float *B_h = (float *)malloc(N * sizeof(float));
    float *C_h = (float *)malloc(N * sizeof(float));

    // Initialize the value of each element in vector A and B
    for (int i = 0; i < N; i++) {
        A_h[i] = 1.0;
        B_h[i] = 2.0;
    }

    // Allocate device memory
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, N * sizeof(float));
    cudaMalloc((void **)&B_d, N * sizeof(float));
    cudaMalloc((void **)&C_d, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N * sizeof(float), cudaMemcpyHostToDevice);

    // Call the kernel
    vecAdd<<<ceil(N / 256.0), 256>>>(A_d, B_d, C_d);

    // Copy result from device back to host
    cudaMemcpy(C_h, C_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check the result. The code snippet was copied from
    // https://github.com/olcf-tutorials/vector_addition_cuda
    double tolerance = 1.0e-14;
    for(int i = 0; i < N; i++)
    {
        if( fabs(C_h[i] - 3.0) > tolerance)
        {
            printf("\nError: value of C_h[%d] = %f instead of 3.0\n\n", i, C_h[i]);
            exit(1);
        }
    }
    
    printf("Compile Successfully!\n");
    printf("Printing the first 10 values:\n");

    // Print the first 10 values 
    for(int i = 0; i < 10; i++)
    {
        printf("C_h[%d] = %f \n", i, C_h[i]);
    }

    // Free host memory
    free(A_h);
    free(B_h);
    free(C_h);
    
    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}