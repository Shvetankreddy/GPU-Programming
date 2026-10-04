#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace wmma;

__global__ void initializeMatrices(half *A, half *B){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 64 * 64){
        A[idx] = __float2half(1.0f);
        B[idx] = __float2half(1.0f);
    }
}

__global__ void tensorCoreMatMul(half *A, half *B, float *C){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid = thread_id / 32;

    if (warpid >= 16) return;

    int tileRow = warpid / 4;
    int tileCol = warpid % 4;

    int row = tileRow * 16;
    int col = tileCol * 16;

    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < 64; k += 16){
        int A_index = row * 64 + k;
        int B_index = k * 64 + col;

        load_matrix_sync(a_frag, A + A_index, 64);
        load_matrix_sync(b_frag, B + B_index, 64);

        mma_sync(c_frag,a_frag,b_frag,c_frag);
    }

    int C_index = row * 64 + col;

    store_matrix_sync(C + C_index,c_frag,64,m_row_major);
}

int main(){
    half *A;
    half *B;
    float *C;

    float h_C[64 * 64];

    cudaMalloc(&A, 64 * 64 * sizeof(half));
    cudaMalloc(&B, 64 * 64 * sizeof(half));
    cudaMalloc(&C, 64 * 64 * sizeof(float));

    int threads = 256;
    int blocks = (64 * 64 + threads - 1) / threads;

    initializeMatrices<<<blocks, threads>>>(A, B);
    cudaDeviceSynchronize();

    tensorCoreMatMul<<<1, 512>>>(A, B, C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C,C,64 * 64 * sizeof(float),cudaMemcpyDeviceToHost);

    bool correct = true;

    for (int i = 0; i < 64; i++){
        for (int j = 0; j < 64; j++){
            if (h_C[i * 64 + j] != 64.0f){
                correct = false;
                printf("Error at C[%d][%d] = %f\n",i,j,h_C[i * 64 + j]);
                break;
            }
        }
        if (!correct) break;
    }

    if (correct){
        printf("Matrix multiplication successfully done r  \n"
                "64x64 matrices multiplied using 16x16 Tensor Core tiles.\n" );
    }
    else{
        printf("Matrix multiplication failed!\n");
    }

    printf("First 4x4 elements of C:\n");

    for (int i = 0; i < 16; i++){
        for (int j = 0; j < 16; j++){
            printf("%6.1f ", h_C[i * 64 + j]);
        }
        printf("\n");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}