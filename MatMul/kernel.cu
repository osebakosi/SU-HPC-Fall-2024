__global__ void matrixMulKernel(float *A, float *B, float *C, int A_height, int A_width, int B_width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < A_height && j < B_width) {
        C[i * B_width + j] = 0;
        for (int k = 0; k < A_width; k++) {
            C[i * B_width + j] += A[i * A_width + k] * B[k * B_width + j];
        }
    }
}