__global__ void vectorSumKernel(int *result, int *vector, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    __shared__ int shared_sum[256];
    int local_sum = 0;
    
    // Каждый поток суммирует свою часть массива
    for (int i = tid; i < n; i += stride) {
        local_sum += vector[i];
    }
    
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Редукция в разделяемой памяти
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Первый поток в блоке записывает результат
    if (threadIdx.x == 0) {
        atomicAdd(result, shared_sum[0]);
    }
}