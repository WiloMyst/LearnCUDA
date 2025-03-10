#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// 核函数：在 GPU 上执行向量加法
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    // 获取线程在 grid 中的全局索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保索引不越界
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 1024; // 向量长度
    int size = n * sizeof(float); // 向量大小（字节）

    // 1. 分配主机内存并初始化
    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_c = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // 2. 分配设备内存
    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 3. 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 4. 调用核函数
    int blockSize = 256; // 每个 block 中的线程数
    int gridSize = (n + blockSize - 1) / blockSize; // block 的数量
    vectorAdd << <gridSize, blockSize >> > (d_a, d_b, d_c, n);

    // 5. 将结果从设备复制回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 6. 验证结果（可选）
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Error at index " << i << std::endl;
            break;
        }
    }
    //如果没有报错就说明结果正确

// 7. 释放内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 等待GPU完成所有操作
    cudaDeviceSynchronize();

    return 0;
}