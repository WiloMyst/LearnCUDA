#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// �˺������� GPU ��ִ�������ӷ�
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    // ��ȡ�߳��� grid �е�ȫ������
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // ȷ��������Խ��
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 1024; // ��������
    int size = n * sizeof(float); // ������С���ֽڣ�

    // 1. ���������ڴ沢��ʼ��
    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_c = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // 2. �����豸�ڴ�
    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 3. �����ݴ��������Ƶ��豸
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 4. ���ú˺���
    int blockSize = 256; // ÿ�� block �е��߳���
    int gridSize = (n + blockSize - 1) / blockSize; // block ������
    vectorAdd << <gridSize, blockSize >> > (d_a, d_b, d_c, n);

    // 5. ��������豸���ƻ�����
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 6. ��֤�������ѡ��
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Error at index " << i << std::endl;
            break;
        }
    }
    //���û�б����˵�������ȷ

// 7. �ͷ��ڴ�
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // �ȴ�GPU������в���
    cudaDeviceSynchronize();

    return 0;
}