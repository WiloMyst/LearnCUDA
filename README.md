# Learn CUDA

**1. 并行计算 (Parallel Computing)**

- **什么是并行计算？**
  - 想象一下你要搬一堆很重的箱子。你可以一个人慢慢搬，也可以找几个人一起搬。并行计算就像找几个人一起搬箱子，将一个大任务分解成多个小任务，然后同时执行这些小任务，从而更快地完成任务。
  - 传统 CPU 串行计算：CPU 就像一个人，一次只能执行一个指令，处理一个任务。
  - GPU 并行计算：GPU 就像一群人，可以同时执行多个指令，处理多个任务。
- **为什么需要并行计算？**
  - **摩尔定律放缓：** 随着芯片制程工艺接近物理极限，CPU 单核性能提升越来越困难。
  - **数据爆炸：** 图像、视频、科学模拟等领域的数据量越来越大，传统的串行计算难以满足需求。
  - **应用需求：** 人工智能、机器学习、深度学习等应用需要大量的计算资源，并行计算是提高效率的关键。
- **并行计算的类型：**
  - **数据并行（Data Parallelism）：** 将数据分成多个部分，每个部分由不同的处理器同时处理。（GPU擅长）
  - **任务并行（Task Parallelism）：** 将任务分成多个子任务，每个子任务由不同的处理器同时执行。（CPU更擅长）

**2. GPU 架构 (GPU Architecture)**

- **GPU 与 CPU 的区别：**

  | 特性     | CPU                          | GPU                                    |
  | -------- | ---------------------------- | -------------------------------------- |
  | 核心数   | 少量（几核到几十核）         | 大量（成百上千个核心）                 |
  | 优化方向 | 低延迟（快速响应单个任务）   | 高吞吐量（同时处理大量任务）           |
  | 控制逻辑 | 复杂（擅长复杂的控制流程）   | 简单（擅长简单、重复的计算）           |
  | 缓存     | 大缓存（减少对内存的访问）   | 小缓存（更多空间用于计算核心）         |
  | 内存带宽 | 相对较低                     | 非常高                                 |
  | 适用场景 | 通用计算、操作系统、复杂逻辑 | 图形渲染、科学计算、机器学习、深度学习 |

- **GPU 的基本组成：**

  - **流多处理器（Streaming Multiprocessor, SM）：** GPU 的基本计算单元，包含多个 CUDA 核心（CUDA Core）、共享内存（Shared Memory）、寄存器（Registers）等。
    - 可以将SM理解成一个“小团队”，团队里的成员（CUDA Core）可以高效地协作完成任务。
  - **CUDA 核心（CUDA Core）：** 执行计算任务的最小单元，每个 CUDA 核心可以执行一个线程。
    - 可以把CUDA Core理解成“小团队”里的一个成员。
  - **共享内存（Shared Memory）：** SM 内部的快速内存，用于线程块内的线程之间共享数据。
    - 类似于“小团队”内部的共享白板，成员可以在上面快速交换信息。
  - **寄存器（Registers）：** 线程私有的快速存储空间，用于存储线程的局部变量。
    - 类似于每个成员自己的小笔记本，用来记录自己的数据。
  - **全局内存（Global Memory）：** GPU 上最大的内存，所有 SM 都可以访问，但访问速度较慢。
    - 类似于整个公司的公共数据库，大家都可以访问，但速度比较慢。

- **GPU 的线程层次结构：**

  - **线程（Thread）：** 执行计算任务的最小单位，由 CUDA 核心执行。
  - **线程块（Thread Block）：** 一组线程的集合，这些线程在同一个 SM 上执行，可以共享 SM 内部的共享内存。
    - 一个Block可以理解为一个执行任务的小队
  - **线程格（Grid）：** 一组线程块的集合，一个 Grid 对应一个 CUDA 核函数的调用。
    - 整个Grid就构成了一次完整的任务
  - **线程束（Warp）：** GPU 执行指令的基本单位，一个 Warp 包含 32 个线程，这些线程以 SIMT（Single Instruction, Multiple Threads）方式执行相同的指令。
    - 实际上干活的时候，是以Warp为单位，可以把Warp理解为32个线程构成的小组。
  - **理解这些层次结构对 CUDA 编程至关重要，因为它决定了如何分配计算任务，以及如何优化内存访问。**

**3. CUDA (Compute Unified Device Architecture)**

- **什么是 CUDA？**
  - CUDA 是 NVIDIA 推出的并行计算平台和编程模型，它允许开发者使用 C/C++ 等语言来编写在 GPU 上运行的程序。
  - CUDA 提供了一套 API（应用程序编程接口）和工具，用于管理 GPU 设备、分配内存、启动核函数、进行数据传输等。
- **CUDA 编程模型：**
  - **主机（Host）：** 指的是 CPU 及其内存。
  - **设备（Device）：** 指的是 GPU 及其内存。
  - **核函数（Kernel）：** 在 GPU 上执行的函数，由多个线程并行执行。
  - **异构计算：** CUDA 程序通常包含 CPU 代码和 GPU 代码，CPU 负责串行部分的执行和控制流程，GPU 负责并行部分的执行。
  - **数据传输：** CUDA 程序需要在主机和设备之间传输数据，使用 cudaMemcpy 等函数。
- **CUDA 程序的基本流程：**
  1. 分配主机内存，并进行数据初始化。
  2. 分配设备内存，并将数据从主机复制到设备。
  3. 调用核函数在设备上执行计算。
  4. 将计算结果从设备复制回主机。
  5. 释放主机和设备内存。

**4. 学习第一个 CUDA 程序**

- **1. 向量加法问题**

  假设我们有两个长度为 N 的向量 A 和 B，我们想要计算它们的和 C，即 C[i] = A[i] + B[i]，其中 i 从 0 到 N-1。

  **2. CUDA 向量加法程序 (vectorAdd.cu)**

  ```c++
  #include <iostream>
  #include <cuda_runtime.h>
  #include "device_launch_parameters.h"
  
  // 核函数：在 GPU 上执行向量加法
  __global__ void vectorAdd(float *a, float *b, float *c, int n) {
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
      float *h_a = new float[n];
      float *h_b = new float[n];
      float *h_c = new float[n];
  
      for (int i = 0; i < n; i++) {
          h_a[i] = i;
          h_b[i] = 2 * i;
      }
  
      // 2. 分配设备内存
      float *d_a, *d_b, *d_c;
      cudaMalloc(&d_a, size);
      cudaMalloc(&d_b, size);
      cudaMalloc(&d_c, size);
  
      // 3. 将数据从主机复制到设备
      cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
  
      // 4. 调用核函数
      int blockSize = 256; // 每个 block 中的线程数
      int gridSize = (n + blockSize - 1) / blockSize; // block 的数量
      vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
  
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
  ```

  **3. 代码解析**

  - **#include <cuda_runtime.h>:** 包含 CUDA 运行时 API 的头文件。

  - **__global__ void vectorAdd(float \*a, float \*b, float \*c, int n):**

    - __global__: CUDA 关键字，表示这是一个核函数，将在 GPU 上执行。

    - void: 核函数没有返回值。

    - float *a, float *b, float *c: 指向设备内存中向量 A、B、C 的指针。

    - int n: 向量的长度。

    - 核函数内部：

      - ```c++
        `int index = blockIdx.x * blockDim.x + threadIdx.x;`
        ```

        - blockIdx.x: 当前线程所在 block 在 grid 中的 x 维度索引。
        - blockDim.x: 一个 block 在 x 维度上的线程数量。
        - threadIdx.x: 当前线程在 block 中的 x 维度索引。
        - 通过这三个内置变量，我们可以计算出当前线程在 grid 中的全局唯一索引。

      - if (index < n): 确保线程索引不越界。

      - c[index] = a[index] + b[index];: 执行向量加法。

  - **int main():**

    - **int n = 1024;**: 定义向量长度。
    - **int size = n \* sizeof(float);**: 计算向量占用的字节数。
    - **1. 分配主机内存并初始化:**
      - float *h_a = new float[n]; 等: 在主机（CPU）上分配内存，并初始化向量 A 和 B。
    - **2. 分配设备内存:**
      - cudaMalloc(&d_a, size); 等: 在设备（GPU）上分配内存。
    - **3. 将数据从主机复制到设备:**
      - cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice); 等: 将数据从主机内存复制到设备内存。
        - cudaMemcpyHostToDevice: 指定数据传输方向。
    - **4. 调用核函数:**
      - int blockSize = 256;: 定义每个 block 中的线程数（通常是 32 的倍数）。
      - int gridSize = (n + blockSize - 1) / blockSize;: 计算 grid 中 block 的数量，确保所有元素都被处理。
      - vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);: 启动核函数。
        - <<<gridSize, blockSize>>>: CUDA 语法，指定核函数的执行配置（grid 大小和 block 大小）。
    - **5. 将结果从设备复制回主机:**
      - cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);: 将结果从设备内存复制回主机内存。
        - cudaMemcpyDeviceToHost指定方向
    - **6. 验证结果**
      检查GPU计算的结果是否正确。
    - **7. 释放内存:**
      - delete[] h_a; 等: 释放主机内存。
      - cudaFree(d_a); 等: 释放设备内存。
      - cudaDeviceSynchronize(); 等待GPU完成所有操作，这是一个好习惯，防止程序在GPU未完成计算时就退出了。

  **4. 编译和运行**

  1. **保存代码：** 将代码保存为 vectorAdd.cu 文件（.cu 扩展名表示这是一个 CUDA 文件）。

  2. **编译代码：** 使用 NVIDIA 的 CUDA 编译器 nvcc 来编译代码：

     ```bash
     nvcc vectorAdd.cu -o vectorAdd
     ```

  3. **运行程序：**

     ```bash
     ./vectorAdd
     ```

  **5. 关键概念回顾**

  - **主机（Host）与设备（Device）：** CPU 和 GPU 的内存是独立的，需要使用 cudaMalloc 和 cudaMemcpy 来进行内存分配和数据传输。
  - **核函数（Kernel）：** 在 GPU 上执行的函数，使用 __global__ 关键字声明。
  - **线程层次结构：**
    - Grid：由多个 Block 组成。
    - Block：由多个 Thread 组成。
    - Thread：执行计算的最小单位。
    - 通过 blockIdx、blockDim 和 threadIdx 来确定每个线程的唯一索引。
  - **执行配置：** 使用 <<<gridSize, blockSize>>> 来指定核函数的执行配置。
  - **内存管理：** 使用 cudaMalloc、cudaMemcpy、cudaFree 等函数来管理设备内存。
  - **错误检查:** 虽然这个例子里没写，但在实际编写中要记得用cudaError_t 和 cudaGetLastError 进行错误检查