
<!-- ---
layout: post
title:  "Optimizing matrix multiplication on NVIDIA Tensor Cores"
date:   2024-06-13 08:52:08 -0600
categories: jekyll update
--- -->

{% include mathjax.html %}

* TOC
{:toc}

# Introduction
This post details my recent efforts to write an optimized matrix multiplication kernel in CUDA using tensor cores on an NVIDIA Tesla T4 GPU. The goal is to compute $D = \alpha * A * B + \beta * C$, where $D,A,B$ and $C$ are large matrices full of half precision floating point numbers, and $\alpha$, $\beta$ are constants. This problem is usually reffered to as a **Ge**neralized **M**atrix **M**ultiply, or **GEMM** for short. My goal is to write a GEMM kernel with comparable throughput to cuBLAS's [Hgemm](https://docs.nvidia.com/cuda/cublas/#cublas-level-3-function-reference) implementation, cuBLAS is a highly optimized, closed source library of hand tuned kernels specific to each GPU architecture released by NVIDIA.

 Tensor Cores are specialized hardware units on NVIDIA chips that implement a small matrix multiplication, with reduced precision operands in hardware. At the lowest level, they multiply two 4x4 matrices in a single clock cycle. I recently became curious about NVIDIA tensor cores after having the following two shower thoughts. First, it seems like [most](https://www.semianalysis.com/p/google-gemini-eats-the-world-gemini) generative AI training and inference these days happens on A100s and H100s. Second, all of this training and inference is almost certainly running on tensor cores, because they offer a **massive** throughput increase for matrix math compared to regular CUDA cores. From [here](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
>An H100 GPU has 989 TFLOPs of half-precision matrix multiply compute, and ~60 TFLOPs of “everything else”. So, every cycle the tensor core is in use, you’re getting at least 94% utilization of the hardware. And every cycle the tensor core is not in use, you’re getting no more than 6% utilization of the hardware.

Given their huge importance in the world today, when I started this project it felt to me like there is disproportionately little info and dialogue on the internet about how to use them directly in CUDA. I quickly learned this lack of info is probably because if you want to write a kernel that uses tensor cores at anywhere close to their full potentional, you need to employ some fairly tricky techniques to keep them fed, which in practice means  efficiently moving bytes through the memory heirarchy of the GPU, and overlapping compute with this data movement. But as with all rewarding engineering endeavors, with a little focus and persistence I started to see the cleverness and elegance in algorithmic details which once seemed uncomfortably complicated and esoteric, which is what moved me to write this article!

The [roofline](https://en.wikipedia.org/wiki/Roofline_model) chart below captures in a nutshell why writing a kernel which achieves peak performance with tensor cores is hard. Roofline charts plot the arithmetic throughput you can achieve on a given piece of hardware as a function of the data reuse of a particular algorithm. The x-axis corresponds to data reuse in units of FLOPs/byte, this is a property of a particular implementation of a kernel. It measures how many FLOPs are performed on each byte read from memory. The y-axis shows FLOP/sec which means at a given level of data reuse, how many floating point operations can this piece of hardware complete in a second. For each roofline, the points on the x axis to the left of the cusp correspond to levels of data reuse that result in an algorithm being "memory bound", that is the floating point throughput we can achieve is limited by how much data we can move onto the chip. The points on the x axis to the right of the cusp correspond to higher levels of data reuse that result in an algorithm being "compute bound", this means we are moving enough data onto the chip to keep our compute units fully fed, and our throughput is limited by the FLOP/s of the device, which is a huge number, rather than the memory bandwidth, which comparitively is a smaller number. In this case for the Tesla T4, the tensor cores are capable of close to $65*10^{12}$ or $65,000,000,000,000$ half precision FLOP/s, which is about 8x the throughput of CUDA cores performing single precision math. The FLOP/s number found on the spec sheet is usually conditioned with the word "theoretical" because in practice, it is physically impossible for the gpu to achieve this throughput for any sustained amount of time without melting or catching on fire, for an excellent explanation of why this is see [here](https://www.thonking.ai/p/strangely-matrix-multiplications) 
![roofline](/images/roofline1.png)



From the perspective of algorithm design, this huge increase in throughput of tensor cores is something of a blessing and a curse. The blessing is that tensor cores are powerful, their theoretical max throughput is 8x that of CUDA cores. The tricky part is that in order to reach peak performance with tensor cores, according to the simplified view of this roofline model we need to write an algorithm that has about 8x more data reuse than an algorithm which achieves peak single precision performance.

In practice, there are lots of details this chart does not capture. It simplifies the memory heirarchy of a GPU down to 2 storage types, one large and slow and the other small and instantaneous. Also, the throughput of our kernel is not only a function of data reuse, the other major factor is the level of concurrency we can achieve between data transfer and compute. All models are wrong, but some are useful, and I find that looking at this chart gives me intuition about why this project is so f***ing hard!

I was inspired to work on this after reading [this](https://siboehm.com/articles/22/CUDA-MMM) excellent article in which the author takes on the same endeavor, but for single precision math. I naively thought something along the lines of "I'll just write something similiar to kernel 10 from Simon's article, swap in some inlined PTX to call tensor cores inside the inner loop, and call it a day!" This turned out to be incorrect, the kernels discussed in this article start from a similiar place to where Simon's article ends. I am trying to make this article as self contained as possible, which necessarily means it is a bit long and meandearing but if you are really interested in this I would highly recommend reading that one first.

# Background
## How to write a fast matrix multiplication, chapter 0
If you are interested in hardware accelerated linear algebra, one of the coolest and most innovative projects happening right now is called NVIDIA [CUTLASS](https://github.com/NVIDIA/cutlass). In their words:
>CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance matrix-matrix multiplication (GEMM) and related computations at all levels and scales within CUDA.

It provides a set of modular and composable building blocks that provide abstractions over some of the finicky lower level details that are essential for performance if you are trying to write fast matmul-like kernels. The abstractions provided by CUTLASS also make it easier to achieve both performance and portability across a range of GPU architectures, which is helpful because there is usually a tradeoff between these two aims. At the top of their README, and at the beggining of almost every CUTLASS slide deck, you usually see something that looks like [this](https://github.com/NVIDIA/cutlass/blob/main/media/images/gemm-hierarchy-with-epilogue-no-labels.png)
![gemm_hierarchy.png](/images/gemm-hierarchy.png)
This is a visualization of algorithmic technique called hierarchical tiling, it is the current paradigm for writing performant number crunching algorithms on computers with multi-level memory hierarchies (which is pretty much every non-embedded computer we have today). I will come back to this image in a bit, but in order to build some intuition about what hierarchical tiling is and why we use it, I think it is helpful to go through an example of how being conscious of our computer's memory heirarchy can speed up a matrix multiplication.

## How to not be memory bound (simple memory hierarchy)

In the 70 or so years it has been since humanity started building transistor based computers, the capacity for performing arithmetic has been growing along the moores law exponential, while the capacity for moving data from where it is stored to where it is computed upon has not been growing exponentially. This problem is called the [memory wall](https://en.wikipedia.org/wiki/Random-access_memory#Memory_wall) and it is one of the central problems in computer architecture today, especially for the heavy number crunching sorts of workloads that are required for running neural networks.

The two dotted lines in the roofline plot above show the "balance point" for CUDA cores (blue) and for tensor cores (red). The balance point is a property of the hardware, it tells us the level of data reuse an algorithm must achieve if want our throughput to be limited by our processors capacity for arithmetic which is a very big number, rather than memory bandwidth which comparitively is a smaller number. Data reuse (also sometimes known as arithmetic intensity) is measured in terms of $\frac{FLOPs}{byte}$, that is how many FLOPs we perform on each byte read from memory. The higher the data reuse, the better chance we have at getting full utilization from our compute cores, as opposed to having them spend idle time waiting for data to arrive from memory.

The balance point for tensor cores according to the simplified roofline model is ~200, this means we need perform ~200 floating point operations for each byte we read from memory. Intuition tells us this should be possible. The computational complexity of a regular matrix multiplication is $O(n^3)$, whereas there is only $O(n^2)$ data involved. The ratio of FLOPs to data we need to move is $O(\frac{n^3}{n^2})=O(n)$. This is good news because $n$ is large, maybe 2048 or 4096 or 8192, way larger than the tensor core's balance point of ~200. So pencil and paper tells us that for a matrix multiplication with large inputs, it should be possible to get our full moneys worth from the tensor cores. Now the question becomes, how can we write an algorithm that exploits this potential for data reuse?

The most basic implementation of matrix multiplication we can write looks like this:
![basic_matmul](/images/basic_matmul.png)

What is the data reuse? This algorithm consists of three nested loops, in the inner loop we are performing 2 reads and 1 write. We are also performing one multiply and one accumulate. Since the memory access and compute is inside three nested loops each going to $N$, this means the algorithm has $2N^3$ compute and $3N^3$ memory access, so the ratio of data reuse is $\frac{2N^3}{3N^3}=0.66ish$. This small number, compared to our balance points of 25 and 200 for CUDA cores and tensor cores respectively, tells us that running this algorithm will result in our compute units spending most of their time idle.

We can do better by considering the memory hierarchy of the computer that this algorithm will run on. As a toy example, lets our imagine our computer has two types of memory, large/slow and small/fast. The former has infinite capacity, but we pay a cost for transfering out of it. The latter can only fit a small amount of data, but access is instantaneous and free. If we modify our initial matrix multiplication so that as many of possible of our memory accesses hit the fast memory rather than the slow memory, since access to the fast memory is free, the denominator of our data reuse ratio $\frac{FLOPs}{byte}$ will get smaller. The number of FLOPs we are doing will remain unchanged, so the overall value of the ratio will get larger, meaning we have improved data reuse. Here is a sketch of the (toy) memory hierarchy aware version: 
![tiled_matmul](/images/tiled_matmul.png)

Like the first matrix multiplication, there are three nested loops, but in this version the three loops are iterating over $t$ by $t$ tiles rather than individual elements. Inside the third loop nest, we are transfering a tile of A and B each with $t^2$ elements from slow memory to fast memory. For each tile of A and B, we then have everything we need to compute a corresponding tile of D from entirely within fast memory, so we aren't paying any cost for memory access. In the diagram above, we only pay for memory accesses made on the left side of the dotted line, which is the part that corresponds to slow memory. Asymtotically, we are making $O((\frac{N}{t})^3 * t^2) = O(\frac{N^3}{t})$ slow memory accesses, and performing $O(N^3)$ compute (as with any regular, i.e. non [strassen](https://en.wikipedia.org/wiki/Strassen_algorithm) matrix multiplication). So our data reuse works out to $O(\frac{N^3}{\frac{N^3}{t}})=O(t)$. This means that the # of FLOPs performed per byte read from memory is proportional to the dimension of our tile size. The larger the tile size, the better chance we have at not being limited by memory bandwidth.

## How to not be memory bound (real GPU memory hierarchy)
Modern GPUs have more than two levels in their memory hierarchy, fast memory is not instantaneous, and slow memory is not infinite. The simplified example above is meant to provide intuition about data reuse, and how we can use out memory hierarchy to keep our compute units busy. But if we wanted to write a tiled matrix multiplication meant to run on a real GPU, how might we go about it? When designing a tiled algorithm for a real GPU, we need to consider these three types of hierarchy that we are dealing with:
* Hierarchy in the matrix multplication: a large matrix multiplication can be broken down into lots of small matrix multiplications, and each of those small matrix multiplications can be further broken down into smaller matrix multiplications.
* Hierarchy in the memory: lets focus on three levels in the memory hierarchy, in order from large/slow to small/fast: off chip DRAM -> on chip SRAM -> register memory.
* Hierarchy in the compute: at the highest level there is the whole GPU. Within the GPU there are multiple streaming multiprocessors (SMs). Within each SM there are multiple [warps](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture). Within each warp, there are 32 threads.

The idea behind hierarchical tiling is to creating mappings between corresponding levels in these hierarchies, such that each level of compute is assigned a portion of the problem that fits inside the type of memory that corresponds to that level of compute. Here is a cropped version of the image from CUTLASS shown again, with some annotations I added ![my_tiles.png](/images/cutlass_tiles.png).

As we go from left to right in the image above, we are moving down levels in the hierarchy. On the far left, we have the whole GPU, responsible for the whole matrix multiplication problem, all of which is stored in DRAM. The matrices are broken down into tiles, each tile of the output and the corresponding tiles of the input is transfered to shared memory, which is fast on chip SRAM memory that is local to a single SM, and consequently local to the thread block which is running on that SM. Each of these tiles in shared memory is further broken down into smaller tiles that can fit in the register memory of a single warp. The compute, which in our case is tensor core operations, happens to/from register memory. In this heirarchical structure, faster memory types are used as a sort of "scratch pad" for slower memory. Data is transfer done in blocks from slower memory to faster memory, amortizing the fixed costs of requesting DRAM/SRAM over lots of elements, and making the most of the limited memory bandwidth. All of the reads and writes of individual elements are done out of and into registers, which is the fastest and smallest level of memory, with negligible latency and efficient granular access to invidiual values.

## How to use Tensor Cores

Part of why GPUs are good for things like matrix multiplication is that instructions are issued to 32 threads at a time, this can be extremely efficient because the overhead of instruction issue is amortized across 32 threads (as opposed to 1 for a CPU). However, the 32 threads in a warp are still independent, [predication](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#control-flow-instructions) and other fancy compiler techniques are used to allow a thread within a warp to take its own execution path dependent on the value of data in its private register memory.

All tensor core operations are performed at the warp level in the compute hierarchy; 32 threads collaboratively load data into their registers and then sychronously execute a small hardware accelerated matrix multiply. When thinking about tensor core algorithms, we should think of the warp as an atomic element of compute, even though in reality a warp contains 32 threads capable of doing their own thing. If we were writing a tensor core free GEMM kernel, threads would be our atomic compute elements. 

Tensor cores are accessible via two different methods. The first is via the `wmma` [api](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma-description) which is part of the CUDA toolkit. `wmma` seems to be regarded as the more portable and less performant way to program tensor cores. I gave up on it pretty quickly, as it abstracts away the loading of input data from shared memory into register memory, and it turns out there are some details here which are critical for performance.

The other route is to use the `mma` family of instructions which are part of PTX, this option is more flexible and performant than the `wmma` route. PTX is an intermediate representation for NVIDIA GPUs that is lower level than CUDA, but higher level than SASS (this is the assembly language that NVIDIA GPUs run). PTX can be inlined in a kernel in order to call tensor cores.

The PTX instruction I used is `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16` (documentation [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k8)), each part of this instruction means something:
* `mma`: we are performing a matrix multiply accumulate operation
* `sync`: this instruction is sychronous, all 32 threads will wait until all 32 threads are done before resuming execution
* `aligned`: all 32 threads in a warp must execute this instruction, if less than 32 threads in a warp were to execute this instruction, behavior is undefined
* `m16n8k8`: this is the identifier for the matrix fragment shape. This means the fragment of matrix 
$A$ has shape (16,8), the fragment of $B$ has shape (8,8), the fragments of $D$ and $C$ have shape (8,8). (Remember, the formula for a GEMM is $D = \alpha * A * B + \beta * C$). If you look at the PTX documentation linked above, there are lots of different shapes to choose from, however the Turing/Volta architectures only support a limited number. Ampere supports more, and Hopper supports even more.
* `row`: the $A$ fragment should be stored in registers in a row-major layout
* `col`: the $B$ fragment should be stored in register in a column-major layout
* `f16`: $D$ is an fp16 matrix 
* `f16`: $A$ is an fp16 matrix
* `f16`: $B$ is an fp16 matrix
* `f16`: $C$ is an fp16 matrix

Each `mma.sync` instruction expects a specific layout of fragment elements across the registers of the 32 threads in a warp, these layouts can be found in the PTX docs. Here is the `m16n8k8` layout:
![matrix_fragments](/images/mma_fragments.png)

These diagrams are describing a mapping between threads, registers, and matrix elements:
* `T0, T1, T2 ...` refers to the index of the thread. Thread indices in these diagrams range from 0-31 since there are 32 threads in a warp.
* `a0, a1, a2, ... b0, b1, b2, ... c0, c1, c2` refer to registers that hold matrix elements.
* The position of each thread/register pair tells us which matrix elements go in which registers of which thread. For example, `T0: {a0,a1}` is at the top left corner of matrix fragment A, this means elements `(0,0)` and `(0,1)` in this fragment are placed in registers `a0` and `a1` of thread 0.

Luckily there is another PTX instruction called `ldmatrix` (docs [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix)) which can efficiently load data from shared memory, and shuffle matrix elements within a warp in order to create this layout for us. It can optionally transpose matrix elements as it moves them from shared memory to register, which is convenient for matrix B above, which is in a column major, or "transposed" layout.

#### (digression)
Given how many people and companies these days are buying NVIDIA GPUs almost exclusively for the purpose of running matrix multiplications, it seems like lots of work goes into improving the tensor cores in terms of programmability and performance between successive architectures. On one hand, the tensor core throughput goes up by an order of magnitude with each new SM architecture. On the other hand, this means that in order to make this increased throughput actually usable (rather than theoretical), new hardware support is required for keeping them fed. For example, Ampere introduced hardware support for [asychronous data copying](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#asynchronous-data-copy-from-global-memory-to-shared-memory) from global memory to shared memory (more on this later). Hopper introduced something even fancier called the Tensor Memory Accelerator or [TMA](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator), this is a dedicated piece of hardware that is purpose built for feeding tensor cores, it can perform index calculation and initiate data copies asychronously with respect to the rest of the SM. Hopper is kind of a new and different beast, if you look at GEMM kernels in CUTLASS that target Hopper the code has a different stucture than all of the other pre `sm_90` kernels. Hopper kernels use a producer/consumer pattern, where a relatively small number of producer threads are initiating asynchronous data copies with the TMA, and then consumer threads are managing the tensor cores. I have never worked on kernels targetting Hopper so I dont know much about this at the moment, [this](https://hazyresearch.stanford.edu/blog/2024-05-12-tk) article provides a great overview of the user experience of writing kernels for Hopper.

**This is all to say** that the kernels discussed here target the Turing architecture, which was SOTA in 2018, and if you are writing kernels targeting Ampere or Hopper, the techniques you employ for latency hiding will be different and easier. I used the Tesla T4 GPU because you can rent them on AWS for ~50 cents/hour, which is about as much money as I want to spend on EC2 instances. Using an older GPU was a blessing and a curse for this project, the curse was that no special hardware support was available for hiding memory latency, the blessing was that I had to use more old school techniques for hiding this latency, which gave me an appreciation for why this hardware support exists now! 

# Kernels

For the rest of this article I will discuss a series of 5 kernels that got me to ~80% of cublas level performance on a tensor core GEMM. Each kernel builds on the previous one, and the themes of each are:
1. hierarchical tiling
2. vectorized/unrolled gmem->smem transfer
3. shared memory swizzling
4. async global memory copies (poor mans version)
5. fast shared memory index calculations
6. tuning tile dimensions

## Kernel 1 - Hierarchical Tiling
The structure of Kernel 1 provides a foundation that I iterated on for the next 4 kernels. It has a 4 level hierarchical tiled structure as shown in the image below. Based on some experiments, for a single precision GEMM calculation not running on tensor cores, a 4 level tiling structure similiar to this one is sufficient to keep the compute units fully fed, and get close to cuBLAS performance. However, for a tensor core GEMM, it is only the starting point! If at this point you find yourself confused, and are interested in reading about a series of 10ish kernels that build up to a kernel like this one, I highly recommend [this](https://siboehm.com/articles/22/CUDA-MMM) article.

![tiling](/images/my_tiles_2.png)

This diagram turned out a bit hectic, but I was trying to show how the loop nests create the tiling structure, and how we start with the whole GPU, full matrix view at the top, and go all the way down to a single tensor core op that produces a 16 by 8 element tile of the output. Each level (except the top one) shows a zoomed in view of a particular position of the tiles which are being looped over at the level above.

 Here is a quick description of each level from the perspective of the compute unit relavent for that level:

* **CUDA Kernel / GPU level**: The GPU is reading the three input matrices, $A$, $B$, and $C$ from global memory, and writing the output matrix $D$ to global memory. Each thread block is looping over the `K` dimension (aka the 'inner' dimension) of $A$ and $B$. This loop is incrementing `block_k` in steps of size `BK`. At each iteration we are copying the blue blocktiles from global memory to shared memory.

* **Thread Block / SM level**: At this point the blue subtiles of $A$ and $B$ that a particular thread block needs to compute a `BM,BN` tile of the output have been copied into shared memory. This thread block is running on one of the 16 SMs on the GPU, and the shared memory is local to that SM and fast to access. Within the thread block there are 256 threads, which is 8 warps containing 32 threads each. Within the thread block, the `BM,BN` tile of the output is partitioned 8 ways, so that each of the 8 warps can work concurrently on the compute. Each of the warps is looping over the inner dimension within the block tile, this loop is incrementing `warp_k` in steps of size `WK`. At each iteration we are copying the green warp tiles from shared memory to register memory.

* **Warp / SM Partition**: At this point the green warp tiles within the blue block tiles have been copied into register memory, and it is the responsibility of a particular warp, running on one of the 4 partitions on the [Turing SM](https://images.app.goo.gl/Z2VVQQgXWTMddBraA) to compute the `WM` by `WN` tile of the output. Each warp computes its tile of the output by taking an outer product between the `WM,WK` tile of A and the `WK,WN` tile of B. Inside the three nested loops that compute the outer product, the we an MMA sync operation.

* **Tensor Core Op**: Finally we get down to the last level of the hierarchy, which is a single tensor core op, this is a single hardware accelerated (16,8) x (8,8) = (16,8) matrix multiply.

Here is some pseduocode all in one place, for readability, I am pretending all of the matrices involved are numpy arrays.


```c++
// outer loop over block tiles
for (block_k = 0; block_k < K; block_k += BK)
{
    // global memory to shared memory transfer
    A_smem[:,:] = A_gmem[block_m:block_m+BM, block_k:block_k+BK]
    B_smem[:,:] = B_gmem[block_k:block_k+BK, block_n:block_n+BN]
    
    // sychronize across the thread block
    __syncthreads();

    for (warp_k = 0; warp_k < BK; warp_k += WK)
    {
        A_reg[: ,:] = A_smem[warp_m:warp_m+WM, warp_k:warp_k+WK]
        B_reg[:, :] = B_smem[warp_k:warp_k+WK, warp_n:warp_n+WN]

        for (mma_k = 0; mma_k < WK; mma_k += MMA_K)
        {
            for (mma_m = 0; mma_m < WM; mma_m += MMA_M)
            {
                for (mma_n = 0; mma_n < WN; mma_n += MMA_N)
                {
                    mma_sync_m16n8k8(
                        acc_reg[mma_m:mma_m+MMA_M, mma_n:mma_n+MMA_N],
                        A_reg[mma_m:mma_m+MMA_M, mma_k:mma_k+MMA_K],
                        B_reg[mma_k:mma_k+MMA_K, mma_n:mma_n+MMA_N],
                        acc_reg[mma_m:mma_m+MMA_M, mma_n:mma_n+MMA_N]
                    )

                }
            }
        }
    }
    __syncthreads();

}
```

This kernel (code [here](https://github.com/alexarmbr/matmul-playground/blob/main/src/kernel1.cu)) is the starting point, it does not get close to cuBLAS level performance.
![table1](/images/table1.png)


## Kernel 2 - Vectorized memory copy and loop unrolling
In order to improve the performance of our code, we need to know why it is slow. When writing CUDA kernels, the best tool to use for this is called NSight Compute, a profiler developed by NVIDIA that gives lots of detailed metrics about how a kernel is interacting with the hardware. The first place I typically look is the section called "Warp State Statistics". As a kernel is executing, each warp is being issued instructions by a scheduler. In an ideal world, the scheduler would be able to issue a new instruction each clock cycle. In the real world, it is very hard to write a kernel that can issue a new instruction every cycle, there are all sorts of reasons why on a given cycle, a warp may not be capable of executing its next instruction and will instead "stall" i.e. do nothing. The reasons for stalling can be due to capacity limits of various hardware pipelines, memory latency, or sychronization points in our kernel which require all the threads running on an SM to wait for all the other threads to catch up. The Warp State Statistics section tells us how many clock cycles the average warp spends stalled, per average instruction issued, broken down across a bunch of different categories. This gives us the information we need to target our optimizations to the least performant parts of our kernel. Here is a screenshot of what the Warp State section for Kernel 1.
![warp_state_kernel1](/images/warp_state_kernel1.png)
The "Warp Cycles Per Issued Instruction" field tells us that on average for each instruction issued, warps spend about ~30 cycles idle, and the table below tells us that 16 of these 30 cycles are due to the "Long Scoreboard" stall category. 

[Scoreboarding](https://en.wikipedia.org/wiki/Scoreboarding) is an algorithm implemented in the hardware of most processors for tracking when the data dependencies for the next instruction have arrived in the registers they need to be in for the instruction to execute. Most modern CPUs are able to reorder instructions on the fly such that instructions whose operands are ready can execute ahead of instructions whose operands have yet to arrive in registers. The reordering is done in hardware, subject to constraints imposed by the data dependencies between subsequent instructions. This is called [out of order execution](https://en.wikipedia.org/wiki/Out-of-order_execution) and it is a rather fancy technique for hiding latency. GPUs generally do not reorder instructions, I would imagine because the logic required for reordering instructions consumes a fair amount of precious transistors on the chip, and since GPUs are designed for [throughput](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-benefits-of-using-gpus) these transistors are better spent on things like tensor cores.

GPUs do however perform scoreboarding in hardware, and when the data required to execute the next instruction has not arrived in register memory, the warp that is executing just waits for its data to arrive. The "Long Scoreboard Stall" counts the average number of cycles that warps spend stalled waiting for data to arrive from global memory. The fact that this stall reason accounts for ~50% of all the cycles that warps spend idle tells us that the performance of Kernel 1 is primarily limited by **memory latency**. This tells us we should focus on the code that is moving data from global memory onto the chip, and figure out how to minimize the latency per byte moved.

Reading a rectangular tile of data from global memory, and writing it to shared memory is the first thing that occurs on each iteration of the outer loop of the kernel. The easiest way to do this is for adjacent threads to access adjacent values in global memory, and write data to shared memory in the same layout that it came from in global memory. This access pattern is optimal both for reading global memory, and writing shared memory. Here is the first data transfer that I wrote:

```c++
__device__ void tileMemcpy(
    half* src,
    half* dst,
    const unsigned int src_stride,
    const unsigned int tile_rows,
    const unsigned int tile_cols
)
{
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int num_threads = blockDim.x * blockDim.y;
    
    // # of threads is multiple of # of columns in the tile
    assert(num_threads % tile_cols == 0);
    
    // assign each thread a row/column in the tile, calculate the row step
    const unsigned int row_step = num_threads / tile_cols;
    const unsigned int thread_row = thread_idx / tile_cols;
    const unsigned int thread_col = thread_idx % tile_cols;
    
    for (unsigned int r = thread_row; r < tile_rows; r+=row_step)
    {
        dst[r * tile_cols + thread_col] =  src[r * src_stride + thread_col];
    }
}
```
Looking at the SASS corresponding to this `tileMemcpy` function in [godbolt](https://godbolt.org/z/1MeavE3GG), we can see that the copy operation inside the loop `dst[...] = src[...]` is actually two operations from the lower level perspective of SASS: first, a load of `sizeof(half) = 2 bytes` from global memory into a register, this is `LDG.U16` in SASS, and then a store of 2 bytes from the register to shared memory, this is STS.U16. The long scoreboard stall prevents the store from taking place until the value we are loading has arrived in the register.

Here is a visualization of how this loop is executing, for a single thread:
![memory_latency](/images/memory_latency.png)
Latency in between the load and the store is inevitable: a request is sent to a DRAM controller, data is fetched from DRAM and then transmitted over bus. Unless we hack the laws of physics or invent a time machine we can't get rid of the latency. But what we can do is **hide** it.

Latency hiding is a central concept in high performance computing, and at its core is very simple. It just means that if we are performing an operation $X$ that has some latency, we want to be doing other useful work while $X$ is happening, rather than wait and do nothing. For example, if I wake up and decide I want an omlette, I would first turn on the burner and let the pan warm up, and while that is happening I would crack the eggs and grate cheese. This order of operations hides the latency of warming up the pan with the cracking of eggs and grating of cheese. If I am hungry and eager to eat the finished omlette as soon as possible, it would be silly to idly stand there and watch as the pan warms up.

The same principle applies to hiding the latency of the global memory loads in `tileMemcpy`. Since the copy operation is happening inside a loop, each thread is performing multiple loads and multiple stores, in an order like `load (stall) store, load (stall) store, ...`. What if we were able to rearrange these so that the order is `load load load (stall) store, store, store`. In this later ordering the loads are happening concurrently, and we can say that the latency of each load is being hidden by the other loads. The easiest way to accomplish the later ordering is by unrolling the loop in `tileMemcpy`. If we can unroll the loop, `nvcc` should be smart enough to reorder the instructions so that the global memory loads are hiding each others latency. In this case the compiler is doing for us what a CPU would do in hardware on the fly. 

If we want to unroll the loop, the number of loop iterations must be known at compile time. The number of loop iterations is a function of the number of threads per block, and the block tile dimensions. Both of these are fixed at compile time, so passing them as template parameters into `tileMemcpy` and calculating the number of iterations as a function of these, and adding a `#pragma unroll` does the trick.

```c++
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpyUnrolled(
    half* src,
    half* dst,
    const unsigned int src_stride
)
{
    // # of threads is multiple of # of columns in the tile
    static_assert(NUM_THREADS % TILE_COLS == 0);
    
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS;
    const unsigned int thread_col = thread_idx % TILE_COLS;
    
    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++)
    {
        dst[thread_row * TILE_COLS + thread_col] =  src[thread_row * src_stride + thread_col];
        thread_row += ROW_STEP;
    }
    
}
```
This gives us something more along the lines of:
![memory_latency_unrolled](/images/memory_latency_unrolled.png)
In the initial version, the total latency of the copy operation is roughly proportional to the memory latency of the device, times the number of loop iterations. After unrolling the loop, compared to the naive version, the total latency is reduced by the number of loads the compiler decides to overlap with eachother.

The other fairly easy optimization we can make here is to increase the number of bytes being loaded per instruction. Our load operation is currently compiling to `LDG.U16`, each of these instructions loads 16 bits/2 bytes from DRAM. The widest load instruction in SASS is `LDG.128`, which loads 128 bits/16 bytes. Since our kernel is bound by memory latency and not memory bandwidth, if we use a wider load instruction will experience the same latency per memory request, but move more bytes per request. We are amortizing the latency over more bytes moved, which is a win for efficiency.

![memory_latency_vectorized](/images/memory_latency_vectorized.png)

A quick and hacky way to accomplish this is by `reinterpret_cast`ing the `src` and `dst` pointers from `half` to `float4`, and updating the index and loop calculations accordingly. Here is a [godbolt link](https://godbolt.org/z/v3T3x14ns) to a kernel with the vectorized and unrolled memory copy, and [here](https://github.com/alexarmbr/matmul-playground/blob/main/src/device_utils.cuh#L73) is the code.

These optimizations to the memcpy increase the throughput over the first kernel by about 3x. But there is still a long way to go before we approach cuBLAS level performance
![table2](/images/table2.png)

## Kernel 3 - Shared Memory Swizzling
Back to the warp state section of NSight Compute
![kernel2_nsight_compute](/images/kernel2_nsight_compute.png)
The long scoreboard stall is no longer the leading offender in terms of warp stalls, and our kernel got about 3x more performant after applying the optimizations described in the last section. Warps are now spending an average of ~19 cycles stalled per issued instruction due to something called "MIO Throttling." What is MIO Throttling, and how do we address it? According to nsight compute [docs](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) this means:
>Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions.

In our case, this stalling is almost certainly due to shared memory instructions, since our kernel has very few dynamic branches, and no trigonometry or any other [special math](https://developer.nvidia.com/cuda-math-library) instructions. Specifically, it is due to shared memory bank conflicts. According to [here] the two main symptoms of shared memory bank conflicts are very high L1/TEX thoughput number (currently at 97% of peak) and MIO Throttle stalls.