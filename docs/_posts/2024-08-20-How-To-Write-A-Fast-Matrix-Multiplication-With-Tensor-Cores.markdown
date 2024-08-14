
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
This post details my recent efforts to write an optimized matrix multiplication kernel in CUDA using tensor cores on a NVIDIA Tesla T4 GPU. The goal is to compute $D = \alpha * A * B + \beta * C$, as fast as possible. In this equation $D,A,B$ and $C$ are large matrices full of half precision floating point numbers, and $\alpha$, $\beta$ are constants. This problem is usually referred to as a **H**alf-precision **Ge**neralized **M**atrix **M**ultiply, or **HGEMM** for short. 

 Tensor Cores are specialized hardware units on NVIDIA chips that implement a small matrix multiplication in hardware. I recently became interested in tensor cores for two reasons. First, it seems like [most](https://www.semianalysis.com/i/136469751/the-gpu-rich) generative AI training and inference these days happens on A100s and H100s. Second, all of this training and inference is almost certainly running on tensor cores, because they offer a massive throughput increase for matrix math as compared to what you get if you dont use them. From [here](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
>An H100 GPU has 989 TFLOPs of half-precision matrix multiply compute, and ~60 TFLOPs of “everything else”. So, every cycle the tensor core is in use, you’re getting at least 94% utilization of the hardware. And every cycle the tensor core is not in use, you’re getting no more than 6% utilization of the hardware.

Given their huge importance in the world today, when I started this project it felt to me like there is disproportionately little info and dialogue on the internet about how to use them directly. I quickly learned this lack of dialogue on the internet is probably because writing algorithms that use them is a bit of a niche interest. The basic mechanics of how to call them are not hard, however writing a kernel that can use them at anywhere close their full potential *is* hard. Their huge throughput means that in order to use them at anywhere close to their full potential, you need to move bytes though the memory hierarchy of the GPU in a maximally efficient way, and overlap the computing with this data movement. There are certain algorithmic techniques that you need to use if you want get your moneys worth from your tensor cores, this article is an exploration of these techniques. I figured out the implementation details mostly by digging around the NVIDIA [CUTLASS](https://github.com/NVIDIA/cutlass/tree/main) forums and source, and I wrote this article in order to make sure I actually understand what I am doing, and also in the hope that some fellow GPU nerds trying to work with tensor cores might find it helpful.

When I started my goal was to write a kernel with comparable performance to the cuBLAS [hgemm](https://docs.nvidia.com/cuda/cublas/#cublas-level-3-function-reference) implementation, which is the closed-source, gold standard implementation released by NVIDIA. I iteratively optimized a series of 6 kernels, with the first achieving a measly 8% of the cuBLAS throughput, and the last achieving a decent 96% of the cuBLAS throughput for 8192x8192 matrices.

# Background
## The memory wall

In the 70 or so years it has been since humanity started building transistor based computers, the capacity for performing arithmetic has been growing along the moores law exponential, while the capacity for moving data from where it is stored to where it is computed upon has not been growing exponentially. This problem is called the [memory wall](https://en.wikipedia.org/wiki/Random-access_memory#Memory_wall) and it is one of the central problems in computer architecture today, [especially](https://horace.io/brrr_intro.html) when it comes to deep learning workloads, and especially especially when it comes to tensor core algorithms. What this means for us is that if we want to be able to use the ~65 trillion FLOPs per second that our tensor cores are capable of, moving the corresponding number of bytes per second from DRAM may be a challenge.

## Roofline charts
The [roofline](https://en.wikipedia.org/wiki/Roofline_model) model allows us to think about this conundrum a bit more precisely. The basic idea is that we imagine a simplified computer with a two level memory hierarchy, fast memory and slow memory. We can only perform computation on data that is resident in fast memory, at a peak rate of $\tau$ FLOP/sec. The slow memory has unlimited size, and it can move $\beta$ bytes/sec of data into the fast memory. Because of the memory wall, $\tau$ is way larger than $\beta$.

![simple_computer](/images/simple_computer.png)

Any given computation has a certain number of FLOPs that need to be performed, for example to multiply a $M$ by $K$ matrix with a $K$ by $N$ matrix we need to perform $2 * M * N * K$ FLOPs. The more FLOP/sec our algorithm can achieve, the faster we can get the matrix multiplication done. The roofline model gives us an upper bound on the FLOP/sec we can achieve, subject to $\tau$ and $\beta$ which are fixed properties of our hardware. We will refer to achieved FLOP/sec as $T$ for throughput, and the upper bound on T as $T_{max}$.

The maximum FLOP/sec we can achieve ($T_{max}$) is modeled as a function of a variable called *computational intensity* or $I$ for short, this is a property of the algorithm we write. This metric measures the "data reuse" of our algorithm in units of FLOP/byte: for each byte moved from slow memory to fast memory, how many FLOPs do we perform on it. According to the roofline model, if you are an algorithm designer, your primary concern is to write an algorithm with high computational intensity, or in other words to maximize $I$. In practice, this means moving a chunk of data from slow memory to fast memory, and then performing as many useful operations on it as allowed by whatever algorithm you are writing. Reusing data in fast memory is important for performance, because our memory bandwidth $\beta$ limited; it is a small number compared to $\tau$, which means the transfer of this chunk of data from slow to fast memory is costly. We make the most of it by performing as many useful FLOPs as possible on it.

The roofline model says the upper bound on FLOP/sec ($T_{max}$) we can achieve is the minimum of our computational intensity times memory bandwidth, and the peak floating point throughput of our hardware.

$$ T_{max}=min(\beta * I, \tau) $$

This model is saying there are two ways $T_{max}$ can be limited:
- $T_{max}$ can never exceed $\tau$. Even if we perform infinity operations on each byte we move into fast memory, we are still limited by the peak floating point throughput of the hardware. $\tau$ is typically a very big number, for example for the T4 GPU, $\tau$ equals 65,000,000,000,000 FLOP/second. If $\tau$ is our limiting factor, we are in good shape, this scenario is referred to as being *compute bound*.
- However, $T_{max}$ may also be limited by the memory bandwidth of the device, times the computational intensity of the algorithm. If $\tau$ were infinite, the achieved floating point throughput would simply be the number of bytes/sec being moved into fast memory, times the number of FLOPs performed per byte moved, this is $\beta * I$ (notice how when you multiply $\beta * I$, the units cancel out to give FLOP/sec). If $\beta * I$ is less than $\tau$, this term becomes the limiting factor on $T_{max}$, this scenario is referred to as being *memory bound*. The thing to do in this situation is to rewrite your algorithm to increase $I$ in the hopes of your algorithm becoming compute bound.

Here is the whole thing in a picture, notice how we can go from being memory bound to being compute bound by varying $I$:
![roofline](/images/roofline.png)

The red dotted line in this picture is referred to as the "balance point" of the hardware, it is the level of arithmetic intensity in units of (FLOP/byte) that we need to surpass in order to go from being memory bound to being compute bound. If we call this value $I^* $, then $I^* * \beta=\tau$ or equivalently $I^*=\frac{\tau}{\beta}$. It is a property of a particular computer, the peak floating point throughput, divided by the memory bandwidth. Because of Moore's law, arithmetic throughput has been improving much faster than memory bandwidth, the consequence of this is that generally speaking, the newer the computer, the higher the balance point.

## Rooflines for the NVIDIA Tesla T4
Plugging in some numbers specific to the GPU we are using, and looking at the resulting roofline can inform our algorithm design, and give us some perspective on what we are in for. On a real computer, there isn't just a single $\tau$ and $\beta$, there are multiple hardware instructions, each with a different peak throughput $\tau$, and different types of memory, each with a different bandwidth $\beta$.

### Tensor Core vs. FFMA
I found it helpful first to compare the balance point of the tensor cores with the balance point for the regular single precision math units, both with respect to global memory. This rooflines provides some intuition about why writing an efficient kernel is more challenging if you are using tensor core instructions, as opposed to the more standard, less specialized math instructions.

First, we need to know the global memory bandwidth $\beta_{gmem}$ of our device. NVIDIA spec sheets report *theoretical* memory bandwidth, which is [never](https://forums.developer.nvidia.com/t/theoretical-bandwidth-vs-effective-bandwidth/48005/3?u=a14armbr) achievable in practice. The real number can be found with a benchmark, according to [this](https://arxiv.org/pdf/1903.07486) whitepaper the achievable memory bandwidth of the T4 is 220 GB/sec (this is 68% of the 320 GB/sec theoretical memory bandwidth).

Next, we need to know the peak floating point throughput with the tensor core, and the peak floating point throughput without it. Similarly to memory, the theoretical numbers are [not actually achievable](https://www.thonking.ai/p/strangely-matrix-multiplications) without the GPU catching fire or melting. I find it reasonable to use the measured throughput of the cuBLAS half precision (uses tensor cores) and single precision (doesn't use tensor cores) GEMM kernels as the achievable floating point throughput numbers. Looking at the assembly of the cuBLAS half precision kernel we can see that the grunt work is done by `HMMA.1688`, this instruction performs a single small hardware accelerated matmul (more on this later). For the single precision GEMM kernel, the instruction that does the work is called `FFMA`, this is a scalar multiply/accumulate operation, $d=a*b+c$. According to my benchmarks, the tensor core HMMA.1688 throughput is 49439 GFLOP/sec, which we will call $\tau_{HMMA}$. The non-tensor core FFMA throughput is 7455 GFLOP/sec, which we will call $\tau_{FFMA}$. These are respectively 76% and 92% of the theoretical peak throughputs, which seems reasonable enough. The resulting rooflines look like this (these plots are typically shown on a log/log scale, this one is not):

![t4_roofline](/images/t4_roofline.png)

This plot should give us some intuition about the comparative hardness of writing a kernel that achieves peak FLOP/sec with tensor core instructions vs. writing a kernel that achieves peak FLOP/sec with fused multiply add instructions. The hardness comes from the fact that if we want to reach a throughput of $\tau_{HMMA}$, we need ~6.6x more arithmetic intensity than we need if our goal is to reach $\tau_{FFMA}$. The two balance points in this plot tell us that with FFMA instructions we can perform ~33 FLOPs in the time it takes a byte to travel from global memory, whereas with tensor cores we can perform 224 FLOPs in this same amount of time. This means that if we took a kernel that reached peak flops achievable with FFMA instructions, simply replacing the fused multiply adds in the inner loop with tensor core instructions would *not* be sufficient to get high tensor core utilization. We would additionally need to improve the code that moves data around to increase the computational intensity by a factor of six. This is one of the things that makes writing a tensor core GEMM interesting!

### Shared memory vs. L2 cache vs. global memory
If we want to write a kernel that can make good use of the tensor cores, we need to be conscious of our computers memory hierarchy. The roofline model simplifies the memory hierarchy down to two storage types, one large and slow, and the other fast and instantaneous. In reality, there are more than two levels, each level has different a bandwidth and capacity, and also different considerations that must be considered in order to facilitate efficient access.

![t4_memory_hierarchy](/images/t4_memory_hierarchy.png)

In these days of the memory wall, using the faster and smaller levels of the memory hierarchy effectively is critical. This requires some ingenuity because of the smallness: for example on the T4 the on chip shared memory has 16.6x the bandwidth of global memory, but on a given streaming multiprocessor (SM for short) it only fits 64 KiB. If we are multiplying large matrices, this is only enough space to fit a tiny portion of the problem.

![t4_memory_roofline](/images/t4_memory_roofline.png)

The plot compares the balance point of tensor cores with respect to:
- global memory or DRAM, the largest and slowest level of the memory hierarchy
- the L2 cache which stores recently accessed data from DRAM, and is shared between the 16 SMs on the T4
- shared memory, per SM fast memory that is explicitly managed.

Global memory has a balance point of 224, this means that if all of our memory accesses go to DRAM, we will need to perform 224 FLOPs for each byte read from DRAM in order to keep our tensor cores busy. This turns out to be a very tall order, as we will see later when we work out how parameters in our algorithm affect the balance point (the sneak preview is that given the amount of fast memory on the T4 and some other performance considerations, achieving this balance point would be counterproductive). However, the L2 cache comes to the rescue, its balance point with respect to tensor cores is 38, which is a much more manageable number. If a good number of our memory accesses can hit the L2 cache rather than going all the way to global memory, we will have a good shot at being compute bound rather than memory bound. The moral of this story is that we need the L2 cache.

Shared memory is used as an explicitly managed cache that will hold small portions of the input matrices local to a particular SM (a SM is kind of analogous to a single CPU core). Within the SM, threads will load their own local portion of the problem from shared memory into register memory, which is where data must reside in order for it to be computed upon. When shared memory is operating at full bandwidth, its balance point with respect to the tensor core is 13, which means we need to cache enough data in registers to perform 13 FLOPs for each byte read from shared memory. It turns out that each SM has enough register memory to make this easily achievable. When we are optimizing this part of the algorithm, the challenge will be to enable shared memory to operate at full bandwidth, which in practice means organizing the data layout in such a way that we can read it and write it without bank conflicts. Once shared memory is at full bandwidth, sufficient arithmetic intensity will be easy to achieve. I think the shared memory balance point of 13 is worth noting though, because it tells us that shared memory alone is not fast enough to achieve peak tensor core throughput. The moral of this story is that we need registers.

<!-- #### digression on registers
Registers are the fastest and highest bandwidth level of the memory hierarchy, and consequently the most precious hardware resource for an algorithm designer. When talking about the memory hierarchy we generally assume that faster means smaller, however this is not the case when comparing registers with shared memory. The [Turing generation](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html) SM has 64KiB of shared memory, and 256KiB of registers. [Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#nvidia-ampere-gpu-architecture-tuning) and [Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) SMs also have more register memory than shared memory, although the gap is smaller. An interesting thought experiment: given that shared memory is smaller and slower than register memory why do we use it at all? Put differently, if we were to write a GEMM kernel without using shared memory, why might it be harder than if we do use shared memory? This question got me thinking about how the tradeoff between different types of memory is more than just size vs. speed. -->

## Theoretical arithmetic intensity
So modern computers generally have an imbalance between their arithmetic throughput and their memory bandwidth, consequently kernels that perform lots of arithmetic relative to data movement make better use of the hardware. At this point we need to think about the algorithm we are running, and forget about hardware for a moment.

### Matrix Multiplication vs Matrix Addition

 Any given algorithm has a maximum amount of arithmetic intensity that is possible, and our goal as an algorithm designer is to write a kernel that achieves an arithmetic intensity as close to this upper bound as we can manage. Comparing the maximum arithmetic intensity that is achievable when adding two $N$ by $N$ matrices, vs. multiplying them, illustrates how different algorithms have different upper bounds in this regard.

![multiplication_vs_addition](/images/multiplication_vs_addition.png)

In the case of matrix addition, computing a single output element requires a single arithmetic operation, which means that when we run this algorithm the amount of data movement and compute will always be directly proportional. If we are adding two $N$x$N$ matrices, the amount of data involved is $O(N^2)$, and the amount of compute required is also $O(N^2)$. So the ratio of compute to data is $\frac{O(N^2)}{O(N^2)}=O(1)$, which means matrix addition will probably be memory bound on any modern device, regardless of how clever an algorithm we write. Relative to the amount of data movement, there just isn't that much math required, so the upper bound on achievable arithmetic intensity is low. Lots of operations in deep learning fall into this low arithmetic intensity category, a technique called kernel fusion can be helpful here.

Matrix multiplication however is not doomed to be memory bound, because there is more arithmetic required relative to the problem size. When multiplying two $N$ by $N$ matrices, the amount of data involved is also $O(N^2)$, but the amount of compute required is $O(N^3)$ ($O(N)$ operations per output element, times $O(N^2)$ output elements). So the ratio of compute to data is $\frac{O(N^3)}{O(N^2)}=O(N)$. There is a factor of $N$ more compute required than data movement. The upper bound on the arithmetic intensity we can achieve grows with the matrix dimension $N$. If we are multiplying sufficiently large matrices, we should be able to write an algorithm that has sufficient arithmetic intensity to be compute bound rather than memory bound.

So in summary the arithmetic intensity we achieve depends on the kernel we write, and it must be less than or equal to an upper bound imposed by the algorithm our kernel is implementing. Achieved arithmetic intensity, given our machine parameters $\tau$ and $\beta$ determines whether we are memory bound or compute bound. If our algorithms upper bound on arithmetic intensity allows it, we want to optimize our kernel until it is compute bound rather than memory bound.

## Achievable arithmetic intensity on a simple computer
For multiplying two $N$ by $N$ matrices, the best possible arithmetic intensity we can achieve is $O(N)$. Now the question is, how do we think about all of this when it comes time to actually write a kernel? To get at this question we need a model of the computer we are running on, to start out we will use the simple computer with fast and slow memory.

### worst case
The first implementation of multiplication between two N x N matrices ($C=A*B$) on the simple computer looks like this. We load each value as soon as we need it, and store each output as soon as we are done with it. What is the ratio of compute to data movement? Is it close to the ideal of $O(N)$?
```
allocate registers a,b,c in fast memory
for i=1...N:
    for j=1...N:
        c = 0
        for k=1...N:
            load A(i,k) into a
            load B(k,j) into b
            c += a * b
        store c into C(i,j) 

```
My mental model of this implementation looks something like this
![simple_computer_matmul_naive](/images/simple_computer_matmul_naive.png)

This arithmetic intensity of this implementation on the simple computer is $O(1)$, because on each iteration of the inner loop a single multiply/accumulate is performed, and only the data operated on during that iteration is loaded. There is $O(N^3)$ data movement, and $O(N^3)$ compute, which means $\frac{O(N^3)}{O(N^3)}=O(1)$ intensity, which is worse than the ideal by a factor of $O(N)$. This turns out to be the worst case.

### best case
The poor intensity of the above implementation is the result of the fact that we load single elements from fast memory one at a time, only when they are needed. Only three matrix elements at a time are stored in fast memory. We can improve intensity by making better use of fast memory. To illustrate the best case scenario, imagine that fast memory was large enough to fit $A,B$ and $C$ in their entirety. If this were the case we could allocate space in fast memory for $C$, transfer the entire $A$ and $B$ upfront, perform the three nested loops with all the data already present in fast memory, and then once we are done store the entire $C$ matrix all at once back to slow memory.
![simple_computer_matmul_best_case](/images/simple_computer_matmul_best_case.png)
In this case, because we move each matrix only once, data movement is $O(N^2)$. Compute is the same as above, $O(N^3)$. Looking at the ratio of the two, we achieve the best case intensity, $\frac{O(N^3)}{O(N^2)}=O(N)$. However, this is unrealistic, because the entire problem will generally not fit in fast memory.

### realistic case
We want to move more than three elements at a time between slow memory and fast memory. But we can't move the full matrices all at once. We can compromise by moving subtiles of $A$ and $B$ from slow memory to fast memory (as large as we can fit). Each pair of input tiles we move to fast memory corresponds to a tile of the output which can be computed with a mini matrix multiplication between the input tiles we have resident in fast memory. We then move the next pair of input tiles to fast memory and then compute again.

![simple_computer_matmul_realistic_case](/images/simple_computer_matmul_realistic_case.png)

Here is some pseudocode corresponding to the above diagram:
```
Allocate A_tile[BN, BN], B_tile[BN,BN], C_tile[BN,BN] in fast memory

# outer loop over tiles of A and B
for i=1...N in steps of size BN:
    for j=1...N in steps of size BN:
        C_tile[: , :] = 0
        for k=1...N in steps of size BN:
            Load A[i : i+BN, k : k+BN] into A_tile
            Load B[k : k+BN, j : j+BN] into B_tile
            
            # inner loop, do a mini matmul between tiles of A and B
            # store the result in C_tile
            for tile_i=1...BN:
                for tile_j=1...BN:
                    for tile_k=1...BN:
                        C_tile[tile_i, tile_j] +=
                            A_tile[tile_i, tile_k] * B_tile[tile_k, tile_j]
            
        # once we have looped over all the tiles along the K dimension of A,B
        # store C_tile back to its place in slow memory
        Store C_tile into C[i : i + BN, j : j+BN]

```
What is the ratio of compute to data movement? How does it compare to the worst cast and the best case? We can answer these questions by looking at the loop structure.

Lets think about data movement first. There are three nested loops on the outside, each of which go from $1$ to $N$ in $BN$ sized steps. Each loop iterates $\frac{N}{BN}$ times, and since we have three levels of nesting, whatever is inside the nested loop body will happen $(\frac{N}{BN})^3$ times. Inside the loop nest, we load two tiles of size $BN^2$, one corresponding to each of the input matrices. Asymptotically this works out to $O((\frac{N}{BN})^3 * BN^2)$ data movement (we can ignore the storing of the `C_tile`, since this is only inside two of the loop nests, it only happens $\frac{N}{BN}^2$ times). Cancelling things out gives us $O(\frac{N^3}{BN})$ data movement. Notice that this a factor of $BN$ less data movement than the naive case.

Now compute. Same as above, we have three nested loops, the inside of this loop body will execute $(\frac{N}{BN})^3$ times. Inside the loop nests, the compute consists of the mini matmul between two $BN$ by $BN$ tiles, the three nested loops have a total of $O(BN^3)$ steps what is what we expect for multiplying together two $BN$ by $BN$ matrices. So the total amount of compute is $O((\frac{N}{BN})^3 * BN^3)$ which simplifies to just $O(N^3)$. This is the number of steps we expect for multiplying two $N$ by $N$ matrices, and it is the same as the naive case.

So this tiled approach has the same number of compute steps as the naive implementation, but a factor of $O(BN)$ less data movement. The arithmetic intensity works out to $O(\frac{N^3}{\frac{N^3}{BN}})=O(BN)$. In english, this is telling us that our achieved arithmetic intensity will scale linearly with the dimension of the tiles that we are fitting in fast memory.

### In Summary

The final takeaway is fairly intuitive. The best possible intensity we can achieve when multiplying two $N$ by $N$ matrices scales with the matrix dimension $N$. However, achieving this upper bound would require fitting the entire $O(N^2)$ sized problem in fast memory, which wont be possible. So we compromise by breaking down the $O(N^2)$ sized problem into lots of smaller $O(BN^2)$ sized problems, and we choose $BN$ such that all of our fast memory is filled up. The intensity we can then achieve scales with $BN$. So in practice, the intensity we can achieve is limited by the size of fast memory on our device.

## Parallelized matrix multiplication on a GPU
Thinking about matrix multiplication on the simple computer helps build intuition about how using the memory hierarchy to our advantage can result in higher arithmetic intensity, which will help for maximizing the performance of our kernel. However the simple computer model is a bit too simple, it consists of a two level memory hierarchy and some compute that can operate at a rate of $\tau$ on the data in fast memory. Our goal is to write a fast matrix multiplication kernel that will run on a GPU, which raises the question of how a GPU is different from the simple computer.

On the most fundamental level that answer is that GPUs, like the simple computer, have a memory hierarchy. But on a GPU the memory hierarchy fits within a hierarchy of concurrent compute units. Here is a diagram of a simple GPU that illustrates this.

![simple_gpu](/images/simple_gpu.png)

On the simple GPU there are three levels to the combined compute/memory hierarchy. 
- At the highest level is the whole GPU, which owns a big piece of DRAM (global memory). The GPU is composed of four multiprocessors, each of which are independent units of a compute, run concurrently with respect to each other and can all read/write to the same DRAM.
- At the middle level there is a multiprocessor which owns a piece of SRAM (shared memory), and is composed of four cores which are independent units of compute that run concurrently and can all read and write the same shared memory that is local to the multiprocessor.
- At the lowest level is a single compute core which owns some private register memory, and can execute a single thread and perform arithmetic independently of the rest of the computer.

### Hierarchical Tiling (simple gpu)
So how do we use this type of computer to perform a matrix multiplication? The first useful observation is that the matrix multiplication problem can be broken down hierarchically into nested tiles. This is good news, because a hierarchical algorithm is a good fit for a hierarchical computer.

![matmul_hierarchies](/images/matmul_hierarchies.png)

If we are computing a matrix multiplication $C=A*B$, we can divide the output matrix $C$ into non-overlapping tiles, and assign each tile to a compute unit. Each of these output tiles can then be computed with a matrix multiplication between corresponding tiles of the input, independently of the other tiles. Since our machine is hierarchical, there are compute units within compute units, and correspondingly there are matrix multiplications within matrix multiplications. We recursively break down the problem into nested tiles, until we end up at an atomic element of compute which physically is usually a single core of some sort, and logically is a single thread of execution. At this level the single thread computes a small matrix multiplication between its tiles of the input.
![hierarchy_combined](/images/hierarchy_combined.png)

### Hierarchical Tiling (real gpu)
The above diagram shows a coarse, high level view of what a GPU implementation of hierarchical tiling looks like. When implementing this in CUDA for an NVIDIA GPU, there are some finer details we need to fill in. This tiling structure is created by:
- a series of global, shared, and register memory allocations of fixed dimension
- nested loops which control the positions of the tiles
- synchronization points between threads running on each multiprocessor
- compute at the lowest level, which in this case is a small matrix multiplication that runs on the tensor core

This kernel was my starting point, but if you are interested in reading about a series of 10 kernels which build up to one like this, I recommend reading [this](https://siboehm.com/articles/22/CUDA-MMM).

![tiling](/images/my_tiles_2.png)

With this diagram my attempt is to show the correspondence between loop nests and the tiling structure. There are four levels, each level corresponds to a level of the compute hierarchy, memory hierarchy, and tile shape.

Here is a quick description of each level from the perspective of the compute unit relevant for that level:

* **CUDA Kernel / GPU level**: The GPU is reading the three input matrices, $A$, $B$, and $C$ from **global memory**, and writing the output matrix $D$ to global memory. Each thread block is looping over the `K` dimension (aka the 'inner' dimension) of $A$ and $B$. This loop is incrementing `block_k` in steps of size `BK`. At each iteration we are copying the blue blocktiles from global memory to shared memory.

* **Thread Block / SM level**: At this point the blue subtiles of $A$ and $B$ that a particular thread block needs to compute a `BM,BN` tile of the output have been copied into **shared memory**. This thread block is running on one of the 16 SMs on the GPU, and the shared memory is local to that SM and fast to access. Within the thread block there are 256 threads, which is 8 warps containing 32 threads each. Within the thread block, the `BM,BN` tile of the output is partitioned 8 ways, so that each of the 8 warps can work concurrently on the compute. Each of the warps is looping over the inner dimension within the block tile, this loop is incrementing `warp_k` in steps of size `WK`. At each iteration we are copying the green warp tiles from shared memory to register memory.

* **Warp / SM Partition**: At this point the green warp tiles within the blue block tiles have been copied into **register memory**, and it is the responsibility of a particular warp, running on one of the 4 partitions on the [Turing SM](https://images.app.goo.gl/Z2VVQQgXWTMddBraA) to compute the `WM` by `WN` tile of the output. Each warp computes its tile of the output by taking an outer product between the `WM,WK` tile of A and the `WK,WN` tile of B. Inside the three nested loops that compute the outer product, the we an MMA sync operation.

* **Tensor Core Op**: Finally we get down to the last level of the hierarchy, which is a single tensor core op, this is a single hardware accelerated (16,8) x (8,8) = (16,8) matrix multiply that takes place in and out of **register memory**.

### Performance considerations on a real GPU
When implementing this structure in a CUDA kernel that targets a particular GPU architecture, there are a number of things that must be considered given that we are trying to squeeze every last drop of performance out of the hardware. I divide the performance considerations into three buckets, each optimization discussed in the rest of this article falls into one or two of these buckets.

#### Arithmetic intensity as a function of tile dimensions
The necessity of achieving high arithmetic intensity is why we have this structure of tiles within tiles, and the tile dimension is the primary knob we can turn that determines the arithmetic intensity of our kernel. In our kernel we are first loading data from global memory to shared memory, and then shared memory into registers. In both cases we are loading two rectangular tiles corresponding to the input data from slower memory to faster memory, and then eventually computing a matrix multiplication between these two inputs at the lowest level of the hierarchy. The arithmetic intensity we should achieve is a function of the tile dimensions we choose (larger is better), this is worked out below.

![intensity_tile_dims](/images/intensity_tile_dims.png)

- **FLOPs**: At each iteration of the inner loop, each thread block multiplies a $(BM,BK)$ shaped matrix with a $(BK,BN)$, to produce a $(BM,BN)$ tile of the output. This matrix product consists of $2 * BM * BK * BN $ FLOPs (three nested loops over the dimensions, with a multiply and accumulate operation in the inner loop)
- **memory**: The $(BM,BK)$ and $(BK,BN)$ shaped matrices are read from global memory each iteration, since each element is two bytes this comes out to a total of $2(BM * BK + BK * BN) = 2BK(BM + BN)$ bytes read, and we don't perform any writes in the inner loop, all writes happen in the kernel epilogue.

Taking the ratio of these two, the arithmetic intensity we should achieve for a given block tile size works out nicely to $\frac{BM*BN}{BM+BN} \frac{FLOP}{byte}$. For the thread block level tiles at the second level of the hierarchy, we will want to choose our tile dimensions such that this ratio is larger than the balance point of the tensor cores with respect to global memory, but we will be limited by the size of shared memory. Likewise for the warp tiles at the next level down in the hierarchy, we will want to choose the tile dimensions such that this ratio is larger than the balance point of the tensor cores with respect to shared memory, but we will be limited by the size of register memory. The former turns out to be a bit more challenging than the later.

#### Overlap between compute and data movement
The roofline model gives us an upper bound on arithmetic throughput $T_{max}=min(\beta * I, \tau)$. In order to achieve this upper bound, we need perfect overlap between compute and data movement. In order to see why this is, imagine we achieve an arithmetic intensity sufficient to put us in the compute bound regime of the roofline model. At this point in order for our achieved throughput to actually equal the upper bound $T_{max}=\tau$, we need to be continuously computing, any time that our compute spends idle will mean that our achieved throughput is less than the machine peak $\tau$. There are a number of reasons why our compute will spend periods of time idle, such as memory latency, data dependencies, and synchronization points.
![compute_data_movement_overlap](/images/compute_data_movement_overlap.png)
As illustrated above our initial loop structure has some inefficiencies in this regard.

#### Maximizing memory bandwidth
According to [unofficial benchmarks](https://arxiv.org/pdf/1903.07486) the best achievable global memory bandwidth on the T4 is ~220 GB/sec, and the best achievable shared memory bandwidth is ~3662 GB/sec. However, an unoptimized kernel will only achieve a fraction of these numbers. The first consideration is access pattern; when groups of adjacent threads are requesting memory, some mappings of threads to data in memory are more efficient than others. The hardware that implements global memory vs. shared memory functions differently, consequently an access pattern that is optimal for reading shared memory may not be optimal for reading global memory. Global memory is discussed here, shared memory is the subject of a later chapter.

NVIDIA streaming multiprocessors execute threads in groups of 32, this group of 32 threads that execute together are called a **warp**. Global memory can only be accessed in aligned and contiguous chunks of 32, 64, or 128 bytes. Aligned here means that the first address of the chunk is a multiple of the size of the chunk. When the 32 threads in a warp make a memory request, the hardware looks at the 32 memory addresses requested, translates them into 32, 64, or 128 byte aligned chunks, which are then provided by the hardware. So in order for memory bandwidth to be maximized, the 32 threads in a warp must request bytes that fit evenly into 32, 64, or 128 byte. (more on this [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses))
![memory_coalescing](/images/memory_coalescing.png)






<!-- - the compute is happening at the bottom level of this hierarchy
- arithmetic intensity matters with respect to global memory, not shared memory

performance considerations: 
- access memory in a pattern that maximizes bandwidth
- achieving the upper bound in the roofline model requires perfect overlap between compute and data movement. Latency hiding -->


<!--

## How to use Tensor Cores

Part of why GPUs are good for things like matrix multiplication is that instructions are issued to 32 threads at a time, this can be extremely efficient because the overhead of instruction issue is amortized across 32 threads (as opposed to 1 for a CPU). However, the 32 threads in a warp are still somewhat [independent](https://developer.nvidia.com/blog/inside-volta/#independent_thread_scheduling). On modern architectures (Volta and later), each thread has its own program counter and call stack, allowing each thread to take its own execution path, although on a given clock cycle the SM partition can be executing no more than one instruction per warp. [Predication](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#control-flow-instructions) is used to mark threads as inactive/active for a given instruction, which for example allows threads within a warp to diverge on a branch based on data in thread local registers.

[but tensor cores are warp level instructions, all threads in a warp do the same thing]

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
Given how many people and companies these days are buying NVIDIA GPUs almost exclusively for the purpose of running matrix multiplications, it seems like lots of work goes into improving the tensor cores in terms of programmability and performance between successive architectures. On one hand, the tensor core throughput goes up by an order of magnitude with each new SM architecture. On the other hand, this means that in order to make this increased throughput actually usable (rather than theoretical), new hardware support is required for keeping them fed. For example, Ampere introduced hardware support for [asychronous data copying](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#asynchronous-data-copy-from-global-memory-to-shared-memory) from global memory to shared memory (more on this later). Hopper introduced something even fancier called the Tensor Memory Accelerator or [TMA](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator), this is a dedicated piece of hardware that is purpose built for feeding tensor cores, it can perform index calculation and initiate data copies asychronously with respect to the rest of the SM. Hopper is kind of a new and different beast, if you look at GEMM kernels in CUTLASS that target Hopper the code has a different stucture than all of the other pre `sm_90` kernels. Hopper kernels use a producer/consumer pattern, where a relatively small number of producer threads are initiating asynchronous data copies with the TMA, and then consumer threads are managing the tensor cores. I have never worked on kernels targetting Hopper so I dont know much about this at the moment, [this](https://hazyresearch.stanford.edu/blog/2024-05-12-tk) article provides an interesting overview of the user experience of writing kernels for Hopper.

This is all to say that the kernels discussed here target the Turing architecture, which was SOTA in 2018, and if you are writing kernels targeting Ampere or Hopper, the techniques you employ for latency hiding will be different and easier. I used the Tesla T4 GPU because you can rent them on AWS for ~50 cents/hour, which is about as much money as I want to spend on EC2 instances. Using an older GPU was a blessing and a curse for this project, the curse was that no special hardware support was available for hiding memory latency, the blessing was that I had to use more old school techniques for hiding this latency, which gave me an appreciation for why this hardware support exists now! -->

# Kernels

For the rest of this article I will discuss a series of 5 kernels that got me to ~80% of cublas level performance on a tensor core GEMM. Each kernel builds on the previous one, and the themes of each are:
1. [hierarchical tiling](#kernel-1---hierarchical-tiling)
2. [vectorized/unrolled gmem->smem transfer](#kernel-2---vectorized-memory-copy-and-loop-unrolling)
3. [shared memory swizzling](#swizzling)
4. [makeshift async copy](#kernel-4---makeshift-async-copy)


## Kernel 1 - Hierarchical Tiling
The first kernel I wrote is an implementation of the hierarchical tiling structure explained above. This loop performs the matrix multiplication

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
The 8% of cublas throughput it achieves is the starting point.

![table1](/images/table1.png)


## Kernel 2 - Vectorized memory copy and loop unrolling
In order to improve the performance of our code, we need to know why it is slow. When writing CUDA kernels, the best tool to use for this is called NSight Compute, a profiler developed by NVIDIA that gives lots of detailed metrics about how a kernel is interacting with the hardware. The first place I typically look is the section called "Warp State Statistics". As a kernel is executing, each warp is being issued instructions by a scheduler. In an ideal world, the scheduler would be able to issue a new instruction each clock cycle. In the real world, it is very hard to write a kernel that can issue a new instruction every cycle, there are all sorts of reasons why on a given cycle, a warp may not be capable of executing its next instruction and will instead "stall" i.e. do nothing. The reasons for stalling can be due to capacity limits of various hardware pipelines, memory latency, or sychronization points in our kernel which require all the threads running on an SM to wait for all the other threads to catch up. The Warp State Statistics section tells us how many clock cycles the average warp spends stalled, per average instruction issued, broken down across a bunch of different categories. This gives us the information we need to target our optimizations to the least performant parts of our kernel. Here is a screenshot of what the Warp State section for Kernel 1.
![warp_state_kernel1](/images/warp_state_kernel1.png)
The "Warp Cycles Per Issued Instruction" field tells us that on average for each instruction issued, warps spend about ~30 cycles idle, and the table below tells us that 16 of these 30 cycles are due to the "Long Scoreboard" stall category. 

[Scoreboarding](https://en.wikipedia.org/wiki/Scoreboarding) is an algorithm implemented in the hardware of most processors for tracking when the data dependencies for the next instruction have arrived in the registers they need to be in for the instruction to execute. Most modern CPUs are able to reorder instructions on the fly such that instructions whose operands are ready can execute ahead of instructions whose operands have yet to arrive in registers. The reordering is done in hardware, subject to constraints imposed by the data dependencies between subsequent instructions. This is called [out of order execution](https://en.wikipedia.org/wiki/Out-of-order_execution) and it is a rather fancy technique for hiding latency. GPUs do not reorder instructions as they are executing, I would imagine because the logic required consumes a fair amount of precious transistors on the chip, and since GPUs are designed for [throughput](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-benefits-of-using-gpus) these transistors are better spent on things like tensor cores.

GPUs do however track data dependencies, but with a lot more help from the compiler as compared to CPUs. When the data required to execute the next instruction has not arrived in register memory, the warp that is executing just waits for its data to arrive. The "Long Scoreboard Stall" approximates the average number of cycles that warps spend stalled waiting for data dependencies. The fact that this stall reason accounts for ~50% of all the cycles that warps spend idle tells us that the performance of Kernel 1 is primarily limited by memory latency. This tells us we should focus on the code that is moving data from global memory onto the chip, and figure out how to minimize the latency per byte moved.

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
Looking at the SASS corresponding to this `tileMemcpy` function in [godbolt](https://godbolt.org/z/1MeavE3GG), we can see that the copy operation inside the loop `dst[...] = src[...]` compiles to two operations from the lower level perspective of SASS, a two byte load from global memory (`LDG.U16` in SASS), followed by a two byte store (`STS.U16`), along with a bunch of index calculations and loop overhead. The long scoreboard stall prevents the store from taking place until the value we are loading has arrived in the register.

Here is a visualization of how this loop is executing, for a single thread:
![memory_latency](/images/memory_latency.png)
Latency in between the load and the store is inevitable: a request is sent to a DRAM controller, data is fetched from DRAM and then transmitted over bus. Unless we hack the laws of physics or invent a time machine we can't get rid of the latency. But what we can do is hide it.

Latency hiding is a central concept in computing, and at its core is very simple. It just means that if we are performing an operation $X$ that has some latency, we want to be doing other useful work while $X$ is happening, rather than wait and do nothing. For example, if I wake up and decide I want an omlette, I would first turn on the burner and let the pan warm up, and while that is happening I would crack the eggs and grate cheese. This order of operations hides the latency of warming up the pan with the cracking of eggs and grating of cheese. If I am hungry and eager to eat the finished omlette as soon as possible, it would be silly to idly stand there and watch as the pan warms up.

The same principle applies to hiding the latency of the global memory loads in `tileMemcpy`. Since the copy operation is happening inside a loop, each thread is performing multiple loads and multiple stores, in an order like `load (stall) store, load (stall) store, ...`. What if we were able to rearrange these so that the order is `load load load (stall) store, store, store`. In this later ordering the data requested by the three loads will be in flight at the same time, and we can say that the latency of each load is being hidden by the other loads. The easiest way to accomplish the later ordering is by unrolling the loop in `tileMemcpy`. If we can unroll the loop, `nvcc` should be smart enough to reorder the instructions so that the global memory loads are hiding each others latency. In this case the compiler is doing for us what a CPU would do in hardware on the fly. 

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
In the initial version, the total latency of the copy operation is roughly proportional to the memory latency of the device, times the number of loop iterations. After unrolling the loop, the total latency compared to the first version should be reduced by a factor of the number of loads the compiler decides to overlap with eachother (ish).

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

In our case, this stalling is almost certainly due to shared memory instructions, since our kernel has very few dynamic branches, and no trigonometry or any other [special math](https://developer.nvidia.com/cuda-math-library) instructions. Specifically, it is due to shared memory bank conflicts. According to [here](https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric/115731/2?u=a14armbr) two symptoms of shared memory bank conflicts are very high L1/TEX thoughput number (currently at 97% of peak) and MIO Throttle stalls, these are both second order effects of shared memory bank conflicts. I learned at this point that if you have a kernel whose performance is being killed due to shared memory bank conflicts, this is not blatantly obvious when you look at NSight Compute, however the information is definetly there. I found that in order to see where shared memory bank conflicts were occuring, and understand their severity, I had to learn the terminology of a "wavefront". In order to understand this term, a bit of background on shared memory is required.

### Background: Bank Conflicts and Wavefronts
From the perspective of a CUDA program, shared memory works as follows ([here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x) is the official guide). If you declare a `__shared__` array in your kernel, it corresponds to physical memory that is located on a specific streaming multiprocessor. Consequently this array is fast to access, but only accessible by threads on the SM, which in the language of CUDA means that shared memory arrays are local to a particular thread block. Physically the memory is spread between 32 "banks" with each bank storing an adjacent 4 bytes, like so:
![shmem_1](/images/shmem_1.png)
Each bank can produce a single 4 byte value per clock cycle. If our goal is to maximize our reading and writing bandwidth from shared memory, we need to keep this in mind when deciding on an access pattern. Full bandwidth is achieved when the 32 threads in a warp spread their access uniformly across the 32 banks. Bank "conflicts" occur when a single bank must produce data for more than one thread for a given request. In order to show how the ideas of bank conflicts and wavefronts tie together, here are 3 scenarios, all in a simplified world where we have 4 threads and 4 memory banks
![bank_conflicts](/images/bank_conflicts_wavefronts.png)
When loading or storing from shared memory, each thread requests a particular memory address that in our simplified world falls into one of the four memory banks. In scenario one, each thread is accessing data in a different bank, and the hardware calculates that these four accesses can be combined into a single transaction for the hardware to process, the word for this transaction is a [wavefront](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#id26). In scenario two, the four threads access addresses that fall into two of the four banks. Since each bank is only capable of sending one word at a time, the hardware groups these four requests into two wavefronts, and the memory hardware processes the two wavefronts one after the other. Scenario three is the worst case scenario, the four threads access addresses that all fall to the 0th memory bank, and in this case the four seperate wavefronts are required to service the transactions from the four threads.

For four threads accessing four bytes, the "ideal" number of wavefronts is one, because (ideally) regardless of which threads are accessing which bytes, we should be able to arange our data such that all of our accesses are spread nicely accross the banks. For example scenario three as shown is less than ideal, but we could make it ideal by transposing the bytes in shared memory, this would result in the four accesses falling evenly accross the four banks. But for the layout as shown, the actual number of wavefronts is four.

NSight Compute will tell us per memory access: 
1. the ideal number of wavefronts
2. the actual number of wavefronts
3. the number of wavefronts that are excessive, which is just 2 - 1

According to the analysis above, if our code has an $n$ way bank conflict, $n$ should be equal to $\frac{actual\ wavefronts}{ideal\ wavefronts}$. We want the actual to equal the ideal, this often requires some careful thinking about how data is being laid out and how threads are accessing it. 

### ldmatrix bank conflicts
Here is a screenshot of the per instruction actual/ideal wavefronts in NSight Compute:
![l1_wavefronts_source_view](/images/l1_wavefronts_source_view.png)
These `ldmatrix` commands are loading data from shared memory into thread local register memory in preparation for the MMA operations. NSight Compute tells us the ratio of actual to ideal is ~8ish, which suggests this memory access results in an 8-way bank conflict. In order to form a strategy for fixing this performance killer, we need to understand why it is happening.

In the tiling structure shown for Kernel 1, in each iteration of the warp loop (the green one), a single warp is responsible for reading a 64x64 tile of data from shared memory, and writing it to registers. The shared memory reads are where the bank conflicts occur. In the visualization below, on the top is a very zoomed out version of one of these 64x64 tiles, the layout across memory banks is visualized by the color of the columns. We can see that a row of 64 elements, which are 2 bytes each, nicely spans the 32 memory banks.
On the bottom is a zoomed in version of a single 8x8 tile that is brought from shared memory into registers by `ldmatrix`. Each warp is iterating over its own local 64x64 tile in 8x8 increments, calling `ldmatrix` on each little tile, this PTX instruction loads values from shared memory, and shuffles the loaded data among the registers in a warp to match the register layout that the tensor core instruction expects.
![mma_tile_zoom_in](/images/mma_tile_zoom_in.png)
The inner workings of `ldmatrix` are a bit opaque, it compiles to a single SASS instruction `LDSM...`, rather than multiple explicit shared memory loads and register shuffles, as one might expect. However, we dont need an understanding of `ldmatrix`s inner workings to see why the 8 way bank conflict is occuring each time we call it. Rather the 8-way bank conflict is an inevitable result of the fact that each row in a given tile is spread across the same four memory banks. One wavefront is required to read each row, and there are eight rows, which means eight wavefronts. Ideally, if the eight rows in each tile were spread evenly across the thirty two memory banks, the entire tile could be read with a single wavefront. Reading these tiles is in the inner loop of the kernel, for $8192$x$8192$ operands we read a total of $ (8192/8)^3=1,073,741,824$ of these tiles which works out to a ~shitload~ of bank conflicts. So if we care about performance, it is worth the time to fix it.

### Padding
In order to have a bank conflict free kernel, we need to rearrange the layout of data in shared memory such that we can read and write to shared memory without any excessive wavefronts. The challenge comes from the fact that the thread to data mapping for shared memory reads is different from that of shared memory writes. When writing, adjacent threads write adjacent values in a row, whereas when reading adjacent threads read adjacent values down a column. 

![row_vs_column_shmem_access](/images/row_vs_column_shmem_access.png)

This is a common situation in kernels that use 2d shared memory tiles, and the standard fix is to add a bit of padding (i.e. empty space) at the end of each row in the shared memory array. If we add this padding in such a way that a single row of our array no longer fits perfectly into the 32 memory banks, adjacent values in a column no longer fall into the same bank, which means we can read columns with no excessive wavefronts. This makes more sense in a picture than in words, here again is a simplified case of a mini-array (4 columns and 4 rows) stored on a mini-gpu with only 4 memory banks:
![simple_smem_padding](/images/simple_smem_padding.png)
Array elements are color coded by column. Notice that in the no padding case, all the array elements in a given column fall into the same memory bank. After adding the column of padding, the array elements in a given column are spread across all 4 memory banks. The padding technique could be used here to fully eliminate bank conflicts. Since we are using [vectorized](#kernel-2---vectorized-memory-copy-and-loop-unrolling) writes to shared memory, we are writing to shared memory in 16 byte chunks at a time, and each chunk must be [aligned](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses). Adding 16 bytes of padding to each row of shared memory would result in each 8x8 mma tile being spread across all 32 memory banks (exercise of convincing yourself of this left to reader). 

The drawback of using the padding technique is that it requires us to allocate extra, unused space in shared memory. In Kernel 2, the shared memory tile for $A$ is 256x64, and the shared memory tile for $B$ is 128x64. If we add an extra 16 byte, or 8 element column to both of these, that will increase the amount of shared memory we allocate by 25%, for a total of increase of 6144 bytes. This wasted space turns out to be a significant drawback, when writing a high performance kernel shared memory is very precious stuff - this becomes especially apparent later down in the road when using a technique called double buffering, each threadblock in future kernels will end up using 100% of the 65536 bytes of shared memory on each SM. So, we should wonder whether there is a way to eliminate bank conflicts without wasting any shared memory space. It turns out this is very possible!

### Swizzling (toy example)
Swizzling is probably my favorite technique that I learned in the process of working on this. The word "swizzle" has several different uses, when used in the context of cocktails it means to [stir](https://en.wikipedia.org/wiki/Swizzle_stick) and when used in the context of GPUs it means to [rearrange](https://en.wikipedia.org/wiki/Swizzling_(computer_graphics)). In our context of eliminating shared memory bank conflicts in 2D tiles of data, swizzling means permuting the elements within a tile of shared memory such that we can access the data without any bank conflicts. This is one of those techniques that seemed like black magic to me until I took the time to understand it, and now I appreciate its cleverness and elegance.

In our 4x4 tile, we add the padding because it shifts the alignment between data and memory banks in a desirable way. Swizzling is based on the observation that we don't need the extra padding bytes to spread column elements evenly over memory banks. Instead we can just figure out a permutation of matrix elements that spreads around the columns in the right way, and apply this permutation when we write to shared memory. Here is an illustration of a "swizzle" i.e. a permutation of elements that can eliminate bank conflicts.
![simple_smem_swizzled](/images/simple_smem_swizzled.png)
It is worth remembering at this point that our shared memory layout must satisfy two requirements, bank conflict free row access for writing, and bank conflict free column access for reading.

In all three cases, each row is consecutive in memory and spread across all four memory banks, which means each row can be written without any bank conflicts. The observation here is that when we apply our permutation or "swizzle", we don't want to permute elements across rows, only within rows; otherwise we might lose this property of bank conflict free writes.

The problem that motivated us to think about shared memory layouts was the bank conflicts that occur when we read columns. Adding the padding fixes the bank conflicts here, but at the expense of wasted shared memory. Swizzling gives us the best of both worlds; we can read columns with no bank conflicts, and no shared memory is wasted. So how do we think about applying this permutation?

The swizzle shown above can be implemented as a function `f` that maps indices to new indices. If `A` is the original array, `A_s` is the swizzled array, and `i` is the index of an element, then `A_s[f(i)] = A[i]`. So what is `f` here?

Since `f` is operating on array indices, we should think about the different ways these indices can be represented and viewed:
![simple_smem_indices](/images/simple_smem_indices.png)
On the far left are the 2D row and column indices. Moving to the middle, these indices can be linearized into a sequential (and in this case row major) ordering of the 16 elements in the array. Moving to the right, when we look at the sequential indices in binary we can see that the 2d structure is present in the index bits. The two least significant bits in the index encode the column and the two other bits encode the row. As a spoiler alert, `f` is going to operate from the perspective of the view on the right, the binary representation of the flat array index. Here are two observations about what `f` needs to do:
* In order to avoid bank conflicts on write, we want to permute elements within a row, or in other words no elements should switch row. This means that `f` should modify the bits which encode the column, and leave alone the bits that encode the row.
* We want to apply a different permutation to each row, and for any given column, we want the elements in that column to be spread across all four columns in the swizzled array.

We can accomplish both of these aims using the XOR function, specifically by XORing the row bits of each element with its column bits, and using the result as the new row bits. Here is a row by row break down that shows how XORing the column bits with the row bits moves around values within a row:
![swizzled_rows](/images/swizzled_rows.png)
The `f` that does this for us is `f(i) = i ^ ((i & 0b1100) >> 2)`. The mask is selecting the two column bits from `i`, these two bits are then shifted right two places so that they line up with the two row bits for `i`, and then we XOR. `i`'s column bits remain unchanged.

 Here is a visualization of the result of applying this function for all rows together:
![2d-swizzle](/images/2d-swizzle.png)

### Swizzling (real world)
Now we need to figure out how to use this technique to permute our shared memory layout in such a way that we can read a single 8x8 mma tile with 0 excessive wavefronts. As a reminder, here is a view of our shared memory layout, with a single tile of interest highlighted.
![mma_tile_zoom_in_blank](/images/mma_tile_zoom_in_blank.png)

Our goal is to figure out a swizzle function that spreads the 8 rows in this tile across all 32 memory banks, rather than having all 8 rows stuffed into 4 memory banks which is the case above. From the view of the full tile, the rows of the tile above would be spread like this.

![mma_tile_zoom_in_swizzle](/images/mma_tile_zoom_in_swizzle.png)

In order to figure out what swizzle function we should use, lets look at the binary representation of an index into this tile, and assign it some structure that corresponds to our tiling scheme.

![swizzle_index_groups](/images/swizzle_index_groups.png)

Some notes about what our swizzling function should do and not do:
* We want to keep the eight elements in each MMA tile row together. In other words, eight adjacent elements in a single row of an 8x8 MMA tile are going to stay together when we apply the swizzle. This means our swizzle function is not going to touch the orange bits.
* Bank conflicts occur because the 8 rows within an MMA tile are all perfectly stacked on top of each other. Within an MMA tile, we want to spread these 8 rows horizontally across the entire warp tile. The blue bits encode where in the 64 element wide warp tile each MMA tile falls, so these blue bits are the ones we want our swizzle function to modify.
* We dont want to move elements between rows, so our swizzle function is not going to modify the green row bits. However, these green row bits provide a nice alternating pattern that we can XOR with the blue bits to mix around the rows of an MMA tile.
* Again, we dont want to be moving elements between rows, and the black bits (the most significant ones shown in this diagram) encode the starting row of each MMA tile. Our swizzle function is going to ignore them.

So what this all means is that for each index, we want to take the blue bits, XOR them with the green bits, and replace the original blue bits with the result of this XOR. If `i` is the index we want to swizzle, this works out to:
![swizzled_vs_unswizzled](/images/swizzled_vs_unswizzled.png)
And just like that, we have no bank conflicts. Swizzling takes a bit more figuring out than the padding technique, the choice of swizzle function depends on the shared memory array dimensions, and the vector width we are using for reads/writes (i.e. `float4`, `float2`, `int`, e.t.c.). As a result, if we use swizzling it adds an extra consideration each time we consider changing either of these. But if you want to eliminate bank conflicts and dont want to increase your shared memory footprint in the process, swizzling becomes necessary. I think it is very elegant and clever, if you compare kernel 2 with kernel 3, there is a total of ~4 lines of code that change, these four lines are the addition of the swizzle into the shared memory index calculation.

I figured all this out by looking at the `Swizzle` class implemented [here](https://github.com/NVIDIA/cutlass/blob/main/python/pycute/swizzle.py) in the CUTLASS repository. Via its three parameters, `bits`, `base`, and `shift`, this class represents a family of swizzle functions that shift and XOR bits of array indices. I have also seen examples of more exotic swizzle functions (see slide 27 [here](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf)) that go beyond what is representable by the implementation in CUTLASS. I found it helpful to visualize the permutations applied by different swizzle functions, to help with this I wrote a bit of python [code](https://github.com/alexarmbr/matmul-playground/blob/main/scripts/shmem_layout_viz.py) that pretty-prints arrays, applies swizzle functions, and counts bank conflicts.

Eliminating bank conflicts results in a ~2x speedup, and gets us to about 50% of the cuBLAS level thoughput.
![table3](/images/table3.png)

## Kernel 4 - Makeshift Async Copy
Each optimization addresses the least performant part of the previoius kernel. After applying each optimization, if it worked, the least performant part of the kernel should change. Before fixing the shared memory bank conflicts, the shared memory operations inside the inner loop were the bottleneck. After eliminating bank conflicts, the inner loop becomes much more efficient, and the bottleneck is once again the latency of the global memory to shared memory transfer. This was addressed with vectorizing and loop unrolling in [Kernel 2](#kernel-2---vectorized-memory-copy-and-loop-unrolling), but after fixing bank conflicts, NSight Compute is telling us that there is more latency here to hide. Here is pseudocode of the current loop nests, with a zoomed in view of the code that needs to be improved:
![long_scoreboard_stall_kernel3](/images/long_scoreboard_stall_kernel3.png)
Once again the issue is that the line which performs the global memory to shared memory copy: 

```c++
dst_float4[dst_index] = src_float4[src_index];
// shared memory        // global memory
```

 This^ is a blocking operation from the perspective of hardware, in the sense that when a given thread executes the resulting assembly the thread will be stalled for the full duration of to data arriving from global memory. The above line is equivalent to this:

```c++
float4 tmp = src_float4[src_index]; // global memory to register
dst_float4[dst_index] =  tmp; // register to shared memory
```
The global memory to register transfer, which is the first line, incurs latency because data is coming from off chip. When it comes time to store from register to shared memory (second line) the hardware detects that the data needed from global memory has not yet arrived in `tmp`, and execution stalls until it arrives. In [Kernel 2](#kernel-2---vectorized-memory-copy-and-loop-unrolling) we addressed this performance issue by amortizing the latency over more data moved per transaction (vectorizing) and helping the compiler to interleave multiple loads/stores, which hides latency (loop unrolling). But NSight Compute tells us that even after these optimizations, this sort of stall, on this line specifically, accounts for about ~20% of the total clock cycles that the kernel spends stalled.

The key observation here is that if we break down the `dst[...] = src[...]` line into its two constituent parts, we can break them apart so that other useful work is being done while the data is in flight from global memory.
The general idea is that we can prefetch data from global memory into register storage, one `block_k` ahead of the `block_k` we are currently computing on. At a very high level we want to go from this:
```c++
float4 tmp = src_float4[src_index]; // global memory to register
// (stall while we wait for data to arrive from memory)
dst_float4[dst_index] =  tmp; // register to shared memory
{
    // compute inner loop for current block tile
}
```

to this:
```c++
float4 tmp = src_float4[src_index]; // global memory to register
{
    // compute inner loop for previous block tile
}
dst_float4[dst_index] =  tmp; // register to shared memory
```

The key improvement being made here is that we are initiating the load of data from global memory corresponding to `block_k`, and performing the compute corresponding to `block_k`-1 concurrently. In doing so we are hiding the latency of loading the `block_k` tiles of $A$ and $B$ with the computation corresponding to the `block_k`-1 tiles.

![concurrent_fetch_compute](/images/concurrent_fetch_compute.png)

This improved overlapping of data movement and compute is accomplished by
- adding new register storage to hold the data that is prefetched from global memory
- breaking up the global to shared memory transfer into its two components, putting these two components on opposite sides of the inner loop (over warp tiles and mma tiles)
- and tweaking the position of the two `__syncthreads()` in the outer loop to allow for the concurrency we want, while still preventing race conditions.

Here is before/after pseudocode which shows how the data movement changes.
![prefetch](/images/prefetch.png)

This produces a nice speedup over the previous kernel, and gets us to performance that is on par with the fastest CUTLASS hgemm kernel.

![table4](/images/table4.png)

### GPU occupancy (digression)
The potential cost of this optimization is that it requires additional register storage, each thread block stores two additional block tiles worth of data in register memory. According to the Launch Statistics section in NSight Compute, we go from using 104 registers per thread in Kernel 3, to 166 registers per thread in Kernel 4. This increase resource usage per thread has the *potential* to hurt kernel performance because it can impact how many threads the hardware is capable of executing concurrently. This is a quick digression on why increasing register use per thread has the potential to hurt performance, but why in this case, it doesn't.

This gets to a topic called **occupancy** which is central to the CUDA hardware implementation. Each streaming multiprocessor (SM) will maintain block, warp, and thread execution state (shared memory, registers, program counter) on chip, for as many thread blocks as can be fit. The amount of thread blocks that can be fit on an SM depends on:
1. how much shared memory, registers per thread, and number of threads each thread block needs to execute (this a property of a given kernel, and the launch configuration of that kernel)
2. how much shared memory, registers per thread, and number of threads the SM can handle at once (this is a property the device, and improves from generation to genereation)

If a given kernel implementation and launch configuration require only a small number of registers, a few threads, and a small amount of shared memory, an SM can execute lots of thread blocks concurrently. When multiple thread blocks are executing concurrently on an SM, context switching between them is free. This allows the hardware to hide stalls and latency simply by tracking which threads are capable of executing their next instruction and which are not, and issuing instructions for whichever threads are ready. The more threads the SM has to choose from, the better this works. This is called [hardware multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-multithreading), and lots of older resources on CUDA performance talk about it as the primary guiding principle for writing fast kernels.

However, this kernel (and many other kernels) require lots of registers per thread, which limits the number of thread blocks that can be resident on the SM at once, which in turn limits the hardwares ability to hide stalls automatically by context switching. Why do we need so many registers?
- [This](#how-to-not-be-memory-bound-simple-memory-hierarchy) part of the background section goes through how a regular matrix multiplication requires $O(n^3)$ compute, but operates on only $O(n^2)$ data. 

### Async Copy on Ampere Architecture (digression)
Ampere arch has a hardware instruction for moving data directly and asynchronously from global memory to shared memory, without going through a register.

## Kernel 5 - Tune Block Tile Dimensions
Kernel 5 achieved ~35 TFLOP/s, which is on par with the fastest CUTLASS HGEMM kernel that I found, but still only achieves about ~62% of the ~51 TFLOP/s throughput of the cuBLAS HGEMM implementation. At this point I thought "darn, my goal was 100% of the cuBLAS throughput, and I have used a good number of performance tricks but I am only at 62%. Where should I focus the rest of my efforts?"

One of the primary challenges when working on kernels such as this one is the (global) memory wall, that is the imbalance betweeen the high throughput of the tensor cores, and comparatively low global memory bandwidth. At this point I became curious, am I still memory bound?

## Kernel 6 - Double Buffering