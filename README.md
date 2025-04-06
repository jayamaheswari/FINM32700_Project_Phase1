# FINM32700_Project_Phase1
High-Performance Linear Algebra Kernels

---

## Team Members
- Aadi Deshpande
- Anand Nakhate
- Jaya Maheswari
- Keyi Wang

---


## Part 3: Discussion Questions 

1. Explain the key differences between pointers and references in C++. When would you choose to use a pointer over a reference, and vice versa, in the context of implementing numerical algorithms?

Ans - Pointers: Hold memory addresses, can be reassigned, and support pointer arithmetic.
      References: Are aliases for other variables and cannot be reassigned once initialized.

   Use cases:
      Pointers are preferred when dynamic memory allocation or pointer arithmetic is necessary.
      References are suitable for simpler cases where passing large objects by reference is needed without reassigning the reference.
  
2. How does the row-major and column-major storage order of matrices affect memory access patterns and cache locality during matrix-vector and matrix-matrix multiplication? Provide specific examples from your implementations and benchmarking results.

Ans - Row-Major: Stores matrix elements row by row, commonly used in C++.
      Column-Major: Stores matrix elements column by column, used by Fortran and MATLAB.

   Impact on Performance:
      Row-major storage benefits sequential row access patterns (good for row-wise traversal).
      Column-major storage is optimal for sequential column access but may result in cache misses for row-wise traversal.
   
   Example: The cache locality analysis in section 2.2 shows that row-major storage outperforms column-major by approximately 1.3× for larger matrices, primarily due to better cache locality resulting from sequential row-wise traversal.
   
3. Describe how CPU caches work (L1, L2, L3) and explain the concepts of temporal and spatial locality. How did you try to exploit these concepts in your optimizations?

Ans - L1 Cache: Fastest and smallest, located closest to the CPU core.
      L2 Cache: Larger and slower than L1, often shared between cores.
      L3 Cache: The largest and slowest, typically shared across multiple cores.

   Optimization via Cache Locality:
      Temporal Locality: Reusing recently accessed data.
      Spatial Locality: Accessing adjacent data in memory to take advantage of cache line fetches.
      Matrix algorithms can benefit from loop reordering and blocking techniques to improve cache hits.
      We implemented Reorder (IKJ), which improves performance by ensuring contiguous accumulation in matrix C, resulting in 7× faster execution for 256×256 matrices. This optimizes spatial locality by reusing data in the cache.
      We also implemented Blocked/Tiled, which divides matrices A and B into cache-friendly blocks, reducing capacity misses and memory traffic, especially for large matrices.
   
4. What is memory alignment, and why is it important for performance? Did you observe a significant performance difference between aligned and unaligned memory in your experiments? Explain your findings.

Ans - Aligning data to cache line boundaries can drastically improve performance, reducing memory access overhead.
      Misaligned memory access can lead to slower performance, especially for large datasets or when working with SIMD instructions.

5. Discuss the role of compiler optimizations (like inlining) in achieving high performance. How did the optimization level affect the performance of your baseline and optimized implementations? What are the potential drawbacks of aggressive optimization?

Ans - Inlining: Replaces function calls with the function body to avoid overhead.
      Other Optimizations: Loop unrolling, vectorization, and instruction reordering improve performance by minimizing redundant operations.

   Effect on Performance:
      Aggressive optimization (-O3) typically improves execution speed but may increase binary size and complicate debugging

6. Based on your profiling experience, what were the main performance bottlenecks in your initial implementations? How did your profiling results guide your optimization efforts?

Ans - Profiling tools were used to identify bottlenecks in the initial implementations:
      Cache misses: Caused by inefficient memory access patterns.
      Function call overhead: Reduced by inlining small functions.
      Inefficient memory access: Optimized by ensuring proper memory alignment and improving data locality.
      
7. Reflect on the teamwork aspect of this assignment. How did dividing the initial implementation tasks and then collaborating on analysis and optimization work? What were the challenges and benefits of this approach?
