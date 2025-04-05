# FINM32700_Project_Phase1
High-Performance Linear Algebra Kernels

---

## Team Members
- Aadi
- Anand
- Jaya
- Keyi

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
   
3. Describe how CPU caches work (L1, L2, L3) and explain the concepts of temporal and spatial locality. How did you try to exploit these concepts in your optimizations?
Ans - L1 Cache: Fastest and smallest, located closest to the CPU core.
      L2 Cache: Larger and slower than L1, often shared between cores.
      L3 Cache: The largest and slowest, typically shared across multiple cores.
      Optimization via Cache Locality:
      Temporal Locality: Reusing recently accessed data.
      Spatial Locality: Accessing adjacent data in memory to take advantage of cache line fetches.
      Matrix algorithms can benefit from loop reordering and blocking techniques to improve cache hits.
   
4. What is memory alignment, and why is it important for performance? Did you observe a significant performance difference between aligned and unaligned memory in your experiments? Explain your findings.
6. Discuss the role of compiler optimizations (like inlining) in achieving high performance. How did the optimization level affect the performance of your baseline and optimized implementations? What are the potential drawbacks of aggressive optimization?
7. Based on your profiling experience, what were the main performance bottlenecks in your initial implementations? How did your profiling results guide your optimization efforts?
8. Reflect on the teamwork aspect of this assignment. How did dividing the initial implementation tasks and then collaborating on analysis and optimization work? What were the challenges and benefits of this approach?
