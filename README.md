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

### 1. Explain the key differences between pointers and references in C++. When would you choose to use a pointer over a reference, and vice versa, in the context of implementing numerical algorithms?
- Pointers: Hold memory addresses, can be reassigned, and support pointer arithmetic.
- References: Are aliases for other variables and cannot be reassigned once initialized.
- Use cases:
  - Pointers are preferred when dynamic memory allocation or pointer arithmetic is necessary.
  - References are suitable for simpler cases where passing large objects by reference is needed without reassigning the reference.
  
### 2. How does the row-major and column-major storage order of matrices affect memory access patterns and cache locality during matrix-vector and matrix-matrix multiplication? Provide specific examples from your implementations and benchmarking results.

- Row-Major: Stores matrix elements row by row, commonly used in C++.
- Column-Major: Stores matrix elements column by column, used by Fortran and MATLAB.
- Impact on Performance:
  - Row-major storage benefits sequential row access patterns (good for row-wise traversal).
  - Column-major storage is optimal for sequential column access but may result in cache misses for row-wise traversal.
- The cache locality analysis in section 2.2 shows that row-major storage outperforms column-major by approximately 1.3× for larger matrices, primarily due to better cache locality resulting from sequential row-wise traversal.
   
### 3. Describe how CPU caches work (L1, L2, L3) and explain the concepts of temporal and spatial locality. How did you try to exploit these concepts in your optimizations?

- L1 Cache: Fastest and smallest, located closest to the CPU core.
- L2 Cache: Larger and slower than L1, often shared between cores.
- L3 Cache: The largest and slowest, typically shared across multiple cores.

Optimization via Cache Locality:
- Temporal Locality: Reusing recently accessed data.
- Spatial Locality: Accessing adjacent data in memory to take advantage of cache line fetches.
- Matrix algorithms can benefit from loop reordering and blocking techniques to improve cache hits.

We implemented Reorder (IKJ), which improves performance by ensuring contiguous accumulation in matrix C, resulting in 7× faster execution for 256×256 matrices. This optimizes spatial locality by reusing data in the cache. We also implemented Blocked/Tiled, which divides matrices A and B into cache-friendly blocks, reducing capacity misses and memory traffic, especially for large matrices.
   
### 4. What is memory alignment, and why is it important for performance? Did you observe a significant performance difference between aligned and unaligned memory in your experiments? Explain your findings.

- Aligning data to cache line boundaries can drastically improve performance, reducing memory access overhead. Misaligned memory access can lead to slower performance, especially for large datasets or when working with SIMD instructions.

Observations

- Aligned row-major can give minor speedups at large sizes (e.g. 1000×1000), but overall gains are inconsistent or small. 
- Column-major sees little or no improvement, likely due to inherently strided access overshadowing alignment benefits. 
- Medium-sized matrices (e.g. 500×500) often benefit the most; alignment reduces memory penalties and cache-line splits.
- Very large or very small problems show negligible difference.

### 5. Discuss the role of compiler optimizations (like inlining) in achieving high performance. How did the optimization level affect the performance of your baseline and optimized implementations? What are the potential drawbacks of aggressive optimization?

Inlining and Other Optimizations

- Inlining: Replaces function calls with the function body to eliminate the overhead associated with function calls.
- Other Optimizations: Techniques like loop unrolling, vectorization, and instruction reordering improve performance by reducing redundant operations.

Effect on Performance

- Aggressive Optimization (-O3): Generally improves execution speed but may increase binary size and complicate debugging.
  
Observations on Performance with Inlining

- Inline Versions: Consistently reduce mean execution time and variance.
- Row-Major Layout: More prominent improvements were observed with Row-Major layout, likely due to the greater impact of function call overhead and its interaction with cache/memory access patterns.
- Column-Major Access: In both inlined and non-inlined versions, column-major access performs better for matrix-vector multiplication. This could be due to better cache locality when the vector is accessed sequentially.

Benefits and Drawbacks of Inlining

- Short, Frequently Called Functions: Inlining is most beneficial for these, as it eliminates function call overhead.
- Large Functions: Inlining increases the code size, which can:
  - Bloat the instruction cache.
  - Increase compilation time.


### 6. Based on your profiling experience, what were the main performance bottlenecks in your initial implementations? How did your profiling results guide your optimization efforts?

- Cache Misses: Inefficient memory access patterns.  
  - Optimization: Improved memory alignment and enhanced data locality to reduce cache misses.

- Function Call Overhead: Small function calls added unnecessary overhead.  
  - Optimization: Inlined small functions to reduce overhead.

- Backend Bound (30.9%): Significant time spent in CPU execution units, indicating a need for better computational efficiency.  
  - Optimization: Refined loop structures, implemented cache optimization techniques (tiling/blocking), and explored more efficient algorithms or vectorization.

- Parallelization: Considered parallelizing the algorithm using threading or OpenMP to utilize multiple CPU cores more effectively.
      
### 7. Reflect on the teamwork aspect of this assignment. How did dividing the initial implementation tasks and then collaborating on analysis and optimization work? What were the challenges and benefits of this approach?

Approach:

- Split tasks by module: 
  - Matrix-Vector (Row-Major)
  - Matrix-Vector (Column-Major)
  - Matrix-Matrix (Naive)
  - Matrix-Matrix (Transposed)

Challenges:

- Different Operating Systems: With two members on Mac, one on Linux, and one on Windows, we encountered performance inconsistencies. On Windows, the standard deviation (std dev) didn’t decrease across runs, likely due to background processes or compiler optimizations.
  
- Hardware Differences: macOS lacks an L3 cache, which caused discrepancies in performance, especially in terms of cache locality.

Outcome:

- Dividing tasks and collaborating allowed us to address these challenges effectively and improve performance.
