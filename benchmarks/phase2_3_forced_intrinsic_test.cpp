#include <iostream>
#include <memory>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <iomanip>

// ALLOC
static inline void* alignedAlloc(std::size_t align, std::size_t size){
    void* p=nullptr;
    posix_memalign(&p, align, size);
    return p;
}
static inline void alignedFree(void* p){ free(p); }

// RNG
static inline void fillRand(double* ptr, std::size_t n, unsigned seed){
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0,1.0);
    for(std::size_t i=0;i<n;i++) ptr[i]=dist(gen);
}

// RESULT
struct Result { double mean; double stddev; };

// X86_64 AVX
#if defined(__x86_64__)
#include <immintrin.h>
static inline void kernelAligned(double* in, double* out, std::size_t n){
    for(std::size_t i=0;i<n;i+=4){
        __m256d v=_mm256_load_pd(&in[i]);
        __m256d m=_mm256_mul_pd(v,v);
        __m256d r=_mm256_add_pd(v,m);
        _mm256_store_pd(&out[i],r);
    }
}
static inline void kernelUnaligned(double* in, double* out, std::size_t n){
    // still using _mm256_load_pd => CPU must fix up unaligned
    for(std::size_t i=0;i<n;i+=4){
        __m256d v=_mm256_load_pd(&in[i]);
        __m256d m=_mm256_mul_pd(v,v);
        __m256d r=_mm256_add_pd(v,m);
        _mm256_store_pd(&out[i],r);
    }
}

// ARM64 NEON (Apple Silicon)
#elif defined(__aarch64__)
#include <arm_neon.h>
static inline void kernelAligned(double* in, double* out, std::size_t n){
    for(std::size_t i=0;i<n;i+=2){
        float64x2_t v=vld1q_f64(&in[i]);
        float64x2_t m=vmulq_f64(v,v);
        float64x2_t r=vaddq_f64(v,m);
        vst1q_f64(&out[i],r);
    }
}
static inline void kernelUnaligned(double* in, double* out, std::size_t n){
    // NEON doesn't strictly differentiate aligned vs unaligned instructions,
    // but offset will still cause potential performance differences on some hardware.
    for(std::size_t i=0;i<n;i+=2){
        float64x2_t v=vld1q_f64(&in[i]);
        float64x2_t m=vmulq_f64(v,v);
        float64x2_t r=vaddq_f64(v,m);
        vst1q_f64(&out[i],r);
    }
}
#else
static inline void kernelAligned(double* in, double* out, std::size_t n){
    for(std::size_t i=0;i<n;i++){
        double x=in[i];
        out[i]=x + x*x;
    }
}
static inline void kernelUnaligned(double* in, double* out, std::size_t n){
    for(std::size_t i=0;i<n;i++){
        double x=in[i];
        out[i]=x + x*x;
    }
}
#endif

static inline Result benchAligned(std::size_t n){
    void* pA=alignedAlloc(64,n*sizeof(double));
    double* in=(double*)pA;
    fillRand(in,n,0);
    void* pB=alignedAlloc(64,n*sizeof(double));
    double* out=(double*)pB;
    std::vector<double> times; times.reserve(20);
    for(int i=0;i<20;i++){
        auto start=std::chrono::high_resolution_clock::now();
        kernelAligned(in,out,n);
        auto end=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(end-start).count());
    }
    alignedFree(pA); alignedFree(pB);
    double sum=0; for(auto&t:times) sum+=t; double mean=sum/times.size(),v=0;
    for(auto&t:times){double d=t-mean;v+=d*d;} v/=times.size();
    return {mean,std::sqrt(v)};
}

static inline Result benchUnaligned(std::size_t n){
    void* pA=alignedAlloc(64,n*sizeof(double)+8);
    double* baseA=(double*)pA;
    double* in=(double*)((char*)baseA+8);
    fillRand(in,n,1);
    void* pB=alignedAlloc(64,n*sizeof(double)+8);
    double* baseB=(double*)pB;
    double* out=(double*)((char*)baseB+8);
    std::vector<double> times; times.reserve(20);
    for(int i=0;i<20;i++){
        auto start=std::chrono::high_resolution_clock::now();
        kernelUnaligned(in,out,n);
        auto end=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(end-start).count());
    }
    alignedFree(pA); alignedFree(pB);
    double sum=0; for(auto&t:times) sum+=t; double mean=sum/times.size(),v=0;
    for(auto&t:times){double d=t-mean;v+=d*d;} v/=times.size();
    return {mean,std::sqrt(v)};
}

int main(){
    std::cout<<std::fixed<<std::setprecision(3);
    std::vector<std::size_t> sizes={200000,400000,800000,1600000};
    std::cout<<"Phase2.3 - Forced Intrinsic Alignment\n";
#if defined(__x86_64__)
    std::cout<<"Architecture: x86_64 (AVX)\n";
#elif defined(__aarch64__)
    std::cout<<"Architecture: ARM64 (NEON)\n";
#else
    std::cout<<"Architecture: Fallback Scalar\n";
#endif
    std::cout<<"Size   Aligned(ms)  Unaligned(ms)\n";
    for(auto sz: sizes){
        auto rA=benchAligned(sz);
        auto rU=benchUnaligned(sz);
        std::cout<<sz<<"  "<<rA.mean<<"("<<rA.stddev<<")   "<<rU.mean<<"("<<rU.stddev<<")\n";
    }
    return 0;
}
