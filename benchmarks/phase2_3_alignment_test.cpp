#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <memory>
#include <iomanip>
#include <random>
#include <cstring>
#include "linalg.hpp"

struct Result { double mean; double stddev; };

inline void* alignedAlloc(std::size_t alignment, std::size_t bytes) {
    void* p=nullptr;
    posix_memalign(&p,alignment,bytes);
    return p;
}

inline void alignedFree(void* p){ free(p); }

inline std::unique_ptr<double[],void(*)(void*)> makeAligned(int n, int m, unsigned s, std::size_t alignment=64){
    void* raw=alignedAlloc(alignment,sizeof(double)*n*m);
    std::unique_ptr<double[],void(*)(void*)> ptr((double*)raw,alignedFree);
    std::mt19937_64 g(s); std::uniform_real_distribution<double>d(-1,1);
    for(int i=0;i<n*m;i++) ptr[i]=d(g);
    return ptr;
}

inline std::unique_ptr<double[]> makeUnaligned(int n,int m,unsigned s){
    std::unique_ptr<double[]> ptr(new double[n*m]);
    std::mt19937_64 g(s); std::uniform_real_distribution<double>d(-1,1);
    for(int i=0;i<n*m;i++) ptr[i]=d(g);
    return ptr;
}

inline Result benchMVRowAligned(int r,int c){
    auto A=makeAligned(r,c,1),x=makeAligned(c,1,2);
    auto y=makeAligned(r,1,9);
    linalg::multiply_mv_row_major(A.get(),r,c,x.get(),y.get());
    std::vector<double> times; times.reserve(30);
    for(int i=0;i<30;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mv_row_major(A.get(),r,c,x.get(),y.get());
        auto e=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0; for(auto v:times) sm+=v; double mean=sm/times.size(),var=0;
    for(auto v:times){ double d=v-mean;var+=d*d;} var/=times.size();
    return {mean,std::sqrt(var)};
}

inline Result benchMVRowUnaligned(int r,int c){
    auto A=makeUnaligned(r,c,1),x=makeUnaligned(c,1,2);
    auto y=makeUnaligned(r,1,9);
    linalg::multiply_mv_row_major(A.get(),r,c,x.get(),y.get());
    std::vector<double> times; times.reserve(30);
    for(int i=0;i<30;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mv_row_major(A.get(),r,c,x.get(),y.get());
        auto e=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0; for(auto v:times) sm+=v; double mean=sm/times.size(),var=0;
    for(auto v:times){ double d=v-mean;var+=d*d;} var/=times.size();
    return {mean,std::sqrt(var)};
}

inline Result benchMVColAligned(int r,int c){
    auto A=makeAligned(r,c,3),x=makeAligned(c,1,4);
    auto y=makeAligned(r,1,10);
    linalg::multiply_mv_col_major(A.get(),r,c,x.get(),y.get());
    std::vector<double> times; times.reserve(30);
    for(int i=0;i<30;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mv_col_major(A.get(),r,c,x.get(),y.get());
        auto e=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0; for(auto v:times) sm+=v; double mean=sm/times.size(),var=0;
    for(auto v:times){ double d=v-mean;var+=d*d;} var/=times.size();
    return {mean,std::sqrt(var)};
}

inline Result benchMVColUnaligned(int r,int c){
    auto A=makeUnaligned(r,c,3),x=makeUnaligned(c,1,4);
    auto y=makeUnaligned(r,1,10);
    linalg::multiply_mv_col_major(A.get(),r,c,x.get(),y.get());
    std::vector<double> times; times.reserve(30);
    for(int i=0;i<30;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mv_col_major(A.get(),r,c,x.get(),y.get());
        auto e=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0; for(auto v:times) sm+=v; double mean=sm/times.size(),var=0;
    for(auto v:times){ double d=v-mean;var+=d*d;} var/=times.size();
    return {mean,std::sqrt(var)};
}

inline Result benchMMNaiveAligned(int n){
    auto A=makeAligned(n,n,5),B=makeAligned(n,n,6);
    auto C=makeAligned(n,n,99);
    linalg::multiply_mm_naive(A.get(),n,n,B.get(),n,n,C.get());
    std::vector<double> times; times.reserve(10);
    for(int i=0;i<10;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mm_naive(A.get(),n,n,B.get(),n,n,C.get());
        auto e=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0; for(auto v:times) sm+=v; double mean=sm/times.size(),var=0;
    for(auto v:times){ double d=v-mean;var+=d*d;} var/=times.size();
    return {mean,std::sqrt(var)};
}

inline Result benchMMNaiveUnaligned(int n){
    auto A=makeUnaligned(n,n,5),B=makeUnaligned(n,n,6);
    std::unique_ptr<double[]> C(new double[n*n]);
    linalg::multiply_mm_naive(A.get(),n,n,B.get(),n,n,C.get());
    std::vector<double> times; times.reserve(10);
    for(int i=0;i<10;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mm_naive(A.get(),n,n,B.get(),n,n,C.get());
        auto e=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0; for(auto v:times) sm+=v; double mean=sm/times.size(),var=0;
    for(auto v:times){ double d=v-mean;var+=d*d;} var/=times.size();
    return {mean,std::sqrt(var)};
}

inline Result benchMMTransAligned(int n){
    auto A=makeAligned(n,n,7),B=makeAligned(n,n,8);
    auto Bt=makeAligned(n,n,100),C=makeAligned(n,n,101);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) Bt.get()[j*n+i]=B.get()[i*n+j];
    linalg::multiply_mm_transposed_b(A.get(),n,n,Bt.get(),n,n,C.get());
    std::vector<double> times; times.reserve(10);
    for(int i=0;i<10;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mm_transposed_b(A.get(),n,n,Bt.get(),n,n,C.get());
        auto e=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0; for(auto v:times) sm+=v; double mean=sm/times.size(),var=0;
    for(auto v:times){ double d=v-mean;var+=d*d;} var/=times.size();
    return {mean,std::sqrt(var)};
}

inline Result benchMMTransUnaligned(int n){
    auto A=makeUnaligned(n,n,7),B=makeUnaligned(n,n,8);
    std::unique_ptr<double[]> Bt(new double[n*n]),C(new double[n*n]);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) Bt[j*n+i]=B[i*n+j];
    linalg::multiply_mm_transposed_b(A.get(),n,n,Bt.get(),n,n,C.get());
    std::vector<double> times; times.reserve(10);
    for(int i=0;i<10;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mm_transposed_b(A.get(),n,n,Bt.get(),n,n,C.get());
        auto e=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0; for(auto v:times) sm+=v; double mean=sm/times.size(),var=0;
    for(auto v:times){ double d=v-mean;var+=d*d;} var/=times.size();
    return {mean,std::sqrt(var)};
}

int main(){
    std::cout<<std::fixed<<std::setprecision(3);
    std::vector<int> sizes={128,256,512,1024};
    std::cout<<"Phase2.3 Alignment Test\n";
    std::cout<<"Size | MV_Row(Aligned/Unaligned) | MV_Col(Aligned/Unaligned) | MM_Naive(Aligned/Unaligned) | MM_Trans(Aligned/Unaligned)\n";
    for(auto n:sizes){
        auto rmA=benchMVRowAligned(n,n), rmU=benchMVRowUnaligned(n,n);
        auto cmA=benchMVColAligned(n,n), cmU=benchMVColUnaligned(n,n);
        auto mmA=benchMMNaiveAligned(n), mmU=benchMMNaiveUnaligned(n);
        auto mtA=benchMMTransAligned(n), mtU=benchMMTransUnaligned(n);
        std::cout<<n<<" x "<<n<<"  "
                 <<rmA.mean<<"("<<rmA.stddev<<")/"<<rmU.mean<<"("<<rmU.stddev<<")  "
                 <<cmA.mean<<"("<<cmA.stddev<<")/"<<cmU.mean<<"("<<cmU.stddev<<")  "
                 <<mmA.mean<<"("<<mmA.stddev<<")/"<<mmU.mean<<"("<<mmU.stddev<<")  "
                 <<mtA.mean<<"("<<mtA.stddev<<")/"<<mtU.mean<<"("<<mtU.stddev<<")\n";
    }
    return 0;
}
