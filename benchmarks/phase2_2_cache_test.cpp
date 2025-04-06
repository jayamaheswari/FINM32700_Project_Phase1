#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>
#include <random>
#include <iomanip>
#include <sys/sysctl.h>
#include "linalg.hpp"

struct CacheInfo {
    size_t l1;
    size_t l2;
    size_t l3;
};

struct Result {
    double mean;
    double stddev;
};

inline CacheInfo getCacheSizes() {
    int64_t l1dc=0,l2c=0,l3c=0; size_t sz=sizeof(int64_t);
    sysctlbyname("hw.l1dcachesize",&l1dc,&sz,nullptr,0);
    sysctlbyname("hw.l2cachesize",&l2c,&sz,nullptr,0);
    sysctlbyname("hw.l3cachesize",&l3c,&sz,nullptr,0);
    return {size_t(l1dc),size_t(l2c),size_t(l3c)};
}

inline std::unique_ptr<double[]> makeRand(int r,int c,unsigned s){
    std::unique_ptr<double[]> p(new double[r*c]);
    std::mt19937_64 g(s); std::uniform_real_distribution<double>d(-1,1);
    for(int i=0;i<r*c;i++)
        p[i]=d(g);
    return p;
}

inline Result benchMVRow(int r,int c){
    auto A=makeRand(r,c,1),x=makeRand(c,1,2); std::unique_ptr<double[]> y(new double[r]);
    linalg::multiply_mv_row_major(A.get(),r,c,x.get(),y.get());
    std::vector<double> t; t.reserve(30);
    for(int i=0;i<30;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mv_row_major(A.get(),r,c,x.get(),y.get());
        auto e=std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0; for(auto&v:t) sm+=v; double m=sm/t.size(),va=0; for(auto&v:t){double d=v-m;va+=d*d;} va/=t.size();
    return {m,std::sqrt(va)};
}

inline Result benchMVCol(int r,int c){
    auto A=makeRand(r,c,3),x=makeRand(c,1,4); std::unique_ptr<double[]> y(new double[r]);
    linalg::multiply_mv_col_major(A.get(),r,c,x.get(),y.get());
    std::vector<double> t; t.reserve(30);
    for(int i=0;i<30;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mv_col_major(A.get(),r,c,x.get(),y.get());
        auto e=std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0;
    for(auto&v:t)
        sm+=v;
    double m=sm/t.size(),va=0;
    for(auto&v:t) {
        double d=v-m;
        va+=d*d;
    }
    va/=t.size();
    return {m,std::sqrt(va)};
}

inline Result benchMMNaive(int n){
    auto A=makeRand(n,n,5),B=makeRand(n,n,6); std::unique_ptr<double[]> C(new double[n*n]);
    linalg::multiply_mm_naive(A.get(),n,n,B.get(),n,n,C.get());
    std::vector<double> t; t.reserve(10);
    for(int i=0;i<10;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mm_naive(A.get(),n,n,B.get(),n,n,C.get());
        auto e=std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0;
    for(auto&v:t)
        sm+=v;
    double m=sm/t.size(),va=0;
    for(auto&v:t) {
        double d=v-m;
        va+=d*d;
    }
    va/=t.size();
    return {m,std::sqrt(va)};
}

inline Result benchMMTrans(int n){
    auto A=makeRand(n,n,7),B=makeRand(n,n,8); std::unique_ptr<double[]> Bt(new double[n*n]),C(new double[n*n]);
    for(int i=0;i<n;i++)for(int j=0;j<n;j++) Bt[j*n+i]=B[i*n+j];
    linalg::multiply_mm_transposed_b(A.get(),n,n,Bt.get(),n,n,C.get());
    std::vector<double> t; t.reserve(10);
    for(int i=0;i<10;i++){
        auto s=std::chrono::high_resolution_clock::now();
        linalg::multiply_mm_transposed_b(A.get(),n,n,Bt.get(),n,n,C.get());
        auto e=std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double,std::milli>(e-s).count());
    }
    double sm=0;
    for(auto&v:t)
        sm+=v;
    double m=sm/t.size(),va=0;
    for(auto&v:t) {
        double d=v-m;
        va+=d*d;
    }
    va/=t.size();
    return {m,std::sqrt(va)};
}

inline int dimLess(size_t bytes){double c=double(bytes)/8.0;return int(std::floor(std::sqrt(c*0.5)));}
inline int dimEqual(size_t bytes){double c=double(bytes)/8.0;return int(std::floor(std::sqrt(c)));}
inline int dimMore(size_t bytes){double c=double(bytes)/8.0;return int(std::floor(std::sqrt(c*2.0)));}

int main(){
    auto c=getCacheSizes();
    std::cout<<std::fixed<<std::setprecision(3);
    std::cout<<"Cache L1="<<c.l1<<" L2="<<c.l2<<" L3="<<c.l3<<"\n";
    std::vector<std::pair<std::string,size_t>> levels;
    levels.push_back({"L1",c.l1});
    levels.push_back({"L2",c.l2});
    if(c.l3>0) levels.push_back({"L3",c.l3});
    for(auto& lev:levels){
        int dl=dimLess(lev.second), de=dimEqual(lev.second), dm=dimMore(lev.second);
        std::vector<int> sizes={dl,de,dm};
        for(auto n: sizes){
            if(n<1) n=1;
            std::cout<<"N="<<n<<" MV(Row/Col) ";
            auto r1=benchMVRow(n,n),r2=benchMVCol(n,n);
            std::cout<<r1.mean<<"("<<r1.stddev<<") "<<r2.mean<<"("<<r2.stddev<<")  MM(Naive/Trans) ";
            auto r3=benchMMNaive(n),r4=benchMMTrans(n);
            std::cout<<r3.mean<<"("<<r3.stddev<<") "<<r4.mean<<"("<<r4.stddev<<")\n";
        }
        std::cout<<"\n";
    }
    return 0;
}
