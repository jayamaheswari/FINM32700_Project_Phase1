#include <iostream>
#include <chrono>
#include <memory>
#include <random>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include "linalg.hpp"

inline double helperMul(double a, double b){return a*b;}
static double helperMulNoInline(double a, double b){return a*b;}

inline void mvRowMajorInline(const double* M, int r, int c, const double* x, double* y){
    if(!M||!x||!y||r<1||c<1) throw std::invalid_argument("Bad args mvRowMajorInline");
    for(int i=0;i<r;i++){
        double acc=0.0;
        for(int j=0;j<c;j++){
            acc+=helperMul(M[i*c+j], x[j]);
        }
        y[i]=acc;
    }
}
static void mvRowMajorNoInline(const double* M, int r, int c, const double* x, double* y){
    if(!M||!x||!y||r<1||c<1) throw std::invalid_argument("Bad args mvRowMajorNoInline");
    for(int i=0;i<r;i++){
        double acc=0.0;
        for(int j=0;j<c;j++){
            acc+=helperMulNoInline(M[i*c+j], x[j]);
        }
        y[i]=acc;
    }
}

inline void mvColMajorInline(const double* M, int r, int c, const double* x, double* y){
    if(!M||!x||!y||r<1||c<1) throw std::invalid_argument("Bad args mvColMajorInline");
    for(int i=0;i<r;i++){
        double acc=0.0;
        for(int j=0;j<c;j++){
            acc+=helperMul(M[j*r+i], x[j]);
        }
        y[i]=acc;
    }
}
static void mvColMajorNoInline(const double* M, int r, int c, const double* x, double* y){
    if(!M||!x||!y||r<1||c<1) throw std::invalid_argument("Bad args mvColMajorNoInline");
    for(int i=0;i<r;i++){
        double acc=0.0;
        for(int j=0;j<c;j++){
            acc+=helperMulNoInline(M[j*r+i], x[j]);
        }
        y[i]=acc;
    }
}

inline void mmNaiveInline(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    if(!A||!B||!C||ra<1||ca<1||rb<1||cb<1||ca!=rb) throw std::invalid_argument("Bad args mmNaiveInline");
    for(int i=0;i<ra;i++){
        for(int j=0;j<cb;j++){
            double acc=0.0;
            for(int k=0;k<ca;k++){
                acc+=helperMul(A[i*ca+k], B[k*cb+j]);
            }
            C[i*cb+j]=acc;
        }
    }
}
static void mmNaiveNoInline(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    if(!A||!B||!C||ra<1||ca<1||rb<1||cb<1||ca!=rb) throw std::invalid_argument("Bad args mmNaiveNoInline");
    for(int i=0;i<ra;i++){
        for(int j=0;j<cb;j++){
            double acc=0.0;
            for(int k=0;k<ca;k++){
                acc+=helperMulNoInline(A[i*ca+k], B[k*cb+j]);
            }
            C[i*cb+j]=acc;
        }
    }
}

inline void mmTransInline(const double* A,int ra,int ca,const double* Bt,int rb,int cb,double* C){
    if(!A||!Bt||!C||ra<1||ca<1||rb<1||cb<1||ca!=rb) throw std::invalid_argument("Bad args mmTransInline");
    for(int i=0;i<ra;i++){
        for(int j=0;j<cb;j++){
            double acc=0.0;
            for(int k=0;k<ca;k++){
                acc+=helperMul(A[i*ca+k], Bt[j*rb+k]);
            }
            C[i*cb+j]=acc;
        }
    }
}
static void mmTransNoInline(const double* A,int ra,int ca,const double* Bt,int rb,int cb,double* C){
    if(!A||!Bt||!C||ra<1||ca<1||rb<1||cb<1||ca!=rb) throw std::invalid_argument("Bad args mmTransNoInline");
    for(int i=0;i<ra;i++){
        for(int j=0;j<cb;j++){
            double acc=0.0;
            for(int k=0;k<ca;k++){
                acc+=helperMulNoInline(A[i*ca+k], Bt[j*rb+k]);
            }
            C[i*cb+j]=acc;
        }
    }
}

struct Result{ double mean; double stddev; };

static std::unique_ptr<double[]> randMat(int n, int m, unsigned s){
    std::unique_ptr<double[]> p(new double[n*m]);
    std::mt19937_64 g(s);
    std::uniform_real_distribution<double> d(-1.0,1.0);
    for(int i=0;i<n*m;i++) p[i]=d(g);
    return p;
}

static Result timeMVInline(bool rowMajor, int r, int c){
    auto M=randMat(r,c,1);
    auto x=randMat(c,1,2);
    auto y=std::make_unique<double[]>(r);
    std::vector<double> times; times.reserve(10);
    for(int run=0;run<10;run++){
        auto start=std::chrono::high_resolution_clock::now();
        if(rowMajor) mvRowMajorInline(M.get(),r,c,x.get(),y.get());
        else mvColMajorInline(M.get(),r,c,x.get(),y.get());
        auto end=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::micro>(end-start).count());
    }
    double sm=0; for(auto&t:times) sm+=t; double mean=sm/times.size(),va=0;
    for(auto&t:times){double d=t-mean;va+=d*d;} va/=times.size();
    return {mean,std::sqrt(va)};
}

static Result timeMVNoInline(bool rowMajor, int r, int c){
    auto M=randMat(r,c,3);
    auto x=randMat(c,1,4);
    auto y=std::make_unique<double[]>(r);
    std::vector<double> times; times.reserve(10);
    for(int run=0;run<10;run++){
        auto start=std::chrono::high_resolution_clock::now();
        if(rowMajor) mvRowMajorNoInline(M.get(),r,c,x.get(),y.get());
        else mvColMajorNoInline(M.get(),r,c,x.get(),y.get());
        auto end=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::micro>(end-start).count());
    }
    double sm=0; for(auto&t:times) sm+=t; double mean=sm/times.size(),va=0;
    for(auto&t:times){double d=t-mean;va+=d*d;} va/=times.size();
    return {mean,std::sqrt(va)};
}

static Result timeMMInline(bool transB, int rA, int cA, int rB, int cB){
    auto A=randMat(rA,cA,5);
    auto B=randMat(rB,cB,6);
    std::unique_ptr<double[]> Bt(nullptr);
    if(transB){
        Bt=std::make_unique<double[]>(rB*cB);
        for(int i=0;i<rB;i++){
            for(int j=0;j<cB;j++){
                Bt[j*rB+i]=B[i*cB+j];
            }
        }
    }
    auto C=std::make_unique<double[]>(rA*cB);
    std::vector<double> times; times.reserve(5);
    for(int run=0;run<5;run++){
        auto start=std::chrono::high_resolution_clock::now();
        if(transB) mmTransInline(A.get(),rA,cA,Bt.get(),rB,cB,C.get());
        else mmNaiveInline(A.get(),rA,cA,B.get(),rB,cB,C.get());
        auto end=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::micro>(end-start).count());
    }
    double sm=0; for(auto&t:times) sm+=t; double mean=sm/times.size(),va=0;
    for(auto&t:times){double d=t-mean;va+=d*d;} va/=times.size();
    return {mean,std::sqrt(va)};
}

static Result timeMMNoInline(bool transB, int rA, int cA, int rB, int cB){
    auto A=randMat(rA,cA,7);
    auto B=randMat(rB,cB,8);
    std::unique_ptr<double[]> Bt(nullptr);
    if(transB){
        Bt=std::make_unique<double[]>(rB*cB);
        for(int i=0;i<rB;i++){
            for(int j=0;j<cB;j++){
                Bt[j*rB+i]=B[i*cB+j];
            }
        }
    }
    auto C=std::make_unique<double[]>(rA*cB);
    std::vector<double> times; times.reserve(5);
    for(int run=0;run<5;run++){
        auto start=std::chrono::high_resolution_clock::now();
        if(transB) mmTransNoInline(A.get(),rA,cA,Bt.get(),rB,cB,C.get());
        else mmNaiveNoInline(A.get(),rA,cA,B.get(),rB,cB,C.get());
        auto end=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::micro>(end-start).count());
    }
    double sm=0; for(auto&t:times) sm+=t; double mean=sm/times.size(),va=0;
    for(auto&t:times){double d=t-mean;va+=d*d;} va/=times.size();
    return {mean,std::sqrt(va)};
}

int main(){
    std::cout<<std::fixed<<std::setprecision(3);
    std::vector<std::pair<int,int>> sizesMV={{500,500},{1000,1000},{1500,1500}};
    std::vector<std::pair<int,int>> sizesMM={{200,200},{500,500},{1000,1000}};
    std::cout<<"ms - micro-seconds, std - standard deviation\n";
    std::cout<<"MATRIX-VECTOR:\nRowsxCols | RowInline(ms/std) RowNoInline(ms/std) ColInline(ms/std) ColNoInline(ms/std)\n";
    for(auto & s: sizesMV){
        auto rI=timeMVInline(true,s.first,s.second);
        auto rN=timeMVNoInline(true,s.first,s.second);
        auto cI=timeMVInline(false,s.first,s.second);
        auto cN=timeMVNoInline(false,s.first,s.second);
        std::cout<<s.first<<"x"<<s.second<<"  "<<rI.mean<<"("<<rI.stddev<<") "<<rN.mean<<"("<<rN.stddev<<") "<<cI.mean<<"("<<cI.stddev<<") "<<cN.mean<<"("<<cN.stddev<<")\n";
    }
    std::cout<<"\nMATRIX-MATRIX:\nRowsAxColsA x RowsBxColsB | NaiveInline(ms/std) NaiveNoInline(ms/std) TransInline(ms/std) TransNoInline(ms/std)\n";
    for(auto & s: sizesMM){
        int ra=s.first, ca=s.second, rb=s.second, cb=s.first;
        auto nI=timeMMInline(false,ra,ca,rb,cb);
        auto nN=timeMMNoInline(false,ra,ca,rb,cb);
        auto tI=timeMMInline(true,ra,ca,rb,cb);
        auto tN=timeMMNoInline(true,ra,ca,rb,cb);
        std::cout<<ra<<"x"<<ca<<" x "<<rb<<"x"<<cb<<"  "<<nI.mean<<"("<<nI.stddev<<") "<<nN.mean<<"("<<nN.stddev<<") "<<tI.mean<<"("<<tI.stddev<<") "<<tN.mean<<"("<<tN.stddev<<")\n";
    }
    return 0;
}
