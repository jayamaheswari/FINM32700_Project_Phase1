#include <iostream>
#include <chrono>
#include <memory>
#include <random>
#include <vector>
#include <cmath>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

inline double fmaLike(double a, double b, double c){return a*b + c;}

static std::unique_ptr<double[]> makeRand(int r, int c, unsigned s){
    std::unique_ptr<double[]> ptr(new double[r*c]);
    std::mt19937_64 gen(s);
    std::uniform_real_distribution<double> dist(-1.0,1.0);
    for(int i=0;i<r*c;i++){
        ptr[i] = dist(gen);
    }
    return ptr;
}

static void naive(const double* A, int ra, int ca, const double* B, int rb, int cb, double* C){
    for(int i=0;i<ra;i++){
        for(int j=0;j<cb;j++){
            double sum=0.0;
            for(int k=0;k<ca;k++){
                sum += A[i*ca+k]*B[k*cb+j];
            }
            C[i*cb+j] = sum;
        }
    }
}

static void reorderIKJ(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    for(int i=0;i<ra;i++){
        for(int k=0;k<ca;k++){
            double val=A[i*ca+k];
            for(int j=0;j<cb;j++){
                C[i*cb+j] = fmaLike(val,B[k*cb+j], C[i*cb+j]);
            }
        }
    }
}

static void blockedCore(double* C,const double* A,const double* B,int ra,int ca,int cb,int blockSize){
    for(int iB=0; iB<ra; iB+=blockSize){
        int iE=(iB+blockSize<ra)?(iB+blockSize):ra;
        for(int jB=0; jB<cb; jB+=blockSize){
            int jE=(jB+blockSize<cb)?(jB+blockSize):cb;
            for(int kB=0; kB<ca; kB+=blockSize){
                int kE=(kB+blockSize<ca)?(kB+blockSize):ca;
                for(int i=iB; i<iE; i++){
                    for(int j=jB; j<jE; j++){
                        double sum=C[i*cb+j];
                        for(int k=kB; k<kE; k++){
                            sum += A[i*ca+k]*B[k*cb+j];
                        }
                        C[i*cb+j] = sum;
                    }
                }
            }
        }
    }
}

static void blockedCoreTrans(double* C,const double* A,const double* BT,int ra,int ca,int cb,int blockSize){
    for(int iB=0; iB<ra; iB+=blockSize){
        int iE=(iB+blockSize<ra)?(iB+blockSize):ra;
        for(int jB=0; jB<cb; jB+=blockSize){
            int jE=(jB+blockSize<cb)?(jB+blockSize):cb;
            for(int kB=0; kB<ca; kB+=blockSize){
                int kE=(kB+blockSize<ca)?(kB+blockSize):ca;
                for(int i=iB; i<iE; i++){
                    for(int j=jB; j<jE; j++){
                        double sum=C[i*cb+j];
                        for(int k=kB; k<kE; k++){
                            sum += A[i*ca+k]*BT[j*ra+k];
                        }
                        C[i*cb+j] = sum;
                    }
                }
            }
        }
    }
}

#ifdef _OPENMP
static void parallelBlockedCore(double* C,const double* A,const double* B,int ra,int ca,int cb,int blockSize){
#pragma omp parallel for collapse(2)
    for(int iB=0; iB<ra; iB+=blockSize){
        for(int jB=0; jB<cb; jB+=blockSize){
            for(int kB=0; kB<ca; kB+=blockSize){
                int iE=(iB+blockSize<ra)?(iB+blockSize):ra;
                int jE=(jB+blockSize<cb)?(jB+blockSize):cb;
                int kE=(kB+blockSize<ca)?(kB+blockSize):ca;
                for(int i=iB; i<iE; i++){
                    for(int j=jB; j<jE; j++){
                        double sum=C[i*cb+j];
                        for(int k=kB; k<kE; k++){
                            sum += A[i*ca+k]*B[k*cb+j];
                        }
                        C[i*cb+j] = sum;
                    }
                }
            }
        }
    }
}
#endif

#if defined(__x86_64__)
static void simdReorderIKJ(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    for(int i=0;i<ra;i++){
        for(int k=0;k<ca;k++){
            __m256d aval=_mm256_set1_pd(A[i*ca+k]);
            for(int j=0;j<cb;j+=4){
                __m256d sum=_mm256_loadu_pd(&C[i*cb+j]);
                __m256d bval=_mm256_loadu_pd(&B[k*cb+j]);
                sum=_mm256_fmadd_pd(aval,bval,sum);
                _mm256_storeu_pd(&C[i*cb+j],sum);
            }
        }
    }
}
#elif defined(__aarch64__)
static void simdReorderIKJ(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    for(int i=0;i<ra;i++){
        for(int k=0;k<ca;k++){
            float64x2_t aval=vdupq_n_f64(A[i*ca+k]);
            for(int j=0;j<cb;j+=2){
                float64x2_t sum=vld1q_f64(&C[i*cb+j]);
                float64x2_t bval=vld1q_f64(&B[k*cb+j]);
                sum=vfmaq_f64(sum,aval,bval);
                vst1q_f64(&C[i*cb+j],sum);
            }
        }
    }
}
#else
static void simdReorderIKJ(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    reorderIKJ(A,ra,ca,B,rb,cb,C); // fallback
}
#endif

// Wrappers that match the function-pointer signature (const double*,int,int,const double*,int,int,double*)
static void reorderWrapper(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    reorderIKJ(A,ra,ca,B,rb,cb,C);
}
static void naiveWrapper(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    naive(A,ra,ca,B,rb,cb,C);
}
static void simdReorderWrapper(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    simdReorderIKJ(A,ra,ca,B,rb,cb,C);
}

// For blocking, we fix the block size in these wrappers or can pass a global
static const int BLOCKSIZE=64;
static void blockedMMWrap(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    blockedCore(C,const_cast<const double*>(A),const_cast<const double*>(B),ra,ca,cb,BLOCKSIZE);
}
static void blockedTransWrap(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    blockedCoreTrans(C,const_cast<const double*>(A),const_cast<const double*>(B),ra,ca,cb,BLOCKSIZE);
}
#ifdef _OPENMP
static void parallelBlockedWrap(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    parallelBlockedCore(C,const_cast<const double*>(A),const_cast<const double*>(B),ra,ca,cb,BLOCKSIZE);
}
#else
static void parallelBlockedWrap(const double* A,int ra,int ca,const double* B,int rb,int cb,double* C){
    naive(A,ra,ca,B,rb,cb,C);
}
#endif

struct R { double mean; double stdv; };

static R bench(void(*fn)(const double*,int,int,const double*,int,int,double*), int n, int runs=5){
    std::vector<double> times; times.reserve(runs);
    for(int r=0;r<runs;r++){
        auto A=makeRand(n,n,r+1);
        auto B=makeRand(n,n,r+11);
        auto C=std::make_unique<double[]>(n*n);
        for(int i=0;i<n*n;i++) C[i]=0.0;
        auto start=std::chrono::high_resolution_clock::now();
        fn(A.get(),n,n,B.get(),n,n,C.get());
        auto end=std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double,std::milli>(end-start).count());
    }
    double s=0; for(auto&t:times) s+=t; double m=s/runs,v=0;
    for(auto&t:times){ double d=t-m; v+=d*d;} v/=runs;
    return {m,std::sqrt(v)};
}

int main(){
    std::cout<<std::fixed<<std::setprecision(3);
    std::cout<<"Phase2.6 HPC - Combined Approaches\n";
#ifdef _OPENMP
    std::cout<<"OpenMP enabled\n";
#else
    std::cout<<"OpenMP not enabled (compile with -fopenmp)\n";
#endif
    std::vector<int> sizes={256,512,1024};
    std::cout<<"Size   naive(ms)  reorder(ms)  blocked(ms)  blockedTrans(ms)  parallelBlocked(ms)  simdReorder(ms)\n";
    for(auto n: sizes){
        auto rNaive=bench(naiveWrapper,n);
        auto rReorder=bench(reorderWrapper,n);
        auto rBlocked=bench(blockedMMWrap,n);
        auto rBlockedT=bench(blockedTransWrap,n);
        auto rParallel=bench(parallelBlockedWrap,n);
        auto rSimd=bench(simdReorderWrapper,n);
        std::cout<<n<<"x"<<n<<"  "
                 <<rNaive.mean<<"("<<rNaive.stdv<<")  "
                 <<rReorder.mean<<"("<<rReorder.stdv<<")  "
                 <<rBlocked.mean<<"("<<rBlocked.stdv<<")  "
                 <<rBlockedT.mean<<"("<<rBlockedT.stdv<<")  "
                 <<rParallel.mean<<"("<<rParallel.stdv<<")  "
                 <<rSimd.mean<<"("<<rSimd.stdv<<")\n";
    }
    return 0;
}
