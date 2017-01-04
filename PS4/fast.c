#include <complex.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <x86intrin.h>

void chemm(complex float* A,
        complex float* B,
        complex float* C,
        int m,
        int n,
        complex float alpha,
        complex float beta){

    float ar;
    float ai;
    float br;
    float bi;
    float cr;
    float ci;
    float dr;
    float di;

    float ar2;
    float ai2;
    float br2;
    float bi2;
    float cr2;
    float ci2;
    float dr2;
    float di2;
    
    complex float* resultarray;
    posix_memalign((void**)&resultarray, 32, sizeof(float) * 8);
    float* thisC;
    float* nextC;
    float* nextC2;
    float* thisC2;
    float betar = creal(beta);
    float betai = cimag(beta);
    //printf("n = %d, m = %d\n", n, m);
        for(int x = 0; x < n; x++){
            for(int y = 0; y < m; y++){

                if(y%4 == 0){

                    thisC = (float*)&C[y*n + x];
                    nextC = (float*)&C[(y+1)*n + x];
                    thisC2 = (float*)&C[(y+2)*n + x];
                    nextC2 = (float*)&C[(y+3)*n + x];


                    // ar = creal(thisC);
                    // ai = cimag(thisC);
                    // br = creal(nextC);
                    // bi = cimag(nextC);
                    // cr = creal(beta);
                    // ci = cimag(beta);
                    // dr = creal(beta);
                    // di = cimag(beta);

                    // ar2 = creal(thisC2);
                    // ai2 = cimag(thisC2);
                    // br2 = creal(nextC2);
                    // bi2 = cimag(nextC2);
                    // cr2 = creal(beta);
                    // ci2 = cimag(beta);
                    // dr2 = creal(beta);
                    // di2 = cimag(beta);

                    ar = thisC[0];
                    ai = thisC[1];
                    br = nextC[0];
                    bi = nextC[1];
                    cr = betar;
                    ci = betai;
                    dr = betar;
                    di = betai;

                    ar2 = thisC2[0];
                    ai2 = thisC2[1];
                    br2 = nextC2[0];
                    bi2 = nextC2[1];
                    cr2 = betar;
                    ci2 = betai;
                    dr2 = betar;
                    di2 = betai;


                    //a*c, b*d, a2*c2, b2*d2
                    __m256 x_avx = _mm256_set_ps(ar, ai, br, bi, ar2, ai2, br2, bi2);
                    __m256 y_avx = _mm256_set_ps(cr, ci, dr, di, cr2, ci2, dr2, di2);
                    __m256 t_avx = _mm256_moveldup_ps(x_avx);
                    __m256 t2_avx = _mm256_mul_ps(t_avx, y_avx);
                    y_avx = _mm256_shuffle_ps(y_avx,y_avx, 0xb1);
                    t_avx = _mm256_movehdup_ps(x_avx);
                    t_avx = _mm256_mul_ps(t_avx, y_avx);
                    x_avx = _mm256_addsub_ps(t2_avx, t_avx);
                    _mm256_store_ps((float*)(resultarray), x_avx);
                    C[y*n + x] = resultarray[0];
                    C[(y+1)*n + x] = resultarray[1];
                    C[(y+2)*n + x] = resultarray[2];
                    C[(y+3)*n + x] = resultarray[3];
                }
                for(int z = 0; z < m; z++){
                    C[y*n + x] += alpha*A[y*m+z]*B[z*n + x];
                }
            }
        }
    }
