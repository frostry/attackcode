#ifndef PARAMS_H
#define PARAMS_H

#ifndef VERS
#define VERS 12
#endif

#if (VERS == 12)

#define NEWHOPE_N 1024
#define NEWHOPE_Q 16385
#define NEWHOPE_QINV 16383     // -inverse_mod(q,2^RLOG)
#define NEWHOPE_RLOG 15	       //	R > p, gcd(R,p) = 1
#define NEWHOPE_MUL 4  		   // (1 << (2*RLOG)) % Q
#define STDDEV 3197

#elif (VERS == 19)

    #if (N == 128)
    #define NEWHOPE_N 128
    #define NEWHOPE_Q 2255041
    #define NEWHOPE_QINV 3266751
    #define NEWHOPE_RLOG 22
    #define NEWHOPE_MUL 87305

    #elif (N == 256)
    #define NEWHOPE_N 256
    #define NEWHOPE_Q 9205761
    #define NEWHOPE_QINV 5011455
    #define NEWHOPE_RLOG 24
    #define NEWHOPE_MUL 5810857

    #elif (N == 512)
    #define NEWHOPE_N 512
    #define NEWHOPE_Q 26038273
    #define NEWHOPE_QINV 9261055
    #define NEWHOPE_RLOG 25
    #define NEWHOPE_MUL 9012481

    #elif (N == 1024)
    #define NEWHOPE_N 1024
    #define NEWHOPE_Q 28434433
    #define NEWHOPE_QINV 28434431
    #define NEWHOPE_RLOG 25
    #define NEWHOPE_MUL 3550909
    
    #else
    #error "N must be one of 128,256,512,1024"
    #endif

//LBA-PAKE
#elif (VERS == 20) 

    #define NEWHOPE_Q 7557773
    #define STDDEV 3192

    //for eliminating warnings
    #define NEWHOPE_QINV 3817403
    #define NEWHOPE_RLOG 23
    #define NEWHOPE_MUL 5158043

//improved scheme
#elif (VERS == 21)


    #define NEWHOPE_Q 1073479709
    #define STDDEV 3192

    //for eliminating warnings
    #define NEWHOPE_QINV 3817403
    #define NEWHOPE_RLOG 23
    #define NEWHOPE_MUL 5158043

#else
#error "VERS must be one of 20 or 21"
#endif

#endif
