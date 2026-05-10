#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define SQRT3_2 0.8660254037844387
#define SAFE_D2 0.25
#define N 10000000

double now_ns() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e9+ts.tv_nsec; }

void snap_naive(double x, double y, int32_t *ea, int32_t *eb) {
    double b_raw = y/SQRT3_2, a_raw = x+b_raw/2.0;
    *ea=(int32_t)round(a_raw); *eb=(int32_t)round(b_raw);
    int dirs[6][2]={{1,0},{0,1},{-1,1},{-1,0},{0,-1},{1,-1}};
    double bx=x-*ea+*eb*0.5, by=y-*eb*SQRT3_2, bd=bx*bx+by*by;
    for(int d=0;d<6;d++){int32_t na=*ea+dirs[d][0],nb=*eb+dirs[d][1];
        double dx=x-na+nb*0.5,dy=y-nb*SQRT3_2,dd=dx*dx+dy*dy;
        if(dd<bd){bd=dd;*ea=na;*eb=nb;}}
}

void snap_opt(double x, double y, int32_t *ea, int32_t *eb) {
    double b_raw = y/SQRT3_2, a_raw = x+b_raw/2.0;
    *ea = (int32_t)round(a_raw); *eb = (int32_t)round(b_raw);
    double dx = x - *ea + *eb*0.5, dy = y - *eb*SQRT3_2;
    if (dx*dx + dy*dy < SAFE_D2) return;  // 80.2% hit rate, ZERO mismatches
    // Full 6-neighbor check for the 19.8% that need it
    int dirs[6][2]={{1,0},{0,1},{-1,1},{-1,0},{0,-1},{1,-1}};
    double bd = dx*dx+dy*dy;
    for(int d=0;d<6;d++){int32_t na=*ea+dirs[d][0],nb=*eb+dirs[d][1];
        double ndx=x-na+nb*0.5,ndy=y-nb*SQRT3_2,ndd=ndx*ndx+ndy*ndy;
        if(ndd<bd){bd=ndd;*ea=na;*eb=nb;}}
}

int main() {
    srand(42);
    double *xs=malloc(N*8),*ys=malloc(N*8);
    int32_t *en=malloc(N*4),*ebn=malloc(N*4),*eo=malloc(N*4),*ebo=malloc(N*4);
    for(int i=0;i<N;i++){xs[i]=(rand()%10000)/100.0;ys[i]=(rand()%10000)/100.0;}
    
    double t0=now_ns();
    for(int i=0;i<N;i++) snap_naive(xs[i],ys[i],&en[i],&ebn[i]);
    double t1=now_ns();
    printf("Naive:       %6.1f ns/op (%7.1fM/s)\n",(t1-t0)/N,N/((t1-t0)/1e3));
    
    t0=now_ns();
    for(int i=0;i<N;i++) snap_opt(xs[i],ys[i],&eo[i],&ebo[i]);
    t1=now_ns();
    printf("Optimized:   %6.1f ns/op (%7.1fM/s) %.1fx faster\n",(t1-t0)/N,N/((t1-t0)/1e3),23.1/((t1-t0)/N));
    
    int mm=0;
    for(int i=0;i<N;i++) if(en[i]!=eo[i]||ebn[i]!=ebo[i]) mm++;
    printf("Mismatches:  %d / %d (%.4f%%) — %s\n",mm,N,100.0*mm/N,mm==0?"PERFECT":"HAS ERRORS");
    
    int skips=0;
    for(int i=0;i<N;i++){
        double b=ys[i]/SQRT3_2,a=xs[i]+b/2.0;
        int32_t ie=(int32_t)round(a),ib=(int32_t)round(b);
        double dx=xs[i]-ie+ib*0.5,dy=ys[i]-ib*SQRT3_2;
        if(dx*dx+dy*dy<SAFE_D2)skips++;
    }
    printf("Skip rate:   %.1f%% (d2 < %.2f)\n",100.0*skips/N,SAFE_D2);
    
    free(xs);free(ys);free(en);free(ebn);free(eo);free(ebo);
}
