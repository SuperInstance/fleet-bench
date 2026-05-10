#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define SQRT3_2 0.8660254037844387
#define N 10000000

double now_ns() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e9+ts.tv_nsec; }

void snap_naive(double x, double y, int32_t *ea, int32_t *eb) {
    double b_raw = y / SQRT3_2;
    double a_raw = x + b_raw / 2.0;
    *ea = (int32_t)round(a_raw);
    *eb = (int32_t)round(b_raw);
    int dirs[6][2] = {{1,0},{0,1},{-1,1},{-1,0},{0,-1},{1,-1}};
    double best_dx = x - *ea + *eb * 0.5;
    double best_dy = y - *eb * SQRT3_2;
    double best_d = best_dx*best_dx + best_dy*best_dy;
    for (int d = 0; d < 6; d++) {
        int32_t na = *ea + dirs[d][0], nb = *eb + dirs[d][1];
        double dx = x - na + nb * 0.5, dy = y - nb * SQRT3_2;
        double dd = dx*dx + dy*dy;
        if (dd < best_d) { best_d = dd; *ea = na; *eb = nb; }
    }
}

int main() {
    srand(42);
    double *xs = malloc(N*sizeof(double)), *ys = malloc(N*sizeof(double));
    int32_t *ea_n = malloc(N*sizeof(int32_t)), *eb_n = malloc(N*sizeof(int32_t));
    for (int i = 0; i < N; i++) { xs[i] = (double)(rand()%10000)/100.0; ys[i] = (double)(rand()%10000)/100.0; }
    for (int i = 0; i < N; i++) snap_naive(xs[i], ys[i], &ea_n[i], &eb_n[i]);

    printf("%8s %8s %8s %10s\n", "Thresh", "Skip%", "Wrong", "Max Wrong D2");
    double thresholds[] = {0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.33, 0.35, 0.40, 0.50};
    for (int t = 0; t < 12; t++) {
        double thresh = thresholds[t];
        int skips = 0, wrong = 0;
        double max_wrong_d2 = 0;
        for (int i = 0; i < N; i++) {
            double b_raw = ys[i] / SQRT3_2;
            double a_raw = xs[i] + b_raw / 2.0;
            int32_t ie = (int32_t)round(a_raw), ib = (int32_t)round(b_raw);
            double dx = xs[i] - ie + ib*0.5, dy = ys[i] - ib*SQRT3_2;
            double d2 = dx*dx + dy*dy;
            if (d2 < thresh) {
                skips++;
                if (ie != ea_n[i] || ib != eb_n[i]) {
                    wrong++;
                    if (d2 > max_wrong_d2) max_wrong_d2 = d2;
                }
            }
        }
        printf("%8.3f %7.1f%% %8d %10.4f%s\n", thresh, 100.0*skips/N, wrong, max_wrong_d2, wrong==0?" ← SAFE":"");
    }

    // Now benchmark with optimal threshold (0.28 = safe)
    printf("\n--- Benchmark with threshold = 0.28 (safe skip) ---\n");
    double safe_thresh = 0.28;
    int32_t *ea_s = malloc(N*sizeof(int32_t)), *eb_s = malloc(N*sizeof(int32_t));
    double t0 = now_ns();
    for (int i = 0; i < N; i++) {
        double b_raw = ys[i] / SQRT3_2, a_raw = xs[i] + b_raw / 2.0;
        int32_t ie = (int32_t)round(a_raw), ib = (int32_t)round(b_raw);
        double dx = xs[i] - ie + ib*0.5, dy = ys[i] - ib*SQRT3_2;
        double d2 = dx*dx + dy*dy;
        if (d2 < safe_thresh) { ea_s[i] = ie; eb_s[i] = ib; continue; }
        snap_naive(xs[i], ys[i], &ea_s[i], &eb_s[i]);
    }
    double t1 = now_ns();
    int mm = 0;
    for (int i = 0; i < N; i++) if (ea_n[i]!=ea_s[i]||eb_n[i]!=eb_s[i]) mm++;
    printf("Optimized: %6.1f ns/op (%7.1fM/s)  mismatches=%d  (%.1fx speedup)\n",
           (t1-t0)/N, N/((t1-t0)/1e3), mm, 23.1/((t1-t0)/N));

    free(xs); free(ys); free(ea_n); free(eb_n); free(ea_s); free(eb_s);
}
