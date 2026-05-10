/*
 * Snap v3: Fixed — no atan2, direct sector from dx/dy comparison.
 * 
 * gcc -O2 -mavx2 -o snap_v3 snap_v3.c -lm && ./snap_v3
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define SQRT3_2 0.8660254037844387
#define N 10000000

double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* Baseline: check all 6 neighbors */
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
        int32_t na = *ea + dirs[d][0];
        int32_t nb = *eb + dirs[d][1];
        double dx = x - na + nb * 0.5;
        double dy = y - nb * SQRT3_2;
        double dd = dx*dx + dy*dy;
        if (dd < best_d) { best_d = dd; *ea = na; *eb = nb; }
    }
}

/* v2: Voronoi skip + direct neighbor check (NO atan2) */
void snap_fast(double x, double y, int32_t *ea, int32_t *eb) {
    double b_raw = y / SQRT3_2;
    double a_raw = x + b_raw / 2.0;
    *ea = (int32_t)round(a_raw);
    *eb = (int32_t)round(b_raw);
    
    double dx = x - *ea + *eb * 0.5;
    double dy = y - *eb * SQRT3_2;
    double d2 = dx*dx + dy*dy;
    
    // Fast path: if within inscribed circle of Voronoi cell, we're done
    if (d2 < 0.08333) return;
    
    // Determine which neighbor(s) to check using sign comparison
    // The Eisenstein lattice has 3 primary axes:
    //   (1,0), (0,1), (1,-1) and their negatives
    // We can determine the sector using only comparisons
    
    // Project onto the 3 axes to find which direction to check
    double p0 = dx;                          // along (1,0) direction
    double p1 = -dx * 0.5 + dy * SQRT3_2;   // along (0,1) direction  
    double p2 = dx * 0.5 + dy * SQRT3_2;    // along (1,-1) direction
    
    // Check the neighbor in the direction of maximum projection
    static const int dirs[6][2] = {{1,0},{0,1},{-1,1},{-1,0},{0,-1},{1,-1}};
    
    // Only check the 2 most likely neighbors
    if (p0 > 0) {
        int32_t na = *ea + 1, nb = *eb;
        double ndx = x - na + nb * 0.5;
        double ndy = y - nb * SQRT3_2;
        double nd2 = ndx*ndx + ndy*ndy;
        if (nd2 < d2) { *ea = na; *eb = nb; d2 = nd2; }
    } else {
        int32_t na = *ea - 1, nb = *eb;
        double ndx = x - na + nb * 0.5;
        double ndy = y - nb * SQRT3_2;
        double nd2 = ndx*ndx + ndy*ndy;
        if (nd2 < d2) { *ea = na; *eb = nb; d2 = nd2; }
    }
    
    if (p1 > 0) {
        int32_t na = *ea, nb = *eb + 1;
        double ndx = x - na + nb * 0.5;
        double ndy = y - nb * SQRT3_2;
        double nd2 = ndx*ndx + ndy*ndy;
        if (nd2 < d2) { *ea = na; *eb = nb; }
    } else {
        int32_t na = *ea, nb = *eb - 1;
        double ndx = x - na + nb * 0.5;
        double ndy = y - nb * SQRT3_2;
        double nd2 = ndx*ndx + ndy*ndy;
        if (nd2 < d2) { *ea = na; *eb = nb; }
    }
}

/* v3: Skip + check only 1 neighbor (the most aggressive optimization) */
void snap_aggressive(double x, double y, int32_t *ea, int32_t *eb) {
    double b_raw = y / SQRT3_2;
    double a_raw = x + b_raw / 2.0;
    *ea = (int32_t)round(a_raw);
    *eb = (int32_t)round(b_raw);
    
    double dx = x - *ea + *eb * 0.5;
    double dy = y - *eb * SQRT3_2;
    double d2 = dx*dx + dy*dy;
    
    // Quick check: if we're well within the cell, done
    if (d2 < 0.08333) return;
    
    // Check only the 3 positive-axis neighbors (covers most cases)
    int best_d = 0; // 0 = current, 1-3 = neighbors
    double best_d2 = d2;
    
    // Neighbor (1,0)
    {
        int32_t na = *ea + 1, nb = *eb;
        double ndx = x - na + nb * 0.5;
        double ndy = y - nb * SQRT3_2;
        double nd2 = ndx*ndx + ndy*ndy;
        if (nd2 < best_d2) { best_d = 1; best_d2 = nd2; }
    }
    // Neighbor (0,1)
    {
        int32_t na = *ea, nb = *eb + 1;
        double ndx = x - na + nb * 0.5;
        double ndy = y - nb * SQRT3_2;
        double nd2 = ndx*ndx + ndy*ndy;
        if (nd2 < best_d2) { best_d = 2; best_d2 = nd2; }
    }
    // Neighbor (-1,1)
    {
        int32_t na = *ea - 1, nb = *eb + 1;
        double ndx = x - na + nb * 0.5;
        double ndy = y - nb * SQRT3_2;
        double nd2 = ndx*ndx + ndy*ndy;
        if (nd2 < best_d2) { best_d = 3; best_d2 = nd2; }
    }
    
    if (best_d == 1) { *ea += 1; }
    else if (best_d == 2) { *eb += 1; }
    else if (best_d == 3) { *ea -= 1; *eb += 1; }
}

int main() {
    printf("=== Snap v3: No-atan2 Optimization ===\n\n");
    
    srand(42);
    double *xs = malloc(N * sizeof(double));
    double *ys = malloc(N * sizeof(double));
    int32_t *ea_n = calloc(N, sizeof(int32_t));
    int32_t *eb_n = calloc(N, sizeof(int32_t));
    int32_t *ea_f = calloc(N, sizeof(int32_t));
    int32_t *eb_f = calloc(N, sizeof(int32_t));
    int32_t *ea_a = calloc(N, sizeof(int32_t));
    int32_t *eb_a = calloc(N, sizeof(int32_t));
    
    for (int i = 0; i < N; i++) {
        xs[i] = (double)(rand() % 10000) / 100.0;
        ys[i] = (double)(rand() % 10000) / 100.0;
    }
    
    /* Naive */
    double t0 = now_ns();
    for (int i = 0; i < N; i++) snap_naive(xs[i], ys[i], &ea_n[i], &eb_n[i]);
    double t1 = now_ns();
    double naive = (t1-t0)/N;
    printf("Naive (6 neighbors):     %6.1f ns/op  (%7.1fM/s) — BASELINE\n", naive, N/((t1-t0)/1e3));
    
    /* Fast (3-axis, no atan2) */
    t0 = now_ns();
    for (int i = 0; i < N; i++) snap_fast(xs[i], ys[i], &ea_f[i], &eb_f[i]);
    t1 = now_ns();
    double fast = (t1-t0)/N;
    int mm_f = 0;
    for (int i = 0; i < N; i++) if (ea_n[i]!=ea_f[i] || eb_n[i]!=eb_f[i]) mm_f++;
    printf("Fast (3-axis, no atan2): %6.1f ns/op  (%7.1fM/s)  %.1fx  mismatches=%d\n", 
           fast, N/((t1-t0)/1e3), naive/fast, mm_f);
    
    /* Aggressive (3 positive neighbors only) */
    t0 = now_ns();
    for (int i = 0; i < N; i++) snap_aggressive(xs[i], ys[i], &ea_a[i], &eb_a[i]);
    t1 = now_ns();
    double aggr = (t1-t0)/N;
    int mm_a = 0;
    for (int i = 0; i < N; i++) if (ea_n[i]!=ea_a[i] || eb_n[i]!=eb_a[i]) mm_a++;
    printf("Aggressive (3 pos nbrs): %6.1f ns/op  (%7.1fM/s)  %.1fx  mismatches=%d\n",
           aggr, N/((t1-t0)/1e3), naive/aggr, mm_a);
    
    printf("\n--- Analysis ---\n");
    printf("Voronoi skip rate (d2<0.08333): ");
    int skips = 0;
    for (int i = 0; i < N; i++) {
        double b_raw = ys[i] / SQRT3_2;
        double a_raw = xs[i] + b_raw / 2.0;
        int32_t ie = (int32_t)round(a_raw);
        int32_t ib = (int32_t)round(b_raw);
        double dx = xs[i] - ie + ib * 0.5;
        double dy = ys[i] - ib * SQRT3_2;
        if (dx*dx + dy*dy < 0.08333) skips++;
    }
    printf("%.1f%%\n", 100.0*skips/N);
    
    // What if we use a larger threshold (less accurate but faster)?
    printf("Skip at d2<0.20:  %.1f%%\n", 100.0*skips/N);  // skip count doesn't change, need to recount
    int skips20 = 0, mismatches20 = 0;
    for (int i = 0; i < N; i++) {
        double b_raw = ys[i] / SQRT3_2;
        double a_raw = xs[i] + b_raw / 2.0;
        int32_t ie = (int32_t)round(a_raw);
        int32_t ib = (int32_t)round(b_raw);
        double dx = xs[i] - ie + ib * 0.5;
        double dy = ys[i] - ib * SQRT3_2;
        if (dx*dx + dy*dy < 0.20) {
            skips20++;
            if (ie != ea_n[i] || ib != eb_n[i]) mismatches20++;
        }
    }
    printf("Skip at d2<0.20:  %d/%d (%.1f%%) with %d wrong (%.3f%%)\n",
           skips20, N, 100.0*skips20/N, mismatches20, 100.0*mismatches20/N);
    
    free(xs); free(ys); free(ea_n); free(eb_n); free(ea_f); free(eb_f); free(ea_a); free(eb_a);
    return 0;
}
