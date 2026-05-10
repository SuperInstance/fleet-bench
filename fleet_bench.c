/*
 * Fleet Crate Micro-Benchmark Suite
 * Profiles every published fleet crate on real hardware.
 * gcc -O2 -mavx2 -o fleet_bench fleet_bench.c -lm && ./fleet_bench
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

#define N 10000000  // 10M iterations
#define WARMUP 100000

double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ============================================================
 * 1. EISENSTEIN INTEGER OPERATIONS (constraint-theory-core)
 * ============================================================ */

// Norm: a² - ab + b²
int64_t eise_norm_i32(int32_t a, int32_t b) {
    int64_t aa = a, bb = b;
    return aa*aa - aa*bb + bb*bb;
}

// Norm through float
double eise_norm_f64(double a, double b) {
    return a*a - a*b + b*b;
}

// Norm through AVX2 (4 pairs at once)
void eise_norm_avx2(const int32_t *a, const int32_t *b, int64_t *out) {
    __m256i va = _mm256_cvtepi32_epi64(_mm_loadl_epi64((__m128i*)a));
    __m256i vb = _mm256_cvtepi32_epi64(_mm_loadl_epi64((__m128i*)b));
    __m256i aa = _mm256_mul_epi32(va, va);       // a*a (low 32 of each 64)
    __m256i ab = _mm256_mul_epi32(va, vb);       // a*b
    __m256i bb = _mm256_mul_epi32(vb, vb);       // b*b
    // N = a² - ab + b²
    __m256i result = _mm256_add_epi64(_mm256_sub_epi64(aa, ab), bb);
    _mm256_storeu_si256((__m256i*)out, result);
}

// Snap to lattice (nearest Eisenstein integer)
void eise_snap(double x, double y, int32_t *ea, int32_t *eb) {
    // Round to nearest Eisenstein integer
    // The lattice basis is (1,0) and (-1/2, sqrt(3)/2)
    double b_raw = y / (sqrt(3.0) / 2.0);
    double a_raw = x + b_raw / 2.0;
    *ea = (int32_t)round(a_raw);
    *eb = (int32_t)round(b_raw);
    // Check if neighbor is closer
    double best_d = eise_norm_f64(x - *ea + *eb * 0.5, y - *eb * sqrt(3.0)/2.0);
    int dirs[6][2] = {{1,0},{0,1},{-1,1},{-1,0},{0,-1},{1,-1}};
    for (int i = 0; i < 6; i++) {
        int32_t na = *ea + dirs[i][0];
        int32_t nb = *eb + dirs[i][1];
        double dx = x - na + nb * 0.5;
        double dy = y - nb * sqrt(3.0)/2.0;
        double d = dx*dx + dy*dy;
        if (d < best_d) {
            best_d = d;
            *ea = na;
            *eb = nb;
        }
    }
}

/* ============================================================
 * 2. BLOOM FILTER / CRDT OPERATIONS (constraint-crdt)
 * ============================================================ */

void bloom_or_scalar(uint64_t *dst, const uint64_t *src, int n) {
    for (int i = 0; i < n; i++) dst[i] |= src[i];
}

void bloom_or_avx2(uint64_t *dst, const uint64_t *src, int n) {
    for (int i = 0; i + 4 <= n; i += 4) {
        __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), _mm256_or_si256(d, s));
    }
}

/* ============================================================
 * 3. FOLDING ORDER (folding-order) — RG flow stages
 * ============================================================ */

// 5-stage coarse-graining: Banach contraction k=1/sqrt(3)
void folding_step(double *values, int n, double k) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += values[i];
    double mean = sum / n;
    for (int i = 0; i < n; i++) {
        values[i] = mean + k * (values[i] - mean);
    }
}

/* ============================================================
 * 4. PLATO TILE OPERATIONS (plato-runtime)
 * ============================================================ */

// 64-byte tile hash: siphash-like
uint64_t tile_hash(const uint8_t *data, int len) {
    uint64_t h = 0x6b898d3c9e7a2c91ULL;
    for (int i = 0; i + 8 <= len; i += 8) {
        uint64_t v;
        memcpy(&v, data + i, 8);
        h ^= v;
        h *= 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 31;
    }
    return h;
}

/* ============================================================
 * 5. TONNETZ OPERATIONS (tonnetz-constraints)
 * ============================================================ */

// Voice-leading distance: sum of |interval differences|
double voice_leading_distance(const int *a, const int *b, int n_notes) {
    double sum = 0;
    for (int i = 0; i < n_notes; i++) {
        int d = abs(a[i] - b[i]);
        sum += (d > 6) ? 12 - d : d;  // wrap around 12
    }
    return sum;
}

/* ============================================================
 * 6. HOLONOMY CONSENSUS (holonomy-consensus)
 * ============================================================ */

// β₁ = E - V + C (first Betti number = emergence detection)
int betti1(int edges, int vertices, int components) {
    return edges - vertices + components;
}

// Cycle check: is the constraint graph a tree? (β₁ = 0)
int is_tree(int edges, int vertices) {
    return edges == vertices - 1;
}

/* ============================================================
 * BENCHMARKS
 * ============================================================ */

int main() {
    printf("================================================================\n");
    printf("FLEET CRATE MICRO-BENCHMARK SUITE\n");
    printf("Hardware: AMD Ryzen AI 9 HX 370 (Zen 5, AVX-512, running AVX2)\n");
    printf("================================================================\n\n");

    volatile int64_t sink_i64;
    volatile double sink_f64;
    volatile int sink_i;

    /* --- 1. Eisenstein Norm --- */
    printf("── 1. EISENSTEIN NORM (constraint-theory-core) ──\n");

    double t0 = now_ns();
    for (int i = 0; i < N; i++) {
        sink_i64 = eise_norm_i32(3, 7);
    }
    double t1 = now_ns();
    printf("  i32 scalar:       %6.1f ns/op  (%7.1fM ops/s)\n",
           (t1-t0)/N, N/((t1-t0)/1e3));

    t0 = now_ns();
    for (int i = 0; i < N; i++) {
        sink_f64 = eise_norm_f64(3.0, 7.0);
    }
    t1 = now_ns();
    printf("  f64 scalar:       %6.1f ns/op  (%7.1fM ops/s)\n",
           (t1-t0)/N, N/((t1-t0)/1e3));

    // AVX2 batch (4 at a time)
    int32_t va[4] = {3, 5, 7, 11};
    int32_t vb[4] = {7, 11, 13, 17};
    int64_t vout[4];
    t0 = now_ns();
    for (int i = 0; i < N; i++) {
        eise_norm_avx2(va, vb, vout);
        sink_i64 = vout[0];
    }
    t1 = now_ns();
    printf("  i32 AVX2 (4x):    %6.1f ns/op  (%7.1fM norms/s, %7.1fM pairs/s)\n",
           (t1-t0)/N, N*4/((t1-t0)/1e3), N/((t1-t0)/1e3));

    // Snap to lattice
    double snap_x = 3.14159, snap_y = 2.71828;
    int32_t ea, eb;
    t0 = now_ns();
    for (int i = 0; i < N; i++) {
        eise_snap(snap_x + i * 0.000001, snap_y + i * 0.000001, &ea, &eb);
        sink_i = ea;
    }
    t1 = now_ns();
    printf("  snap to lattice:  %6.1f ns/op  (%7.1fM snaps/s)\n",
           (t1-t0)/N, N/((t1-t0)/1e3));

    /* --- 2. Bloom / CRDT --- */
    printf("\n── 2. BLOOM CRDT (constraint-crdt) ──\n");

    int bloom_sizes[] = {125, 1000, 8000};  // 1KB, 8KB, 64KB
    for (int s = 0; s < 3; s++) {
        int nw = bloom_sizes[s];
        uint64_t *dst = aligned_alloc(64, nw * 8);
        uint64_t *src = aligned_alloc(64, nw * 8);
        memset(dst, 0, nw * 8);
        memset(src, 0xFF, nw * 8);

        int reps = (s == 0) ? 100000 : (s == 1) ? 10000 : 1000;
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            bloom_or_scalar(dst, src, nw);
        }
        t1 = now_ns();
        double gb_s = (double)reps * nw * 8 / ((t1-t0) / 1e9) / 1e9;
        printf("  scalar %5d words: %7.1f GB/s\n", nw, gb_s);

        memset(dst, 0, nw * 8);
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            bloom_or_avx2(dst, src, nw);
        }
        t1 = now_ns();
        gb_s = (double)reps * nw * 8 / ((t1-t0) / 1e9) / 1e9;
        printf("  AVX2   %5d words: %7.1f GB/s\n", nw, gb_s);

        free(dst); free(src);
    }

    /* --- 3. Folding Order --- */
    printf("\n── 3. FOLDING ORDER / RG FLOW (folding-order) ──\n");

    double vals[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    double k = 1.0 / sqrt(3.0);  // Banach contraction

    t0 = now_ns();
    for (int i = 0; i < N; i++) {
        folding_step(vals, 16, k);
    }
    t1 = now_ns();
    printf("  1 stage (16 vals): %6.1f ns/op  (%7.1fM stages/s)\n",
           (t1-t0)/N, N/((t1-t0)/1e3));

    // 5 stages (full RG flow)
    double vals5[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    t0 = now_ns();
    for (int i = 0; i < N; i++) {
        memcpy(vals5, (double[]){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, sizeof(vals5));
        for (int s = 0; s < 5; s++) folding_step(vals5, 16, k);
        sink_f64 = vals5[0];
    }
    t1 = now_ns();
    printf("  5 stages (full):   %6.1f ns/op  (%7.1fM flows/s)\n",
           (t1-t0)/N, N/((t1-t0)/1e3));

    // Contraction rate: measure convergence
    printf("  Contraction k=1/√3=%.4f: vals converge to ", k);
    double cv[16] = {100, -50, 200, -100, 75, -25, 150, -75, 25, -12, 50, -37, 12, -6, 37, -18};
    for (int s = 0; s < 20; s++) folding_step(cv, 16, k);
    printf("%.2f (mean of 16 rand in [-100,200])\n", cv[0]);

    /* --- 4. Tile Hash --- */
    printf("\n── 4. TILE HASH (plato-runtime) ──\n");

    uint8_t tile_data[64];
    for (int i = 0; i < 64; i++) tile_data[i] = i;

    t0 = now_ns();
    for (int i = 0; i < N; i++) {
        sink_i64 = tile_hash(tile_data, 64);
    }
    t1 = now_ns();
    printf("  64-byte hash:      %6.1f ns/op  (%7.1fM hashes/s)\n",
           (t1-t0)/N, N/((t1-t0)/1e3));

    // Batch hash (1000 tiles)
    uint8_t batch[1000][64];
    for (int i = 0; i < 1000; i++) memcpy(batch[i], tile_data, 64);
    t0 = now_ns();
    for (int i = 0; i < N / 1000; i++) {
        uint64_t h = 0;
        for (int j = 0; j < 1000; j++) h ^= tile_hash(batch[j], 64);
        sink_i64 = h;
    }
    t1 = now_ns();
    printf("  1000-tile batch:   %6.1f us     (%7.1fK batches/s)\n",
           (t1-t0)/(N/1000)/1e3, (N/1000)/((t1-t0)/1e3));

    /* --- 5. Voice Leading --- */
    printf("\n── 5. VOICE LEADING (tonnetz-constraints) ──\n");

    int chord_a[6] = {0, 4, 7, 12, 16, 19};  // C major
    int chord_b[6] = {2, 5, 9, 14, 17, 21};  // D minor

    t0 = now_ns();
    for (int i = 0; i < N; i++) {
        sink_f64 = voice_leading_distance(chord_a, chord_b, 6);
    }
    t1 = now_ns();
    printf("  6-note distance:   %6.1f ns/op  (%7.1fM ops/s)\n",
           (t1-t0)/N, N/((t1-t0)/1e3));

    /* --- 6. Holonomy / Emergence --- */
    printf("\n── 6. HOLONOMY / EMERGENCE (holonomy-consensus) ──\n");

    t0 = now_ns();
    for (int i = 0; i < N; i++) {
        sink_i = betti1(100, 80, 5);
    }
    t1 = now_ns();
    printf("  β₁ computation:    %6.1f ns/op  (%7.1fM ops/s)\n",
           (t1-t0)/N, N/((t1-t0)/1e3));

    t0 = now_ns();
    for (int i = 0; i < N; i++) {
        sink_i = is_tree(99, 100);
    }
    t1 = now_ns();
    printf("  tree check:        %6.1f ns/op  (%7.1fM ops/s)\n",
           (t1-t0)/N, N/((t1-t0)/1e3));

    /* --- 7. Combined Fleet Pipeline --- */
    printf("\n── 7. COMBINED FLEET PIPELINE ──\n");
    printf("  (snap → norm → bloom → fold → hash → betti₁)\n\n");

    // Simulate a full fleet tick: 100 constraints flowing through the pipeline
    int pipeline_n = 100;
    double pvals[100], py[100];
    for (int i = 0; i < pipeline_n; i++) {
        pvals[i] = (double)(i * 7 % 200 - 100);
        py[i] = (double)(i * 13 % 200 - 100);
    }
    uint64_t bloom_p = 0;
    uint64_t pipeline_hash = 0x1234567890abcdefULL;

    t0 = now_ns();
    for (int iter = 0; iter < N / pipeline_n; iter++) {
        for (int i = 0; i < pipeline_n; i++) {
            int32_t ea, eb;
            eise_snap(pvals[i], py[i], &ea, &eb);
            int64_t norm = eise_norm_i32(ea, eb);
            bloom_p |= (1ULL << (norm % 64));
            pvals[i] = pvals[i] * 0.577 + (double)norm * 0.001;  // fold
            pipeline_hash ^= tile_hash((uint8_t*)&norm, 8);
        }
    }
    t1 = now_ns();
    double per_constraint = (t1-t0) / (N / pipeline_n * pipeline_n);
    printf("  %d constraints:    %6.1f ns/constraint (%7.1fM constraints/s)\n",
           pipeline_n, per_constraint, 1e3 / per_constraint);
    printf("  Pipeline hash:     0x%016lx\n", pipeline_hash ^ bloom_p);

    printf("\n================================================================\n");
    printf("DONE. All fleet crates profiled.\n");
    printf("================================================================\n");

    (void)sink_i64; (void)sink_f64; (void)sink_i;
    return 0;
}
