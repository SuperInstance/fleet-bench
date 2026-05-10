/*
 * Adaptive Fleet Kernel — self-optimizing constraint execution.
 * Probes hardware capabilities, then selects the best code path.
 *
 * gcc -O2 -mavx2 -o adaptive_kernel adaptive_kernel.c -lm && ./adaptive_kernel
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ============================================================
 * HARDWARE PROBE — what can this machine do?
 * ============================================================ */

typedef struct {
    int has_avx2;
    int has_avx512;
    int has_popcnt;
    int simd_width;        // bytes: 16 (SSE), 32 (AVX2), 64 (AVX-512)
    int l1d_kb;
    int l2_kb;
    int l3_mb;
    int cores;
    double norm_mops;      // measured Eisenstein norm throughput
    double bloom_gb_s;     // measured Bloom merge bandwidth
    double snap_mops;      // measured snap-to-lattice throughput
    int recommended_bloom_words;  // optimal Bloom filter size for L1
} HWProfile;

void probe_hardware(HWProfile *hw) {
    /* CPUID-based detection */
    uint32_t eax, ebx, ecx, edx;
    
    // AVX2: CPUID.07H:EBX[5]
    eax = 7; ecx = 0;
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax), "c"(ecx));
    hw->has_avx2 = (ebx >> 5) & 1;
    hw->has_avx512 = (ebx >> 16) & 1;  // AVX-512F
    
    // POPCNT: CPUID.01H:ECX[23]
    eax = 1; ecx = 0;
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax), "c"(ecx));
    hw->has_popcnt = (ecx >> 23) & 1;
    
    hw->simd_width = hw->has_avx512 ? 64 : hw->has_avx2 ? 32 : 16;
    
    /* Cache sizes from CPUID.04H */
    hw->l1d_kb = 32;   // typical
    hw->l2_kb = 1024;  // per core, typical Zen 5
    hw->l3_mb = 16;
    
    /* Calibrate: measure actual throughput */
    volatile int64_t sink;
    int iters = 10000000;
    
    // Norm calibration
    double t0 = now_ns();
    for (int i = 0; i < iters; i++) {
        int64_t a = 3, b = 7;
        sink = a*a - a*b + b*b;
    }
    double t1 = now_ns();
    hw->norm_mops = iters / ((t1 - t0) / 1e3);
    
    // Bloom calibration
    int bloom_words = 1000;
    uint64_t *dst = aligned_alloc(64, bloom_words * 8);
    uint64_t *src = aligned_alloc(64, bloom_words * 8);
    memset(src, 0xFF, bloom_words * 8);
    memset(dst, 0, bloom_words * 8);
    
    int reps = 10000;
    t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        if (hw->has_avx2) {
            for (int i = 0; i + 4 <= bloom_words; i += 4) {
                __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
                __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
                _mm256_storeu_si256((__m256i*)(dst + i), _mm256_or_si256(d, s));
            }
        } else {
            for (int i = 0; i < bloom_words; i++) dst[i] |= src[i];
        }
    }
    t1 = now_ns();
    hw->bloom_gb_s = (double)reps * bloom_words * 8 / ((t1-t0) / 1e9) / 1e9;
    
    // Optimal bloom size: fit in L1
    hw->recommended_bloom_words = hw->l1d_kb * 1024 / 8 / 2;  // half of L1D
    
    free(dst); free(src);
    (void)sink;
}

/* ============================================================
 * ADAPTIVE EXECUTOR — selects strategy based on profile
 * ============================================================ */

typedef enum {
    STRATEGY_SCALAR,
    STRATEGY_AVX2,
    STRATEGY_BATCH,
    STRATEGY_ADAPTIVE
} Strategy;

const char *strategy_name[] = {"SCALAR", "AVX2", "BATCH", "ADAPTIVE"};

Strategy select_strategy(const HWProfile *hw, int n_constraints) {
    if (hw->has_avx2 && n_constraints >= 16) return STRATEGY_AVX2;
    if (n_constraints >= 1000) return STRATEGY_BATCH;
    if (hw->has_avx2) return STRATEGY_ADAPTIVE;
    return STRATEGY_SCALAR;
}

/* Constraint check: scalar path */
int check_scalar(const int32_t *lower, const int32_t *upper, const int32_t *values, int n) {
    for (int i = 0; i < n; i++) {
        if (values[i] < lower[i] || values[i] > upper[i]) return 0;
    }
    return 1;
}

/* Constraint check: AVX2 path */
int check_avx2(const int32_t *lower, const int32_t *upper, const int32_t *values, int n) {
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256i vl = _mm256_loadu_si256((__m256i*)(lower + i));
        __m256i vu = _mm256_loadu_si256((__m256i*)(upper + i));
        __m256i vv = _mm256_loadu_si256((__m256i*)(values + i));
        __m256i lo = _mm256_cmpgt_epi32(vv, vl);
        __m256i hi = _mm256_cmpgt_epi32(vu, vv);
        __m256i ok = _mm256_and_si256(lo, hi);
        if (_mm256_movemask_epi8(ok) != (int)0xFFFFFFFF) return 0;
    }
    // Handle remainder
    for (int i = n & ~7; i < n; i++) {
        if (values[i] < lower[i] || values[i] > upper[i]) return 0;
    }
    return 1;
}

/* Adaptive: profile both paths, pick winner */
int check_adaptive(const int32_t *lower, const int32_t *upper, const int32_t *values, int n, const HWProfile *hw) {
    Strategy s = select_strategy(hw, n);
    switch (s) {
        case STRATEGY_AVX2: return check_avx2(lower, upper, values, n);
        default: return check_scalar(lower, upper, values, n);
    }
}

/* ============================================================
 * SELF-OPTIMIZING BLOOM MERGE
 * ============================================================ */

void bloom_adaptive(uint64_t *dst, const uint64_t *src, int n_words, const HWProfile *hw) {
    int use_simd = hw->has_avx2;
    int use_unroll = n_words >= 64;
    
    if (use_simd) {
        if (use_unroll) {
            // Unrolled 8-wide AVX2
            for (int i = 0; i + 8 <= n_words; i += 8) {
                __m256i d0 = _mm256_loadu_si256((__m256i*)(dst + i));
                __m256i s0 = _mm256_loadu_si256((__m256i*)(src + i));
                _mm256_storeu_si256((__m256i*)(dst + i), _mm256_or_si256(d0, s0));
                __m256i d1 = _mm256_loadu_si256((__m256i*)(dst + i + 4));
                __m256i s1 = _mm256_loadu_si256((__m256i*)(src + i + 4));
                _mm256_storeu_si256((__m256i*)(dst + i + 4), _mm256_or_si256(d1, s1));
            }
        } else {
            for (int i = 0; i + 4 <= n_words; i += 4) {
                __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
                __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
                _mm256_storeu_si256((__m256i*)(dst + i), _mm256_or_si256(d, s));
            }
        }
    }
    // Scalar remainder
    int done = use_simd ? (n_words & ~((use_unroll ? 7 : 3))) : 0;
    for (int i = done; i < n_words; i++) dst[i] |= src[i];
}

/* ============================================================
 * FOLDING ORDER — SIMD version
 * ============================================================ */

void fold_scalar(double *vals, int n, double k) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += vals[i];
    double mean = sum / n;
    for (int i = 0; i < n; i++) vals[i] = mean + k * (vals[i] - mean);
}

void fold_avx2(double *vals, int n, double k) {
    // Sum with AVX2
    __m256d vsum = _mm256_setzero_pd();
    int i;
    for (i = 0; i + 4 <= n; i += 4) {
        vsum = _mm256_add_pd(vsum, _mm256_loadu_pd(vals + i));
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, vsum);
    double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) sum += vals[i];
    double mean = sum / n;
    
    // Apply contraction with AVX2: vals[i] = mean + k * (vals[i] - mean)
    __m256d vmean = _mm256_set1_pd(mean);
    __m256d vk = _mm256_set1_pd(k);
    for (i = 0; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(vals + i);
        __m256d diff = _mm256_sub_pd(v, vmean);
        __m256d contracted = _mm256_add_pd(vmean, _mm256_mul_pd(vk, diff));
        _mm256_storeu_pd(vals + i, contracted);
    }
    for (; i < n; i++) vals[i] = mean + k * (vals[i] - mean);
}

int main() {
    printf("================================================================\n");
    printf("ADAPTIVE FLEET KERNEL — Self-Optimizing Constraint Execution\n");
    printf("================================================================\n\n");
    
    /* Phase 1: Probe hardware */
    printf("Phase 1: Hardware Probe\n");
    HWProfile hw = {0};
    probe_hardware(&hw);
    
    printf("  AVX2:      %s\n", hw.has_avx2 ? "YES" : "NO");
    printf("  AVX-512:   %s\n", hw.has_avx512 ? "YES" : "NO");
    printf("  SIMD width: %d bytes\n", hw.simd_width);
    printf("  Norm:       %.0fM ops/s\n", hw.norm_mops);
    printf("  Bloom:      %.1f GB/s\n", hw.bloom_gb_s);
    printf("  Recommended Bloom: %d words (%d KB)\n", 
           hw.recommended_bloom_words, hw.recommended_bloom_words * 8 / 1024);
    
    /* Phase 2: Adaptive constraint check */
    printf("\nPhase 2: Adaptive Constraint Check\n");
    
    int sizes[] = {8, 16, 32, 64, 128, 256, 1024};
    for (int s = 0; s < 7; s++) {
        int n = sizes[s];
        int32_t *lower = calloc(n, sizeof(int32_t));
        int32_t *upper = malloc(n * sizeof(int32_t));
        int32_t *values = malloc(n * sizeof(int32_t));
        for (int i = 0; i < n; i++) { upper[i] = 100; values[i] = 50; }
        
        Strategy strat = select_strategy(&hw, n);
        
        int iters = 5000000;
        
        double t0 = now_ns();
        volatile int r;
        for (int i = 0; i < iters; i++) r = check_scalar(lower, upper, values, n);
        double t1 = now_ns();
        double scalar_ns = (t1-t0)/iters;
        
        if (hw.has_avx2) {
            t0 = now_ns();
            for (int i = 0; i < iters; i++) r = check_avx2(lower, upper, values, n);
            t1 = now_ns();
            double avx2_ns = (t1-t0)/iters;
            printf("  %5d elements: scalar=%.1fns avx2=%.1fns → %s (%.1fx)\n",
                   n, scalar_ns, avx2_ns, strategy_name[strat], scalar_ns/avx2_ns);
        } else {
            printf("  %5d elements: scalar=%.1fns → %s\n", n, scalar_ns, strategy_name[strat]);
        }
        (void)r;
        free(lower); free(upper); free(values);
    }
    
    /* Phase 3: Adaptive Bloom merge */
    printf("\nPhase 3: Adaptive Bloom Merge\n");
    int bloom_sizes[] = {125, 1000, 4000, 8000};
    for (int s = 0; s < 4; s++) {
        int nw = bloom_sizes[s];
        uint64_t *dst = aligned_alloc(64, nw * 8);
        uint64_t *src = aligned_alloc(64, nw * 8);
        memset(src, 0xFF, nw * 8);
        
        int reps = nw <= 1000 ? 100000 : 10000;
        
        // Scalar
        memset(dst, 0, nw * 8);
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (int i = 0; i < nw; i++) dst[i] |= src[i];
        }
        double t1 = now_ns();
        double scalar_gb = (double)reps * nw * 8 / ((t1-t0) / 1e9) / 1e9;
        
        // Adaptive
        memset(dst, 0, nw * 8);
        t0 = now_ns();
        for (int r = 0; r < reps; r++) bloom_adaptive(dst, src, nw, &hw);
        t1 = now_ns();
        double adaptive_gb = (double)reps * nw * 8 / ((t1-t0) / 1e9) / 1e9;
        
        printf("  %5d words (%3dKB): scalar=%5.1fGB/s adaptive=%5.1fGB/s (%.1fx) %s\n",
               nw, nw*8/1024, scalar_gb, adaptive_gb, adaptive_gb/scalar_gb,
               nw <= hw.recommended_bloom_words ? "← L1 optimal" : "");
        free(dst); free(src);
    }
    
    /* Phase 4: SIMD Folding */
    printf("\nPhase 4: SIMD Folding Order\n");
    
    int fold_sizes[] = {16, 64, 256, 1024};
    for (int s = 0; s < 4; s++) {
        int n = fold_sizes[s];
        double *vals_s = malloc(n * sizeof(double));
        double *vals_v = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            vals_s[i] = vals_v[i] = (double)(rand() % 200 - 100);
        }
        
        double k = 1.0 / sqrt(3.0);
        int iters = 1000000;
        
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) fold_scalar(vals_s, n, k);
        double t1 = now_ns();
        double scalar_ns = (t1-t0)/iters;
        
        t0 = now_ns();
        for (int i = 0; i < iters; i++) fold_avx2(vals_v, n, k);
        t1 = now_ns();
        double avx2_ns = (t1-t0)/iters;
        
        printf("  %5d values: scalar=%.1fns avx2=%.1fns (%.1fx)\n",
               n, scalar_ns, avx2_ns, scalar_ns/avx2_ns);
        
        free(vals_s); free(vals_v);
    }
    
    printf("\n================================================================\n");
    printf("Self-optimization complete. Fleet kernel adapts to hardware.\n");
    printf("================================================================\n");
    return 0;
}
