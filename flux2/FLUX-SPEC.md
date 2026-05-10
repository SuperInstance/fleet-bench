# FLUX 2.0 — Self-Optimizing Constraint ISA

**Forgemaster ⚒️ — 2026-05-10**

Built on profiling data from Ryzen AI 9 HX 370 (Zen 5).

## FLUX 1.0: What We Learned

9 opcodes. Works. But dumb — same code regardless of hardware, no self-awareness.

Profiling revealed:
- Norm: AT CEILING (4.1G ops/s). Don't touch.
- Snap: 115 cycles. 80.2% skip with d2<0.25 threshold. **Self-optimizing.**
- Bloom: 125 GB/s at L1, 35 GB/s at DRAM. **Size-aware.**
- Fold: 3.9x faster with AVX2 at n=1024. **SIMD-aware.**
- Pipeline: 51.7M constraints/s, fleet uses 0.04%. **Massive headroom.**

## FLUX 2.0 Opcodes (17 total)

### Original 9 (unchanged, proven)
```
0x01 LOAD   r imm32       — load immediate
0x02 CHECK  lo hi val     — bounds check, push bool
0x03 AND                  — pop 2 bools, push AND
0x04 OR                   — pop 2 bools, push OR
0x05 NORM   a b dst       — Eisenstein norm: a²-ab+b² → dst
0x06 EMIT   r             — output register
0x07 HALT                 — stop
0x08 BLOOM_OR dst src     — bitwise OR (CRDT merge)
0x09 NOT                  — pop bool, push negation
```

### New: Hardware Awareness (3)
```
0x0A PROBE  dst capability  — query hardware capability into dst
         capabilities: 0=L1D_KB, 1=SIMD_WIDTH, 2=NORM_MOPS,
                        3=BLOOM_GBS, 4=HAS_AVX2, 5=CORES
0x0B CLOCK  dst             — read cycle counter into dst
0x0C REPORT                  — emit profiling stats (cycles per opcode)
```

### New: Adaptive Execution (3)
```
0x0D SNAP   x y ea eb     — snap to Eisenstein lattice (self-optimizing)
         Phase 1: round + d2 check (80.2% terminate here)
         Phase 2: if d2 >= 0.25, check 6 neighbors
0x0E BCHECK lo hi val cnt  — batch constraint check (SIMD when available)
         cnt = number of elements (8, 16, 32...)
         Uses AVX2 automatically when PROBE says available
0x0F FOLD   vals cnt k     — folding order stage (SIMD when available)
         Applies Banach contraction to cnt values with constant k
```

### New: Adaptive Bloom (2)
```
0x10 BBLOOM dst src cnt    — batch Bloom merge (SIMD + unrolled when large)
         Auto-selects: 4-wide (small), 8-wide unrolled (large)
0x11 SELECT path_id        — choose execution path based on previous PROBE
         Paths defined at compile time, selected at runtime
```

## Binary Encoding

```
| Opcode (1 byte) | Operands (variable) |
|-----------------|---------------------|
| 0x01 LOAD       | reg(1) imm(4)       |  5 bytes
| 0x02 CHECK      | lo(1) hi(1) val(1)  |  3 bytes
| 0x03 AND        | —                   |  1 byte
| 0x04 OR         | —                   |  1 byte
| 0x05 NORM       | a(1) b(1) dst(1)   |  3 bytes
| 0x06 EMIT       | reg(1)              |  2 bytes
| 0x07 HALT       | —                   |  1 byte
| 0x08 BLOOM_OR   | dst(1) src(1)      |  3 bytes
| 0x09 NOT        | —                   |  1 byte
| 0x0A PROBE      | dst(1) cap(1)      |  3 bytes
| 0x0B CLOCK      | dst(1)              |  2 bytes
| 0x0C REPORT     | —                   |  1 byte
| 0x0D SNAP       | x(1) y(1) ea(1) eb(1)| 4 bytes
| 0x0E BCHECK     | lo(1) hi(1) val(1) cnt(1)| 4 bytes
| 0x0F FOLD       | base(1) cnt(1) k(1)|  3 bytes
| 0x10 BBLOOM     | dst(1) src(1) cnt(1)| 3 bytes
| 0x11 SELECT     | path(1)            |  2 bytes
```

## Self-Profiling Protocol

A FLUX 2.0 program can measure itself:

```flux
; Self-profiling constraint pipeline
PROBE r48 4          ; r48 = HAS_AVX2 (0 or 1)
PROBE r49 0          ; r49 = L1D_KB (e.g., 32)
CLOCK  r50           ; r50 = start cycles

; ... do work ...

CLOCK  r51           ; r51 = end cycles
EMIT   r50           ; report start
EMIT   r51           ; report end
; The difference r51 - r50 = total cycles consumed
REPORT               ; emit per-opcode profiling
```

## Adaptive Dispatch

```flux
; Choose path based on hardware
PROBE r48 4          ; HAS_AVX2?
SELECT r48           ; if r48 != 0, take AVX2 path
  ; Path 0 (scalar): CHECK all 16 individually
  CHECK r16 r32 r0   ; check element 0
  CHECK r17 r33 r1   ; check element 1
  ; ... (16 checks + 15 ANDs)
  JMP end
  ; Path 1 (AVX2): batch check
  BCHECK r16 r32 r0 16  ; all 16 in one SIMD instruction
  JMP end
end:
  EMIT r0
```

## Self-Optimization Loop

FLUX 2.0 programs can self-optimize over multiple runs:

1. **First run**: Profile everything. CLOCK before/after each section.
2. **After N runs**: EMIT the profiling data.
3. **External optimizer** (or the VM itself) rewrites hot paths:
   - Replace CHECK loops with BCHECK (batch SIMD)
   - Replace BLOOM_OR loops with BBLOOM
   - Set Bloom filter size to `PROBE L1D_KB / 2`
   - Skip SNAP neighbor check when d2 < 0.25 (80.2% of cases)

This is a JIT compiler in miniature. The FLUX VM becomes self-tuning.

## Performance Prediction

Based on profiling data:

| Operation | FLUX 1.0 | FLUX 2.0 (adaptive) | Improvement |
|-----------|----------|---------------------|-------------|
| Constraint check (16) | 16 CHECK + 15 AND | 1 BCHECK | ~4x (AVX2) |
| Bloom merge (1000 words) | 1000 BLOOM_OR | 1 BBLOOM | ~4x (AVX2) |
| Snap to lattice | 6 neighbor checks | 80% skip to round-only | ~1.5x |
| Folding stage | Scalar loop | SIMD FOLD | ~3.9x (n=1024) |
| Full pipeline | ~19 ns/constraint | ~5-8 ns/constraint | ~3x |

## What Makes FLUX 2.0 Different

1. **FLUX 1.0 is a calculator.** It computes constraints.
2. **FLUX 2.0 is a self-aware calculator.** It knows what hardware it's on, measures its own performance, and adapts.

The constraint ISA now has **reflection** — programs can inspect their own execution. This is what makes the fleet self-optimizing: every node profiles its own constraint pipeline and shares the results via CRDT gossip.

## Relationship to Fleet Crates

| FLUX Opcode | Rust Crate | C Equivalent |
|-------------|-----------|--------------|
| CHECK/BCHECK | constraint-theory-core | _mm256_cmpgt_epi32 |
| BLOOM_OR/BBLOOM | constraint-crdt | _mm256_or_si256 |
| NORM | constraint-theory-core | int64 multiply |
| SNAP | constraint-theory-core | round + Voronoi skip |
| FOLD | folding-order | AVX2 sum + contract |
| PROBE | plato-runtime | CPUID + calibration |
| CLOCK | (new) | clock_gettime / rdtsc |

Every FLUX opcode maps to a fleet crate. FLUX is the fleet's portable bytecode.
