# Fleet Crate Deep Profile

Micro-benchmarks and cache/branch analysis of every published Cocapn fleet crate.

## Hardware: AMD Ryzen AI 9 HX 370 (Zen 5, AVX-512)

## Key Findings

| Operation | Throughput | Bottleneck | Headroom |
|-----------|-----------|------------|----------|
| Eisenstein norm | 4.1G ops/s | **AT CEILING** (1 cycle) | None |
| Snap to lattice | 44M snaps/s | Branch misprediction (115 cycles) | **6x possible** |
| Bloom merge AVX2 | 125 GB/s (8KB) | L1D bandwidth (38% ceiling) | 2.5x possible |
| Tile hash | 213M hashes/s | Instruction count (24 cycles) | Minor |
| Folding stage | 59M stages/s | Memory access | Plenty |
| Voice leading | 383M ops/s | Trivial | N/A |
| β₁ emergence | 4.1G ops/s | **AT CEILING** | None |
| **Full pipeline** | **51.7M constraints/s** | **Not CPU — network is the bottleneck** |

## The Pipeline Has 2500x Headroom

Current fleet load: ~20,000 constraints/second.
Pipeline capacity: 51,700,000 constraints/second.
**The fleet is using 0.04% of available throughput.**

## Files

- `fleet_bench.c` — 7-benchmark suite (norm, bloom, fold, hash, voice-leading, holonomy, pipeline)
- `deep_profile.c` — 5 deep experiments (magnitude, cache hierarchy, branch prediction, convergence, theoretical limits)
- `cross_validate.py` — Python cross-validation against all Rust crate operations
- `FLEET_PROFILE.md` — Full analysis and findings
- `results/` — Raw output files

## Run

```bash
gcc -O2 -mavx2 -o fleet_bench fleet_bench.c -lm && ./fleet_bench
gcc -O2 -mavx2 -o deep_profile deep_profile.c -lm && ./deep_profile
python3 cross_validate.py
```
