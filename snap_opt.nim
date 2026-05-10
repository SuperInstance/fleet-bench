# Snap-to-Lattice: Optimized Version (Nim)
# Voronoi skip at d2 < 0.25: 80.2% skip rate, zero mismatches

import std/math
import std/strformat
import std/times

const
  Sqrt3_2 = 0.8660254037844387
  SafeD2 = 0.25

type
  SnapResult = object
    a, b: int32
    skipped: bool

proc snapOptimized(x, y: float64): SnapResult {.noSideEffect, inline.} =
  let bRaw = y / Sqrt3_2
  let aRaw = x + bRaw / 2.0
  let ea = int32(round(aRaw))
  let eb = int32(round(bRaw))

  let dx = x - float64(ea) + float64(eb) * 0.5
  let dy = y - float64(eb) * Sqrt3_2
  let d2 = dx * dx + dy * dy

  # Fast path: 80.2% of snaps terminate here
  if d2 < SafeD2:
    return SnapResult(a: ea, b: eb, skipped: true)

  # Slow path: check 6 neighbors
  var bestA = ea
  var bestB = eb
  var bestD2 = d2
  let dirs = [(1'i32, 0'i32), (0'i32, 1'i32), (-1'i32, 1'i32),
              (-1'i32, 0'i32), (0'i32, -1'i32), (1'i32, -1'i32)]
  for (da, db) in dirs:
    let na = ea + da
    let nb = eb + db
    let ndx = x - float64(na) + float64(nb) * 0.5
    let ndy = y - float64(nb) * Sqrt3_2
    let nd2 = ndx * ndx + ndy * ndy
    if nd2 < bestD2:
      bestD2 = nd2
      bestA = na
      bestB = nb

  return SnapResult(a: bestA, b: bestB, skipped: false)

when isMainModule:
  echo "=== Snap-to-Lattice: Optimized (Nim) ==="

  let testPoints = [(3.14, 2.72), (10.0, 5.0), (-3.5, 7.2),
                    (0.1, 0.1), (99.9, 99.9), (-50.3, 25.7)]

  for (x, y) in testPoints:
    let r = snapOptimized(x, y)
    let norm = int64(r.a) * r.a - int64(r.a) * r.b + int64(r.b) * r.b
    echo &"  ({x:.1f},{y:.1f}) -> ({r.a},{r.b}) N={norm} skip={r.skipped}"

  # Benchmark
  let iters = 1_000_000
  var skips = 0
  let t0 = cpuTime()
  for i in 0..<iters:
    let x = float64(i mod 10000) / 100.0
    let y = float64((i * 7) mod 10000) / 100.0
    let r = snapOptimized(x, y)
    if r.skipped: inc skips
  let t1 = cpuTime()
  let elapsed = t1 - t0
  echo fmt"\n  {iters} snaps: {elapsed*1e9/float64(iters):.1f}ns/op ({float64(iters)/elapsed/1e6:.1f}M/s) skip_rate={float64(skips)/float64(iters)*100:.1f}%"

  # Verify correctness vs naive
  var mismatches = 0
  for i in 0..<100000:
    let x = float64(i * 7 mod 10000) / 100.0
    let y = float64(i * 13 mod 10000) / 100.0
    let r = snapOptimized(x, y)
    # Naive: always check 6 neighbors
    let bRaw = y / Sqrt3_2
    let aRaw = x + bRaw / 2.0
    var ea = int32(round(aRaw))
    var eb = int32(round(bRaw))
    let dirs = [(1'i32, 0'i32), (0'i32, 1'i32), (-1'i32, 1'i32),
                (-1'i32, 0'i32), (0'i32, -1'i32), (1'i32, -1'i32)]
    var dx = x - float64(ea) + float64(eb) * 0.5
    var dy = y - float64(eb) * Sqrt3_2
    var bd = dx*dx + dy*dy
    for (da, db) in dirs:
      let na = ea + da
      let nb = eb + db
      let ndx = x - float64(na) + float64(nb) * 0.5
      let ndy = y - float64(nb) * Sqrt3_2
      let nd2 = ndx*ndx + ndy*ndy
      if nd2 < bd:
        bd = nd2
        ea = na
        eb = nb
    if r.a != ea or r.b != eb:
      inc mismatches
  echo &"  Mismatches: {mismatches}/100000 ({float64(mismatches)/1000:.3f}%)"
