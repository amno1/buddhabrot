# Buddhabrot AVX-512 Benchmarks & Tests

This directory contains empirical benchmarks evaluating thread scaling and register saturation on modern AVX-512 x86-64 hardware (specifically tested on the AMD Ryzen AI 9 365, 10 physical cores: 4 Zen 5 + 6 Zen 5c, 20 SMT threads).

---

## Benchmarks

### 1. Thread Scaling (`bench-threads.lisp`)

Evaluates scaling performance across thread counts (**4, 8, 10, 12, 16, 20**) on the full 103.55 MP canvas (12,944 × 8,000, 80,000 escaped orbits), measuring both SIMD orbit generation and OpenEXR RLE export time.

**8 Threads**: Leaves 2 physical cores idle (+44% execution time).
**10 Threads** (1 per physical core): Leaves the dual 512-bit FPU pipelines of Zen 5 performance cores under-utilized (+14% execution time).
**16 Threads (Default)**: Sweet spot maximizing performance core SMT saturation while avoiding Zen 5c compact core cache thrashing.
**20 Threads**: Maximizes EXR export throughput, but causes slight memory contention during orbit generation (+3% execution time).

### 2. Register Saturation & ILP (`bench-registers.lisp`)

Compares active ZMM vector register usage across three unrolling configurations:

**1 Vector  (16 lanes)**: ~10 active ZMM registers (current baseline).
**2 Vectors (32 lanes)**: ~19 active ZMM registers.
**3 Vectors (48 lanes)**: ~26 active ZMM registers (near architectural limit of 32 ZMMs).

Includes three tests:

1. **Raw Mandelbrot Math (50M steps)**: Measures pure ALU pipeline ILP without memory or branching overhead. Saturating 19 to 26 registers delivers **1.84x to 2.24x raw throughput** by interleaving independent FMAs and hiding Zen 5's 3–4 cycle FMA latency.

2. **Real Buddhabrot Worker (Single-Thread, 25k orbits)**: Measures the impact of asynchronous lane escaping and 1.24 GB canvas writes.

3. **Multi-Threaded Buddhabrot Worker (16 Threads, 80k orbits)**: Evaluates full concurrent scaling.

---

## How to Run

### Use Make

From the project root:

```bash
# Run all benchmarks
make test

# Run thread scaling benchmark
make test-threads

# Run register saturation & ILP benchmark
make test-registers
```

### Use the Test Runner Script

```bash
# Run all benchmarks
./tests/run-tests.sh all

# Run thread scaling only
./tests/run-tests.sh threads

# Run register saturation only
./tests/run-tests.sh registers
```

### Direct SBCL Shell Command

```bash
# Run thread benchmark directly
/path/to/your/sb-simd-512/run-sbcl.sh --dynamic-space-size 8000 --script tests/bench-threads.lisp

# Run register benchmark directly
/path/to/your/sb-simd-512/run-sbcl.sh --dynamic-space-size 8000 --script tests/bench-registers.lisp
```

# Results on My Machine

### Testing Fewer Threads (8 ~ 10 threads vs 16 vs 20)

Full end-to-end rendering on the 103.55 MP canvas (12,944 × 8,000, 80,000 target escaped orbits, HALF EXR with SIMD RLE compression):

| Threads | Configuration                     | SIMD Compute | Orbit Rate | EXR Export | Total Time | vs 16 Threads     |
|:--------|:----------------------------------|:-------------|:-----------|:-----------|:-----------|:------------------|
| 8       | Under-subscribed                  | 0.206 s      | 0.8 M/s    | 0.215 s    | 0.421 s    | +44% slower       |
| 10      | 1 thread / physical core (no SMT) | 0.160 s      | 1.0 M/s    | 0.175 s    | 0.335 s    | +14% slower       |
| 12      | Partial SMT                       | 0.167 s      | 1.0 M/s    | 0.154 s    | 0.321 s    | +10% slower       |
| 16      | Current Default (Sweet Spot)      | 0.146 s      | 1.1 M/s    | 0.147 s    | 0.293 s    | Baseline(Fastest) |
| 20      | 100% of SMT threads               | 0.162 s      | 1.0 M/s    | 0.139 s    | 0.301 s    | +3% slower        |

#### Why does this happen?

**8 Threads**: The CPU has 10 physical cores. At 8 threads, 2 physical cores sit completely idle (20% of the hardware is unutilized).

**10 Threads (1 per physical core)**: Runs cleanly without SMT resource contention, but under-utilizes the 4 Zen 5 performance cores, which feature dual 512-bit FPU pipelines and massive reorder buffers capable of executing two threads simultaneously with almost zero penalty.

**16 Threads**: The sweet spot for the hybrid Zen 5 + Zen 5c topology:
   - The 4 Zen 5 performance cores take 2 threads each (8 threads), saturating their dual 512-bit execution pipes and overlapping memory latency during orbit retrace.
   - The 6 compact Zen 5c cores take the remaining 8 threads without thrashing their narrower L2/L3 caches.

**20 Threads**: While the purely sequential EXR RLE compression scales slightly better to 20 threads (0.139 s vs 0.147 s), the orbit compute phase slows down (0.162 s vs 0.146 s) due to SMT contention on the Zen 5c dense cores and atomic/write bandwidth into the 1.24 GB canvas.

---

### Saturating More Registers (20 ~ 24 registers vs ~10)

AVX-512 provides 32 architectural vector registers (`zmm0`–`zmm31`). The current 16-lane loop keeps **~9–10 ZMM registers** active (`zr, zi, cr, ci, four, zr2, zi2, zri, mag2`). We implemented and tested 2x unrolling (32 lanes, **~19 active registers**) and 3x unrolling (48 lanes, **~26 active registers**).

#### Benchmark: Raw Compute / Pipeline ILP Saturation (50,000,000 steps)
*Measuring pure ALU instruction-level parallelism without branching or memory overhead:*

| Unrolling     | Active ZMM Regs | Execution Time | Throughput           | Speedup |
|:--------------|:----------------|:---------------|:---------------------|:--------|
| 1x (16 lanes) | ~10 registers   | 0.106 s        | 7.55 G lane-iters/s  | 1.00x   |
| 2x (32 lanes) | ~19 registers   | 0.115 s        | 13.91 G lane-iters/s | 1.84x   |
| 3x (48 lanes) | ~26 registers   | 0.142 s        | 16.90 G lane-iters/s | 2.24x   |
|               |                 |                |                      |         |

**Why 2x and 3x excel in raw compute:**

Zen 5 has two 512-bit FMA units with a 3–4 cycle latency. In a single vector loop, consecutive iterations have a strict Read-After-Write (RAW) data dependency on z_{n+1} = z_n^2 + c, causing the FMA pipeline to stall for ~3 cycles between steps. Unrolling to 2x (19 registers) and 3x (26 registers) interleaves independent vector streams, completely hiding pipeline latency and delivering a 2.24x raw throughput boost.

---

#### Benchmark: Real Buddhabrot Workload (Continuous Lane Refill + Orbit Retrace)

*Measuring actual escaped orbit generation into the 103.55 MP canvas:*

| Configuration | Active ZMM Regs | Single-Thread Rate (25k orb) | 16-Thread Rate (80k orb) | Multi-Thread Time |
|:--------------|:----------------|:-----------------------------|:-------------------------|:------------------|
| 1x (16 lanes) | ~10 registers   | 133.7 k orb/s (0.187 s)      | 1.8 M orb/s              | 0.091 s           |
| 2x (32 lanes) | ~19 registers   | 132.3 k orb/s (0.189 s)      | 1.7 M orb/s              | 0.092 s           |
| 3x (48 lanes) | ~26 registers   | 123.8 k orb/s (0.202 s)      | 1.6 M orb/s              | 0.098 s           |

**Why saturating more registers does not speed up the real Buddhabrot worker:**

1. **Branch Divergence Frequency**:
   In Buddhabrot, orbits escape at varying step counts. The zero-overhead register fast path only runs when *no* lane escapes.
   - With 16 lanes, P(\text{no escape}) = (1 - p)^{16}.
   - With 48 lanes, P(\text{no escape}) = (1 - p)^{48}.
   Wider vectors trigger the slow scalar refill branch roughly 3x more frequently, interrupting SIMD execution.

2. **Scalar Metadata Overhead**:
   Maintaining candidate coordinates and cycle detection snapshots (`cr, ci, start-steps, snap-zr, snap-zi, snap-valid`) for 48 lanes triples the scalar state per thread, increasing L1 cache pressure and spill costs during the refill path.

3. **Memory Retrace Amdahl's Law**:
   Whenever an orbit escapes, the worker must retrace it and scatter increments into the 1.24 GB canvas. Memory latency to DDR5 RAM dominates this phase, making raw FMA throughput secondary.

---

### Conclusion

**Threads**: **16 threads** are optimal on this CPU. 8 and 10 threads leave CPU cores/pipelines under-utilized (+14% to +44% slower), while 20 threads suffers slight SMT contention on the compact Zen 5c cores (+3% slower).

**Registers**: **1x vector (16 lanes, ~10 registers)** are optimal for this Buddhabrot generator. Saturating 19–26 registers yields a 2.24x boost in pure mathematical loops (great for Mandelbrot set rendering), but the Buddhabrot's asynchronous lane escaping and 1.24 GB memory retrace negate the benefit.
