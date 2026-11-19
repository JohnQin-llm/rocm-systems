# Machine-parameter microbenchmarks

`configs/gfx950_mi355x_kmd_timing.json` describes the shape of the machine. The timing plane also needs measured parameters: latencies, rates, port counts and launch costs.

Without them, every latency resolves to 10 000 cycles and every rate to one byte or one line per cycle. A run on those defaults is about 1000x slow and only usable after heavy rescaling. With the MI355X overlay in `configs/tuning/gfx950_mi355x_measured.json`, uncalibrated runs land at a median of 15-20% from hardware.

`mb.hip` measures those parameters, one section per argument:

```bash
hipcc -O3 --offload-arch=gfx950 -std=c++17 mb.hip -o mb
./mb chain    # dependent VALU / transcendental / SALU chains
./mb mfma     # matrix latency, per-wave and device throughput
./mb lds      # LDS chase, bank conflicts by width and stride, LDS throughput
./mb chase    # vector and scalar pointer chases from 8 KB to 512 MB
./mb mlp      # one wavefront's memory-level parallelism
./mb bw       # device read and write bandwidth by footprint
./mb l1       # per-CU L1 read bandwidth
./mb issue    # VALU and transcendental issue rate per CU
./mb dispatch # empty-kernel launch cost and workgroup launch rate
```

Times are `s_memtime` shader cycles; the shader clock is measured against the 100 MHz constant clock. Run each section on an otherwise idle GPU.

Merge the overlay into a config with:

```bash
python3 ../merge-tuning.py --config ../../../configs/gfx950_mi355x_kmd_timing.json \
    --tuning ../../../configs/tuning/gfx950_mi355x_measured.json --out mi355x_measured.json
```

## Where each MI355X value comes from (one MI355X, 2026-11-18, shader clock 2390 MHz)

| parameter | value | measurement |
|---|---|---|
| `vector_alu.result_cycles` | 9 | dependent `v_add_u32` 9.0, `v_fma_f32` 9-13 |
| `transcendental.result_cycles` / `.issue_cycles` | 13 / 3 | dependent `v_exp_f32` 13.0; throughput 0.37 per CU-cycle at 4 waves per SIMD |
| `scalar_alu.result_cycles` | 9 | dependent `s_add_u32` 9.5 |
| `matrix_multiply.result_cycles` | 44 | dependent 32x32x16 bf16 44.2, 16x16x32 bf16 45.8 |
| `matrix_multiply.macs_per_cycle.bf16` / `.f16` | 512 | independent 32x32x16 bf16: 34 cycles each; 16x16x32: 20 |
| `matrix_multiply.macs_per_cycle.f32` | 32 | dependent 32x32x2 f32: 64 cycles |
| `matrix_multiply.macs_per_cycle.narrow` / `.integer` | 1024 | twice bf16, as the ISA rates |
| `*.ports` | 4 for VALU, transcendental, matrix; 1 otherwise | four SIMDs per CU |
| `*.issue_cycles`, `front_end.issue_cycles` | 1 | VALU 0.93 wave-instructions per CU-cycle at 4 waves per SIMD |
| `lds.latency_cycles` | 51 | LDS chase step 61.6, less one VALU |
| `lds.granule_bytes` / `.max_ways` / `.cycles_per_way` | 8 / 32 / 1 | `lds_fit.py mb_lds.out`: 65.4 cycles + 2.0 per way, rmse 0.8 cycles over b32/b64/b128 and stride 4-256 B (a way is charged on the port and on the return) |
| `l1_vector.hit_cycles` / `.lines_per_cycle` | 120 / 0.5 | vector chase 8 KB 119; 63 B per CU-cycle |
| `l1_scalar.hit_cycles` | 80 | scalar chase 8 KB 80 |
| `l2.hit_cycles` / `.lines_per_cycle` | 210 / 10 | chase 256 KB-2 MB 200-222; 10 455 B per cycle over 8 XCDs |
| `mall.hit_cycles` | 535 | chase 8-128 MB 533-538 |
| `dram.latency_cycles` | 896 | chase 512 MB |
| `fabric.bytes_per_cycle`, `mall.bytes_per_cycle` | 2800 | MALL-resident reads 2706-2792 B per cycle |
| `dram.bytes_per_cycle` | 2550 | 4 GB read 2547 B per cycle |
| `vector_memory.lane_addresses_per_cycle` | 16 | one wave64 address per 4 cycles |
| `miss_status_registers_per_cu` | 4096 | no saturation up to 32 loads x 64 lines from one wavefront |
| `dispatch.*_cycles` | 1800 + 600 + 1270 | empty kernel back to back: 3669 cycles |
| `dispatch.workgroups_per_cycle` | 1.9 | 1M empty workgroups |
