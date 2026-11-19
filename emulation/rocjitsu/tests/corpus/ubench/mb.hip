// MI355X machine-parameter microbenchmarks for the rocjitsu timing overlay.
// Every test prints "name value unit". Cycles are s_memtime (shader clock) cycles.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x16 __attribute__((ext_vector_type(16)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));

// ---- shader clock against the 100 MHz constant clock
__global__ void k_clock(long long *out) {
  long long w0 = wall_clock64(), c0 = clock64();
  while (wall_clock64() - w0 < 1000000) {}  // 10 ms
  out[0] = clock64() - c0;
  out[1] = wall_clock64() - w0;
}

// ---- dependent chains, one wave
template <int OP>
__global__ void k_chain(long long *out, float *sink, int n) {
  float x = threadIdx.x * 1e-3f, a = 1.0001f, b = 0.5f;
  unsigned s = threadIdx.x, ss = (unsigned)n;
  long long t0 = clock64();
  for (int i = 0; i < n; i += 16) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (OP == 0) asm volatile("v_fma_f32 %0, %0, %1, %2" : "+v"(x) : "v"(a), "v"(b));
      if (OP == 1) asm volatile("v_exp_f32 %0, %0" : "+v"(x));
      if (OP == 2) asm volatile("s_add_u32 %0, %0, 1" : "+s"(ss) :: "scc");
      if (OP == 3) asm volatile("v_add_u32 %0, %0, %1" : "+v"(s) : "v"(s));
    }
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  sink[threadIdx.x] = x + s + ss;
}

// ---- matrix: dependent chain (latency) and four independent chains (throughput), one wave
template <int SHAPE, int INDEP>
__global__ void k_mfma(long long *out, float *sink, int n) {
  bf16x8 a, b;
  for (int i = 0; i < 8; ++i) { a[i] = (__bf16)(threadIdx.x * 1e-3f); b[i] = (__bf16)1.0f; }
  f32x16 c[4] = {};
  f32x4 d[4] = {};
  long long t0 = clock64();
  for (int i = 0; i < n; ++i) {
#pragma unroll
    for (int k = 0; k < INDEP; ++k) {
      if (SHAPE == 0) c[k] = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c[k], 0, 0, 0);
      if (SHAPE == 1) d[k] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, d[k], 0, 0, 0);
    }
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  float s = 0;
  for (int k = 0; k < 4; ++k) s += c[k][0] + d[k][0];
  sink[threadIdx.x] = s;
}

__global__ void k_mfma_f32(long long *out, float *sink, int n) {
  float a = threadIdx.x * 1e-3f, b = 1.0f;
  f32x16 c = {};
  long long t0 = clock64();
  for (int i = 0; i < n; ++i) c = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, c, 0, 0, 0);
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  sink[threadIdx.x] = c[0];
}

// ---- LDS pointer chase (one VALU shift per step is part of the loop)
__global__ void k_lds_chase(long long *out, unsigned *sink, int n) {
  __shared__ unsigned idx[4096];
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) idx[i] = (i * 1031 + 7) % 4096;
  __syncthreads();
  unsigned j = threadIdx.x;
  long long t0 = clock64();
  for (int i = 0; i < n; ++i) j = idx[j];
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  sink[threadIdx.x] = j;
}

// ---- LDS throughput and bank conflicts: each lane reads WIDTH bytes at lane*stride, n times, one wave
template <int WIDTH>
__global__ void k_lds_stride(long long *out, float *sink, int stride_bytes, int n) {
  __shared__ __attribute__((aligned(16))) char buf[65536];
  for (int i = threadIdx.x; i < 65536 / 4; i += blockDim.x) ((float *)buf)[i] = i;
  __syncthreads();
  unsigned off = (threadIdx.x * stride_bytes) % 65536;
  off &= ~(WIDTH - 1);
  float acc = 0;
  long long t0 = clock64();
  for (int i = 0; i < n; ++i) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      if (WIDTH == 4) { float v; asm volatile("ds_read_b32 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(off)); acc += v; }
      if (WIDTH == 8) { float v[2]; asm volatile("ds_read_b64 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(*(double *)v) : "v"(off)); acc += v[0]; }
      if (WIDTH == 16) { f32x4 v; asm volatile("ds_read_b128 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(off)); acc += v[0]; }
    }
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  sink[threadIdx.x] = acc;
}

// Same, unwaited: 8 independent reads per wait, many waves per CU -> LDS throughput
template <int WIDTH>
__global__ void k_lds_tput(long long *out, float *sink, int n) {
  __shared__ __attribute__((aligned(16))) char buf[65536];
  for (int i = threadIdx.x; i < 65536 / 4; i += blockDim.x) ((float *)buf)[i] = i;
  __syncthreads();
  unsigned off = ((threadIdx.x & 63) * WIDTH + (threadIdx.x >> 6) * 4096) % 65536;
  f32x4 acc = {};
  long long t0 = clock64();
  for (int i = 0; i < n; ++i) {
    f32x4 v0, v1, v2, v3;
    asm volatile("ds_read_b128 %0, %4\n ds_read_b128 %1, %4 offset:1024\n ds_read_b128 %2, %4 offset:2048\n ds_read_b128 %3, %4 offset:3072\n s_waitcnt lgkmcnt(0)"
                 : "=v"(v0), "=v"(v1), "=v"(v2), "=v"(v3) : "v"(off));
    acc += v0 + v1 + v2 + v3;
  }
  long long t1 = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = t1 - t0;
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc[0];
}

// ---- vector pointer chase, two identical passes, second timed
__global__ void k_vchase(long long *out, const unsigned *next, unsigned n, unsigned *sink) {
  unsigned j = 0;
  long long t = 0;
  for (int pass = 0; pass < 2; ++pass) {
    asm volatile("" : "+v"(j));
    long long t0 = clock64();
    for (unsigned i = 0; i < n; ++i) j = next[j];
    t = clock64() - t0;
  }
  out[0] = t;
  sink[0] = j;
}

// ---- scalar pointer chase (s_load through the scalar cache)
__global__ void k_schase(long long *out, const unsigned *next, unsigned n, unsigned *sink) {
  unsigned j = 0;
  long long t = 0;
  for (int pass = 0; pass < 2; ++pass) {
    j = __builtin_amdgcn_readfirstlane(j * 0u);
    long long t0 = clock64();
    for (unsigned i = 0; i < n; ++i) {
      const unsigned *p = next + j;
      asm volatile("s_load_dword %0, %1, 0x0\n s_waitcnt lgkmcnt(0)" : "=s"(j) : "s"(p));
    }
    t = clock64() - t0;
  }
  out[0] = t;
  sink[0] = j;
}

// ---- memory-level parallelism of one wave: K loads in flight, then wait
template <int K>
__global__ void k_mlp(long long *out, const float *buf, unsigned lines_mask, int n, int one_lane, float *sink) {
  float acc = 0;
  unsigned lane = threadIdx.x;
  long long t0 = clock64();
  for (int i = 0; i < n; ++i) {
    float v[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
      unsigned line = ((unsigned)(i * K + k) * 64u + (one_lane ? 0u : lane)) * 2654435761u & lines_mask;
      v[k] = (one_lane && lane != 0) ? 0.f : __builtin_nontemporal_load(&buf[line * 32]);
    }
#pragma unroll
    for (int k = 0; k < K; ++k) acc += v[k];
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  sink[threadIdx.x] = acc;
}

// ---- device-wide read / write bandwidth over a power-of-two footprint
__global__ void k_read(const f32x4 *p, size_t mask, size_t total, float *sink) {
  size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
  f32x4 acc = {};
  for (size_t i = gid; i < total; i += stride) acc += p[i & mask];
  if (acc[0] == 12345.f) sink[0] = acc[1];
}
__global__ void k_write(f32x4 *p, size_t mask, size_t total) {
  size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
  f32x4 v = {1.f, 2.f, 3.f, 4.f};
  for (size_t i = gid; i < total; i += stride) p[i & mask] = v;
}
// each workgroup re-reads its own small window: L1 bandwidth per CU
__global__ void k_read_l1(const f32x4 *p, unsigned window_vecs, int reps, float *sink) {
  const f32x4 *w = p + (size_t)blockIdx.x * window_vecs;
  f32x4 a0 = {}, a1 = {}, a2 = {}, a3 = {};
  unsigned m = window_vecs - 1;
  for (int r = 0; r < reps; ++r) {
    unsigned b = (threadIdx.x + r * 4 * blockDim.x);
    a0 += w[b & m];
    a1 += w[(b + blockDim.x) & m];
    a2 += w[(b + 2 * blockDim.x) & m];
    a3 += w[(b + 3 * blockDim.x) & m];
  }
  f32x4 acc = a0 + a1 + a2 + a3;
  if (acc[0] == 12345.f) sink[0] = acc[1];
}

// ---- VALU throughput: 8 independent FMA chains per wave
__global__ void k_valu_tput(float *sink, int n) {
  float x[8];
  for (int k = 0; k < 8; ++k) x[k] = threadIdx.x * 1e-3f + k;
  float a = 1.0001f, b = 0.5f;
  for (int i = 0; i < n; ++i) {
#pragma unroll
    for (int k = 0; k < 8; ++k) asm volatile("v_fma_f32 %0, %0, %1, %2" : "+v"(x[k]) : "v"(a), "v"(b));
  }
  float s = 0;
  for (int k = 0; k < 8; ++k) s += x[k];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
__global__ void k_trans_tput(float *sink, int n) {
  float x[8];
  for (int k = 0; k < 8; ++k) x[k] = threadIdx.x * 1e-6f + k * 1e-3f;
  for (int i = 0; i < n; ++i) {
#pragma unroll
    for (int k = 0; k < 8; ++k) asm volatile("v_exp_f32 %0, %0" : "+v"(x[k]));
  }
  float s = 0;
  for (int k = 0; k < 8; ++k) s += x[k];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
__global__ void k_mfma_tput(float *sink, int n) {
  bf16x8 a, b;
  for (int i = 0; i < 8; ++i) { a[i] = (__bf16)(threadIdx.x * 1e-3f); b[i] = (__bf16)1.0f; }
  f32x16 c0 = {}, c1 = {};
  for (int i = 0; i < n; ++i) {
    c0 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c0, 0, 0, 0);
    c1 = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c1, 0, 0, 0);
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = c0[0] + c1[0];
}
__global__ void k_empty() {}

static long long *g_out;
static float *g_sink;
static long long host_out() { long long h; CHECK(hipMemcpy(&h, g_out, 8, hipMemcpyDeviceToHost)); return h; }

template <class F> static float time_ms(F launch, int reps = 5) {
  hipEvent_t a, b;
  hipEventCreate(&a); hipEventCreate(&b);
  launch(); CHECK(hipDeviceSynchronize());
  std::vector<float> t;
  for (int r = 0; r < reps; ++r) {
    hipEventRecord(a); launch(); hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b); t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

static unsigned *make_chase(size_t bytes) {
  size_t n = bytes / 128;
  std::vector<unsigned> perm(n), next(bytes / 4, 0);
  for (size_t i = 0; i < n; ++i) perm[i] = i;
  std::shuffle(perm.begin() + 1, perm.end(), std::mt19937(1));
  for (size_t i = 0; i < n; ++i) next[perm[i] * 32] = perm[(i + 1) % n] * 32;
  unsigned *d; CHECK(hipMalloc(&d, bytes));
  CHECK(hipMemcpy(d, next.data(), bytes, hipMemcpyHostToDevice));
  return d;
}

#define RUN2(launch) { launch; CHECK(hipDeviceSynchronize()); launch; CHECK(hipDeviceSynchronize()); }

int main(int argc, char **argv) {
  std::string only = argc > 1 ? argv[1] : "all";
  auto want = [&](const char *t) { return only == "all" || only == t; };
  CHECK(hipMalloc(&g_out, 64));
  CHECK(hipMalloc(&g_sink, 64 << 20));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int cus = prop.multiProcessorCount;
  printf("cus %d count\n", cus);
  double mhz = 2390.0;
  if (!getenv("MB_NO_CLOCK")) {  // the emulator does not advance the constant clock: skip the spin there
    k_clock<<<1, 64>>>(g_out);
    CHECK(hipDeviceSynchronize());
    long long cw[2];
    CHECK(hipMemcpy(cw, g_out, 16, hipMemcpyDeviceToHost));
    mhz = cw[0] / (cw[1] / 100.0);
  }
  printf("shader_clock_mhz %.1f MHz\n", mhz);

  if (want("chain")) {
    const int n = 1 << 14;
    RUN2((k_chain<0><<<1, 64>>>(g_out, g_sink, n))) printf("valu_fma_latency %.2f cycles\n", host_out() / double(n));
    RUN2((k_chain<1><<<1, 64>>>(g_out, g_sink, n))) printf("transcendental_exp_latency %.2f cycles\n", host_out() / double(n));
    RUN2((k_chain<2><<<1, 64>>>(g_out, g_sink, n))) printf("salu_add_latency %.2f cycles\n", host_out() / double(n));
    RUN2((k_chain<3><<<1, 64>>>(g_out, g_sink, n))) printf("valu_add_u32_latency %.2f cycles\n", host_out() / double(n));
  }
  if (want("mfma")) {
    const int n = 4096;
    RUN2((k_mfma<0, 1><<<1, 64>>>(g_out, g_sink, n))) printf("mfma_32x32x16_bf16_dependent %.2f cycles\n", host_out() / double(n));
    RUN2((k_mfma<0, 4><<<1, 64>>>(g_out, g_sink, n))) printf("mfma_32x32x16_bf16_independent %.2f cycles\n", host_out() / double(4 * n));
    RUN2((k_mfma<1, 1><<<1, 64>>>(g_out, g_sink, n))) printf("mfma_16x16x32_bf16_dependent %.2f cycles\n", host_out() / double(n));
    RUN2((k_mfma<1, 4><<<1, 64>>>(g_out, g_sink, n))) printf("mfma_16x16x32_bf16_independent %.2f cycles\n", host_out() / double(4 * n));
    RUN2((k_mfma_f32<<<1, 64>>>(g_out, g_sink, n))) printf("mfma_32x32x2_f32_dependent %.2f cycles\n", host_out() / double(n));
    for (int wps : {1, 2}) {
      int nn = 2048;
      float ms = time_ms([&] { k_mfma_tput<<<cus, 64 * 4 * wps>>>(g_sink, nn); });
      double macs = double(cus) * 4 * wps * nn * 2 * 32 * 32 * 16;
      printf("mfma_bf16_macs_per_cycle_per_cu_wps%d %.1f macs\n", wps, macs / (ms * 1e-3 * mhz * 1e6) / cus);
    }
  }
  if (want("lds")) {
    const int n = 4096;
    RUN2((k_lds_chase<<<1, 64>>>(g_out, (unsigned *)g_sink, n))) printf("lds_chase_step %.2f cycles\n", host_out() / double(n));
    for (int stride : {4, 8, 16, 32, 64, 128, 256}) {
      RUN2((k_lds_stride<4><<<1, 64>>>(g_out, g_sink, stride, 512))) printf("lds_b32_waited_stride%d %.2f cycles\n", stride, host_out() / double(512 * 8));
    }
    for (int stride : {8, 16, 32, 64, 128, 256}) {
      RUN2((k_lds_stride<8><<<1, 64>>>(g_out, g_sink, stride, 512))) printf("lds_b64_waited_stride%d %.2f cycles\n", stride, host_out() / double(512 * 8));
    }
    for (int stride : {16, 32, 64, 128, 256}) {
      RUN2((k_lds_stride<16><<<1, 64>>>(g_out, g_sink, stride, 512))) printf("lds_b128_waited_stride%d %.2f cycles\n", stride, host_out() / double(512 * 8));
    }
    for (int waves : {1, 4, 8, 16}) {
      int nn = 2048;
      float ms = time_ms([&] { k_lds_tput<16><<<cus, 64 * waves>>>(g_out, g_sink, nn); });
      double bytes = double(cus) * waves * 64 * 16 * 4 * nn;
      printf("lds_b128_bytes_per_cycle_per_cu_waves%d %.1f bytes\n", waves, bytes / (ms * 1e-3 * mhz * 1e6) / cus);
    }
  }
  if (want("chase")) {
    for (size_t kb : {8, 64, 256, 2048, 8192, 32768, 131072, 524288}) {
      unsigned *d = make_chase(kb << 10);
      unsigned n = unsigned((kb << 10) / 128);
      k_vchase<<<1, 1>>>(g_out, d, n, (unsigned *)g_sink);
      CHECK(hipDeviceSynchronize());
      printf("vector_chase_%zuKB %.1f cycles\n", kb, host_out() / double(n));
      if (kb <= 2048) {
        k_schase<<<1, 64>>>(g_out, d, n, (unsigned *)g_sink);
        CHECK(hipDeviceSynchronize());
        printf("scalar_chase_%zuKB %.1f cycles\n", kb, host_out() / double(n));
      }
      CHECK(hipFree(d));
    }
  }
  if (want("mlp")) {
    size_t bytes = size_t(1) << 20;
    float *buf;
    CHECK(hipMalloc(&buf, bytes));
    CHECK(hipMemset(buf, 0, bytes));
    unsigned mask = unsigned(bytes / 128 - 1);
    for (int one : {1, 0}) {
#define MLP(K) { RUN2((k_mlp<K><<<1, 64>>>(g_out, buf, mask, 256, one, g_sink))) printf("mlp_%s_K%d %.1f cycles_per_batch\n", one ? "1lane" : "64lanes", K, host_out() / 256.0); }
      MLP(1) MLP(2) MLP(4) MLP(8) MLP(16) MLP(32)
    }
    CHECK(hipFree(buf));
  }
  if (want("bw") || want("l1")) {
    size_t big = size_t(4) << 30;
    f32x4 *p;
    CHECK(hipMalloc(&p, big));
    CHECK(hipMemset(p, 0, big));
    int grid = cus * 8, block = 256;
    if (only != "l1") for (size_t fp : {size_t(2) << 20, size_t(16) << 20, size_t(64) << 20, size_t(128) << 20, size_t(1) << 30, big}) {
      size_t mask = fp / 16 - 1, total = std::max<size_t>(fp / 16, (size_t(256) << 20) >> 4) * 4;
      float ms = time_ms([&] { k_read<<<grid, block>>>(p, mask, total, g_sink); });
      printf("read_bw_%zuMB %.1f bytes_per_cycle %.0f GBps\n", fp >> 20, total * 16.0 / (ms * 1e-3 * mhz * 1e6), total * 16.0 / ms / 1e6);
      float msw = time_ms([&] { k_write<<<grid, block>>>(p, mask, total); });
      printf("write_bw_%zuMB %.1f bytes_per_cycle %.0f GBps\n", fp >> 20, total * 16.0 / (msw * 1e-3 * mhz * 1e6), total * 16.0 / msw / 1e6);
    }
    for (unsigned kb : {4u, 16u}) {
      for (int threads : {256, 1024}) {
        unsigned vecs = kb * 1024 / 16;
        int reps = 4000;
        float ms = time_ms([&] { k_read_l1<<<cus, threads>>>(p, vecs, reps, g_sink); });
        double bytes = double(cus) * threads * 4 * 16.0 * reps;
        printf("l1_read_bw_%uKB_window_%dthreads %.1f bytes_per_cycle_per_cu\n", kb, threads, bytes / (ms * 1e-3 * mhz * 1e6) / cus);
      }
    }
    CHECK(hipFree(p));
  }
  if (want("issue")) {
    for (int wps : {1, 2, 4}) {
      int nn = 4096;
      float ms = time_ms([&] { k_valu_tput<<<cus, 64 * 4 * wps>>>(g_sink, nn); });
      double insts = double(cus) * 4 * wps * nn * 8;
      printf("valu_wave_insts_per_cycle_per_cu_wps%d %.2f insts\n", wps, insts / (ms * 1e-3 * mhz * 1e6) / cus);
      float mst = time_ms([&] { k_trans_tput<<<cus, 64 * 4 * wps>>>(g_sink, nn); });
      printf("trans_wave_insts_per_cycle_per_cu_wps%d %.2f insts\n", wps, insts / (mst * 1e-3 * mhz * 1e6) / cus);
    }
  }
  if (want("dispatch")) {
    const int launches = 2000;
    float ms = time_ms([&] { for (int i = 0; i < launches; ++i) k_empty<<<1, 64>>>(); }, 3);
    printf("empty_kernel_back_to_back %.0f cycles %.2f us\n", ms * 1e-3 * mhz * 1e6 / launches, ms * 1e3 / launches);
    for (int wgs : {1 << 16, 1 << 20}) {
      float msw = time_ms([&] { k_empty<<<wgs, 64>>>(); });
      printf("empty_workgroups_%d %.4f workgroups_per_cycle %.2f us\n", wgs, wgs / (msw * 1e-3 * mhz * 1e6), msw * 1e3);
    }
  }
  return 0;
}
