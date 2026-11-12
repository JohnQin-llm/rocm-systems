# RocJITsu corpus policy

`gfx1250_b0_a0_semantic_tests.json` selects semantic programs whose instruction
forms have implemented runtime translations. The companion
`gfx1250_b0_a0_semantic_rewrites.json` pins the exact non-negative offline
rewrite count for every selected executable. A zero count qualifies a
deliberate copy-path fix through runtime comparison.

These source-coverage programs are intentionally outside this qualification
until their translations or semantic fixtures are ready:

- `barrier_id_minus1_scc_test`
- `barrier_id_minus2_scc_test`
- `flat_scratch_scalar_hi_test`
- `flat_scratch_scalar_lo_test`
- `flat_scratch_vector_hi_test`
- `flat_scratch_vector_lo_test`
- `fp8_e5m3_pack_test`
- `monitor_sleep_bounded_test`
- `monitor_sleep_unbounded_test`
- `permute_pk16_test`

The offline translator currently rejects those forms as unsupported. They are
not silently accepted or treated as passing translations.

## Offline translation SHA baseline

The `rocjitsu-test-corpus` workflow records the SHA-256 of every successful
gfx1250 B0-to-A0 input and translated output after the release-lane corpus
tests succeed. A develop push uploads the canonical manifest as the
`gfx1250-b0-a0-sha-pairs` workflow artifact for 45 days. A manual workflow
dispatch on the `develop` branch also refreshes the baseline when the official
corpus repository and pinned gfx1250 corpus ref are left unchanged. The
manifest includes the fixed translation profile, pinned corpus commit,
input-manifest hash, package-lock hash, ROCm SDK version, and source commit; it
never contains the code objects themselves.

Pull-request runs upload a seven-day candidate manifest without receiving
write access. A separate trusted workflow selects the newest completed,
unexpired artifact from a successful develop push or canonical manual run,
then compares output hashes only when the corpus, input manifest, package lock,
and SDK provenance match. Before using a candidate artifact, the trusted
workflow also requires the expected source workflow path and an exact match to
the current head of a same-repository PR. Changed outputs or incompatible
provenance create or update one non-blocking warning comment on the PR. When a
later run matches, the workflow removes its stale comment.

`record-gfx1250-dbt-sha-pairs.py` runs a bounded, translation-only collection
pass after the corpus tests and validates the resulting manifest. The external
corpus harness does not expose the translated bytes, so keeping collection
separate avoids coupling its interface to this workflow. The preceding
harness run qualifies the included input set under its timeout and memory
policy; the collector repeats the per-object timeout and does not rerun
declared exclusions. It streams each output through a temporary file and
retains only its size and SHA-256. Its `finalize` command requires every pinned
corpus input to have either a successful pair or a matching declared
exclusion, which prevents partial runs from becoming a develop baseline.

## HIP, race, file-limit, and DBT corpus pin

`ROCJITSU_DBT_CORPUS_REF` pins the HIP, race, file-limit, and DBT suites together in
`.github/workflows/rocjitsu-corpus-tests.yml`. Keep it equal to the `corpus_ref`
manual-dispatch default. When advancing the pin, update both references to the
selected merged revision and validate that pair in corpus CI before landing
the systems pin update. Matching references also preserves canonical manual
collection of the gfx1250 translation SHA baseline.

## HIP file-limit matrix

`run-file-limit-tests.sh` is the shared local and CI entry point for the corpus's
18 local/daemon file-limit cases. It builds the HIP client for `gfx950`, uses
`tests/race-detector/race_test_config.json`, and runs CTest under `tini -s` so
detached memory keepers are reaped. Install `tini` on `PATH`, CMake, Ninja, and a
ROCm SDK, then build rocjitsu before invoking the script from its source root:

```sh
ROCM_PATH=/path/to/rocm-sdk \
ROCJITSU_BUILD_DIR="$PWD/build" \
ROCJITSU_CORPUS_DIR=/path/to/rocjitsu-test-corpus \
  bash tests/corpus/run-file-limit-tests.sh
```

Use the corpus revision selected by the shared pin above. CI requires
`corpus/runtime/file-limits/CMakeLists.txt` at the shared pin, including after
a pin rollback. A manual dispatch may compare against an earlier pin that
predates the matrix: if its resolved SHA differs from the shared pin and is
listed in `ROCJITSU_FILE_LIMIT_LEGACY_CORPUS_SHAS`, the step logs a notice with
the SHA and missing path and runs no tests. All other missing-suite cases
fail the file-limit step. The script reads
`RJ_ENABLE_ASAN` and `RJ_ENABLE_TSAN` from the rocjitsu build's `CMakeCache.txt`
to select `address,undefined` or `thread` instrumentation for the HIP client's
host code. For ASan/TSan, it also uses the cached `CMAKE_CXX_COMPILER` as the
HIP compiler so the client's sanitizer runtime matches the launcher's preload.
That compiler must be Clang with HIP/AMDGPU support for `gfx950` and the selected
ROCm SDK. Use `RJ_SANITIZER_RUNTIME=AUTO` or `SHARED`; static sanitizer builds
and non-Clang ASan/TSan compilers are rejected before configuring the client.
The runner checks that the cached compiler accepts HIP for `gfx950` with the
selected SDK before configuring the client. Both SDK `amdclang++` and compatible
system Clang builds are supported. The
client's build RPATH includes `$ROCM_PATH/lib` so system Clang can find HIP's
shared libraries. Do not add a different compiler's sanitizer directory to
`LD_LIBRARY_PATH` to work around a runtime mismatch.

Default and UBSan-only builds use an uninstrumented SDK HIP client, matching
CI. Point `ROCJITSU_BUILD_DIR` at a separate matching rocjitsu build to run each
sanitizer configuration. The suite build and CTest logs are retained under
`$ROCJITSU_BUILD_DIR/file-limit-corpus`. Build parallelism uses Ninja's default;
set `CMAKE_BUILD_PARALLEL_LEVEL` to limit it. CTest defaults to four concurrent
tests to bound memory use. Extra arguments go to CTest, for example
`-R local.early -V`.

The client first looks for an installed GoogleTest. When the rocjitsu build's
GoogleTest sources are the release the suite requests, the script passes them
as `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` for the client's FetchContent fallback.
If no installed package is found, the client uses those sources or fetches its
pinned release when matching sources are unavailable.

The script uses its own source tree by default; set `ROCJITSU_SOURCE_DIR` to
override it. It does not check out or update the corpus revision and does not
run the pytest corpus suites.

## Per-test deadlines and near-timeout reporting

`run-corpus-tests.sh` uses `--soft-timeout` for the first pass and
`--hard-timeout` for failed-test retries (30 and 60 seconds by default).
For race cases, it sets `ROCJITSU_RACE_CASE_TIMEOUT` to the same per-test
deadline as the launch wrapper, overriding any inherited value on both
passes. The Release workflow explicitly selects 360 seconds for the first
pass and 600 seconds for retries. `--rerun-timeout` separately bounds each
target's entire retry invocation; a successful retry retains the original CI
failure.

With `--warn-perf`, `run-corpus-tests.sh` warns about passing tests whose
runtime approaches the first-pass per-test deadline. Reporting uses
`--soft-timeout`, including for the race suite.

With `--ubsan`, which the workflow passes for UBSan builds, the simulator suites
omit cases listed in the script's `ubsan_excluded_cases`. Those cases, such as
the 2048x2048 rocBLAS SGEMM, exceed the deadline under UBSan instrumentation.
Uninstrumented lanes still run them.

## gfx1201 simulator exclusions

`fpsan_global_load_tr_gfx12_w64_test` is excluded because its wave64
`global_load_tr` path currently attempts to write a lane outside the gfx1201
simulator wavefront. This is a simulator modeling gap, not a change introduced
by the corpus SDK nightly.

## gfx1250 simulator exclusions

`rocblas_sgemm::square_64x64` is excluded because the pinned ROCm SDK kernel
reads a scalar-memory result after a nonzero KMCNT wait.

## Vulkan compute and buffer CTS

The release configuration runs the corpus's opt-in `vulkan` suite separately on
`gfx1100` (RDNA3) and `gfx1201` (RDNA4). It installs Mesa RADV and libdrm in
the job container and builds an unmodified pinned headless Vulkan CTS. CI selects 735
compute and 2,543 texel-buffer cases, leaving twelve expensive
maximum-workgroup/LDS stress cases available for local runs. Both targets run
in the main corpus workflow's release configuration, after the HIP, DBT, and
file-limit corpus steps. Vulkan setup and execution use their own prerequisite
checks, so a Vulkan setup failure cannot suppress that earlier coverage and
Vulkan can still run after a HIP, DBT, or file-limit test failure. Any CI
coverage including the stress cases must use a separate workflow. Graphics and
image sampling are out of scope. Exact hardware-baseline unsupported cases are
reported as skips; unexpected unsupported results and incomplete logs fail.
Four cases that initialize images through graphics commands are excluded from
the original compute selection and recorded in `corpus/vulkan/excluded-graphics.txt`.

The Vulkan corpus has its own commit pin and checkout so the existing HIP and
DBT corpus, including its translation baseline, retain their current inputs.
For a coordinated corpus change, publish its commit first, then update both
`ROCJITSU_VULKAN_CORPUS_REF` and the `vulkan_corpus_ref` dispatch default.
A manual workflow dispatch can override the Vulkan corpus ref independently.

For local testing, install Mesa 25.2.8+ and libdrm 2.4.125+, activate a Python
environment, and build CTS once from the corpus checkout:

```sh
cd /path/to/rocjitsu-test-corpus
uv pip install -r corpus/vulkan/requirements.txt
bash scripts/build_vulkan_cts.sh
export ROCJITSU_CORPUS_DIR="$PWD"
export ROCJITSU_BUILD_DIR=/path/to/rocjitsu/build
bash /path/to/rocm-systems/emulation/rocjitsu/tests/corpus/run-vulkan-cts.sh all --case smoke
```

The RocJITsu build should use `CMAKE_BUILD_TYPE=Release`. An explicit
`ROCJITSU_BUILD_DIR` selects its launcher; otherwise the wrapper looks for
`rocjitsu` on `PATH`, then under `emulation/rocjitsu/build`. Set
`ROCJITSU_SOURCE_DIR` to override the source directory used for target configs.
Use `--case ci` to reproduce the release workflow, omit `--case smoke` for
all 3,290 cases, or use
`--case stress` for just the twelve expensive cases. Replace `all` with one
target to narrow the run. The wrapper bounds pytest to sixteen workers by default
and divides `VULKAN_CTS_CPU_BUDGET` among them through RocJITsu's
`--cpu-thread-budget` option. By default the total budget is the smaller of 64
and the available CPU count, so a host with at least 64 CPUs uses sixteen
processes with four execution threads each. Set
`VULKAN_CTS_WORKERS` and `VULKAN_CTS_CPU_BUDGET` to compare other allocations;
runtime service threads and Mesa compilation are outside this budget.
Each launch uses the shipped target config and passes
`--cpu-thread-budget floor(VULKAN_CTS_CPU_BUDGET / VULKAN_CTS_WORKERS)`.
For example, budget 64 with sixteen workers gives each simulator four execution
threads. Budget three with two workers leaves one execution thread unused.
The wrapper selects the system RADV ICD, avoiding SDK library overrides.
CTS verifies the requested device and driver in each result. Logs
and JUnit reports go to `.pytest-artifacts/vulkan-<target>` in the corpus and
are uploaded even after test failures. Set `VULKAN_CTS_ARTIFACT_ROOT` to a
different repo-local prefix to retain separate configurations and logs when
comparing runs. See the corpus's
`corpus/vulkan/README.md` for dependencies, individual case selection and
existing CTS binary overrides. That guide also documents direct hardware runs;
the corpus adapter has no simulator dependency. Simulator configuration,
thread budgets and launcher selection belong to this wrapper.

## Sanitizer simulator coverage

The Clang and GCC ASan+UBSan lanes run the same target-qualified corpus as the
release lane against a sanitizer-instrumented RocJITsu build. They use longer
per-test timeouts to accommodate the instrumented simulator.

Each case runs through `env` → `setpriv` → the process supervisor → `timeout` →
`rocjitsu` → the optional HIP preload helper → the corpus executable. The
run-wrapper `timeout` owns the per-test deadline and retains command output;
pytest gets 15 seconds of cleanup headroom as a failsafe. Each target's
failed-test rerun also has a 20-minute budget in CI, within the 60-minute
workflow step.

Clang sanitizer runs load HIP at child startup through the corpus-only helper.
This keeps the shared Clang ASan runtime, simulator interposer, and HIP runtime
in loader order without adding corpus-specific policy to the `rocjitsu`
command-line interface. The GCC lane uses its executable-linked ASan runtime
and leaves HIP loading to the corpus executable.

Leak detection is disabled for the sanitizer corpus subprocesses because each
target group mixes HIP-backed and pure simulator cases, while LeakSanitizer's
stop-the-world scan stalls on HIP's multi-gigabyte mappings. The current corpus
wrapper cannot vary that setting per individual case; ASan and UBSan remain
enabled throughout.

## Temporarily disabled gfx1250 integer and memory CTS tests

`gfx1250_skip_tests.json` excludes all 22 integer CTS cases added by
[rocjitsu-test-corpus#41](https://github.com/ROCm/rocjitsu-test-corpus/pull/41)
and all 15 memory CTS cases added by
[rocjitsu-test-corpus#42](https://github.com/ROCm/rocjitsu-test-corpus/pull/42),
including their linked-image coverage checks.

Reason: These tests are really slow and haven't been verified for their usefulness. Someone needs to investigate these before they are enabled

## What each file here is for

The directory holds two unrelated harnesses that happen to share a name. One
qualifies the *functional* corpus (does the simulator compute the right answer
on real programs); the other measures *time* (does a timing model agree with
hardware). Nothing in the first group reads a timing model, and nothing in the
second group asserts a result value.

### The functional corpus

| File | Purpose |
|---|---|
| `run-corpus-tests.sh` | Runs the pytest corpus under each simulated target. The entry point for everything in this group. |
| `corpus-process-supervisor.sh` | Runs one case below a pinned session leader so no process-group member outlives it. |
| `corpus-hip-preload.sh` | Loads HIP at child startup for the Clang sanitizer lane, keeping the ASan runtime, the interposer and HIP in loader order. |
| `report-near-timeout-tests.py` | Ranks passing tests by how close they came to their per-test timeout. Wired to `--warn-perf`. |
| `gfx*_skip_tests.json` | Per-target exclusions, with the reason each one carries. |
| `gfx1250_b0_a0_semantic_*.json`, `dbt/` | The gfx1250 B0-to-A0 translation corpus: selected programs, pinned rewrite counts, expected failures, package lock. |
| `validate-gfx1250-semantic-translations.py` | Checks a translation run against those pinned expectations. |
| `record-gfx1250-dbt-sha-pairs.py` | Collects the input/output SHA-256 manifest described above, and validates it. |
| `test-validate-gfx1250-semantic-translations.py`, `test-record-gfx1250-dbt-sha-pairs.py` | Unit tests for the two scripts above. |

### The timing-model harness

| File | Purpose |
|---|---|
| `rocm-meter.py` | Runs a suite of torch and Triton kernels and writes a JSON report: per case, a `torch.cuda.Event` bracket and a torch-profiler operator breakdown. Runs on hardware and, unmodified, under the emulator. |
| `run-meter-emulated.sh` | Runs that suite under `rocjitsu`, one emulator process per kernel family, writing one shard report each. A single serial pass is too slow to iterate a model against. |
| `meter_extract.py` | Pulls per-case timings out of one report, or diffs two, for looking at a single family by hand. Explains why `kernel_us` and `device_timing` differ. |
| `test_meter_extract.py` | Unit tests for the extractor. Run by ctest. |
| `meter_score.py` | Compares an emulated run against a hardware run and scores it. Owns the decision of *what* is compared; read its module docstring before reading any number it prints. `--json` emits the scored comparison. |
| `meter_report.py` | Turns one or more of those JSON payloads into a self-contained Markdown or HTML report: coverage, accuracy per category, the log2-ratio spread and its histogram, what the model gets wrong, model-versus-model, and the wall-clock cost of running the model. Adds no data to the comparison. |
| `meter_rank.py` | Scores an emulated run on *ranking*: among variants of one case (eager / compiled / triton), does the model order them as hardware does, and how far off is its speedup. See `docs/timing-mi355x.md`. |
| `meter_fit.py` | Fits `meter_refit.py`'s composition knobs to a hardware reference by differential evolution, with a train/test split by variant group, and reports absolute and ranking accuracy on both. |
| `streamk_gemv_repro.py` | Minimal hipBLASLt Stream-K reproducer (1x4096 @ 4096x4096) for the scalar-memory flag-polling hang described in `docs/timing-mi355x.md`. |

`meter_report.py --selftest` renders a synthetic payload in a temporary
directory, checks that the hand-computed statistics reach both formats, that
the HTML references no external host, and that two renders of the same input
are byte-identical.

## Validating a timing model end to end

The whole sequence, from a clean tree to a report. It is written for the
gfx950 MI355X config; swap the config and the reference run together, never
one of them.

**1. Record the reference run on hardware.** On a real MI355X, with a ROCm
torch on the path:

```sh
python3 tests/corpus/rocm-meter.py \
    --tier standard --samples 5 --warmups 5 \
    --output ~/meter/real-gfx950.json
```

This is the only step that needs the physical part, and its output is the
thing everything downstream is scored against. Keep it; re-recording it under
a different torch or a different ROCm changes the target.

**2. Build rocjitsu, including the timing models.**

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24
```

The models load as their own shared objects and the loader refuses one whose
ABI version does not match, so a partial rebuild is a diagnostic rather than a
mystery. Build the whole target list.

**3. Select the model in the architecture config.** The `timing` block of
`configs/gfx950_mi355x_kmd_timing.json` names it. Copy the config per model rather
than editing one in place, so a run's config is recoverable afterwards:

```sh
sed 's/"model": "leaky"/"model": "des"/' \
    configs/gfx950_mi355x_kmd_timing.json > /tmp/gfx950-des.json
```

For the untimed baseline of step 6, take a copy with the whole `timing` block
removed. There is no "model": "none" -- a model is loaded or it is not.

**4. Run the suite under the emulator, once per model.** Time it: the wall
clock is an input to the report.

```sh
start=$SECONDS
tests/corpus/run-meter-emulated.sh \
    --config /tmp/gfx950-des.json \
    --out ~/meter/emulated-des
echo "des wall: $((SECONDS - start))s"
```

`METER_PYTHON` (or `--python`) selects the interpreter; it needs a ROCm torch,
which is not the system python. conv2d NCHW is excluded by default because it
reaches MIOpen, which does not terminate under emulation within any useful
budget; `--with-nchw` includes it behind the per-shard timeout. Every case the
run does not produce is reported by step 6 as a coverage gap, which is the
intended treatment: a dropped case is not a passed one.

**5. Score each emulated run against the reference.**

```sh
python3 tests/corpus/meter_score.py \
    --real ~/meter/real-gfx950.json \
    --emulated ~/meter/emulated-des/*.json \
    --tolerance-pct 20 \
    --json ~/meter/scored-des.json
```

Repeat for every model. The shard reports are merged here, so overlapping
`--kernel` filters are harmless.

**6. Render the report.**

```sh
python3 tests/corpus/meter_report.py \
    --scored "des=$HOME/meter/scored-des.json" \
    --scored "leaky=$HOME/meter/scored-leaky.json" \
    --wall none=744 --wall des=3120 --wall leaky=982 \
    --title "gfx950 MI355X: des vs leaky" \
    --out ~/meter/report
```

`--scored` takes `LABEL=PATH`, so the path is written with `$HOME` rather
than `~`: a tilde after `=` is not expanded by every shell.

`--wall` takes the seconds from step 4, plus the untimed baseline run, and
turns them into the multiplier the timing model costs. Omit them and the
report says the cost was not measured rather than leaving the section out;
an unmeasured cost must not read as a free one. `--generated-at TEXT` stamps a
date; without it the output carries none and two runs of the same input are
byte-identical.

The report opens with coverage, not accuracy. A suite that quietly drops cases
flatters whatever is left, so the count of reference cases that were never
scored belongs beside the accuracy claim.

### Reading the result

The score is the **spread** of `log2(model / measured)`, in octaves. Not the
bias, which is a median and cancels a model whose errors are large in both
directions; not a correlation, which is nearly free on a corpus spanning four
orders of magnitude. See `docs/timing-model.md` for the argument, and the
module docstring of `meter_score.py` for which two quantities are compared and
why the emulated bracket is scored against the hardware *kernel* time rather
than against the hardware bracket.
