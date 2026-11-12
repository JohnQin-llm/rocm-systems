// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_HSA_CLOCK_H_
#define ROCJITSU_VM_AMDGPU_HSA_CLOCK_H_

/// @file hsa_clock.h
/// @brief Shared synthetic HSA system-clock timestamp helper.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/vm/timing/simulated_clock.h"

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Return a monotonic timestamp in the guest HSA system-clock domain.
///
/// @details `LinuxKfd::fill_get_clock_counters_ioctl` exposes a synthetic
/// 1 GHz nanosecond clock. Dispatch profiling timestamps stored in
/// `amd_signal_t` use the same HSA system-clock domain, so HIP event timing can
/// subtract values written by the command processor and completion tracker
/// directly.
///
/// The value comes from SimulatedClock, which reports host wall time until a
/// timing model is installed and that model's simulated time afterwards.
/// Routing every writer of this domain through one clock is what keeps a HIP
/// event pair subtractable: the two ends are written by different components,
/// and they have to agree on what time means.
inline uint64_t hsa_system_timestamp() { return SimulatedClock::instance().nanoseconds(); }

/// Architectural RTC frequency in MHz. Other models use a nanosecond device clock.
constexpr uint32_t gpu_clock_mhz(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
    return 25;
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return 100;
  default:
    return 1000;
  }
}

constexpr uint64_t gpu_clock_from_ns(uint64_t ns, uint32_t frequency_mhz) {
  return ns / (1000 / frequency_mhz);
}

/// GPU profiling packets and clock calibration share the steady-clock epoch.
inline uint64_t gpu_timestamp(rj_code_arch_t arch) {
  return gpu_clock_from_ns(hsa_system_timestamp(), gpu_clock_mhz(arch));
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_HSA_CLOCK_H_
