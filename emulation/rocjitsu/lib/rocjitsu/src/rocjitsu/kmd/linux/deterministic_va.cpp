// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/deterministic_va.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <link.h>
#include <sys/mman.h>

namespace rocjitsu::kmd {
namespace {

std::uint64_t round_up(std::uint64_t value, std::uint64_t granule) {
  return (value + granule - 1) / granule * granule;
}

/// @brief Executable ranges of the GPU runtime. Written once per KFD open (and
///        then only ever extended), read on every candidate mmap.
struct TextRanges {
  static constexpr std::size_t kMax = 16;
  std::array<std::atomic<std::uintptr_t>, kMax> begin{};
  std::array<std::atomic<std::uintptr_t>, kMax> end{};
  std::atomic<std::size_t> count{0};
};

TextRanges &text_ranges() {
  static TextRanges ranges;
  return ranges;
}

int note_object(dl_phdr_info *info, size_t, void *) {
  if (info->dlpi_name == nullptr || !is_gpu_runtime_library(info->dlpi_name))
    return 0;
  TextRanges &ranges = text_ranges();
  for (int index = 0; index < info->dlpi_phnum; ++index) {
    const ElfW(Phdr) &header = info->dlpi_phdr[index];
    if (header.p_type != PT_LOAD || (header.p_flags & PF_X) == 0)
      continue;
    const std::uintptr_t begin = info->dlpi_addr + header.p_vaddr;
    const std::uintptr_t end = begin + header.p_memsz;
    const std::size_t known = ranges.count.load(std::memory_order_acquire);
    bool seen = false;
    for (std::size_t slot = 0; slot < known && !seen; ++slot)
      seen = ranges.begin[slot].load(std::memory_order_relaxed) == begin;
    if (seen || known >= TextRanges::kMax)
      continue;
    ranges.begin[known].store(begin, std::memory_order_relaxed);
    ranges.end[known].store(end, std::memory_order_relaxed);
    ranges.count.store(known + 1, std::memory_order_release);
  }
  return 0;
}

std::uint64_t parse_base(const char *text) {
  if (text == nullptr || *text == '\0')
    return DeterministicVaArena::kDefaultBase;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 0);
  if (end == text || value == 0)
    return DeterministicVaArena::kDefaultBase;
  return round_up(value, DeterministicVaArena::kGranule);
}

} // namespace

DeterministicVaArena::DeterministicVaArena(std::uint64_t base, std::uint64_t limit)
    : base_(round_up(base, kGranule)), limit_(limit), cursor_(base_) {}

bool DeterministicVaArena::is_reservation(const void *addr, size_t length, int prot, int flags,
                                          int fd) {
  constexpr int kRequired = MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE;
  constexpr int kPlacement = MAP_FIXED | MAP_FIXED_NOREPLACE;
  return addr == nullptr && fd == -1 && length != 0 && prot == PROT_NONE &&
         (flags & kRequired) == kRequired && (flags & kPlacement) == 0;
}

void *DeterministicVaArena::reserve(size_t length, int prot, int flags, MapFn map) {
  if (length == 0 || map == nullptr) {
    errno = EINVAL;
    return MAP_FAILED;
  }
  const std::uint64_t span = round_up(length, kGranule);
  std::lock_guard<std::mutex> guard(mutex_);
  // A bounded number of skips: one occupied granule is an accident, a run of
  // them means something else owns this range and the ordinary placement is
  // the better answer.
  for (int attempt = 0; attempt < 64; ++attempt) {
    if (cursor_ > limit_ || span > limit_ - cursor_) {
      errno = ENOMEM;
      return MAP_FAILED;
    }
    void *const hint = reinterpret_cast<void *>(static_cast<std::uintptr_t>(cursor_));
    void *const placed = map(hint, length, prot, flags | MAP_FIXED_NOREPLACE, -1, 0);
    if (placed == hint) {
      cursor_ += span;
      return placed;
    }
    // A kernel too old for MAP_FIXED_NOREPLACE treats it as a hint and maps
    // elsewhere; hand that back rather than unmapping it and trying again.
    if (placed != MAP_FAILED)
      return placed;
    if (errno != EEXIST)
      return MAP_FAILED;
    cursor_ += span;
  }
  errno = EEXIST;
  return MAP_FAILED;
}

std::uint64_t DeterministicVaArena::cursor() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return cursor_;
}

DeterministicVaArena *runtime_va_arena() {
  static DeterministicVaArena *const arena = []() -> DeterministicVaArena * {
    const char *enabled = std::getenv("ROCJITSU_DETERMINISTIC_VA");
    if (enabled != nullptr && enabled[0] == '0')
      return nullptr;
    // Leaked on purpose: reservations are made until the process's last
    // instruction, including from destructors that run after static teardown.
    return new DeterministicVaArena(parse_base(std::getenv("ROCJITSU_DETERMINISTIC_VA_BASE")));
  }();
  return arena;
}

void note_gpu_runtime_text() { dl_iterate_phdr(note_object, nullptr); }

bool is_gpu_runtime_caller(const void *return_address) {
  const TextRanges &ranges = text_ranges();
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(return_address);
  const std::size_t count = ranges.count.load(std::memory_order_acquire);
  for (std::size_t slot = 0; slot < count; ++slot)
    if (address >= ranges.begin[slot].load(std::memory_order_relaxed) &&
        address < ranges.end[slot].load(std::memory_order_relaxed))
      return true;
  return false;
}

bool is_gpu_runtime_library(const char *name) {
  if (name == nullptr)
    return false;
  const char *slash = std::strrchr(name, '/');
  const char *base = slash != nullptr ? slash + 1 : name;
  return std::strncmp(base, "libhsa-runtime64.so", 19) == 0 ||
         std::strncmp(base, "libhsakmt.so", 12) == 0;
}

} // namespace rocjitsu::kmd
