// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/deterministic_va.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <sys/mman.h>
#include <vector>

namespace rocjitsu::kmd {
namespace {

constexpr int kReserveFlags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE;
constexpr std::uint64_t kGranule = DeterministicVaArena::kGranule;

// MapFn is a plain function pointer, so the fake keeps its script in statics.
struct FakeMap {
  static inline std::vector<std::uintptr_t> hints;
  static inline std::vector<int> flags;
  // Granules (by hint) that report EEXIST; everything else succeeds at the hint.
  static inline std::vector<std::uintptr_t> occupied;
  // When set, every call fails with this errno instead.
  static inline int fail_errno = 0;
  // When non-zero, the call ignores the hint and returns this address.
  static inline std::uintptr_t elsewhere = 0;

  static void reset() {
    hints.clear();
    flags.clear();
    occupied.clear();
    fail_errno = 0;
    elsewhere = 0;
  }

  static void *map(void *hint, size_t, int, int flag, int, off_t) {
    const auto at = reinterpret_cast<std::uintptr_t>(hint);
    hints.push_back(at);
    flags.push_back(flag);
    if (fail_errno != 0) {
      errno = fail_errno;
      return MAP_FAILED;
    }
    for (const std::uintptr_t taken : occupied)
      if (taken == at) {
        errno = EEXIST;
        return MAP_FAILED;
      }
    if (elsewhere != 0)
      return reinterpret_cast<void *>(elsewhere);
    return hint;
  }
};

class DeterministicVaArenaTest : public ::testing::Test {
protected:
  void SetUp() override { FakeMap::reset(); }
};

TEST_F(DeterministicVaArenaTest, RecognisesOnlyTheRuntimesReservationShape) {
  EXPECT_TRUE(DeterministicVaArena::is_reservation(nullptr, 4096, PROT_NONE, kReserveFlags, -1));
  EXPECT_FALSE(DeterministicVaArena::is_reservation(reinterpret_cast<void *>(0x1000), 4096,
                                                    PROT_NONE, kReserveFlags, -1));
  EXPECT_FALSE(DeterministicVaArena::is_reservation(nullptr, 4096, PROT_READ, kReserveFlags, -1));
  EXPECT_FALSE(DeterministicVaArena::is_reservation(nullptr, 4096, PROT_NONE,
                                                    MAP_ANONYMOUS | MAP_PRIVATE, -1));
  EXPECT_FALSE(DeterministicVaArena::is_reservation(nullptr, 4096, PROT_NONE,
                                                    kReserveFlags | MAP_FIXED, -1));
  EXPECT_FALSE(DeterministicVaArena::is_reservation(nullptr, 4096, PROT_NONE, kReserveFlags, 3));
  EXPECT_FALSE(DeterministicVaArena::is_reservation(nullptr, 0, PROT_NONE, kReserveFlags, -1));
}

TEST_F(DeterministicVaArenaTest, PlacesReservationsAtFixedGranulesInRequestOrder) {
  const std::uint64_t base = 0x200000000000ULL;
  // Two arenas fed the same requests hand out the same addresses.
  for (int run = 0; run < 2; ++run) {
    FakeMap::reset();
    DeterministicVaArena arena(base);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                  arena.reserve(4096, PROT_NONE, kReserveFlags, &FakeMap::map)),
              base);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                  arena.reserve(3 * kGranule + 1, PROT_NONE, kReserveFlags, &FakeMap::map)),
              base + kGranule);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                  arena.reserve(kGranule, PROT_NONE, kReserveFlags, &FakeMap::map)),
              base + 5 * kGranule);
    EXPECT_EQ(arena.cursor(), base + 6 * kGranule);
    for (const int flag : FakeMap::flags)
      EXPECT_NE(flag & MAP_FIXED_NOREPLACE, 0);
  }
}

TEST_F(DeterministicVaArenaTest, RoundsAnUnalignedBaseUpToAGranule) {
  DeterministicVaArena arena(0x200000000000ULL + 1);
  EXPECT_EQ(arena.base(), 0x200000000000ULL + kGranule);
}

TEST_F(DeterministicVaArenaTest, SkipsAGranuleSomethingElseOccupies) {
  const std::uint64_t base = 0x200000000000ULL;
  FakeMap::occupied = {base, base + kGranule};
  DeterministicVaArena arena(base);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                arena.reserve(4096, PROT_NONE, kReserveFlags, &FakeMap::map)),
            base + 2 * kGranule);
  EXPECT_EQ(FakeMap::hints.size(), 3u);
}

TEST_F(DeterministicVaArenaTest, GivesUpWhenTheFixedRangeIsUnavailable) {
  const std::uint64_t base = 0x200000000000ULL;
  // A long occupied run: bounded retries, then MAP_FAILED/EEXIST for the
  // caller's ordinary placement.
  for (std::uint64_t granule = 0; granule < 100; ++granule)
    FakeMap::occupied.push_back(base + granule * kGranule);
  DeterministicVaArena busy(base);
  errno = 0;
  EXPECT_EQ(busy.reserve(4096, PROT_NONE, kReserveFlags, &FakeMap::map), MAP_FAILED);
  EXPECT_EQ(errno, EEXIST);
  EXPECT_EQ(FakeMap::hints.size(), 64u);

  // Exhausted: no call at all.
  FakeMap::reset();
  DeterministicVaArena full(base, base + kGranule);
  EXPECT_NE(full.reserve(kGranule, PROT_NONE, kReserveFlags, &FakeMap::map), MAP_FAILED);
  errno = 0;
  EXPECT_EQ(full.reserve(4096, PROT_NONE, kReserveFlags, &FakeMap::map), MAP_FAILED);
  EXPECT_EQ(errno, ENOMEM);
  EXPECT_EQ(FakeMap::hints.size(), 1u);

  // Any other failure is passed through without retrying.
  FakeMap::reset();
  FakeMap::fail_errno = ENOMEM;
  DeterministicVaArena failing(base);
  EXPECT_EQ(failing.reserve(4096, PROT_NONE, kReserveFlags, &FakeMap::map), MAP_FAILED);
  EXPECT_EQ(errno, ENOMEM);
  EXPECT_EQ(FakeMap::hints.size(), 1u);
  EXPECT_EQ(failing.cursor(), base);
}

TEST_F(DeterministicVaArenaTest, HandsBackAPlacementAKernelMadeElsewhere) {
  // A kernel without MAP_FIXED_NOREPLACE treats it as a hint.
  FakeMap::elsewhere = 0x7f0000000000ULL;
  DeterministicVaArena arena(0x200000000000ULL);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                arena.reserve(4096, PROT_NONE, kReserveFlags, &FakeMap::map)),
            FakeMap::elsewhere);
}

TEST_F(DeterministicVaArenaTest, RealMmapSkipsAnOccupiedGranule) {
  const std::uint64_t base = 0x3a0000000000ULL;
  void *const blocker = mmap(reinterpret_cast<void *>(base), kGranule, PROT_NONE,
                             kReserveFlags | MAP_FIXED_NOREPLACE, -1, 0);
  if (blocker != reinterpret_cast<void *>(base))
    GTEST_SKIP() << "could not place the blocking mapping at the test base";
  DeterministicVaArena arena(base);
  void *const placed = arena.reserve(4096, PROT_NONE, kReserveFlags, &::mmap);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(placed), base + kGranule);
  if (placed != MAP_FAILED)
    munmap(placed, 4096);
  munmap(blocker, kGranule);
}

TEST(DeterministicVaEnvTest, OnByDefaultAtTheDefaultBase) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        unsetenv("ROCJITSU_DETERMINISTIC_VA");
        unsetenv("ROCJITSU_DETERMINISTIC_VA_BASE");
        DeterministicVaArena *arena = runtime_va_arena();
        std::_Exit(arena != nullptr && arena->base() == DeterministicVaArena::kDefaultBase ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(DeterministicVaEnvTest, ZeroDisablesTheArena) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        setenv("ROCJITSU_DETERMINISTIC_VA", "0", 1);
        std::_Exit(runtime_va_arena() == nullptr ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(DeterministicVaEnvTest, BaseVariableMovesTheArena) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        unsetenv("ROCJITSU_DETERMINISTIC_VA");
        setenv("ROCJITSU_DETERMINISTIC_VA_BASE", "0x300000000001", 1);
        DeterministicVaArena *arena = runtime_va_arena();
        std::_Exit(arena != nullptr && arena->base() == 0x300000000000ULL + kGranule ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(DeterministicVaRuntimeTest, RecognisesTheRuntimeLibrariesByBaseName) {
  EXPECT_TRUE(is_gpu_runtime_library("/opt/rocm/lib/libhsa-runtime64.so.1.15.0"));
  EXPECT_TRUE(is_gpu_runtime_library("libhsakmt.so.1"));
  EXPECT_FALSE(is_gpu_runtime_library("/usr/lib/libc.so.6"));
  EXPECT_FALSE(is_gpu_runtime_library("/opt/libhsa-runtime64.so.d/libfoo.so"));
  EXPECT_FALSE(is_gpu_runtime_library(nullptr));
}

} // namespace
} // namespace rocjitsu::kmd
