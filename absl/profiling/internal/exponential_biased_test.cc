// Copyright 2019 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/profiling/internal/exponential_biased.h"

#include <cmath>
#include <cstdint>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/attributes.h"
#include "absl/base/config.h"
#include "absl/strings/str_cat.h"

using ::testing::Ge;

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace profiling_internal {
namespace {

MATCHER_P2(IsBetween, a, b,
           absl::StrCat(std::string(negation ? "isn't" : "is"), " between ", a,
                        " and ", b)) {
  return a <= arg && arg <= b;
}

TEST(ExponentialBiasedTest, NonPositiveParametersAndClamp) {
  ExponentialBiased eb;
  EXPECT_EQ(eb.GetSkipCount(0), 0);
  EXPECT_EQ(eb.GetSkipCount(-5), 0);
  EXPECT_EQ(eb.GetStride(1), 1);
  EXPECT_EQ(eb.GetStride(0), 1);
  EXPECT_EQ(eb.GetStride(-5), 1);

  for (int i = 0; i < 100; ++i) {
    EXPECT_GE(eb.GetStride(int64_t{1} << 62), 1);
  }
}

TEST(ExponentialBiasedTest, DistributionShapeAndMean) {
  ExponentialBiased eb;

  // Verify both sample mean and 5 quantile CDF buckets for `mean = 9`
  // (`period = 10`, p = 0.1) over 200,000 draws. All intervals are +/-10 SD
  // around Geometric(0.1) (`p_flake < 1e-22`).
  constexpr int64_t kMean = 9;
  constexpr int kDraws = 200000;
  int b0 = 0;      // k == 0:      p = 0.10000
  int b1_2 = 0;    // 1 <= k <= 2: p = 0.17100
  int b3_6 = 0;    // 3 <= k <= 6: p = 0.25070
  int b7_13 = 0;   // 7 <= k <= 13: p = 0.24953
  int b14_up = 0;  // k >= 14:     p = 0.22877
  int64_t sum = 0;
  for (int i = 0; i < kDraws; ++i) {
    const int64_t k = eb.GetSkipCount(kMean);
    ASSERT_GE(k, 0);
    sum += k;
    if (k == 0) {
      ++b0;
    } else if (k <= 2) {
      ++b1_2;
    } else if (k <= 6) {
      ++b3_6;
    } else if (k <= 13) {
      ++b7_13;
    } else {
      ++b14_up;
    }
  }
  // std(sum) = sqrt(200000 * 90) ~= 4243 (+/-10 SD -> [1757570, 1842430]).
  EXPECT_THAT(sum, IsBetween(1757570, 1842430));
  EXPECT_THAT(b0, IsBetween(18658, 21341));
  EXPECT_THAT(b1_2, IsBetween(32516, 35883));
  EXPECT_THAT(b3_6, IsBetween(48202, 52078));
  EXPECT_THAT(b7_13, IsBetween(47970, 51841));
  EXPECT_THAT(b14_up, IsBetween(43874, 47632));

  // Verify sample mean across small, medium, and large means (+/-10 SD).
  for (const int64_t mean : {int64_t{1}, int64_t{999}, int64_t{1} << 30}) {
    constexpr int kMeanDraws = 50000;
    double total = 0;
    for (int i = 0; i < kMeanDraws; ++i) {
      const int64_t k = eb.GetSkipCount(mean);
      ASSERT_GE(k, 0);
      total += static_cast<double>(k);
    }
    // At mean = 1 (`period = 2`), `floor(-ln(u) * 1.5)` has exact expectation
    // `1 / expm1(2/3) ~= 1.0551`; for larger means `1 / (12 * (mean + 0.5))`
    // is negligible (< 1e-4).
    const double expected_per_draw =
        1.0 / std::expm1(1.0 / (static_cast<double>(mean) + 0.5));
    const double expected_total = kMeanDraws * expected_per_draw;
    const double sd =
        std::sqrt(kMeanDraws * expected_per_draw * (expected_per_draw + 1.0));
    EXPECT_THAT(total,
                IsBetween(expected_total - 10 * sd, expected_total + 10 * sd))
        << "mean = " << mean;
  }
}

TEST(ExponentialBiasedTest, FirstDrawUnbiased) {
  // Freshly constructed generators must sample event #1 (`GetStride(kPeriod)
  // == 1`) with probability `1 - exp(-1 / (kPeriod - 0.5)) ~= 1 / kPeriod`
  // (not ~0.5 / kPeriod from `std::rint`'s half-width `[0, 0.5)` zero bin).
  constexpr int64_t kPeriod = 16;
  constexpr int kTrials = 160000;
  int unit_strides = 0;
  for (int i = 0; i < kTrials; ++i) {
    ExponentialBiased fresh;
    if (fresh.GetStride(kPeriod) == 1) {
      ++unit_strides;
    }
  }
  // Expected ~= 10316 (for 1 - exp(-1/15.5) = 0.06448; or 10000 for 1/16),
  // SD ~= 98; +/-10 SD easily fits inside [9000, 11500], whereas `std::rint`
  // produced ~5245.
  EXPECT_THAT(unit_strides, IsBetween(9000, 11500));
}

TEST(ExponentialBiasedTest, MemorylessIndependence) {
  // Verify consecutive draws are independent with no post-sample refractory
  // dip: `P(S_{t+1} == 1 | S_t == 1)` must match `P(S == 1)`.
  ExponentialBiased eb;
  constexpr int64_t kPeriod = 8;
  constexpr int kDraws = 640000;
  bool prev_was_one = false;
  int prev_one_count = 0;
  int consecutive_ones = 0;
  for (int i = 0; i < kDraws; ++i) {
    const int64_t s = eb.GetStride(kPeriod);
    ASSERT_GE(s, 1);
    if (prev_was_one) {
      ++prev_one_count;
      if (s == 1) ++consecutive_ones;
    }
    prev_was_one = (s == 1);
  }
  const double expected_p = 1.0 - std::exp(-1.0 / (kPeriod - 0.5));
  const double observed_p =
      static_cast<double>(consecutive_ones) / prev_one_count;
  // SD(observed_p) ~= sqrt(0.125 * 0.875 / 80000) ~= 0.00117; +/-10 SD is
  // +/-0.012 (old `std::rint` + `bias_` produced ~0.068).
  EXPECT_THAT(observed_p, IsBetween(expected_p - 0.015, expected_p + 0.015));
}


// Verifies 1D marginal uniformity and 2D lag-1 pair independence of
// `NextRandom` via chi-squared bucket tests.
TEST(ExponentialBiasedTest, TestNextRandom) {
  uint64_t x = 1;
  for (int i = 0; i < 20; ++i) {
    x = ExponentialBiased::NextRandom(x);
  }

  constexpr int kBuckets = 64;
  constexpr int kPairDim = 8;  // 8 x 8 = 64 lag-1 pair buckets
  constexpr int kDraws = 65536;
  constexpr double kExpected = static_cast<double>(kDraws) / kBuckets;

  int counts_1d[kBuckets] = {};
  int counts_2d[kPairDim][kPairDim] = {};
  uint64_t prev_pair_bin = x >> (ExponentialBiased::kPrngNumBits - 3);

  for (int i = 0; i < kDraws; ++i) {
    x = ExponentialBiased::NextRandom(x);
    ASSERT_EQ(x >> ExponentialBiased::kPrngNumBits, 0u);
    ++counts_1d[x >> (ExponentialBiased::kPrngNumBits - 6)];
    const uint64_t curr_pair_bin = x >> (ExponentialBiased::kPrngNumBits - 3);
    ++counts_2d[prev_pair_bin][curr_pair_bin];
    prev_pair_bin = curr_pair_bin;
  }

  double chi2_1d = 0;
  for (int c : counts_1d) {
    const double diff = c - kExpected;
    chi2_1d += (diff * diff) / kExpected;
  }
  double chi2_2d = 0;
  for (int i = 0; i < kPairDim; ++i) {
    for (int j = 0; j < kPairDim; ++j) {
      const double diff = counts_2d[i][j] - kExpected;
      chi2_2d += (diff * diff) / kExpected;
    }
  }
  // For df = 63, E[chi^2] = 63, SD = sqrt(126) ~= 11.2.
  EXPECT_THAT(chi2_1d, IsBetween(20.0, 120.0));
  EXPECT_THAT(chi2_2d, IsBetween(20.0, 120.0));
}

// The generator needs to be available as a thread_local and as a static
// variable.
TEST(ExponentialBiasedTest, InitializationModes) {
  ABSL_CONST_INIT static ExponentialBiased eb_static;
  EXPECT_THAT(eb_static.GetSkipCount(2), Ge(0));

#ifdef ABSL_HAVE_THREAD_LOCAL
  thread_local ExponentialBiased eb_thread;
  EXPECT_THAT(eb_thread.GetSkipCount(2), Ge(0));
#endif

  ExponentialBiased eb_stack;
  EXPECT_THAT(eb_stack.GetSkipCount(2), Ge(0));
}

}  // namespace
}  // namespace profiling_internal
ABSL_NAMESPACE_END
}  // namespace absl
