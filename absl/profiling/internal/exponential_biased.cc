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

#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

#include "absl/base/attributes.h"
#include "absl/base/config.h"
#include "absl/base/optimization.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace profiling_internal {

// The algorithm generates a random number between 0 and 1 and applies the
// inverse cumulative distribution function for an exponential. Specifically:
// Let m be the inverse of the sample period, then the probability
// distribution function is m*exp(-mx) so the CDF is
// p = 1 - exp(-mx), so
// q = 1 - p = exp(-mx)
// log_e(q) = -mx
// -log_e(q)/m = x
// log_2(q) * (-log_e(2) * 1/m) = x
// In the code, q is actually in the range 1 to 2**26, hence the -26 below
int64_t ExponentialBiased::GetSkipCount(int64_t mean) {
  if (ABSL_PREDICT_FALSE(!initialized_)) {
    Initialize();
  }

  uint64_t rng = NextRandom(rng_);
  rng_ = rng;

  // Take the top 26 bits as the random number
  // (This plus the 1<<58 sampling bound give a max possible step of
  // 5194297183973780480 bytes.)
  // The uint32_t cast is to prevent a (hard-to-reproduce) NAN
  // under piii debug for some binaries.
  double q = static_cast<uint32_t>(rng >> (kPrngNumBits - 26)) + 1.0;
  // Put the computed p-value through the CDF of a geometric.
  double interval = bias_ + (std::log2(q) - 26) * (-std::log(2.0) * mean);
  // Very large values of interval overflow int64_t. To avoid that, we will
  // cheat and clamp any huge values to (int64_t max)/2. This is a potential
  // source of bias, but the mean would need to be such a large value that it's
  // not likely to come up. For example, with a mean of 1e18, the probability of
  // hitting this condition is about 1/1000. For a mean of 1e17, standard
  // calculators claim that this event won't happen.
  if (interval > static_cast<double>(std::numeric_limits<int64_t>::max() / 2)) {
    // Assume huge values are bias neutral, retain bias for next call.
    return std::numeric_limits<int64_t>::max() / 2;
  }
  double value = std::rint(interval);
  bias_ = interval - value;
  return static_cast<int64_t>(value);
}

int64_t ExponentialBiased::GetStride(int64_t mean) {
  return GetSkipCount(mean - 1) + 1;
}

int64_t ExponentialBiased::ConfirmSampleSlow(SamplingState& state,
                                             int64_t period, int64_t step,
                                             int64_t disabled_stride) {
  if (period <= 0) {
    state = {disabled_stride > 0 ? disabled_stride : (int64_t{1} << 16), 0};
    return 0;
  }
  if (ABSL_PREDICT_FALSE(step < 1)) {
    step = 1;
  }
  if (state.sample_period <= 0) {
    state.sample_period = period;
    const int64_t initial_stride = (period == 1) ? 1 : GetStride(period);
    state.next_sample = initial_stride - step;
    if (ABSL_PREDICT_TRUE(state.next_sample > 0)) {
      return 0;
    }
  }

  int64_t weight;
  const int64_t max_int64 = (std::numeric_limits<int64_t>::max)();
  if (ABSL_PREDICT_FALSE(state.next_sample < 0 &&
                         state.sample_period > max_int64 + state.next_sample)) {
    weight = max_int64;
  } else {
    weight = state.sample_period - state.next_sample;
  }

  state.sample_period = period;
  state.next_sample = (period == 1) ? 1 : GetStride(period);
  return weight;
}

void ExponentialBiased::Initialize() {
  // We don't get well distributed numbers from `this` so we call NextRandom() a
  // bunch to mush the bits around. We use a global_rand to handle the case
  // where the same thread (by memory address) gets created and destroyed
  // repeatedly.
  ABSL_CONST_INIT static std::atomic<uint32_t> global_rand(0);
  uint64_t r = reinterpret_cast<uint64_t>(this) +
               global_rand.fetch_add(1, std::memory_order_relaxed);
  for (int i = 0; i < 20; ++i) {
    r = NextRandom(r);
  }
  rng_ = r;
  initialized_ = true;
}

}  // namespace profiling_internal
ABSL_NAMESPACE_END
}  // namespace absl
