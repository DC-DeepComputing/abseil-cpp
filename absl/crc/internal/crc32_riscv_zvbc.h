// Copyright 2026 The Abseil Authors.
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

#ifndef ABSL_CRC_INTERNAL_CRC32_RISCV_ZVBC_H_
#define ABSL_CRC_INTERNAL_CRC32_RISCV_ZVBC_H_

#include "absl/base/config.h"
#include <cstddef>
#include <cstdint>

#if defined(__riscv) && defined(__riscv_xlen) && __riscv_xlen == 64 && \
    defined(__riscv_zvbc) && defined(ABSL_IS_LITTLE_ENDIAN) && ABSL_IS_LITTLE_ENDIAN
#define ABSL_CRC_INTERNAL_HAVE_RISCV_ZVBC 1

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace crc_internal {
namespace riscv_zvbc {

// Hardware-accelerated CRC32C using RISC-V Zvbc vector carry-less multiply.
// Bit-exact with the portable slicing-by-8 implementation.  Compile-time
// gated on Zvbc (mirrors x86 PCLMULQDQ / arm PMULL gating).
uint32_t Extend(uint32_t crc, const char* buf, size_t len);

}  // namespace riscv_zvbc
}  // namespace crc_internal
ABSL_NAMESPACE_END
}  // namespace absl

#endif  // Zvbc + little-endian riscv64
#endif  // ABSL_CRC_INTERNAL_CRC32_RISCV_ZVBC_H_
