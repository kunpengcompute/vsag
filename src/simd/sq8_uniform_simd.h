/*
    Copyright 2026 Huawei Technologies Co., Ltd.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>

#include "simd_marco.h"

namespace vsag {

#define DECLARE_SQ8_UNIFORM_FUNCTIONS(ns)                          \
    namespace ns {                                                 \
    float                                                          \
    SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,       \
                             const uint8_t* RESTRICT codes2,       \
                             uint64_t dim);                        \
    void                                                           \
    SQ8UniformComputeCodesIPBatch(const uint8_t* RESTRICT query,   \
                                  const uint8_t* RESTRICT codes,   \
                                  uint64_t dim,                    \
                                  uint64_t n_codes,                \
                                  uint64_t code_stride,            \
                                  float* RESTRICT out);            \
    void                                                           \
    SQ8UniformComputeCodesIPBatch4(const uint8_t* RESTRICT query,  \
                                   const uint8_t* RESTRICT codes1, \
                                   const uint8_t* RESTRICT codes2, \
                                   const uint8_t* RESTRICT codes3, \
                                   const uint8_t* RESTRICT codes4, \
                                   uint64_t dim,                   \
                                   float* RESTRICT out);           \
    }  // namespace ns
DECLARE_SQ8_UNIFORM_FUNCTIONS(generic)
DECLARE_SQ8_UNIFORM_FUNCTIONS(sse)
DECLARE_SQ8_UNIFORM_FUNCTIONS(avx)
DECLARE_SQ8_UNIFORM_FUNCTIONS(avx2)
DECLARE_SQ8_UNIFORM_FUNCTIONS(avx512)
DECLARE_SQ8_UNIFORM_FUNCTIONS(amx)
DECLARE_SQ8_UNIFORM_FUNCTIONS(neon)
DECLARE_SQ8_UNIFORM_FUNCTIONS(sve)

#undef DECLARE_SQ8_UNIFORM_FUNCTIONS

using SQ8UniformComputeCodesType = float (*)(const uint8_t* RESTRICT codes1,
                                             const uint8_t* RESTRICT codes2,
                                             uint64_t dim);
extern SQ8UniformComputeCodesType SQ8UniformComputeCodesIP;

// Batch inner-product: one query against n_codes SQ8-uniform codes
// of `dim` bytes each, with row stride `code_stride` (>= dim).
// `code_stride` lets callers pass codes that have trailing per-row
// metadata bytes (e.g. norm/sum) without copying.  Use code_stride==dim
// for tightly-packed input.  Implementations may require dim % some_factor
// == 0 only when it would change semantics; the generic implementation
// handles every dim.
using SQ8UniformComputeCodesIPBatchType = void (*)(const uint8_t* RESTRICT query,
                                                   const uint8_t* RESTRICT codes,
                                                   uint64_t dim,
                                                   uint64_t n_codes,
                                                   uint64_t code_stride,
                                                   float* RESTRICT out);
extern SQ8UniformComputeCodesIPBatchType SQ8UniformComputeCodesIPBatch;

// Batch4 inner-product: one query against 4 SQ8-uniform codes given as
// independent pointers (e.g. HGraph reorder passes neighbor-list entries
// that are not contiguous).  Writes the 4 raw u8*u8 dot products to
// out[0..3]; callers fold in the per-code metric correction (norm/sum),
// mirroring the single-code SQ8UniformComputeCodesIP entry point.
using SQ8UniformComputeCodesIPBatch4Type = void (*)(const uint8_t* RESTRICT query,
                                                    const uint8_t* RESTRICT codes1,
                                                    const uint8_t* RESTRICT codes2,
                                                    const uint8_t* RESTRICT codes3,
                                                    const uint8_t* RESTRICT codes4,
                                                    uint64_t dim,
                                                    float* RESTRICT out);
extern SQ8UniformComputeCodesIPBatch4Type SQ8UniformComputeCodesIPBatch4;
}  // namespace vsag
