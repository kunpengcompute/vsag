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

// SQ8 quantized distance kernels.
//
// SQ8ComputeIP:  sum += query[i] * (codes[i]/255 * diff[i] + lower_bound[i])
// SQ8ComputeL2Sqr: sum += (query[i] - (codes[i]/255 * diff[i] + lower_bound[i]))^2
// SQ8ComputeCodesIP: sum += adj1[i] * adj2[i]   where adj = codes/255*diff+lb
// SQ8ComputeCodesL2Sqr: sum += (adj1[i] - adj2[i])^2
//
// Parameterized on SQ8Traits<ISA>, which must expose:
//   FloatVec        - SIMD float vector type
//   Width           - number of float elements per vector
//   load_u8_as_float(const uint8_t* p) -> FloatVec
//                    Load Width uint8 values, zero-extend, convert to float
//   zero()          -> FloatVec
//   set1(float)     -> FloatVec
//   load(const float* p) -> FloatVec
//   mul(FloatVec, FloatVec) -> FloatVec
//   add(FloatVec, FloatVec) -> FloatVec
//   sub(FloatVec, FloatVec) -> FloatVec
//   fmadd(FloatVec a, FloatVec b, FloatVec c) -> FloatVec  (a*b+c)
//   reduce_add(FloatVec) -> float

namespace vsag::simd {

// --- SQ8ComputeIP ---

template <typename T>
inline float
SQ8ComputeIPImpl(const float* query,
                 const uint8_t* codes,
                 const float* lower_bound,
                 const float* diff,
                 uint64_t dim,
                 float (*fallback)(const float*,
                                   const uint8_t*,
                                   const float*,
                                   const float*,
                                   uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(query, codes, lower_bound, diff, dim);
        }
    }

    V sum = T::zero();
    V inv255 = T::set1(1.0f / 255.0f);

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V code_floats = T::load_u8_as_float(codes + i);
        V query_vals = T::load(query + i);
        V diff_vals = T::load(diff + i);
        V lb_vals = T::load(lower_bound + i);

        V normalized = T::mul(code_floats, inv255);
        V adjusted = T::fmadd(normalized, diff_vals, lb_vals);
        sum = T::fmadd(query_vals, adjusted, sum);
    }

    float result = T::reduce_add(sum);

    if constexpr (W > 1) {
        if (i < dim) {
            result += fallback(query + i, codes + i, lower_bound + i, diff + i, dim - i);
        }
    }
    return result;
}

// --- SQ8ComputeL2Sqr ---

template <typename T>
inline float
SQ8ComputeL2SqrImpl(const float* query,
                    const uint8_t* codes,
                    const float* lower_bound,
                    const float* diff,
                    uint64_t dim,
                    float (*fallback)(const float*,
                                      const uint8_t*,
                                      const float*,
                                      const float*,
                                      uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(query, codes, lower_bound, diff, dim);
        }
    }

    V sum = T::zero();
    V inv255 = T::set1(1.0f / 255.0f);

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V code_floats = T::load_u8_as_float(codes + i);
        V query_vals = T::load(query + i);
        V diff_vals = T::load(diff + i);
        V lb_vals = T::load(lower_bound + i);

        V normalized = T::mul(code_floats, inv255);
        V adjusted = T::fmadd(normalized, diff_vals, lb_vals);
        V d = T::sub(query_vals, adjusted);
        sum = T::fmadd(d, d, sum);
    }

    float result = T::reduce_add(sum);

    if constexpr (W > 1) {
        if (i < dim) {
            result += fallback(query + i, codes + i, lower_bound + i, diff + i, dim - i);
        }
    }
    return result;
}

// --- SQ8ComputeCodesIP ---

template <typename T>
inline float
SQ8ComputeCodesIPImpl(const uint8_t* codes1,
                      const uint8_t* codes2,
                      const float* lower_bound,
                      const float* diff,
                      uint64_t dim,
                      float (*fallback)(const uint8_t*,
                                        const uint8_t*,
                                        const float*,
                                        const float*,
                                        uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(codes1, codes2, lower_bound, diff, dim);
        }
    }

    V sum = T::zero();
    V inv255 = T::set1(1.0f / 255.0f);

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V c1_floats = T::load_u8_as_float(codes1 + i);
        V c2_floats = T::load_u8_as_float(codes2 + i);
        V diff_vals = T::load(diff + i);
        V lb_vals = T::load(lower_bound + i);

        V adj1 = T::fmadd(T::mul(c1_floats, inv255), diff_vals, lb_vals);
        V adj2 = T::fmadd(T::mul(c2_floats, inv255), diff_vals, lb_vals);
        sum = T::fmadd(adj1, adj2, sum);
    }

    float result = T::reduce_add(sum);

    if constexpr (W > 1) {
        if (i < dim) {
            result += fallback(codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
        }
    }
    return result;
}

// --- SQ8ComputeCodesL2Sqr ---

template <typename T>
inline float
SQ8ComputeCodesL2SqrImpl(const uint8_t* codes1,
                         const uint8_t* codes2,
                         const float* lower_bound,
                         const float* diff,
                         uint64_t dim,
                         float (*fallback)(const uint8_t*,
                                           const uint8_t*,
                                           const float*,
                                           const float*,
                                           uint64_t) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            return fallback(codes1, codes2, lower_bound, diff, dim);
        }
    }

    V sum = T::zero();
    V inv255 = T::set1(1.0f / 255.0f);

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V c1_floats = T::load_u8_as_float(codes1 + i);
        V c2_floats = T::load_u8_as_float(codes2 + i);
        V diff_vals = T::load(diff + i);
        V lb_vals = T::load(lower_bound + i);

        V adj1 = T::fmadd(T::mul(c1_floats, inv255), diff_vals, lb_vals);
        V adj2 = T::fmadd(T::mul(c2_floats, inv255), diff_vals, lb_vals);
        V d = T::sub(adj1, adj2);
        sum = T::fmadd(d, d, sum);
    }

    float result = T::reduce_add(sum);

    if constexpr (W > 1) {
        if (i < dim) {
            result += fallback(codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
        }
    }
    return result;
}

// --- SQ8ComputeIPBatch4 / SQ8ComputeL2SqrBatch4 ---

enum class SQ8Batch4Kind { IP, L2 };

// One query against four SQ8 codes. The four codes share the query /
// lower_bound / diff loads, giving 4x reuse of every non-code cacheline.
// Results are accumulated into r1..r4 (the caller must initialise them
// before invocation, e.g. to 0).
template <typename T, SQ8Batch4Kind Kind>
inline void
SQ8ComputeBatch4Impl(const float* RESTRICT query,
                     uint64_t dim,
                     const uint8_t* RESTRICT c1,
                     const uint8_t* RESTRICT c2,
                     const uint8_t* RESTRICT c3,
                     const uint8_t* RESTRICT c4,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     float& r1,
                     float& r2,
                     float& r3,
                     float& r4,
                     void (*fallback)(const float* RESTRICT,
                                      uint64_t,
                                      const uint8_t* RESTRICT,
                                      const uint8_t* RESTRICT,
                                      const uint8_t* RESTRICT,
                                      const uint8_t* RESTRICT,
                                      const float* RESTRICT,
                                      const float* RESTRICT,
                                      float&,
                                      float&,
                                      float&,
                                      float&) = nullptr) {
    using V = typename T::FloatVec;
    constexpr int W = T::Width;

    if constexpr (W > 1) {
        if (dim < static_cast<uint64_t>(W)) {
            fallback(query, dim, c1, c2, c3, c4, lower_bound, diff, r1, r2, r3, r4);
            return;
        }
    }

    V s1 = T::zero();
    V s2 = T::zero();
    V s3 = T::zero();
    V s4 = T::zero();
    V inv255 = T::set1(1.0f / 255.0f);

    uint64_t i = 0;
    for (; i + W <= dim; i += W) {
        V q = T::load(query + i);
        V lb = T::load(lower_bound + i);
        V d = T::load(diff + i);

        V adj1 = T::fmadd(T::mul(T::load_u8_as_float(c1 + i), inv255), d, lb);
        V adj2 = T::fmadd(T::mul(T::load_u8_as_float(c2 + i), inv255), d, lb);
        V adj3 = T::fmadd(T::mul(T::load_u8_as_float(c3 + i), inv255), d, lb);
        V adj4 = T::fmadd(T::mul(T::load_u8_as_float(c4 + i), inv255), d, lb);

        if constexpr (Kind == SQ8Batch4Kind::IP) {
            s1 = T::fmadd(q, adj1, s1);
            s2 = T::fmadd(q, adj2, s2);
            s3 = T::fmadd(q, adj3, s3);
            s4 = T::fmadd(q, adj4, s4);
        } else {
            V e1 = T::sub(q, adj1);
            V e2 = T::sub(q, adj2);
            V e3 = T::sub(q, adj3);
            V e4 = T::sub(q, adj4);
            s1 = T::fmadd(e1, e1, s1);
            s2 = T::fmadd(e2, e2, s2);
            s3 = T::fmadd(e3, e3, s3);
            s4 = T::fmadd(e4, e4, s4);
        }
    }

    r1 += T::reduce_add(s1);
    r2 += T::reduce_add(s2);
    r3 += T::reduce_add(s3);
    r4 += T::reduce_add(s4);

    if constexpr (W > 1) {
        if (dim > i) {
            fallback(query + i,
                     dim - i,
                     c1 + i,
                     c2 + i,
                     c3 + i,
                     c4 + i,
                     lower_bound + i,
                     diff + i,
                     r1,
                     r2,
                     r3,
                     r4);
        }
    }
}

}  // namespace vsag::simd
