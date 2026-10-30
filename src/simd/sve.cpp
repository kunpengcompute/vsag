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

#include "simd/int8_simd.h"
#if defined(ENABLE_SVE)
#include <arm_sve.h>
#endif

#include <alloca.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include "simd.h"
constexpr auto
generate_bit_lookup_table() {
    std::array<std::array<uint8_t, 8>, 256> table{};
    for (int byte_value = 0; byte_value < 256; ++byte_value) {
        for (int bit_pos = 0; bit_pos < 8; ++bit_pos) {
            table[byte_value][bit_pos] = ((byte_value >> bit_pos) & 1) ? 1 : 0;
        }
    }
    return table;
}

static constexpr auto g_bit_lookup_table = generate_bit_lookup_table();

namespace vsag::sve {

namespace {

float
BinaryIPFromLookupLocal(const float* lookup, const uint8_t* bits, uint64_t dim) {
    const uint64_t plane_bytes = (dim + 7) / 8;
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0F);
    uint64_t block = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(block, plane_bytes);
    while (svptest_first(svptrue_b32(), predicate)) {
        svuint32_t block_ids = svadd_n_u32_x(
            predicate, svindex_u32(0, 1), static_cast<uint32_t>(block));
        svuint32_t lookup_ids = svlsl_n_u32_x(predicate, block_ids, 8);
        lookup_ids = svadd_u32_x(predicate, lookup_ids, svld1ub_u32(predicate, bits + block));

        svfloat32_t values = svld1_gather_u32index_f32(predicate, lookup, lookup_ids);
        sum = svadd_f32_m(predicate, sum, values);

        block += step;
        predicate = svwhilelt_b32(block, plane_bytes);
    }
    return svaddv_f32(svptrue_b32(), sum);
#else
    float result = 0.0F;
    for (uint64_t block = 0; block < plane_bytes; ++block) {
        result += lookup[(block << 8) + bits[block]];
    }
    return result;
#endif
}

void
BinaryIPBatch4FromLookupLocal(const float* lookup,
                              const uint8_t* bits1,
                              const uint8_t* bits2,
                              const uint8_t* bits3,
                              const uint8_t* bits4,
                              uint64_t dim,
                              float* results) {
    const uint64_t plane_bytes = (dim + 7) / 8;
    uint64_t sq = 0;
    float res1 = 0, res2 = 0, res3 = 0, res4 = 0;
    const float* table = lookup;
    for(; sq + 3 < plane_bytes; sq += 4) {
        uint8_t idx1[4], idx2[4], idx3[4], idx4[4];
        memcpy(idx1, bits1 + sq, 4);
        memcpy(idx2, bits2 + sq, 4);
        memcpy(idx3, bits3 + sq, 4);
        memcpy(idx4, bits4 + sq, 4);
        res1 += table[idx1[0]];
        res2 += table[idx2[0]];
        res3 += table[idx3[0]];
        res4 += table[idx4[0]];
        table += 256;
        res1 += table[idx1[1]];
        res2 += table[idx2[1]];
        res3 += table[idx3[1]];
        res4 += table[idx4[1]];
        table += 256;
        res1 += table[idx1[2]];
        res2 += table[idx2[2]];
        res3 += table[idx3[2]];
        res4 += table[idx4[2]];
        table += 256;
        res1 += table[idx1[3]];
        res2 += table[idx2[3]];
        res3 += table[idx3[3]];
        res4 += table[idx4[3]];
        table += 256;
    }
    for (; sq < plane_bytes; ++sq) {
        res1 += table[bits1[sq]];
        res2 += table[bits2[sq]];
        res3 += table[bits3[sq]];
        res4 += table[bits4[sq]];
        table += 256;
    }
    results[0] = res1;
    results[1] = res2;
    results[2] = res3;
    results[3] = res4;
}

}  // namespace

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::FP32ComputeL2Sqr(pVect1, pVect2, qty);
}
float
INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::INT8ComputeL2Sqr(pVect1, pVect2, qty);
}
float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::FP32ComputeIP(pVect1, pVect2, qty);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - sve::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;
    auto qty = *((uint64_t*)qty_ptr);
    return sve::INT8ComputeIP(pVect1, pVect2, qty);
}

float
INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svint64_t sum = svdup_s64(0);
    uint64_t i = 0;
    const uint64_t step = svcnth();

    svbool_t predicate = svwhilelt_b16(i, dim);
    do {
        svint16_t vec_query = svld1sb_s16(predicate, query + i);
        svint16_t vec_codes = svld1sb_s16(predicate, codes + i);

        svint16_t diff = svsub_s16_x(predicate, vec_query, vec_codes);

        sum = svdot_s64(sum, diff, diff);

        i += step;
        predicate = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b16(), predicate));

    return static_cast<float>(svaddv_s64(svptrue_b64(), sum));
#else
    return neon::INT8ComputeL2Sqr(query, codes, dim);
#endif
}

float
INT8ComputeIP(const int8_t* __restrict query, const int8_t* __restrict codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svint32_t sum = svdup_s32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t predicate = svwhilelt_b8(i, dim);
    do {
        svint8_t vec1 = svld1_s8(predicate, query + i);
        svint8_t vec2 = svld1_s8(predicate, codes + i);
        sum = svdot_s32(sum, vec1, vec2);
        i += step;
        predicate = svwhilelt_b8(i, dim);
    } while (svptest_first(svptrue_b8(), predicate));

    return static_cast<float>(svaddv_s32(svptrue_b32(), sum));
#else
    return neon::INT8ComputeIP(query, codes, dim);
#endif
}

float
INT8InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -sve::INT8InnerProduct(pVect1, pVect2, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_SVE)
    const auto* float_centers = (const float*)single_dim_centers;
    auto* float_result = (float*)result;
    uint64_t num_floats_per_vector = svcntw();
    svfloat32_t value = svdup_f32(single_dim_val);
    int i = 0;
    do {
        svbool_t predicate = svwhilelt_b32(i, 256);
        svfloat32_t centers = svld1_f32(predicate, float_centers + i);
        svfloat32_t results = svld1_f32(predicate, float_result + i);
        svfloat32_t diff = svsub_f32_m(predicate, centers, value);
        results = svmad_f32_m(predicate, diff, diff, results);
        svst1_f32(predicate, float_result + i, results);
        i += num_floats_per_vector;
    } while (i < 256);
#else
    neon::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;

    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t codes_vec = svld1_f32(predicate, codes + i);

        sum = svmla_f32_m(predicate, sum, query_vec, codes_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t codes_vec = svld1_f32(predicate, codes + i);

        svfloat32_t diff = svsub_f32_z(predicate, query_vec, codes_vec);

        sum = svmla_f32_m(predicate, sum, diff, diff);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP32ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP32SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const float* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
    return neon::FP32SparseAccumulate(dists, ids, vals, query_val, num);
}

void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4) {
#if defined(ENABLE_SVE)

    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);
    svfloat32_t sum4 = svdup_f32(0.0f);

    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);

        svfloat32_t codes1_vec = svld1_f32(predicate, codes1 + i);
        sum1 = svmla_f32_m(predicate, sum1, query_vec, codes1_vec);

        svfloat32_t codes2_vec = svld1_f32(predicate, codes2 + i);
        sum2 = svmla_f32_m(predicate, sum2, query_vec, codes2_vec);

        svfloat32_t codes3_vec = svld1_f32(predicate, codes3 + i);
        sum3 = svmla_f32_m(predicate, sum3, query_vec, codes3_vec);

        svfloat32_t codes4_vec = svld1_f32(predicate, codes4 + i);
        sum4 = svmla_f32_m(predicate, sum4, query_vec, codes4_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    result1 = svaddv_f32(svptrue_b32(), sum1);
    result2 = svaddv_f32(svptrue_b32(), sum2);
    result3 = svaddv_f32(svptrue_b32(), sum3);
    result4 = svaddv_f32(svptrue_b32(), sum4);
#else
    neon::FP32ComputeIPBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4) {
#if defined(ENABLE_SVE)
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);
    svfloat32_t sum4 = svdup_f32(0.0f);

    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t query_vec = svld1_f32(predicate, query + i);

        svfloat32_t codes1_vec = svld1_f32(predicate, codes1 + i);
        svfloat32_t diff1 = svsub_f32_z(predicate, query_vec, codes1_vec);
        sum1 = svmla_f32_m(predicate, sum1, diff1, diff1);

        svfloat32_t codes2_vec = svld1_f32(predicate, codes2 + i);
        svfloat32_t diff2 = svsub_f32_z(predicate, query_vec, codes2_vec);
        sum2 = svmla_f32_m(predicate, sum2, diff2, diff2);

        svfloat32_t codes3_vec = svld1_f32(predicate, codes3 + i);
        svfloat32_t diff3 = svsub_f32_z(predicate, query_vec, codes3_vec);
        sum3 = svmla_f32_m(predicate, sum3, diff3, diff3);

        svfloat32_t codes4_vec = svld1_f32(predicate, codes4 + i);
        svfloat32_t diff4 = svsub_f32_z(predicate, query_vec, codes4_vec);
        sum4 = svmla_f32_m(predicate, sum4, diff4, diff4);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    result1 = svaddv_f32(svptrue_b32(), sum1);
    result2 = svaddv_f32(svptrue_b32(), sum2);
    result3 = svaddv_f32(svptrue_b32(), sum3);
    result4 = svaddv_f32(svptrue_b32(), sum4);
#else
    neon::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svsub_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svadd_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svmul_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);
        svfloat32_t y_vec = svld1_f32(predicate, y + i);
        svfloat32_t result = svdiv_f32_z(predicate, x_vec, y_vec);
        svst1_f32(predicate, z + i, result);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(predicate, x + i);

        sum = svadd_f32_m(predicate, sum, x_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP32ReduceAdd(x, dim);
#endif
}

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t query_u32 = svld1uh_u32(predicate, &query_bf16[i]);
        svuint32_t codes_u32 = svld1uh_u32(predicate, &codes_bf16[i]);

        query_u32 = svlsl_n_u32_x(predicate, query_u32, 16);
        codes_u32 = svlsl_n_u32_x(predicate, codes_u32, 16);

        svfloat32_t query_f32 = svreinterpret_f32_u32(query_u32);
        svfloat32_t codes_f32 = svreinterpret_f32_u32(codes_u32);

        sum = svmla_f32_x(predicate, sum, query_f32, codes_f32);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t query_u32 = svld1uh_u32(predicate, &query_bf16[i]);
        svuint32_t codes_u32 = svld1uh_u32(predicate, &codes_bf16[i]);

        query_u32 = svlsl_n_u32_x(predicate, query_u32, 16);
        codes_u32 = svlsl_n_u32_x(predicate, codes_u32, 16);

        svfloat32_t query_f32 = svreinterpret_f32_u32(query_u32);
        svfloat32_t codes_f32 = svreinterpret_f32_u32(codes_u32);

        svfloat32_t diff = svsub_f32_x(predicate, query_f32, codes_f32);
        sum = svmla_f32_x(predicate, sum, diff, diff);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    auto* query_fp16 = reinterpret_cast<const __fp16*>(query);
    auto* codes_fp16 = reinterpret_cast<const __fp16*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcnth();
    svbool_t predicate = svwhilelt_b16(i, dim);
    do {
        svfloat16_t query_f16 = svld1_f16(predicate, &query_fp16[i]);
        svfloat16_t codes_f16 = svld1_f16(predicate, &codes_fp16[i]);

        svbool_t half_predicate = svptrue_pat_b16(SV_POW2);
        svfloat32_t query_f32_low = svcvt_f32_f16_x(half_predicate, query_f16);
        svfloat32_t codes_f32_low = svcvt_f32_f16_x(half_predicate, codes_f16);

        svfloat16_t query_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(query_f16), 16));
        svfloat16_t codes_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(codes_f16), 16));
        svfloat32_t query_f32_high = svcvt_f32_f16_x(half_predicate, query_f16_high);
        svfloat32_t codes_f32_high = svcvt_f32_f16_x(half_predicate, codes_f16_high);

        sum = svmla_f32_x(svptrue_b32(), sum, query_f32_low, codes_f32_low);
        sum = svmla_f32_x(svptrue_b32(), sum, query_f32_high, codes_f32_high);
        i += step;
        predicate = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)

    auto* query_fp16 = reinterpret_cast<const __fp16*>(query);
    auto* codes_fp16 = reinterpret_cast<const __fp16*>(codes);

    svfloat32_t sum = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t step = svcnth();
    svbool_t predicate = svwhilelt_b16(i, dim);
    do {
        svfloat16_t query_f16 = svld1_f16(predicate, &query_fp16[i]);
        svfloat16_t codes_f16 = svld1_f16(predicate, &codes_fp16[i]);

        svbool_t half_predicate = svptrue_pat_b16(SV_POW2);
        svfloat32_t query_f32_low = svcvt_f32_f16_x(half_predicate, query_f16);
        svfloat32_t codes_f32_low = svcvt_f32_f16_x(half_predicate, codes_f16);

        svfloat16_t query_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(query_f16), 16));
        svfloat16_t codes_f16_high = svreinterpret_f16_u32(
            svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(codes_f16), 16));
        svfloat32_t query_f32_high = svcvt_f32_f16_x(half_predicate, query_f16_high);
        svfloat32_t codes_f32_high = svcvt_f32_f16_x(half_predicate, codes_f16_high);

        svfloat32_t diff_low = svsub_f32_x(svptrue_b32(), query_f32_low, codes_f32_low);
        svfloat32_t diff_high = svsub_f32_x(svptrue_b32(), query_f32_high, codes_f32_high);

        sum = svmla_f32_x(svptrue_b32(), sum, diff_low, diff_low);
        sum = svmla_f32_x(svptrue_b32(), sum, diff_high, diff_high);
        i += step;
        predicate = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::FP16ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP16SparseAccumulate(float* RESTRICT dists,
                     const uint16_t* RESTRICT ids,
                     const uint16_t* RESTRICT vals,
                     float query_val,
                     uint32_t num) {
    return neon::FP16SparseAccumulate(dists, ids, vals, query_val, num);
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes_u32 = svld1ub_u32(predicate, codes + i);
        svfloat32_t codes_f32 = svcvt_f32_u32_z(predicate, codes_u32);

        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes_f32, scale_factor), diff_vec);

        sum = svmla_f32_m(predicate, sum, query_vec, dequantized);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes_u32 = svld1ub_u32(predicate, codes + i);
        svfloat32_t codes_f32 = svcvt_f32_u32_z(predicate, codes_u32);

        svfloat32_t query_vec = svld1_f32(predicate, query + i);
        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes_f32, scale_factor), diff_vec);
        svfloat32_t delta = svsub_f32_z(predicate, query_vec, dequantized);
        sum = svmla_f32_m(predicate, sum, delta, delta);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes1_u32 = svld1ub_u32(predicate, codes1 + i);
        svfloat32_t codes1_f32 = svcvt_f32_u32_z(predicate, codes1_u32);
        svuint32_t codes2_u32 = svld1ub_u32(predicate, codes2 + i);
        svfloat32_t codes2_f32 = svcvt_f32_u32_z(predicate, codes2_u32);

        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized1 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes1_f32, scale_factor), diff_vec);
        svfloat32_t dequantized2 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes2_f32, scale_factor), diff_vec);

        sum = svmla_f32_m(predicate, sum, dequantized1, dequantized2);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t codes1_u32 = svld1ub_u32(predicate, codes1 + i);
        svfloat32_t codes1_f32 = svcvt_f32_u32_z(predicate, codes1_u32);
        svuint32_t codes2_u32 = svld1ub_u32(predicate, codes2 + i);
        svfloat32_t codes2_f32 = svcvt_f32_u32_z(predicate, codes2_u32);

        svfloat32_t lower_bound_vec = svld1_f32(predicate, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(predicate, diff + i);

        svfloat32_t dequantized1 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes1_f32, scale_factor), diff_vec);
        svfloat32_t dequantized2 = svmla_f32_m(
            predicate, lower_bound_vec, svmul_f32_m(predicate, codes2_f32, scale_factor), diff_vec);

        svfloat32_t delta = svsub_f32_z(predicate, dequantized1, dequantized2);
        sum = svmla_f32_m(predicate, sum, delta, delta);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

void
SQ8SparseAccumulate(float* RESTRICT dists,
                    const uint16_t* RESTRICT ids,
                    const uint8_t* RESTRICT vals,
                    float query_val,
                    uint32_t num) {
    return neon::SQ8SparseAccumulate(dists, ids, vals, query_val, num);
}

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    uint64_t i = 0;
    const svbool_t predicate = svwhilelt_b32(i, dim);

    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t query_pair = svld2_f32(predicate, &query[i]);
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t query_even = svget2_f32(query_pair, 0);
        svfloat32_t query_odd = svget2_f32(query_pair, 1);
        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes = svld1ub_u32(predicate, &codes[i / 2]);
        svuint32_t codes_even_u32 = svand_n_u32_x(predicate, packed_codes, 0x0F);
        svuint32_t codes_odd_u32 = svlsr_n_u32_x(predicate, packed_codes, 4);
        svfloat32_t codes_even_f32 = svcvt_f32_u32_x(predicate, codes_even_u32);
        svfloat32_t codes_odd_f32 = svcvt_f32_u32_x(predicate, codes_odd_u32);

        svfloat32_t dequantized_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes_odd_f32, scale_factor),
                        diff_odd);

        sum = svmla_f32_x(predicate, sum, query_even, dequantized_even);
        sum = svmla_f32_x(predicate, sum, query_odd, dequantized_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeIP(&query[i], &codes[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    const svbool_t predicate = svptrue_b32();

    uint64_t i = 0;
    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t query_pair = svld2_f32(predicate, &query[i]);
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t query_even = svget2_f32(query_pair, 0);
        svfloat32_t query_odd = svget2_f32(query_pair, 1);
        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes = svld1ub_u32(predicate, &codes[i / 2]);
        svuint32_t codes_even_u32 = svand_n_u32_x(predicate, packed_codes, 0x0F);
        svuint32_t codes_odd_u32 = svlsr_n_u32_x(predicate, packed_codes, 4);
        svfloat32_t codes_even_f32 = svcvt_f32_u32_x(predicate, codes_even_u32);
        svfloat32_t codes_odd_f32 = svcvt_f32_u32_x(predicate, codes_odd_u32);

        svfloat32_t dequantized_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes_odd_f32, scale_factor),
                        diff_odd);

        svfloat32_t delta_even = svsub_f32_x(predicate, query_even, dequantized_even);
        svfloat32_t delta_odd = svsub_f32_x(predicate, query_odd, dequantized_odd);

        sum = svmla_f32_x(predicate, sum, delta_even, delta_even);
        sum = svmla_f32_x(predicate, sum, delta_odd, delta_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeL2Sqr(&query[i], &codes[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    const svbool_t predicate = svptrue_b32();

    uint64_t i = 0;
    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes1 = svld1ub_u32(predicate, &codes1[i / 2]);
        svuint32_t packed_codes2 = svld1ub_u32(predicate, &codes2[i / 2]);

        svuint32_t codes1_even_u32 = svand_n_u32_x(predicate, packed_codes1, 0x0F);
        svuint32_t codes1_odd_u32 = svlsr_n_u32_x(predicate, packed_codes1, 4);
        svuint32_t codes2_even_u32 = svand_n_u32_x(predicate, packed_codes2, 0x0F);
        svuint32_t codes2_odd_u32 = svlsr_n_u32_x(predicate, packed_codes2, 4);

        svfloat32_t codes1_even_f32 = svcvt_f32_u32_x(predicate, codes1_even_u32);
        svfloat32_t codes1_odd_f32 = svcvt_f32_u32_x(predicate, codes1_odd_u32);
        svfloat32_t codes2_even_f32 = svcvt_f32_u32_x(predicate, codes2_even_u32);
        svfloat32_t codes2_odd_f32 = svcvt_f32_u32_x(predicate, codes2_odd_u32);

        svfloat32_t dequantized1_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes1_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized1_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes1_odd_f32, scale_factor),
                        diff_odd);
        svfloat32_t dequantized2_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes2_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized2_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes2_odd_f32, scale_factor),
                        diff_odd);

        sum = svmla_f32_x(predicate, sum, dequantized1_even, dequantized2_even);
        sum = svmla_f32_x(predicate, sum, dequantized1_odd, dequantized2_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeCodesIP(
                   &codes1[i / 2], &codes2[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum = svdup_f32(0.0f);
    const svfloat32_t scale_factor = svdup_f32(1.0f / 15.0f);
    const uint64_t step = svcntw();
    const svbool_t predicate = svptrue_b32();

    uint64_t i = 0;
    for (; i + 2 * step <= dim; i += 2 * step) {
        svfloat32x2_t lower_bound_pair = svld2_f32(predicate, &lower_bound[i]);
        svfloat32x2_t diff_pair = svld2_f32(predicate, &diff[i]);

        svfloat32_t lower_bound_even = svget2_f32(lower_bound_pair, 0);
        svfloat32_t lower_bound_odd = svget2_f32(lower_bound_pair, 1);
        svfloat32_t diff_even = svget2_f32(diff_pair, 0);
        svfloat32_t diff_odd = svget2_f32(diff_pair, 1);

        svuint32_t packed_codes1 = svld1ub_u32(predicate, &codes1[i / 2]);
        svuint32_t packed_codes2 = svld1ub_u32(predicate, &codes2[i / 2]);

        svuint32_t codes1_even_u32 = svand_n_u32_x(predicate, packed_codes1, 0x0F);
        svuint32_t codes1_odd_u32 = svlsr_n_u32_x(predicate, packed_codes1, 4);
        svuint32_t codes2_even_u32 = svand_n_u32_x(predicate, packed_codes2, 0x0F);
        svuint32_t codes2_odd_u32 = svlsr_n_u32_x(predicate, packed_codes2, 4);

        svfloat32_t codes1_even_f32 = svcvt_f32_u32_x(predicate, codes1_even_u32);
        svfloat32_t codes1_odd_f32 = svcvt_f32_u32_x(predicate, codes1_odd_u32);
        svfloat32_t codes2_even_f32 = svcvt_f32_u32_x(predicate, codes2_even_u32);
        svfloat32_t codes2_odd_f32 = svcvt_f32_u32_x(predicate, codes2_odd_u32);

        svfloat32_t dequantized1_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes1_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized1_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes1_odd_f32, scale_factor),
                        diff_odd);
        svfloat32_t dequantized2_even =
            svmla_f32_x(predicate,
                        lower_bound_even,
                        svmul_f32_x(predicate, codes2_even_f32, scale_factor),
                        diff_even);
        svfloat32_t dequantized2_odd =
            svmla_f32_x(predicate,
                        lower_bound_odd,
                        svmul_f32_x(predicate, codes2_odd_f32, scale_factor),
                        diff_odd);

        svfloat32_t delta_even = svsub_f32_x(predicate, dequantized1_even, dequantized2_even);
        svfloat32_t delta_odd = svsub_f32_x(predicate, dequantized1_odd, dequantized2_odd);

        sum = svmla_f32_x(predicate, sum, delta_even, delta_even);
        sum = svmla_f32_x(predicate, sum, delta_odd, delta_odd);
    }

    if (i < dim) {
        return svaddv_f32(predicate, sum) +
               neon::SQ4ComputeCodesL2Sqr(
                   &codes1[i / 2], &codes2[i / 2], &lower_bound[i], &diff[i], dim - i);
    }

    return svaddv_f32(predicate, sum);
#else
    return neon::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}
float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE)
    svuint32_t sum = svdup_u32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb() * 2;
    svbool_t predicate = svwhilelt_b8(i / 2, (dim + 1) / 2);
    do {
        svuint8_t packed_codes1 = svld1_u8(predicate, codes1 + i / 2);
        svuint8_t packed_codes2 = svld1_u8(predicate, codes2 + i / 2);

        svuint8_t codes1_low = svand_u8_z(predicate, packed_codes1, svdup_u8(0x0F));
        svuint8_t codes1_high = svlsr_n_u8_z(predicate, packed_codes1, 4);
        svuint8_t codes2_low = svand_u8_z(predicate, packed_codes2, svdup_u8(0x0F));
        svuint8_t codes2_high = svlsr_n_u8_z(predicate, packed_codes2, 4);

        sum = svdot_u32(sum, codes1_low, codes2_low);
        sum = svdot_u32(sum, codes1_high, codes2_high);

        i += step;
        predicate = svwhilelt_b8(i / 2, (dim + 1) / 2);
    } while (svptest_first(svptrue_b8(), predicate));

    return static_cast<float>(svaddv_u32(svptrue_b32(), sum));
#else
    return neon::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE)
    svuint32_t sum = svdup_u32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t predicate = svwhilelt_b8(i, dim);
    do {
        svuint8_t codes1_vec = svld1_u8(predicate, codes1 + i);
        svuint8_t codes2_vec = svld1_u8(predicate, codes2 + i);

        sum = svdot_u32(sum, codes1_vec, codes2_vec);

        i += step;
        predicate = svwhilelt_b8(i, dim);
    } while (svptest_first(svptrue_b8(), predicate));

    return static_cast<float>(svaddv_u32(svptrue_b32(), sum));
#else
    return neon::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

void
SQ8UniformComputeCodesIPBatch(const uint8_t* RESTRICT query,
                              const uint8_t* RESTRICT codes,
                              uint64_t dim,
                              uint64_t n_codes,
                              uint64_t code_stride,
                              float* RESTRICT out) {
    for (uint64_t i = 0; i < n_codes; ++i) {
        out[i] = sve::SQ8UniformComputeCodesIP(query, codes + i * code_stride, dim);
    }
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
    if (std::isnan(inv_sqrt_d)) {
        return BinaryIPFromLookupLocal(vector, bits, dim);
    }

#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 0.0f;
    }

    auto predicate_array = std::make_unique<uint8_t[]>(dim);

    const uint64_t num_bytes = dim / 8;
    for (uint64_t i = 0; i < num_bytes; ++i) {
        memcpy(&predicate_array[i * 8], g_bit_lookup_table[bits[i]].data(), 8);
    }

    if (dim % 8 != 0) {
        const uint64_t remaining_bits = dim % 8;
        memcpy(&predicate_array[num_bytes * 8],
               g_bit_lookup_table[bits[num_bytes]].data(),
               remaining_bits);
    }

    uint64_t i = 0;
    const uint64_t step = svcntw();
    svfloat32_t sum = svdup_f32(0.0f);

    const svfloat32_t positive_val = inv_sqrt_d < 1e-3 ? svdup_f32(1.0f) : svdup_f32(inv_sqrt_d);
    const svfloat32_t negative_val = inv_sqrt_d < 1e-3 ? svdup_f32(0.0f) : svdup_f32(-inv_sqrt_d);
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t predicate_values = svld1ub_u32(predicate, &predicate_array[i]);

        svbool_t bit_mask = svcmpne_n_u32(predicate, predicate_values, 0);

        svfloat32_t binary_vec = svsel_f32(bit_mask, positive_val, negative_val);
        svfloat32_t vector_values = svld1_f32(predicate, &vector[i]);
        sum = svmla_f32_m(predicate, sum, vector_values, binary_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return svaddv_f32(svptrue_b32(), sum);
#else
    return neon::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
#endif
}

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results) {
    if (std::isnan(inv_sqrt_d)) {
        BinaryIPBatch4FromLookupLocal(vector, bits1, bits2, bits3, bits4, dim, results);
        return;
    }

    neon::RaBitQFloatBinaryIPBatch4(vector, bits1, bits2, bits3, bits4, dim, inv_sqrt_d, results);
}

void
RaBitQFloatThreeBitIPBatch4(const float* vector,
                            const uint8_t* bits1,
                            const uint8_t* bits2,
                            const uint8_t* bits3,
                            const uint8_t* bits4,
                            uint64_t dim,
                            uint32_t reorder_bits,
                            float* results) {
    neon::RaBitQFloatThreeBitIPBatch4(
        vector, bits1, bits2, bits3, bits4, dim, reorder_bits, results);
}

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits) {
    return neon::RaBitQFloatSplitCodeIP(
        vector, one_bit_code, supplement_code, dim, supplement_bits);
}

float
RaBitQFloatSupplementCodeIP(const float* vector,
                            const uint8_t* supplement_code,
                            uint64_t dim,
                            uint32_t supplement_bits) {
    return neon::RaBitQFloatSupplementCodeIP(vector, supplement_code, dim, supplement_bits);
}

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 0;
    }

    uint32_t result = 0;
    uint64_t num_bytes = (dim + 7) / 8;

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        uint64_t i = 0;
        uint64_t sum = 0;

        const uint8_t* current_codes = codes + bit_pos * num_bytes;

        svbool_t predicate = svwhilelt_b8(i, num_bytes);
        do {
            svuint8_t codes_vec = svld1_u8(predicate, current_codes + i);
            svuint8_t bits_vec = svld1_u8(predicate, bits + i);

            svuint8_t and_result = svand_u8_x(predicate, codes_vec, bits_vec);

            svuint8_t popcount = svcnt_u8_x(predicate, and_result);

            sum += svaddv_u8(predicate, popcount);

            i += svcntb();
            predicate = svwhilelt_b8(i, num_bytes);
        } while (svptest_first(svptrue_b8(), predicate));

        result += sum << bit_pos;
    }

    return result;
#else
    return neon::RaBitQSQ4UBinaryIP(codes, bits, dim);
#endif
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t y_vec = svld1_u8(predicate, y + i);
        svuint8_t result_vec = svand_u8_z(predicate, x_vec, y_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t y_vec = svld1_u8(predicate, y + i);
        svuint8_t result_vec = svorr_u8_z(predicate, x_vec, y_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t y_vec = svld1_u8(predicate, y + i);
        svuint8_t result_vec = sveor_u8_z(predicate, x_vec, y_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t predicate = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(predicate, x + i);
        svuint8_t result_vec = svnot_u8_z(predicate, x_vec);
        svst1_u8(predicate, result + i, result_vec);

        i += step;
        predicate = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), predicate));
#else
    neon::BitNot(x, num_byte, result);
#endif
}

void
Prefetch(const void* data) {
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return;
    }
    if (scalar == 0) {
        scalar = 1.0f;
    }
    svfloat32_t divisor = svdup_f32(scalar);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t values = svld1_f32(predicate, from + i);
        svfloat32_t result = svdiv_f32_z(predicate, values, divisor);
        svst1_f32(predicate, to + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
#if defined(ENABLE_SVE)
    float norm = std::sqrt(sve::FP32ComputeIP(from, from, dim));
    if (norm == 0) {
        norm = 1.0f;
    }
    sve::DivScalar(from, to, dim, norm);
    return norm;
#else
    return neon::Normalize(from, to, dim);
#endif
}

__attribute__((no_sanitize("address", "undefined"))) void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t total_bytes = pq_dim * 16;
    auto step = svcntb();

    const svuint8_t mask_low = svdup_u8(0x0F);
    const svuint16_t mask_low16 = svdup_u16(0x00FF);

    svuint16_t accum0 = svdup_u16(0);
    svuint16_t accum1 = svdup_u16(0);
    svuint16_t accum2 = svdup_u16(0);
    svuint16_t accum3 = svdup_u16(0);

    uint8_t offsets_data[svcntb()];
    for (uint64_t c = 0; c < svcntb() / 16; ++c) std::memset(offsets_data + c * 16, c * 16, 16);

    const svuint8_t index_offsets = svld1_u8(svptrue_b8(), offsets_data);

    svbool_t predicate = svwhilelt_b8(i, total_bytes);
    do {
        svuint8_t table_data = svld1_u8(predicate, lookup_table + i);
        svuint8_t code_data = svld1_u8(predicate, codes + i);

        svuint8_t low_nibbles = svand_u8_z(predicate, code_data, mask_low);
        svuint8_t high_nibbles = svlsr_n_u8_z(predicate, code_data, 4);

        svuint8_t adjusted_low_indices = svadd_u8_z(predicate, low_nibbles, index_offsets);
        svuint8_t adjusted_high_indices = svadd_u8_z(predicate, high_nibbles, index_offsets);

        svuint8_t low_values = svtbl_u8(table_data, adjusted_low_indices);
        svuint8_t high_values = svtbl_u8(table_data, adjusted_high_indices);

        svbool_t predicate_u16 = svwhilelt_b16(i / 2, total_bytes / 2);

        accum0 =
            svadd_u16_m(predicate_u16,
                        accum0,
                        svand_u16_z(predicate_u16, svreinterpret_u16_u8(low_values), mask_low16));
        accum1 = svadd_u16_m(predicate_u16,
                             accum1,
                             svlsr_n_u16_z(predicate_u16, svreinterpret_u16_u8(low_values), 8));
        accum2 =
            svadd_u16_m(predicate_u16,
                        accum2,
                        svand_u16_z(predicate_u16, svreinterpret_u16_u8(high_values), mask_low16));
        accum3 = svadd_u16_m(predicate_u16,
                             accum3,
                             svlsr_n_u16_z(predicate_u16, svreinterpret_u16_u8(high_values), 8));

        i += step;
        predicate = svwhilelt_b8(i, total_bytes);
    } while (svptest_first(svptrue_b8(), predicate));

    uint16_t temp[svcntb() / 2];

    // Segment 0
    svst1_u16(svptrue_b16(), temp, accum0);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[0 * 8 + j] += temp[j + 8 * (k)];

    // Segment 1
    svst1_u16(svptrue_b16(), temp, accum1);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[1 * 8 + j] += temp[j + 8 * k];

    // Segment 2
    svst1_u16(svptrue_b16(), temp, accum2);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[2 * 8 + j] += temp[j + 8 * k];

    // Segment 3
    svst1_u16(svptrue_b16(), temp, accum3);
    for (int j = 0; j < 8; ++j)
        for (int k = 0; k < svcntb() / 16; k++) result[3 * 8 + j] += temp[j + 8 * k];

#else
    neon::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
KacsWalk(float* data, uint64_t len) {
#if defined(ENABLE_SVE)
    uint64_t n = len / 2;
    uint64_t offset = (len % 2) + n;
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, n);
    do {
        svfloat32_t vec1 = svld1_f32(predicate, data + i);
        svfloat32_t vec2 = svld1_f32(predicate, data + i + offset);
        svfloat32_t sum_vec = svadd_f32_z(predicate, vec1, vec2);
        svfloat32_t diff_vec = svsub_f32_z(predicate, vec1, vec2);
        svst1_f32(predicate, data + i, sum_vec);
        svst1_f32(predicate, data + i + offset, diff_vec);
        i += step;
        predicate = svwhilelt_b32(i, n);
    } while (svptest_first(svptrue_b32(), predicate));

    if (len % 2 != 0) {
        data[n] *= std::sqrt(2.0F);
    }
#else
    neon::KacsWalk(data, len);
#endif
}

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint8_t* predicate_array = (uint8_t*)alloca(dim);
    const uint64_t num_bytes = dim / 8;
    for (uint64_t j = 0; j < num_bytes; ++j) {
        memcpy(&predicate_array[j * 8], g_bit_lookup_table[flip[j]].data(), 8);
    }
    if (dim % 8 != 0) {
        const uint64_t remaining_bits = dim % 8;
        memcpy(&predicate_array[num_bytes * 8],
               g_bit_lookup_table[flip[num_bytes]].data(),
               remaining_bits);
    }

    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svuint32_t predicate_values = svld1ub_u32(predicate, &predicate_array[i]);
        svbool_t bit_mask = svcmpne_n_u32(predicate, predicate_values, 0);

        svfloat32_t data_vec = svld1_f32(predicate, data + i);
        svfloat32_t result_vec = svneg_f32_m(data_vec, bit_mask, data_vec);
        svst1_f32(predicate, data + i, result_vec);

        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::FlipSign(flip, data, dim);
#endif
}

void
VecRescale(float* data, uint64_t dim, float val) {
#if defined(ENABLE_SVE)
    svfloat32_t scale = svdup_f32(val);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t data_vec = svld1_f32(predicate, data + i);
        svfloat32_t result = svmul_f32_z(predicate, data_vec, scale);
        svst1_f32(predicate, data + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_SVE)
    for (int i = idx; i < dim_; i += 2 * step) {
        uint64_t j = 0;
        const uint64_t sve_step = svcntw();
        svbool_t predicate = svwhilelt_b32(j, (uint64_t)step);
        do {
            svfloat32_t x = svld1_f32(predicate, data + i + j);
            svfloat32_t y = svld1_f32(predicate, data + i + j + step);
            svst1_f32(predicate, data + i + j, svadd_f32_z(predicate, x, y));
            svst1_f32(predicate, data + i + j + step, svsub_f32_z(predicate, x, y));
            j += sve_step;
            predicate = svwhilelt_b32(j, (uint64_t)step);
        } while (svptest_first(svptrue_b32(), predicate));
    }
#else
    neon::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, uint64_t dim_) {
#if defined(ENABLE_SVE)
    uint64_t n = dim_;
    uint64_t step = 1;
    while (step < n) {
        sve::RotateOp(data, 0, dim_, step);
        step *= 2;
    }
#else
    neon::FHTRotate(data, dim_);
#endif
}

float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 1.0f;
    }

    svfloat32_t sum = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t f_vec = svld1_f32(predicate, from + i);
        svfloat32_t c_vec = svld1_f32(predicate, centroid + i);
        svfloat32_t diff = svsub_f32_z(predicate, f_vec, c_vec);
        sum = svmla_f32_m(predicate, sum, diff, diff);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    float norm_sq = svaddv_f32(svptrue_b32(), sum);
    float norm = 0;
    if (norm_sq < 1e-5f) {
        norm = 1.0f;
    } else {
        norm = std::sqrt(norm_sq);
    }

    svfloat32_t normVec = svdup_f32(norm);
    i = 0;
    predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t f_vec = svld1_f32(predicate, from + i);
        svfloat32_t c_vec = svld1_f32(predicate, centroid + i);
        svfloat32_t diff = svsub_f32_z(predicate, f_vec, c_vec);
        svfloat32_t result = svdiv_f32_z(predicate, diff, normVec);
        svst1_f32(predicate, to + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));

    return norm;
#else
    return neon::NormalizeWithCentroid(from, centroid, to, dim);
#endif
}

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return;
    }

    svfloat32_t normVec = svdup_f32(norm);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t predicate = svwhilelt_b32(i, dim);
    do {
        svfloat32_t f_vec = svld1_f32(predicate, from + i);
        svfloat32_t c_vec = svld1_f32(predicate, centroid + i);
        svfloat32_t result = svmad_f32_z(predicate, f_vec, normVec, c_vec);
        svst1_f32(predicate, to + i, result);
        i += step;
        predicate = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), predicate));
#else
    neon::InverseNormalizeWithCentroid(from, centroid, to, dim, norm);
#endif
}

#if defined(ENABLE_SVE)

void RaBitQFloatBuild4BitLookupTable(const float* query, uint64_t dim, float inv_sqrt_d,
                                float16_t* table) {
    // The vectorized path below assumes a 256-bit SVE vector (svcntw() == 8):
    // the f16 stores write 16 lanes each and the per-byte table stride is 32.
    // On any other vector length it would write out of bounds, so fall back
    // to the generic implementation.
    // The SVE table layout is also only consumed exactly when dim is a
    // multiple of 64 (the paired lookup routine consumes 256 f16 per 64
    // dims), so keep the layout pairing consistent with the consumer and use
    // the generic implementation otherwise.
    if (svcntw() != 8 || (dim & 63) != 0) {
        generic::RaBitQFloatBuild4BitLookupTable(query, dim, inv_sqrt_d, table);
        return;
    }
    const float pos = inv_sqrt_d < 1e-3F ? 1.0F : inv_sqrt_d;
    const float neg = inv_sqrt_d < 1e-3F ? 0.0F : -inv_sqrt_d;
    const float pattern[8] = { neg, neg, neg, neg,
                               pos, pos, pos, pos};
    svbool_t all_true = svptrue_b16();
    svfloat32_t inv_sqrt_d1 = svdupq_n_f32(neg, pos, neg, pos);
    svfloat32_t inv_sqrt_d2 = svdupq_n_f32(neg, neg, pos, pos);
    svfloat32_t inv_sqrt_d3 = svld1_f32(all_true, pattern);
    svfloat32_t inv_sqrt_d4_lo = svdup_n_f32(neg);
    svfloat32_t inv_sqrt_d4_hi = svdup_n_f32(pos);
    const uint64_t num_bytes = (dim / 8);
    const uint64_t plus_dim = (dim & 7);
    for (uint64_t nb = 0; nb < num_bytes; ++nb) {
        const float* q = query + 8 * nb;
        svfloat32_t low_res = svdup_n_f32(0);
        svfloat32_t hig_res = svdup_n_f32(0);
        low_res =  svmla_n_f32_m(all_true, low_res,  inv_sqrt_d1, q[0]);
        hig_res =  svmla_n_f32_m(all_true, hig_res,  inv_sqrt_d1, q[4]);
        low_res =  svmla_n_f32_m(all_true, low_res,  inv_sqrt_d2, q[1]);
        hig_res =  svmla_n_f32_m(all_true, hig_res,  inv_sqrt_d2, q[5]);
        low_res =  svmla_n_f32_m(all_true, low_res,  inv_sqrt_d3, q[2]);
        hig_res =  svmla_n_f32_m(all_true, hig_res,  inv_sqrt_d3, q[6]);
        svfloat32_t dim_lowhalflow = svmla_n_f32_m(all_true, low_res, inv_sqrt_d4_lo, q[3]);
        svfloat32_t dim_lowhalfhig = svmla_n_f32_m(all_true, low_res, inv_sqrt_d4_hi, q[3]);
        svfloat32_t dim_highalflow = svmla_n_f32_m(all_true, hig_res, inv_sqrt_d4_lo, q[7]);
        svfloat32_t dim_highalfhig = svmla_n_f32_m(all_true, hig_res, inv_sqrt_d4_hi, q[7]);
        svfloat16_t complate_dim_low = svuzp1_f16(svcvt_f16_f32_x(all_true, dim_lowhalflow), svcvt_f16_f32_x(all_true, dim_lowhalfhig));
        svfloat16_t complate_dim_hig = svuzp1_f16(svcvt_f16_f32_x(all_true, dim_highalflow), svcvt_f16_f32_x(all_true, dim_highalfhig));
        // store
        svst1_f16(all_true, table, complate_dim_low);
        svst1_f16(all_true, table + 16, complate_dim_hig);
        table += 32;
    }
    if (plus_dim > 0) {
        // The remaining plus_dim (= dim & 7) dims start at query + 8 * num_bytes.
        // The first min(plus_dim, 4) of them fill the low-nibble table; the rest
        // fill the high-nibble table, which is only part of the allocated buffer
        // when plus_dim >= 5 (the buffer has 16 * ceil(dim / 4) entries). Lanes
        // beyond dim contribute 0, matching the generic implementation, so the
        // query is never read out of bounds and the table is never overrun.
        const int low_count = plus_dim < 4 ? static_cast<int>(plus_dim) : 4;
        const int high_count = plus_dim > 4 ? static_cast<int>(plus_dim) - 4 : 0;
        const float* q_tail = query + 8 * num_bytes;
        for (int byte_value = 0; byte_value < 16; ++byte_value) {
            float low = 0.0F;
            float high = 0.0F;
            for (int bit_pos = 0; bit_pos < low_count; ++bit_pos) {
                low += q_tail[bit_pos] * (((byte_value >> bit_pos) & 0x1) != 0 ? pos : neg);
            }
            for (int bit_pos = 0; bit_pos < high_count; ++bit_pos) {
                high += q_tail[4 + bit_pos] * (((byte_value >> bit_pos) & 0x1) != 0 ? pos : neg);
            }
            table[byte_value] = static_cast<float16_t>(low);
            if (high_count > 0) {
                table[byte_value + 16] = static_cast<float16_t>(high);
            }
        }
    }
}

inline void prefetch_L2(const void* address) {
    __builtin_prefetch(address, 0, 2);
}

#define TBL_DOUBLE_LOOP(input_code0, input_code1, preload) \
    c0 = svbext_n_u16(input_code0, 0x000F); \
    c4 = svbext_n_u16(input_code1, 0x000F); \
    c1 = svbext_n_u16(input_code0, 0x00F0); \
    c5 = svbext_n_u16(input_code1, 0x00F0); \
    c2 = svbext_n_u16(input_code0, 0x0F00); \
    c6 = svbext_n_u16(input_code1, 0x0F00); \
    c3 = svbext_n_u16(input_code0, 0xF000); \
    c7 = svbext_n_u16(input_code1, 0xF000); \
    asm volatile("tbl %[index].h, %[ftable0].h, %[index].h"     \
    : [index] "+w"(c0) : [ftable0] "w"(table0):);               \
    asm volatile("tbl %[index].h, %[ftable0].h, %[index].h"     \
    : [index] "+w"(c4) : [ftable0] "w"(table0):);               \
    if constexpr (preload) table0 = svld1_f16(all_true, sim_table);        \
    asm volatile("tbl %[index].h, %[ftable1].h, %[index].h"     \
    : [index] "+w"(c1) : [ftable1] "w"(table1):);               \
    asm volatile("tbl %[index].h, %[ftable1].h, %[index].h"     \
    : [index] "+w"(c5) : [ftable1] "w"(table1):);               \
    if constexpr (preload) table1 = svld1_f16(all_true, sim_table + 16);   \
    asm volatile("tbl %[index].h, %[ftable2].h, %[index].h"     \
    : [index] "+w"(c2) : [ftable2] "w"(table2):);               \
    asm volatile("tbl %[index].h, %[ftable2].h, %[index].h"     \
    : [index] "+w"(c6) : [ftable2] "w"(table2):);               \
    if constexpr (preload) table2 = svld1_f16(all_true, sim_table + 32);   \
    asm volatile("tbl %[index].h, %[ftable3].h, %[index].h"     \
    : [index] "+w"(c3) : [ftable3] "w"(table3):);               \
    asm volatile("tbl %[index].h, %[ftable3].h, %[index].h"     \
    : [index] "+w"(c7) : [ftable3] "w"(table3):);               \
    if constexpr (preload) table3 = svld1_f16(all_true, sim_table + 48);   \
    if constexpr (preload) sim_table += 64;                                \
    res0 = svadd_f16_m(all_true, res0, svreinterpret_f16_u16(c0));   \
    res4 = svadd_f16_m(all_true, res4, svreinterpret_f16_u16(c4));   \
    res1 = svadd_f16_m(all_true, res1, svreinterpret_f16_u16(c1));   \
    res5 = svadd_f16_m(all_true, res5, svreinterpret_f16_u16(c5));   \
    res2 = svadd_f16_m(all_true, res2, svreinterpret_f16_u16(c2));   \
    res6 = svadd_f16_m(all_true, res6, svreinterpret_f16_u16(c6));   \
    res3 = svadd_f16_m(all_true, res3, svreinterpret_f16_u16(c3));   \
    res7 = svadd_f16_m(all_true, res7, svreinterpret_f16_u16(c7));

#define SVE_INNER_LOOP_DOUBLE_BODY(preload) \
    c0 = svzip1_u16(rgba0, rgba1); \
    c1 = svzip1_u16(rgba2, rgba3); \
    c2 = svzip2_u16(rgba0, rgba1); \
    c3 = svzip2_u16(rgba2, rgba3); \
    c4 = svzip1_u16(rgba4, rgba5); \
    c5 = svzip1_u16(rgba6, rgba7); \
    c6 = svzip2_u16(rgba4, rgba5); \
    c7 = svzip2_u16(rgba6, rgba7); \
    rgba0 = svreinterpret_u16_u32(svzip1_u32(svreinterpret_u32_u16(c0), svreinterpret_u32_u16(c1))); \
    rgba1 = svreinterpret_u16_u32(svzip2_u32(svreinterpret_u32_u16(c0), svreinterpret_u32_u16(c1))); \
    rgba2 = svreinterpret_u16_u32(svzip1_u32(svreinterpret_u32_u16(c2), svreinterpret_u32_u16(c3))); \
    rgba3 = svreinterpret_u16_u32(svzip2_u32(svreinterpret_u32_u16(c2), svreinterpret_u32_u16(c3))); \
    rgba4 = svreinterpret_u16_u32(svzip1_u32(svreinterpret_u32_u16(c4), svreinterpret_u32_u16(c5))); \
    rgba5 = svreinterpret_u16_u32(svzip2_u32(svreinterpret_u32_u16(c4), svreinterpret_u32_u16(c5))); \
    rgba6 = svreinterpret_u16_u32(svzip1_u32(svreinterpret_u32_u16(c6), svreinterpret_u32_u16(c7))); \
    rgba7 = svreinterpret_u16_u32(svzip2_u32(svreinterpret_u32_u16(c6), svreinterpret_u32_u16(c7))); \
    TBL_DOUBLE_LOOP(rgba0, rgba4, true) \
    if constexpr (preload) rgba0 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p0  + m), svld1uh_u64(all_true, p2  + m), 16), svsli_n_u64(svld1uh_u64(all_true, p4  + m), svld1uh_u64(all_true, p6  + m), 16), 32)); \
    if constexpr (preload) rgba4 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p16 + m), svld1uh_u64(all_true, p18 + m), 16), svsli_n_u64(svld1uh_u64(all_true, p20 + m), svld1uh_u64(all_true, p22 + m), 16), 32)); \
    TBL_DOUBLE_LOOP(rgba1, rgba5, true) \
    if constexpr (preload) rgba1 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p8  + m), svld1uh_u64(all_true, p10 + m), 16), svsli_n_u64(svld1uh_u64(all_true, p12 + m), svld1uh_u64(all_true, p14 + m), 16), 32)); \
    if constexpr (preload) rgba5 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p24 + m), svld1uh_u64(all_true, p26 + m), 16), svsli_n_u64(svld1uh_u64(all_true, p28 + m), svld1uh_u64(all_true, p30 + m), 16), 32)); \
    TBL_DOUBLE_LOOP(rgba2, rgba6, true) \
    if constexpr (preload) rgba2 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p1  + m), svld1uh_u64(all_true, p3  + m), 16), svsli_n_u64(svld1uh_u64(all_true, p5  + m), svld1uh_u64(all_true, p7  + m), 16), 32)); \
    if constexpr (preload) rgba6 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p17 + m), svld1uh_u64(all_true, p19 + m), 16), svsli_n_u64(svld1uh_u64(all_true, p21 + m), svld1uh_u64(all_true, p23 + m), 16), 32)); \
    TBL_DOUBLE_LOOP(rgba3, rgba7, preload)  \
    if constexpr (preload) rgba3 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p9  + m), svld1uh_u64(all_true, p11 + m), 16), svsli_n_u64(svld1uh_u64(all_true, p13 + m), svld1uh_u64(all_true, p15 + m), 16), 32)); \
    if constexpr (preload) rgba7 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p25 + m), svld1uh_u64(all_true, p27 + m), 16), svsli_n_u64(svld1uh_u64(all_true, p29 + m), svld1uh_u64(all_true, p31 + m), 16), 32));

#define TBL_MAIN_LOOP(input_code0, preload) \
    c0 = svbext_n_u16(input_code0, 0x000F); \
    c1 = svbext_n_u16(input_code0, 0x00F0); \
    c2 = svbext_n_u16(input_code0, 0x0F00); \
    c3 = svbext_n_u16(input_code0, 0xF000); \
    asm volatile("tbl %[index].h, %[ftable0].h, %[index].h"         \
    : [index] "+w"(c0) : [ftable0] "w"(table0):);                   \
    if constexpr (preload) table0 = svld1_f16(all_true, sim_table); \
    asm volatile("tbl %[index].h, %[ftable1].h, %[index].h"         \
    : [index] "+w"(c1) : [ftable1] "w"(table1):);                   \
    if constexpr (preload) table1 = svld1_f16(all_true, sim_table + 16);   \
    asm volatile("tbl %[index].h, %[ftable2].h, %[index].h"         \
    : [index] "+w"(c2) : [ftable2] "w"(table2):);                   \
    if constexpr (preload) table2 = svld1_f16(all_true, sim_table + 32);   \
    asm volatile("tbl %[index].h, %[ftable3].h, %[index].h"         \
    : [index] "+w"(c3) : [ftable3] "w"(table3):);                   \
    if constexpr (preload) table3 = svld1_f16(all_true, sim_table + 48);   \
    if constexpr (preload) sim_table += 64;                           \
    res0 = svadd_f16_m(all_true, res0, svreinterpret_f16_u16(c0));  \
    res1 = svadd_f16_m(all_true, res1, svreinterpret_f16_u16(c1));  \
    res2 = svadd_f16_m(all_true, res2, svreinterpret_f16_u16(c2));  \
    res3 = svadd_f16_m(all_true, res3, svreinterpret_f16_u16(c3));

#define SVE_INNER_LOOP_BODY(preload) \
    c0 = svzip1_u16(rgba0, rgba1); \
    c1 = svzip1_u16(rgba2, rgba3); \
    c2 = svzip2_u16(rgba0, rgba1); \
    c3 = svzip2_u16(rgba2, rgba3); \
    rgba0 = svreinterpret_u16_u32(svzip1_u32(svreinterpret_u32_u16(c0), svreinterpret_u32_u16(c1))); \
    rgba1 = svreinterpret_u16_u32(svzip2_u32(svreinterpret_u32_u16(c0), svreinterpret_u32_u16(c1))); \
    rgba2 = svreinterpret_u16_u32(svzip1_u32(svreinterpret_u32_u16(c2), svreinterpret_u32_u16(c3))); \
    rgba3 = svreinterpret_u16_u32(svzip2_u32(svreinterpret_u32_u16(c2), svreinterpret_u32_u16(c3))); \
    TBL_MAIN_LOOP(rgba0, true) \
    if constexpr (preload) rgba0 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p0  + m), svld1uh_u64(all_true, p2  + m), 16), svsli_n_u64(svld1uh_u64(all_true, p4  + m), svld1uh_u64(all_true, p6  + m), 16), 32)); \
    TBL_MAIN_LOOP(rgba1, true) \
    if constexpr (preload) rgba1 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p8  + m), svld1uh_u64(all_true, p10 + m), 16), svsli_n_u64(svld1uh_u64(all_true, p12 + m), svld1uh_u64(all_true, p14 + m), 16), 32)); \
    TBL_MAIN_LOOP(rgba2, true) \
    if constexpr (preload) rgba2 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p1  + m), svld1uh_u64(all_true, p3  + m), 16), svsli_n_u64(svld1uh_u64(all_true, p5  + m), svld1uh_u64(all_true, p7  + m), 16), 32)); \
    TBL_MAIN_LOOP(rgba3, preload)  \
    if constexpr (preload) rgba3 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p9  + m), svld1uh_u64(all_true, p11 + m), 16), svsli_n_u64(svld1uh_u64(all_true, p13 + m), svld1uh_u64(all_true, p15 + m), 16), 32));

#define SVE_HALF_INNER_LOOP_BODY(preload) \
    c0 = svzip1_u16(rgba0, rgba1);  \
    c2 = svzip2_u16(rgba0, rgba1);  \
    rgba0 = svzip1_u16(c0, zeros);  \
    rgba1 = svzip2_u16(c0, zeros);  \
    rgba2 = svzip1_u16(c2, zeros);  \
    rgba3 = svzip2_u16(c2, zeros);  \
    TBL_MAIN_LOOP(rgba0, true) \
    if constexpr (preload) rgba0 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p0 + m), svld1uh_u64(all_true, p2 + m), 16), svsli_n_u64(svld1uh_u64(all_true, p4 + m), svld1uh_u64(all_true, p6 + m), 16), 32)); \
    TBL_MAIN_LOOP(rgba1, true)      \
    if constexpr (preload) rgba1 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p1 + m), svld1uh_u64(all_true, p3 + m), 16), svsli_n_u64(svld1uh_u64(all_true, p5 + m), svld1uh_u64(all_true, p7 + m), 16), 32)); \
    TBL_MAIN_LOOP(rgba2, true)      \
    TBL_MAIN_LOOP(rgba3, preload)

#define L2_PREFETCH_OFFSET 28
#define PREFETCH_STEP 12

void
RaBitQFloatBatchBinaryIPBy4BitLookup(const uint8_t* codes, uint64_t ncode, uint64_t dim,
                                      uint64_t code_stride, const float16_t* table, float* out,
                                      const uint32_t* idx) {
    // This routine is a 256-bit SVE port of the FAISS PQFastScan layout: the
    // TBL lookups, the packed code loads (svsli shifts 16/32) and the result
    // stores (out / out + 8 / out + 16 / out + 24) all assume svcntw() == 8.
    // On any other vector length the stores would write out of bounds and the
    // arithmetic would be wrong, so fall back to the generic implementation.
    // The main loops also walk the lookup table in fixed 256-entry (64-dim)
    // blocks and only consume the table exactly when dim is a multiple of
    // 64; for other dims they would read past the table allocation, so use
    // the generic implementation (which pairs with the generic table
    // layout) in that case as well.
    if (svcntw() != 8 || (dim & 63) != 0) {
        generic::RaBitQFloatBatchBinaryIPBy4BitLookup(
            codes, ncode, dim, code_stride, table, out, idx);
        return;
    }
    svbool_t all_true = svptrue_b16();
    svbool_t all_true32 = svptrue_b32();
    uint64_t nsq = (dim / 16);
    uint64_t nc = 0;
    svfloat16_t table0, table1, table2, table3;
    svfloat16_t res0, res1, res2, res3;
    svfloat16_t res4, res5, res6, res7;
    for (; nc + 31 < ncode; nc += 32) {
        {
            svuint16_t c0, c1, c2, c3;
            svuint16_t c4, c5, c6, c7;
            const float16_t* sim_table = (float16_t*)table;
            table0 = svld1_f16(all_true, sim_table);
            table1 = svld1_f16(all_true, sim_table + 16);
            table2 = svld1_f16(all_true, sim_table + 32);
            table3 = svld1_f16(all_true, sim_table + 48);
            sim_table += 64;
            res0 = svdup_n_f16(0);
            res1 = svdup_n_f16(0);
            res2 = svdup_n_f16(0);
            res3 = svdup_n_f16(0);
            res4 = svdup_n_f16(0);
            res5 = svdup_n_f16(0);
            res6 = svdup_n_f16(0);
            res7 = svdup_n_f16(0);

            const uint16_t *p0  = (const uint16_t*)(codes + (uint64_t)idx[nc + 0 ] * code_stride);
            const uint16_t *p1  = (const uint16_t*)(codes + (uint64_t)idx[nc + 1 ] * code_stride);
            const uint16_t *p2  = (const uint16_t*)(codes + (uint64_t)idx[nc + 2 ] * code_stride);
            const uint16_t *p3  = (const uint16_t*)(codes + (uint64_t)idx[nc + 3 ] * code_stride);
            const uint16_t *p4  = (const uint16_t*)(codes + (uint64_t)idx[nc + 4 ] * code_stride);
            const uint16_t *p5  = (const uint16_t*)(codes + (uint64_t)idx[nc + 5 ] * code_stride);
            const uint16_t *p6  = (const uint16_t*)(codes + (uint64_t)idx[nc + 6 ] * code_stride);
            const uint16_t *p7  = (const uint16_t*)(codes + (uint64_t)idx[nc + 7 ] * code_stride);
            const uint16_t *p8  = (const uint16_t*)(codes + (uint64_t)idx[nc + 8 ] * code_stride);
            const uint16_t *p9  = (const uint16_t*)(codes + (uint64_t)idx[nc + 9 ] * code_stride);
            const uint16_t *p10 = (const uint16_t*)(codes + (uint64_t)idx[nc + 10] * code_stride);
            const uint16_t *p11 = (const uint16_t*)(codes + (uint64_t)idx[nc + 11] * code_stride);
            const uint16_t *p12 = (const uint16_t*)(codes + (uint64_t)idx[nc + 12] * code_stride);
            const uint16_t *p13 = (const uint16_t*)(codes + (uint64_t)idx[nc + 13] * code_stride);
            const uint16_t *p14 = (const uint16_t*)(codes + (uint64_t)idx[nc + 14] * code_stride);
            const uint16_t *p15 = (const uint16_t*)(codes + (uint64_t)idx[nc + 15] * code_stride);
            const uint16_t *p16 = (const uint16_t*)(codes + (uint64_t)idx[nc + 16] * code_stride);
            const uint16_t *p17 = (const uint16_t*)(codes + (uint64_t)idx[nc + 17] * code_stride);
            const uint16_t *p18 = (const uint16_t*)(codes + (uint64_t)idx[nc + 18] * code_stride);
            const uint16_t *p19 = (const uint16_t*)(codes + (uint64_t)idx[nc + 19] * code_stride);
            const uint16_t *p20 = (const uint16_t*)(codes + (uint64_t)idx[nc + 20] * code_stride);
            const uint16_t *p21 = (const uint16_t*)(codes + (uint64_t)idx[nc + 21] * code_stride);
            const uint16_t *p22 = (const uint16_t*)(codes + (uint64_t)idx[nc + 22] * code_stride);
            const uint16_t *p23 = (const uint16_t*)(codes + (uint64_t)idx[nc + 23] * code_stride);
            const uint16_t *p24 = (const uint16_t*)(codes + (uint64_t)idx[nc + 24] * code_stride);
            const uint16_t *p25 = (const uint16_t*)(codes + (uint64_t)idx[nc + 25] * code_stride);
            const uint16_t *p26 = (const uint16_t*)(codes + (uint64_t)idx[nc + 26] * code_stride);
            const uint16_t *p27 = (const uint16_t*)(codes + (uint64_t)idx[nc + 27] * code_stride);
            const uint16_t *p28 = (const uint16_t*)(codes + (uint64_t)idx[nc + 28] * code_stride);
            const uint16_t *p29 = (const uint16_t*)(codes + (uint64_t)idx[nc + 29] * code_stride);
            const uint16_t *p30 = (const uint16_t*)(codes + (uint64_t)idx[nc + 30] * code_stride);
            const uint16_t *p31 = (const uint16_t*)(codes + (uint64_t)idx[nc + 31] * code_stride);

            svuint16_t rgba0 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p0 ), svld1uh_u64(all_true, p2 ), 16), svsli_n_u64(svld1uh_u64(all_true, p4 ), svld1uh_u64(all_true, p6 ), 16), 32));
            svuint16_t rgba1 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p8 ), svld1uh_u64(all_true, p10), 16), svsli_n_u64(svld1uh_u64(all_true, p12), svld1uh_u64(all_true, p14), 16), 32));
            svuint16_t rgba2 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p1 ), svld1uh_u64(all_true, p3 ), 16), svsli_n_u64(svld1uh_u64(all_true, p5 ), svld1uh_u64(all_true, p7 ), 16), 32));
            svuint16_t rgba3 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p9 ), svld1uh_u64(all_true, p11), 16), svsli_n_u64(svld1uh_u64(all_true, p13), svld1uh_u64(all_true, p15), 16), 32));
            svuint16_t rgba4 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p16), svld1uh_u64(all_true, p18), 16), svsli_n_u64(svld1uh_u64(all_true, p20), svld1uh_u64(all_true, p22), 16), 32));
            svuint16_t rgba5 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p24), svld1uh_u64(all_true, p26), 16), svsli_n_u64(svld1uh_u64(all_true, p28), svld1uh_u64(all_true, p30), 16), 32));
            svuint16_t rgba6 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p17), svld1uh_u64(all_true, p19), 16), svsli_n_u64(svld1uh_u64(all_true, p21), svld1uh_u64(all_true, p23), 16), 32));
            svuint16_t rgba7 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p25), svld1uh_u64(all_true, p27), 16), svsli_n_u64(svld1uh_u64(all_true, p29), svld1uh_u64(all_true, p31), 16), 32));
            
            uint64_t m = 4;
            for (uint64_t x = 4; x + L2_PREFETCH_OFFSET < nsq; x += PREFETCH_STEP) {
                prefetch_L2(p0  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p1  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p2  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p3  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p4  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p5  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p6  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p7  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p8  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p9  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p10 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p11 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p12 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p13 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p14 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p15 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p16 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p17 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p18 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p19 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p20 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p21 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p22 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p23 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p24 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p25 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p26 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p27 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p28 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p29 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p30 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p31 + (x + L2_PREFETCH_OFFSET));
                for (m = x; m < x + PREFETCH_STEP; m += 4) {
                    SVE_INNER_LOOP_DOUBLE_BODY(true)
                }
            }
            for (; m < nsq; m += 4) {
                SVE_INNER_LOOP_DOUBLE_BODY(true)
            }
            SVE_INNER_LOOP_DOUBLE_BODY(false)
        }
        svfloat32_t ans0 = svdup_n_f32(0);
        svfloat32_t ans1 = svdup_n_f32(0);
        svfloat32_t ans2 = svdup_n_f32(0);
        svfloat32_t ans3 = svdup_n_f32(0);
        ans0 = svmlalb_n_f32(ans0, res0, 1.0);
        ans1 = svmlalt_n_f32(ans1, res0, 1.0);
        ans2 = svmlalb_n_f32(ans2, res4, 1.0);
        ans3 = svmlalt_n_f32(ans3, res4, 1.0);
        ans0 = svmlalb_n_f32(ans0, res1, 1.0);
        ans1 = svmlalt_n_f32(ans1, res1, 1.0);
        ans2 = svmlalb_n_f32(ans2, res5, 1.0);
        ans3 = svmlalt_n_f32(ans3, res5, 1.0);
        ans0 = svmlalb_n_f32(ans0, res2, 1.0);
        ans1 = svmlalt_n_f32(ans1, res2, 1.0);
        ans2 = svmlalb_n_f32(ans2, res6, 1.0);
        ans3 = svmlalt_n_f32(ans3, res6, 1.0);
        ans0 = svmlalb_n_f32(ans0, res3, 1.0);
        ans1 = svmlalt_n_f32(ans1, res3, 1.0);
        ans2 = svmlalb_n_f32(ans2, res7, 1.0);
        ans3 = svmlalt_n_f32(ans3, res7, 1.0);

        svst1_f32(all_true32, out     , ans0);
        svst1_f32(all_true32, out + 8 , ans1);
        svst1_f32(all_true32, out + 16, ans2);
        svst1_f32(all_true32, out + 24, ans3);
        out += 32;
    }
    if (ncode & 16) {
        svuint16_t c0, c1, c2, c3;
        const float16_t* sim_table = (float16_t*)table;
        table0 = svld1_f16(all_true, sim_table);
        table1 = svld1_f16(all_true, sim_table + 16);
        table2 = svld1_f16(all_true, sim_table + 32);
        table3 = svld1_f16(all_true, sim_table + 48);
        sim_table += 64;
        res0 = svdup_n_f16(0);
        res1 = svdup_n_f16(0);
        res2 = svdup_n_f16(0);
        res3 = svdup_n_f16(0);

        svfloat32_t ans0 = svdup_n_f32(0);
        svfloat32_t ans1 = svdup_n_f32(0);
        const uint16_t *p0  = (const uint16_t*)(codes + (uint64_t)idx[nc + 0 ] * code_stride);
        const uint16_t *p1  = (const uint16_t*)(codes + (uint64_t)idx[nc + 1 ] * code_stride);
        const uint16_t *p2  = (const uint16_t*)(codes + (uint64_t)idx[nc + 2 ] * code_stride);
        const uint16_t *p3  = (const uint16_t*)(codes + (uint64_t)idx[nc + 3 ] * code_stride);
        const uint16_t *p4  = (const uint16_t*)(codes + (uint64_t)idx[nc + 4 ] * code_stride);
        const uint16_t *p5  = (const uint16_t*)(codes + (uint64_t)idx[nc + 5 ] * code_stride);
        const uint16_t *p6  = (const uint16_t*)(codes + (uint64_t)idx[nc + 6 ] * code_stride);
        const uint16_t *p7  = (const uint16_t*)(codes + (uint64_t)idx[nc + 7 ] * code_stride);
        const uint16_t *p8  = (const uint16_t*)(codes + (uint64_t)idx[nc + 8 ] * code_stride);
        const uint16_t *p9  = (const uint16_t*)(codes + (uint64_t)idx[nc + 9 ] * code_stride);
        const uint16_t *p10 = (const uint16_t*)(codes + (uint64_t)idx[nc + 10] * code_stride);
        const uint16_t *p11 = (const uint16_t*)(codes + (uint64_t)idx[nc + 11] * code_stride);
        const uint16_t *p12 = (const uint16_t*)(codes + (uint64_t)idx[nc + 12] * code_stride);
        const uint16_t *p13 = (const uint16_t*)(codes + (uint64_t)idx[nc + 13] * code_stride);
        const uint16_t *p14 = (const uint16_t*)(codes + (uint64_t)idx[nc + 14] * code_stride);
        const uint16_t *p15 = (const uint16_t*)(codes + (uint64_t)idx[nc + 15] * code_stride);
        svuint16_t rgba0 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p0), svld1uh_u64(all_true, p2 ), 16), svsli_n_u64(svld1uh_u64(all_true, p4 ), svld1uh_u64(all_true, p6 ), 16), 32));
        svuint16_t rgba1 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p8), svld1uh_u64(all_true, p10), 16), svsli_n_u64(svld1uh_u64(all_true, p12), svld1uh_u64(all_true, p14), 16), 32));
        svuint16_t rgba2 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p1), svld1uh_u64(all_true, p3 ), 16), svsli_n_u64(svld1uh_u64(all_true, p5 ), svld1uh_u64(all_true, p7 ), 16), 32));
        svuint16_t rgba3 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p9), svld1uh_u64(all_true, p11), 16), svsli_n_u64(svld1uh_u64(all_true, p13), svld1uh_u64(all_true, p15), 16), 32));
        uint64_t m = 4;
        for (uint64_t x = 4; x + L2_PREFETCH_OFFSET < nsq; x += PREFETCH_STEP) {
            prefetch_L2(p0  + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p1  + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p2  + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p3  + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p4  + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p5  + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p6  + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p7  + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p8  + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p9  + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p10 + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p11 + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p12 + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p13 + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p14 + (x + L2_PREFETCH_OFFSET));
            prefetch_L2(p15 + (x + L2_PREFETCH_OFFSET));
            for (m = x; m < x + PREFETCH_STEP; m += 4) {
                SVE_INNER_LOOP_BODY(true)
            }
        }
        for (; m < nsq; m += 4) {
            SVE_INNER_LOOP_BODY(true)
        }
        SVE_INNER_LOOP_BODY(false)

        ans0 = svmlalb_n_f32(ans0, res0, 1.0);
        ans1 = svmlalt_n_f32(ans1, res0, 1.0);
        ans0 = svmlalb_n_f32(ans0, res1, 1.0);
        ans1 = svmlalt_n_f32(ans1, res1, 1.0);
        ans0 = svmlalb_n_f32(ans0, res2, 1.0);
        ans1 = svmlalt_n_f32(ans1, res2, 1.0);
        ans0 = svmlalb_n_f32(ans0, res3, 1.0);
        ans1 = svmlalt_n_f32(ans1, res3, 1.0);

        svst1_f32(all_true32, out     , ans0);
        svst1_f32(all_true32, out + 8 , ans1);
        out += 16;
        nc += 16;
    }
    if (nc < ncode) {
        svuint16_t c0, c1, c2, c3;
        uint64_t remaining = ncode - nc;
        if (remaining > 8) {
            svbool_t pred_hi = svwhilelt_b32(8, (int32_t)remaining);
            const float16_t* sim_table = (float16_t*)table;
            table0 = svld1_f16(all_true, sim_table);
            table1 = svld1_f16(all_true, sim_table + 16);
            table2 = svld1_f16(all_true, sim_table + 32);
            table3 = svld1_f16(all_true, sim_table + 48);
            sim_table += 64;
            res0 = svdup_n_f16(0);
            res1 = svdup_n_f16(0);
            res2 = svdup_n_f16(0);
            res3 = svdup_n_f16(0);
            svfloat32_t ans0 = svdup_n_f32(0);
            svfloat32_t ans1 = svdup_n_f32(0);
            const uint16_t *p0  = (const uint16_t*)(codes + (uint64_t)idx[nc + 0]  * code_stride);
            const uint16_t *p1  = (const uint16_t*)(codes + (uint64_t)idx[nc + 1]  * code_stride);
            const uint16_t *p2  = (const uint16_t*)(codes + (uint64_t)idx[nc + 2]  * code_stride);
            const uint16_t *p3  = (const uint16_t*)(codes + (uint64_t)idx[nc + 3]  * code_stride);
            const uint16_t *p4  = (const uint16_t*)(codes + (uint64_t)idx[nc + 4]  * code_stride);
            const uint16_t *p5  = (const uint16_t*)(codes + (uint64_t)idx[nc + 5]  * code_stride);
            const uint16_t *p6  = (const uint16_t*)(codes + (uint64_t)idx[nc + 6]  * code_stride);
            const uint16_t *p7  = (const uint16_t*)(codes + (uint64_t)idx[nc + 7]  * code_stride);
    #define TAIL_PTR(k) ((k) < remaining ? idx[nc + (k)] : idx[0])
            const uint16_t *p8  = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(8)  * code_stride);
            const uint16_t *p9  = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(9)  * code_stride);
            const uint16_t *p10 = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(10) * code_stride);
            const uint16_t *p11 = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(11) * code_stride);
            const uint16_t *p12 = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(12) * code_stride);
            const uint16_t *p13 = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(13) * code_stride);
            const uint16_t *p14 = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(14) * code_stride);
            const uint16_t *p15 = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(15) * code_stride);
    #undef TAIL_PTR
            svuint16_t rgba0 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p0), svld1uh_u64(all_true, p2 ), 16), svsli_n_u64(svld1uh_u64(all_true, p4 ), svld1uh_u64(all_true, p6 ), 16), 32));
            svuint16_t rgba1 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p8), svld1uh_u64(all_true, p10), 16), svsli_n_u64(svld1uh_u64(all_true, p12), svld1uh_u64(all_true, p14), 16), 32));
            svuint16_t rgba2 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p1), svld1uh_u64(all_true, p3 ), 16), svsli_n_u64(svld1uh_u64(all_true, p5 ), svld1uh_u64(all_true, p7 ), 16), 32));
            svuint16_t rgba3 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p9), svld1uh_u64(all_true, p11), 16), svsli_n_u64(svld1uh_u64(all_true, p13), svld1uh_u64(all_true, p15), 16), 32));
            uint64_t m = 4;
            for (uint64_t x = 4; x + L2_PREFETCH_OFFSET < nsq; x += PREFETCH_STEP) {
                prefetch_L2(p0  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p1  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p2  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p3  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p4  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p5  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p6  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p7  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p8  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p9  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p10 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p11 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p12 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p13 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p14 + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p15 + (x + L2_PREFETCH_OFFSET));
                for (m = x; m < x + PREFETCH_STEP; m += 4) {
                    SVE_INNER_LOOP_BODY(true)
                }
            }
            for (; m < nsq; m += 4) {
                SVE_INNER_LOOP_BODY(true)
            }
            SVE_INNER_LOOP_BODY(false)

            ans0 = svmlalb_n_f32(ans0, res0, 1.0);
            ans1 = svmlalt_n_f32(ans1, res0, 1.0);
            ans0 = svmlalb_n_f32(ans0, res1, 1.0);
            ans1 = svmlalt_n_f32(ans1, res1, 1.0);
            ans0 = svmlalb_n_f32(ans0, res2, 1.0);
            ans1 = svmlalt_n_f32(ans1, res2, 1.0);
            ans0 = svmlalb_n_f32(ans0, res3, 1.0);
            ans1 = svmlalt_n_f32(ans1, res3, 1.0);

            svst1_f32(all_true, out, ans0);
            svst1_f32(pred_hi, out + 8, ans1);
        } else {
            svuint16_t zeros = svdup_n_u16(0);
            svbool_t pred_lo = svwhilelt_b32(0, (int32_t)remaining);
            const float16_t* sim_table = (float16_t*)table;
            table0 = svld1_f16(all_true, sim_table);
            table1 = svld1_f16(all_true, sim_table + 16);
            table2 = svld1_f16(all_true, sim_table + 32);
            table3 = svld1_f16(all_true, sim_table + 48);
            sim_table += 64;
            res0 = svdup_n_f16(0);
            res1 = svdup_n_f16(0);
            res2 = svdup_n_f16(0);
            res3 = svdup_n_f16(0);
            svfloat32_t ans0 = svdup_n_f32(0);
            svfloat32_t ans1 = svdup_n_f32(0);
    #define TAIL_PTR(k) ((k) < remaining ? idx[nc + (k)] : idx[0])
            const uint16_t *p0  = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(0)  * code_stride);
            const uint16_t *p1  = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(1)  * code_stride);
            const uint16_t *p2  = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(2)  * code_stride);
            const uint16_t *p3  = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(3)  * code_stride);
            const uint16_t *p4  = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(4)  * code_stride);
            const uint16_t *p5  = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(5)  * code_stride);
            const uint16_t *p6  = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(6)  * code_stride);
            const uint16_t *p7  = (const uint16_t*)(codes + (uint64_t)TAIL_PTR(7)  * code_stride);
    #undef TAIL_PTR
            svuint16_t rgba0 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p0), svld1uh_u64(all_true, p2), 16), svsli_n_u64(svld1uh_u64(all_true, p4), svld1uh_u64(all_true, p6), 16), 32));
            svuint16_t rgba1 = svreinterpret_u16_u64(svsli_n_u64(svsli_n_u64(svld1uh_u64(all_true, p1), svld1uh_u64(all_true, p3), 16), svsli_n_u64(svld1uh_u64(all_true, p5), svld1uh_u64(all_true, p7), 16), 32));
            svuint16_t rgba2, rgba3;
            uint64_t m = 4;
            for (uint64_t x = 4; x + L2_PREFETCH_OFFSET < nsq; x += PREFETCH_STEP) {
                prefetch_L2(p0  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p1  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p2  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p3  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p4  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p5  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p6  + (x + L2_PREFETCH_OFFSET));
                prefetch_L2(p7  + (x + L2_PREFETCH_OFFSET));
                for (m = x; m < x + PREFETCH_STEP; m += 4) {
                    SVE_HALF_INNER_LOOP_BODY(true)
                }
            }
            for (; m < nsq; m += 4) {
                SVE_HALF_INNER_LOOP_BODY(true)
            }
            SVE_HALF_INNER_LOOP_BODY(false)

            ans0 = svmlalb_n_f32(ans0, res0, 1.0);
            ans1 = svmlalb_n_f32(ans1, res1, 1.0);
            ans0 = svmlalb_n_f32(ans0, res2, 1.0);
            ans1 = svmlalb_n_f32(ans1, res3, 1.0);
            ans0 = svadd_f32_m(pred_lo, ans0, ans1);

            svst1_f32(pred_lo, out, ans0);
        }
    }
}

#else

void
RaBitQFloatBuild4BitLookupTable(const float* query,
                                uint64_t dim,
                                float inv_sqrt_d,
                                float16_t* table) {
    generic::RaBitQFloatBuild4BitLookupTable(query, dim, inv_sqrt_d, table);
}

void
RaBitQFloatBatchBinaryIPBy4BitLookup(const uint8_t* codes,
                                     uint64_t ncode,
                                     uint64_t dim,
                                     uint64_t code_stride,
                                     const float16_t* table,
                                     float* out,
                                     const uint32_t* idx) {
    generic::RaBitQFloatBatchBinaryIPBy4BitLookup(codes, ncode, dim, code_stride, table, out, idx);
}

#endif  // ENABLE_SVE

void
BuildBinaryQueryLookupTable(const float* vector, uint64_t dim, float inv_sqrt_d, float* lookup) {
#if defined(ENABLE_SVE)
    // The vectorised path below requires a 256-bit SVE vector (svcntw() == 8): it
    // performs whole-vector loads/stores on the 8- and 16-element stack buffers and
    // assumes every svst1_f32 writes exactly 8 table entries. On other vector
    // lengths it would read past `pattern`, leave table entries unwritten or write
    // past `tmp_pattern`, so fall back to the portable implementation there.
    if (svcntw() == 8) {
        const float pos = inv_sqrt_d < 1e-3F ? 1.0F : inv_sqrt_d;
        const float neg = inv_sqrt_d < 1e-3F ? 0.0F : -inv_sqrt_d;
        const float pattern[8] = {neg, neg, neg, neg, pos, pos, pos, pos};
        svbool_t all_true = svptrue_b32();
        svfloat32_t inv_sqrt_d1 = svdupq_n_f32(neg, pos, neg, pos);
        svfloat32_t inv_sqrt_d2 = svdupq_n_f32(neg, neg, pos, pos);
        svfloat32_t inv_sqrt_d3 = svld1_f32(all_true, pattern);
        const uint64_t num_bytes = dim / 8;
        const uint64_t plus_dim = dim & 7;
        float tmp_pattern[16];
        for (uint64_t nb = 0; nb < num_bytes; ++nb) {
            const float* q = vector + 8 * nb;
            svfloat32_t dim_lowhalf = svdup_n_f32(0);
            svfloat32_t dim_highhalf = svdup_n_f32(0);
            dim_lowhalf = svmla_n_f32_m(all_true, dim_lowhalf, inv_sqrt_d1, q[0]);
            dim_highhalf = svmla_n_f32_m(all_true, dim_highhalf, inv_sqrt_d1, q[4]);
            dim_lowhalf = svmla_n_f32_m(all_true, dim_lowhalf, inv_sqrt_d2, q[1]);
            dim_highhalf = svmla_n_f32_m(all_true, dim_highhalf, inv_sqrt_d2, q[5]);
            dim_lowhalf = svmla_n_f32_m(all_true, dim_lowhalf, inv_sqrt_d3, q[2]);
            dim_highhalf = svmla_n_f32_m(all_true, dim_highhalf, inv_sqrt_d3, q[6]);

            svfloat32_t dim_lowhalf_low = svadd_n_f32_m(all_true, dim_lowhalf, q[3] * neg);
            svfloat32_t dim_lowhalf_high = svadd_n_f32_m(all_true, dim_lowhalf, q[3] * pos);
            svfloat32_t dim_highhalf_low = svadd_n_f32_m(all_true, dim_highhalf, q[7] * neg);
            svfloat32_t dim_highhalf_high = svadd_n_f32_m(all_true, dim_highhalf, q[7] * pos);

            svst1_f32(all_true, tmp_pattern, dim_highhalf_low);
            svst1_f32(all_true, tmp_pattern + 8, dim_highhalf_high);
            for (int i = 0; i < 16; ++i) {
                const svfloat32_t dim_low_half =
                    svadd_n_f32_m(all_true, dim_lowhalf_low, tmp_pattern[i]);
                const svfloat32_t dim_high_half =
                    svadd_n_f32_m(all_true, dim_lowhalf_high, tmp_pattern[i]);
                svst1_f32(all_true, lookup + i * 16, dim_low_half);
                svst1_f32(all_true, lookup + i * 16 + 8, dim_high_half);
            }
            lookup += 256;
        }
        if (plus_dim != 0) {
            const float inv_sqrt_d_list[2] = {neg, pos};
            for (int byte_value = 0; byte_value < 256; ++byte_value) {
                lookup[byte_value] = 0;
                for (uint64_t bit_pos = 0; bit_pos < plus_dim; ++bit_pos) {
                    lookup[byte_value] += vector[8 * num_bytes + bit_pos] *
                                          inv_sqrt_d_list[(byte_value >> bit_pos) & 0x1];
                }
            }
        }
        return;
    }
#endif
    const uint64_t plane_bytes = (dim + 7) / 8;
    const float pos = inv_sqrt_d < 1e-3F ? 1.0F : inv_sqrt_d;
    const float neg = inv_sqrt_d < 1e-3F ? 0.0F : -inv_sqrt_d;
    const float delta = pos - neg;

    for (uint64_t block = 0; block < plane_bytes; ++block) {
        float lane_values[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
        float base_sum = 0.0F;
        const uint64_t dim_base = block * 8;
        for (uint64_t lane = 0; lane < 8; ++lane) {
            const uint64_t dim_id = dim_base + lane;
            if (dim_id < dim) {
                lane_values[lane] = vector[dim_id];
                base_sum += lane_values[lane];
            }
        }

        auto* block_lookup = lookup + (block << 8);
        block_lookup[0] = neg * base_sum;
        for (uint32_t mask = 1; mask < 256; ++mask) {
            const uint32_t low_bit = mask & (~mask + 1U);
            const uint32_t lane = static_cast<uint32_t>(__builtin_ctz(low_bit));
            block_lookup[mask] = block_lookup[mask ^ low_bit] + delta * lane_values[lane];
        }
    }
}

}  // namespace vsag::sve
