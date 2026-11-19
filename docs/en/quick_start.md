# Quick Start

<!-- md-trans-meta sourceCommit=8085279f6cf821bce464666389b13ac09eee25ce translatedAt=2026-11-13T09:18:54.835Z pushedAt=2026-11-16T09:18:54.144Z -->

See [Installation Guide](./installation_guide.md) to complete the source code installation of vsag first, and then follow the examples in this document to verify the core functions of the vsag optimized for the Kunpeng platform.

## Background

The version described in this document is the Kunpeng platform optimized version based on vsag v1.0.0, which applies the Kunpeng optimization patch `0001-vsag_1.0.0-optimize-rabitq_sq8.patch` on top of the open-source code. The patch contains four optimizations: RaBitQ 4-bit table lookup, Computer object pooling and reuse in the preprocessing stage, BFS graph-order base vector reordering, and batch4 distance computation for SQ8 and SQ8 Uniform. For the principles and enabling conditions of each optimization, see [Feature Introduction](./feature_introduction.md).

Compared with the standard version, the optimized version keeps the public APIs unchanged, with changes concentrated in the internal search and distance computation paths of the library. The main differences are as follows:

- **Applicable scope**: The optimizations target the HGraph index that uses the combination of RaBitQ 1-bit coarse ranking and SQ8 or SQ8 Uniform fine ranking. Other indexes or quantization combinations are not affected.
- **Enabling method**: Computer object pooling and reuse takes effect by default; batch4 distance computation for SQ8 and SQ8 Uniform takes effect automatically when the index uses SQ8 or SQ8 Uniform quantization; RaBitQ 4-bit table lookup requires setting `"rabitq_enable_batch": true` in `index_param`; BFS graph-order base vector reordering is executed automatically after the index is loaded.
- **Platform dependency**: RaBitQ 4-bit table lookup requires an Arm platform with CPU support for SVE2 + SVE-bitperm; the SVE paths of BFS graph-order base vector reordering and batch4 distance computation take effect in the SVE branch. On hardware that does not meet the conditions, the implementation automatically falls back to the generic implementation, with functionality consistent with the standard version.

The reason for using this version on the Kunpeng platform is that Kunpeng processors provide vector instructions such as SVE2 and dot product, and the preceding optimizations can fully utilize their vector registers and instruction-level parallel capabilities to improve distance computation throughput and reduce search latency. For testing and parameter tuning methods, see [Best Practices](./best_practices.md).

## HGraph Example with RaBitQ 1-bit Coarse and SQ8 Uniform Fine Ranking Quantization

This section uses a quantized index with RaBitQ 1-bit coarse codes and SQ8 Uniform precise codes to run the HGraph graph algorithm, providing a getting started example for vsag.

```cpp
#include <vsag/vsag.h>

#include <iostream>
#include <random>
#include <vector>

int main() {
    vsag::init();

    // 1. Prepare data
    int64_t dim = 128;
    int64_t num_base = 10000;
    int64_t num_queries = 5;

    std::vector<int64_t> ids(num_base);
    std::vector<float> base_vectors(dim * num_base);
    std::vector<float> query_vectors(dim * num_queries);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < num_base; ++i) {
        ids[i] = i;
    }
    for (auto& v : base_vectors) v = dist(rng);
    for (auto& v : query_vectors) v = dist(rng);

    auto base = vsag::Dataset::Make();
    base->NumElements(num_base)->Dim(dim)->Ids(ids.data())
        ->Float32Vectors(base_vectors.data())->Owner(false);

    // 2. Create an index (HGraph: RaBitQ 1-bit coarse codes for graph traversal, SQ8 Uniform precise codes for result reordering)
    std::string hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "rabitq",
            "rabitq_bits_per_dim_base": 1,
            "rabitq_bits_per_dim_query": 32,
            "rabitq_enable_batch": true,
            "precise_quantization_type": "sq8_uniform",
            "use_reorder": true,
            "max_degree": 32,
            "ef_construction": 200
        }
    }
    )";
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto index_result = engine.CreateIndex("hgraph", hgraph_build_parameters);
    if (not index_result.has_value()) {
        std::cerr << "Failed to create index: " << index_result.error().message << std::endl;
        return -1;
    }
    auto index = index_result.value();

    // 3. Build the index (quantize the base vectors and build the graph)
    if (auto build_result = index->Build(base); not build_result.has_value()) {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        return -1;
    }
    std::cout << "Number of vectors in the index: " << index->GetNumElements() << std::endl;

    // 4. Perform search (KnnSearch of vsag processes one query at a time)
    std::string hgraph_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100,
            "rabitq_one_bit_search": true
        }
    }
    )";
    int64_t topk = 4;
    std::vector<vsag::DatasetPtr> results(num_queries);
    for (int64_t i = 0; i < num_queries; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)
            ->Float32Vectors(query_vectors.data() + i * dim)->Owner(false);
        auto search_result = index->KnnSearch(query, topk, hgraph_search_parameters);
        if (not search_result.has_value()) {
            std::cerr << "Search failed: " << search_result.error().message << std::endl;
            return -1;
        }
        results[i] = search_result.value();
    }

    // 5. Parse results
    std::cout << "Nearest neighbor labels and corresponding distances:" << std::endl;
    for (int64_t i = 0; i < num_queries; ++i) {
        const auto& result = results[i];
        std::cout << "Query " << i << ": ";
        for (int64_t j = 0; j < result->GetDim(); ++j) {
            std::cout << result->GetIds()[j] << "(" << result->GetDistances()[j] << ") ";
        }
        std::cout << std::endl;
    }

    engine.Shutdown();
    return 0;
}
```

The expected result is as follows (the specific values of nearest neighbor labels and distances are subject to the actual run).

```output
Number of vectors in the index: 10000
Nearest neighbor labels and corresponding distances:
Query 0: 6281(14.2039) 1547(14.6793) 2298(15.0812) 9042(15.6477)
Query 1: 3724(13.7549) 895(14.1627) 4100(14.7528) 199(15.5407)
Query 2: 5806(13.9291) 2505(14.3303) 7298(15.0447) 1183(15.4026)
Query 3: 7067(14.6191) 2746(15.2389) 5392(15.8332) 964(16.0873)
Query 4: 1954(13.6117) 6865(14.0496) 2931(14.9112) 8674(15.2730)
```

## Change History

| Version | Date | Description |
| ---- | ---- | ---- |
| 01 | 2026-12-30 | This is the first official release. |
