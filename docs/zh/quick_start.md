# 快速入门

请先参照[安装指南](./installation_guide.md)完成vsag源码安装后，参考本文档示例内容验证基于鲲鹏平台优化后的vsag的核心功能。

## 背景说明

本文档所述版本为基于vsag v1.0.0的鲲鹏平台优化版本，在开源代码基础上应用了鲲鹏优化补丁0001-vsag_1.0.0-optimize-rabitq_sq8.patch。补丁包含4项优化：RaBitQ 4bit查表、预处理阶段Computer对象池化复用、BFS图序底库重排、SQ8和SQ8 Uniform batch4距离计算，各优化点的原理与开启条件参见[特性介绍](./feature_introduction.md)。

与标准版相比，优化版本保持对外公共API不变，改动集中在库内部的搜索与距离计算路径，主要差异如下：

- **适用范围**：优化面向采用RaBitQ 1bit粗排与SQ8或SQ8 Uniform精排组合的hgraph索引，其他索引或量化组合不受影响。
- **开启方式**：Computer对象池化复用默认生效；SQ8和SQ8 Uniform batch4距离计算在索引采用SQ8或SQ8 Uniform量化时自动生效；RaBitQ 4bit查表需在index_param中设置`"rabitq_enable_batch": true`；BFS图序底库重排在索引加载后自动执行。
- **平台依赖**：RaBitQ 4bit查表需ARM平台且CPU支持SVE2+SVE-bitperm；BFS图序底库重排及batch4距离计算的SVE路径在SVE分支生效。不满足条件的硬件自动回退到通用实现，功能与标准版一致。

在鲲鹏平台使用该版本的原因是：鲲鹏处理器提供SVE2、点积等向量指令，上述优化可充分利用其向量寄存器和指令级并行能力，提升距离计算吞吐并降低搜索时延。测试与参数调优方法参见[最佳实践](./best_practices.md)。

## RaBitQ 1bit粗码与SQ8 Uniform精排码量化索引运行hgraph图算法示例

本章节以RaBitQ 1bit粗码和SQ8 Uniform精排码的量化索引运行hgraph图算法为例，提供vsag入门示例。

```cpp
#include <vsag/vsag.h>

#include <iostream>
#include <random>
#include <vector>

int main() {
    vsag::init();

    // 1. 准备数据
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

    // 2. 创建索引（hgraph：RaBitQ 1bit粗码用于图遍历，SQ8 Uniform精排码用于结果重排）
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
        std::cerr << "创建索引失败: " << index_result.error().message << std::endl;
        return -1;
    }
    auto index = index_result.value();

    // 3. 构建索引（量化底库向量并建图）
    if (auto build_result = index->Build(base); not build_result.has_value()) {
        std::cerr << "构建索引失败: " << build_result.error().message << std::endl;
        return -1;
    }
    std::cout << "索引中向量数量: " << index->GetNumElements() << std::endl;

    // 4. 执行搜索（vsag的KnnSearch每次处理一条查询）
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
            std::cerr << "搜索失败: " << search_result.error().message << std::endl;
            return -1;
        }
        results[i] = search_result.value();
    }

    // 5. 结果解析
    std::cout << "最近邻标签和对应距离:" << std::endl;
    for (int64_t i = 0; i < num_queries; ++i) {
        const auto& result = results[i];
        std::cout << "查询 " << i << ": ";
        for (int64_t j = 0; j < result->GetDim(); ++j) {
            std::cout << result->GetIds()[j] << "(" << result->GetDistances()[j] << ") ";
        }
        std::cout << std::endl;
    }

    engine.Shutdown();
    return 0;
}
```

预期结果如下（最近邻标签和距离的具体数值以实际运行为准）。

```output
索引中向量数量: 10000
最近邻标签和对应距离:
查询 0: 6281(14.2039) 1547(14.6793) 2298(15.0812) 9042(15.6477)
查询 1: 3724(13.7549) 895(14.1627) 4100(14.7528) 199(15.5407)
查询 2: 5806(13.9291) 2505(14.3303) 7298(15.0447) 1183(15.4026)
查询 3: 7067(14.6191) 2746(15.2389) 5392(15.8332) 964(16.0873)
查询 4: 1954(13.6117) 6865(14.0496) 2931(14.9112) 8674(15.2730)
```

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| ---- | ---- | ---- |
| 01 | 2026-12-30 | 第一次正式发布。|
