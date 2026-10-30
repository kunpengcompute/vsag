# 安装指南

## 已验证环境

为保证您可以顺利安全地使用vsag，请确保所使用的环境信息在已验证环境范围内。

表1 已验证环境

| 操作系统 | CPU类型 | 内存 | 编译器 | 其他 |
| --- | --- | --- | --- | --- |
| openEuler 24.03 LTS SP3 | 鲲鹏950 7592C处理器 | 24 * 64GB | GCC 12.3.1 | CMake>=3.22.0 |

## 编译安装

先从GitHub下载vsag开源代码，然后从GitCode下载vsag补丁代码，应用补丁。

1. 获取开源代码，假设安装路径为“/path/to”。

    ```bash
    cd /path/to
    git clone -b v1.0.0 https://github.com/antgroup/vsag.git
    ```

2. 下载hnswlib的Patch文件，假设安装路径为“/path/to”。

    ```bash
    git clone -b v1.0.0 https://gitcode.com/boostkit/vsag.git vsag-patch
    ```

3. 应用Patch。

    ```bash
    cd /path/to/vsag
    patch -p1 < ../vsag-patch/0001-vsag_1.0.0-optimize-rabitq_sq8.patch
    ```

    使能补丁后的目录结构如下所示：

    ```output
    vsag/
    ├─ benchs/                                     // 基准测试
    ├─ cmake/                                      // CMake配置模块
    ├─ docker/                                     // Docker构建镜像
    ├─ docs/                                       // 项目文档
    ├─ examples/                                   // 示例程序
    ├─ extern/                                     // 第三方依赖（DiskANN、antlr4等）
    ├─ include/
    │   └─ vsag/                                   // 对外API头文件
    │       ├─ index.h                             // 索引抽象接口
    │       ├─ factory.h                           // 索引工厂接口
    │       ├─ dataset.h                           // 数据集接口
    │       ├─ constants.h                         // 参数键常量
    │       ├─ filter.h                            // 过滤器接口
    │       └─ resource.h                          // 资源管理接口
    ├─ mockimpl/                                   // 简易索引实现
    ├─ node_bindings/                              // Node.js绑定
    ├─ python/                                     // pyvsag包与打包配置
    ├─ python_bindings/                            // Python绑定
    ├─ scripts/                                    // 构建与测试脚本
    ├─ src/
    │   ├─ algorithm/                              // 索引算法实现
    │   │   ├─ bruteforce/                         // 暴力检索
    │   │   ├─ hgraph/                             // HGraph图索引
    │   │   │   ├─ hgraph.h                        // HGraph类定义
    │   │   │   ├─ hgraph.cpp                      // HGraph公共流程
    │   │   │   ├─ hgraph_build.cpp                // 建图流程
    │   │   │   ├─ hgraph_search.cpp               // 搜索流程
    │   │   │   ├─ hgraph_param_mapping.cpp        // JSON参数映射
    │   │   │   └─ hgraph_serialize.cpp            // 序列化与反序列化
    │   │   ├─ hnswlib/                            // HNSW图索引
    │   │   ├─ ivf/                                // IVF系列索引
    │   │   ├─ pyramid/                            // Pyramid索引
    │   │   ├─ simq/                               // SIMQ索引
    │   │   └─ sindi/                              // SINDI稀疏索引
    │   ├─ analyzer/                               // 索引分析器
    │   ├─ attr/                                   // 属性过滤（表达式解析与执行）
    │   ├─ datacell/                               // 底层数据存储单元
    │   │   ├─ flatten_datacell.h                  // 扁平量化码存储
    │   │   ├─ flatten_interface.h                 // 扁平存储接口
    │   │   ├─ graph_datacell.h                    // 图邻居存储
    │   │   ├─ graph_interface.h                   // 图存储接口
    │   │   ├─ compressed_graph_datacell.h         // 压缩图存储
    │   │   ├─ sparse_graph_datacell.h             // 稀疏图存储
    │   │   ├─ rabitq_split_datacell.h             // RaBitQ切分码存储
    │   │   └─ extra_info_datacell.h               // 额外信息存储
    │   ├─ factory/                                // 索引工厂实现
    │   ├─ impl/                                   // 公共组件实现
    │   │   ├─ allocator/                          // 内存分配器
    │   │   ├─ bitset/                             // 位图
    │   │   ├─ blas/                               // BLAS接口封装
    │   │   ├─ cluster/                            // KMeans聚类
    │   │   ├─ filter/                             // 过滤器
    │   │   ├─ heap/                               // 距离堆
    │   │   ├─ label_table/                        // 标签映射表
    │   │   ├─ logger/                             // 日志
    │   │   ├─ odescent/                           // ODescent图构建
    │   │   ├─ reasoning/                          // 搜索推理
    │   │   ├─ reorder/                            // 结果重排
    │   │   │   └─ flatten_reorder.cpp             // 基于扁平存储的精排
    │   │   ├─ searcher/                           // 搜索器
    │   │   │   ├─ basic_searcher.cpp              // 基础搜索器
    │   │   │   └─ parallel_searcher.cpp           // 并行搜索器
    │   │   ├─ thread_pool/                        // 线程池
    │   │   ├─ transform/                          // 向量变换
    │   │   ├─ query_computer_pool.h               // 请求级Computer对象池（新增）
    │   │   └─ query_computer_pool_test.cpp        // Computer对象池单元测试（新增）
    │   ├─ index/                                  // 顶层索引封装（DiskANN、HNSW等）
    │   ├─ io/                                     // 索引IO实现（内存/文件/异步）
    │   ├─ quantization/                           // 量化器
    │   │   ├─ computer.h                          // 查询计算机基类
    │   │   ├─ quantizer.h                         // 量化器基类
    │   │   ├─ product_quantization/               // 乘积量化器
    │   │   ├─ rabitq_quantization/                // RaBitQ量化器
    │   │   │   ├─ rabitq_quantizer.h              // 量化器接口
    │   │   │   └─ rabitq_quantizer.cpp            // 量化器实现
    │   │   ├─ scalar_quantization/                // 标量量化器
    │   │   │   ├─ scalar_quantizer.h              // SQ量化器接口
    │   │   │   ├─ scalar_quantizer.cpp            // SQ量化器实现
    │   │   │   └─ sq8_uniform_quantizer.cpp       // SQ8Uniform量化器实现
    │   │   └─ sparse_quantization/                // 稀疏量化器
    │   ├─ simd/                                   // SIMD距离计算
    │   │   ├─ kernels/                            // 跨指令集共用kernel
    │   │   │   └─ sq8_compute.h                   // SQ8距离计算kernel
    │   │   ├─ rabitq_simd.cpp                     // RaBitQ查表距离计算
    │   │   ├─ sq8_simd.cpp                        // SQ8距离计算
    │   │   ├─ sq8_uniform_simd.cpp                // SQ8Uniform距离计算
    │   │   ├─ amx.cpp                             // AMX实现
    │   │   ├─ avx.cpp                             // AVX实现
    │   │   ├─ avx2.cpp                            // AVX2实现
    │   │   ├─ avx512.cpp                          // AVX512实现
    │   │   ├─ generic.cpp                         // 通用标量实现
    │   │   ├─ neon.cpp                            // NEON实现
    │   │   ├─ sse.cpp                             // SSE实现
    │   │   └─ sve.cpp                             // SVE实现
    │   ├─ storage/                                // 索引序列化框架
    │   ├─ utils/                                  // 通用工具
    │   │   └─ visited_list.h                      // 图搜索访问标记
    │   ├─ CMakeLists.txt                          // 构建配置
    │   ├─ constants.cpp                           // 参数键常量实现
    │   ├─ inner_string_params.h                   // 内部参数键定义
    │   └─ query_context.h                         // 查询上下文
    ├─ tests/                                      // 单元测试
    ├─ tools/                                      // 辅助工具（eval评测、索引分析等）
    ├─ typescript/                                 // TypeScript绑定
    ├─ CMakeLists.txt                              // 顶层构建配置
    ├─ Makefile                                    // 构建命令入口
    ├─ CODE_OF_CONDUCT.md
    ├─ CONTRIBUTING.md
    ├─ DEVELOPMENT.md
    ├─ LICENSE
    └─ README.md
    ```

4. 编译vsag源码，获取libvsag.so。

    ```bash
    cd /path/to/vsag
    VSAG_ENABLE_TOOLS=ON make release
    ```
