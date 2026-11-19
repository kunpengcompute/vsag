# Installation Guide

<!-- md-trans-meta sourceCommit=8085279f6cf821bce464666389b13ac09eee25ce translatedAt=2026-11-13T09:18:54.835Z pushedAt=2026-11-16T09:18:54.144Z -->

This document describes the build and installation procedure of vsag in verified environments, covering open-source code acquisition, patch application, and build, for users who need to install vsag on Kunpeng servers.

## Verified Environments

To ensure that you can use vsag smoothly and securely, ensure that your environment is within the range of verified environments.

Table 1 Verified environments

| OS | CPU | Memory | Compiler | Others |
| --- | --- | --- | --- | --- |
| openEuler 24.03 LTS SP3 | Kunpeng 950 processor | 24 x 64 GB | GCC 12.3.1 | CMake 3.22.0 or later |

## Build and Installation

First download the vsag open-source code from GitHub, then download the vsag patch code and apply the patch.

1. Obtain the open-source code. Assume the installation path is `/path/to`.

    ```bash
    cd /path/to
    git clone -b v1.0.0 https://github.com/antgroup/vsag.git
    ```

2. Download the patch file of vsag. Assume the installation path is `/path/to`.

    ```bash
    git clone -b v1.0.0 https://gitcode.com/boostkit/vsag.git vsag-patch
    ```

3. Apply the patch.

    ```bash
    cd /path/to/vsag
    patch -p1 < ../vsag-patch/0001-vsag_1.0.0-optimize-rabitq_sq8.patch
    ```

    After the patch is applied, the directory structure is as follows:

    ```output
    vsag/
    ├─ benchs/                                     // Benchmarks
    ├─ cmake/                                      // CMake configuration modules
    ├─ docker/                                     // Docker build image
    ├─ docs/                                       // Project documents
    ├─ examples/                                   // Example programs
    ├─ extern/                                     // Third-party dependencies (DiskANN, antlr4, etc.)
    ├─ include/
    │   └─ vsag/                                   // Public API header files
    │       ├─ index.h                             // Index abstraction interface
    │       ├─ factory.h                           // Index factory interface
    │       ├─ dataset.h                           // Dataset interface
    │       ├─ constants.h                         // Parameter key constants
    │       ├─ filter.h                            // Filter interface
    │       └─ resource.h                          // Resource management interface
    ├─ mockimpl/                                   // Simple index implementation
    ├─ node_bindings/                              // Node.js bindings
    ├─ python/                                     // pyvsag package and packaging configuration
    ├─ python_bindings/                            // Python bindings
    ├─ scripts/                                    // Build and test scripts
    ├─ src/
    │   ├─ algorithm/                              // Index algorithm implementations
    │   │   ├─ bruteforce/                         // Brute-force search
    │   │   ├─ hgraph/                             // HGraph graph index
    │   │   │   ├─ hgraph.h                        // HGraph class definition
    │   │   │   ├─ hgraph.cpp                      // HGraph common flow
    │   │   │   ├─ hgraph_build.cpp                // Graph build flow
    │   │   │   ├─ hgraph_search.cpp               // Search flow
    │   │   │   ├─ hgraph_param_mapping.cpp        // JSON parameter mapping
    │   │   │   └─ hgraph_serialize.cpp            // Serialization and deserialization
    │   │   ├─ hnswlib/                            // HNSW graph index
    │   │   ├─ ivf/                                // IVF series indexes
    │   │   ├─ pyramid/                            // Pyramid index
    │   │   ├─ simq/                               // SIMQ index
    │   │   └─ sindi/                              // SINDI sparse index
    │   ├─ analyzer/                               // Index analyzer
    │   ├─ attr/                                   // Attribute filtering (expression parsing and execution)
    │   ├─ datacell/                               // Underlying data storage units
    │   │   ├─ flatten_datacell.h                  // Flat quantized code storage
    │   │   ├─ flatten_interface.h                 // Flat storage interface
    │   │   ├─ graph_datacell.h                    // Graph neighbor storage
    │   │   ├─ graph_interface.h                   // Graph storage interface
    │   │   ├─ compressed_graph_datacell.h         // Compressed graph storage
    │   │   ├─ sparse_graph_datacell.h             // Sparse graph storage
    │   │   ├─ rabitq_split_datacell.h             // RaBitQ split code storage
    │   │   └─ extra_info_datacell.h               // Extra information storage
    │   ├─ factory/                                // Index factory implementation
    │   ├─ impl/                                   // Common component implementations
    │   │   ├─ allocator/                          // Memory allocator
    │   │   ├─ bitset/                             // Bitmap
    │   │   ├─ blas/                               // BLAS interface wrapper
    │   │   ├─ cluster/                            // KMeans clustering
    │   │   ├─ filter/                             // Filters
    │   │   ├─ heap/                               // Distance heaps
    │   │   ├─ label_table/                        // Label mapping table
    │   │   ├─ logger/                             // Logging
    │   │   ├─ odescent/                           // ODescent graph build
    │   │   ├─ reasoning/                          // Search reasoning
    │   │   ├─ reorder/                            // Result reordering
    │   │   │   └─ flatten_reorder.cpp             // Precise reordering based on flat storage
    │   │   ├─ searcher/                           // Searchers
    │   │   │   ├─ basic_searcher.cpp              // Basic searcher
    │   │   │   └─ parallel_searcher.cpp           // Parallel searcher
    │   │   ├─ thread_pool/                        // Thread pool
    │   │   ├─ transform/                          // Vector transform
    │   │   ├─ query_computer_pool.h               // Request-level Computer object pool (new)
    │   │   └─ query_computer_pool_test.cpp        // Computer object pool unit test (new)
    │   ├─ index/                                  // Top-level index wrappers (DiskANN, HNSW, etc.)
    │   ├─ io/                                     // Index I/O implementations (memory/file/async)
    │   ├─ quantization/                           // Quantizers
    │   │   ├─ computer.h                          // Base class of query computers
    │   │   ├─ quantizer.h                         // Quantizer base class
    │   │   ├─ product_quantization/               // Product quantizer
    │   │   ├─ rabitq_quantization/                // RaBitQ quantizer
    │   │   │   ├─ rabitq_quantizer.h              // Quantizer interface
    │   │   │   └─ rabitq_quantizer.cpp            // Quantizer implementation
    │   │   ├─ scalar_quantization/                // Scalar quantizers
    │   │   │   ├─ scalar_quantizer.h              // SQ quantizer interface
    │   │   │   ├─ scalar_quantizer.cpp            // SQ quantizer implementation
    │   │   │   └─ sq8_uniform_quantizer.cpp       // SQ8 Uniform quantizer implementation
    │   │   └─ sparse_quantization/                // Sparse quantizers
    │   ├─ simd/                                   // SIMD distance computation
    │   │   ├─ kernels/                            // Kernels shared across instruction sets
    │   │   │   └─ sq8_compute.h                   // SQ8 distance computation kernel
    │   │   ├─ rabitq_simd.cpp                     // RaBitQ lookup-based distance computation
    │   │   ├─ sq8_simd.cpp                        // SQ8 distance computation
    │   │   ├─ sq8_uniform_simd.cpp                // SQ8 Uniform distance computation
    │   │   ├─ amx.cpp                             // AMX implementation
    │   │   ├─ avx.cpp                             // AVX implementation
    │   │   ├─ avx2.cpp                            // AVX2 implementation
    │   │   ├─ avx512.cpp                          // AVX512 implementation
    │   │   ├─ generic.cpp                         // Generic scalar implementation
    │   │   ├─ neon.cpp                            // NEON implementation
    │   │   ├─ sse.cpp                             // SSE implementation
    │   │   └─ sve.cpp                             // SVE implementation
    │   ├─ storage/                                // Index serialization framework
    │   ├─ utils/                                  // Common utilities
    │   │   └─ visited_list.h                      // Visited flags for graph search
    │   ├─ CMakeLists.txt                          // Build configuration
    │   ├─ constants.cpp                           // Parameter key constants implementation
    │   ├─ inner_string_params.h                   // Internal parameter key definitions
    │   └─ query_context.h                         // Query context
    ├─ tests/                                      // Unit tests
    ├─ tools/                                      // Auxiliary tools (eval, index analysis, etc.)
    ├─ typescript/                                 // TypeScript bindings
    ├─ CMakeLists.txt                              // Top-level build configuration
    ├─ Makefile                                    // Build command entry
    ├─ CODE_OF_CONDUCT.md
    ├─ CONTRIBUTING.md
    ├─ DEVELOPMENT.md
    ├─ LICENSE
    └─ README.md
    ```

4. Build the vsag source code to obtain `libvsag.so`.

    ```bash
    cd /path/to/vsag
    VSAG_ENABLE_TOOLS=ON make release
    ```

    After the build is complete, the output is as follows.

    ```output
    [100%] Linking CXX executable eval_performance
    [100%] Built target eval_performance
    [100%] Linking CXX executable analyze_index
    [100%] Built target analyze_index
    gmake[1]: Leaving directory '/path/to/vsag/build-release'
    ```

## Change History

| Version | Date | Description |
| ---- | ---- | ---- |
| 01 | 2026-12-30 | This is the first official release. |
