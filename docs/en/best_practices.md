# Best Practices

<!-- md-trans-meta sourceCommit=8085279f6cf821bce464666389b13ac09eee25ce translatedAt=2026-11-13T09:18:54.835Z pushedAt=2026-11-16T09:18:54.144Z -->

This document describes the testing methods of the optimized vsag on the Kunpeng platform. The vsag used in testing depends on the Kunpeng optimization patch file `0001-vsag_1.0.0-optimize-rabitq_sq8.patch`. This patch file applies to the HGraph index that uses RaBitQ 1-bit coarse ranking and SQ8/SQ8 Uniform fine ranking, and contains four optimizations. For details about each optimization and its enabling conditions, see [Feature Introduction](./feature_introduction.md).

## vsag Test Example

The following example uses the `gist-960-euclidean.hdf5` dataset.

1. See [Installation Guide](./installation_guide.md) to complete the installation and build of vsag.

2. Download the `gist-960-euclidean.hdf5` dataset. Assume the installation path is `/path/to/data`.

    ```bash
    cd /path/to/data
    wget http://ann-benchmarks.com/gist-960-euclidean.hdf5 --no-check-certificate
    ```

3. Create a configuration file for the gist dataset. Assume the path is `/path/to/config`. Enter `vi gist.yaml` in this path to create the configuration file, and then enter the following content in the file.

    ```yaml
    global:
        num_threads_building: 32
        num_threads_searching: 32
        exporters:
            print-directly:
                format: "table"
                to: "stdout"
            save-to-file:
                format: "json"
                to: "file:///path/to/vsag/gist.json"

    eval_case1:
        datapath: "/path/to/data/gist-960-euclidean.hdf5"
        type: "build,search"
        index_name: "hgraph"
        create_params: '{"dim":960,"dtype":"float32","metric_type":"l2","index_param":{"base_quantization_type":"rabitq","rabitq_enable_batch": true,"max_degree":192,"ef_construction":2000,"precise_quantization_type":"sq8_uniform", "use_reorder":true}}'
        search_params: '{"hgraph":{"ef_search":450}}'
        index_path: "/path/to/index/gist/hgraph_index"
        search_mode: "knn"
        topk: 10
        range: 0.5
        delete_index_after_search: false
    ```

    Press **ESC** to exit the insert mode, and type `:wq` to save and exit the file.

    >![icon note](public_sys-resources/icon-note.gif) **NOTE:**
    > - During the first run, set the `type` field to `build,search` to build and save the index. The index path is specified in the `index_path` field, and the saved index file is `hgraph_index`. In subsequent runs, set the `type` field to `search` to directly load the built index and perform performance tests.
    > - The `datapath` field specifies the path to the dataset. Ensure that the path points to the correct dataset file.
    > - The `max_degree`, `ef_construction`, and `ef_search` fields specify the maximum degree of the HGraph graph algorithm, the number of candidates during build, and the number of candidates during search, respectively. These three parameters are related to the quality of index build, and directly affect the recall and the QPS result of performance tests.
    > - The `rabitq_enable_batch` field specifies whether to enable the batch computation feature of RaBitQ. It is available only on the Arm platform when the CPU supports SVE2 + SVE-bitperm. This parameter is not required when running on x86.

4. Run the performance test.

    ```bash
    cd /path/to/vsag
    ./build-release/tools/eval/eval_performance /path/to/config/gist.yaml
    ```

    After the performance test is complete, the output is as follows. Pay attention to the `QPS` and `LatencyAvg` metrics.

    ```output
    +------------+--------+------------+-----+----------+------------+----------------------------------------+---------------+-----------+-----+------------------------------+----------------+--------------+----------------+-----------+
    | Name       | Index  | NumVectors | Dim | DataType | MetricType | IndexParam                             | Memory(build) | BuildTime | TPS | SearchParam                  | Memory(search) | QPS          | LatencyAvg(ms) | RecallAvg |
    +------------+--------+------------+-----+----------+------------+----------------------------------------+---------------+-----------+-----+------------------------------+----------------+--------------+----------------+-----------+
    | eval_case1 | hgraph | 1000000    | 960 | float32  | l2         | {"base_quantization_type":"rabitq","e- | N/A           | N/A       | N/A | {"hgraph":{"ef_search":450}} | 1.87 GB        | 11916.567383 | 2.685337       | 0.956100  |
    |            |        |            |     |          |            | f_construction":2000,"max_degree":192- |               |           |     |                              |                |              |                |           |
    |            |        |            |     |          |            | ,"precise_quantization_type":"sq8_uni- |               |           |     |                              |                |              |                |           |
    |            |        |            |     |          |            | form","rabitq_enable_batch":true,"use- |               |           |     |                              |                |              |                |           |
    |            |        |            |     |          |            | _reorder":true}                        |               |           |     |                              |                |              |                |           |
    +------------+--------+------------+-----+----------+------------+----------------------------------------+---------------+-----------+-----+------------------------------+----------------+--------------+----------------+-----------+

    ```

## Performance Test Parameter Tuning Guide

In the configuration file, adjust the values of the `max_degree`, `ef_construction`, and `ef_search` fields to optimize the performance of index build and search. The parameters are described in [**Table 1** Tuning parameters](#tuning-parameters).

**Table 1** Tuning parameters<a id="tuning-parameters"></a>

| Parameter | Description | Tuning Suggestion |
| -------- | -------- | -------- |
| max_degree | Maximum degree of the HGraph graph algorithm | A smaller value reduces the index size but may reduce the search accuracy; a larger value improves the search accuracy but increases the index size and build time. |
| ef_construction | Number of candidates during graph build | A smaller value speeds up index build but may reduce the index quality; a larger value improves the index quality but increases the index build time. Use a larger value during build to obtain better index quality. |
| ef_search | Number of candidates during graph search | A smaller value speeds up the search but may reduce the search accuracy; a larger value improves the search accuracy but increases the search time, resulting in a QPS drop. |

>![icon note](public_sys-resources/icon-note.gif) **NOTE:**
> When the `type` field is set to `build,search`, perform the above parameter tuning to find the parameter combination with the highest QPS while ensuring that the recall is greater than 0.95.
> For the gist dataset, the recommended tuning parameters are `max_degree=192`, `ef_construction=2000`, and `ef_search=450`.

## Unit Test

Run the following unit test script.

```bash
cd /path/to/vsag
BUILD_DIR=build-ft bash scripts/coverage/run_ut.sh --no-coverage \
    --filter '[ut][RaBitQuantizer],[ut][RaBitQuantizerParameter],[ut][RaBitQSplitDataCell],[ut][query_computer_pool],[ut][SQ8Quantizer],[ut][SQ8QuantizerParameter],[ut][SQ8UniformQuantizer],[ut][SQ8UniformQuantizerParameter],[ut][simd]~[!benchmark],[ft][hgraph][rabitq],[ut][LabelTable]'
```

After the unit test is complete, the output is as follows.

```output
==================== TEST RESULTS ====================
  suite  : unittests: PASS
  suite  : functests: PASS
  suite  : tests_mockimpl: PASS
  filter : [ut][RaBitQuantizer],[ut][RaBitQuantizerParameter],[ut][RaBitQSplitDataCell],[ut][query_computer_pool],[ut][SQ8Quantizer],[ut][SQ8QuantizerParameter],[ut][SQ8UniformQuantizer],[ut][SQ8UniformQuantizerParameter],[ut][simd]~[!benchmark],[ft][hgraph][rabitq],[ut][LabelTable]
======================================================
RESULT: PASSED - all test suites passed
```

## Change History

| Version | Date | Description |
| ---- | ---- | ---- |
| 01 | 2026-12-30 | This is the first official release. |
