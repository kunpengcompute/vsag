# 最佳实践

本文档介绍优化后的vsag在鲲鹏平台上的测试方法，测试所用vsag依赖鲲鹏的优化补丁文件0001-vsag_1.0.0-optimize-rabitq_sq8.patch。该补丁文件适用于采用RaBitQ 1bit粗排和SQ8/SQ8 Uniform精排的hgraph索引，共包含4项优化，各优化点的详细介绍和开启条件详见《[特性介绍](./feature_introduction.md)》。

## vsag测试实例

下方使用示例以gist-960-euclidean.hdf5数据集为例。

1. 请参见《[安装指南](./installation_guide.md)》完成vsag的安装和编译。

2. 下载gist-960-euclidean.hdf5数据集，假设安装路径为“/path/to/data”。

    ```bash
    cd /path/to/data
    wget http://ann-benchmarks.com/gist-960-euclidean.hdf5 --no-check-certificate
    ```

3. 创建gist数据集的配置文件，假设路径为“/path/to/config”，在该路径下输入`vi gist.yaml`创建配置文件后，在文件中输入以下配置信息。

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

    按ESC键退出编辑模式，输入`:wq`保存并退出文件。
    
    >![表示说明的图片](public_sys-resources/icon-note.gif) **说明：**
    > - 第一次运行时，需将type字段设置为"build,search"以构建并保存索引，索引路径在index_path字段中指定，保存的索引文件为hgraph_index。后续运行时，将type字段设置为"search"以直接加载构建完成的索引，进行性能测试。
    > - datapath字段指定数据集的路径，确保路径指向正确的数据集文件。
    > - max_degree、ef_construction、ef_search字段分别指定hgraph图算法的最大度数、构建时的候选数和搜索时的候选数，这三个参数与索引构建的质量相关，直接影响召回率和性能测试的QPS结果。
    > - rabitq_enable_batch字段指定是否启用RaBitQ的批量计算功能，仅在ARM平台且CPU支持SVE2+SVE-bitperm时可用，x86上运行时无需设置该参数。

4. 运行性能测试。

    ```bash
    cd /path/to/vsag
    ./build-release/tools/eval/eval_performance /path/to/config/gist.yaml
    ```

    性能测试完成后回显如下所示，性能关注QPS和LatencyAvg指标。

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

## 性能测试参数调优指导

在配置文件中，调整max_degree、ef_construction、ef_search字段的值，以优化索引构建和搜索的性能。调优参数如[**表 1** 调优参数表](#调优参数表)所示。

**表 1** 调优参数表<a id="调优参数表"></a>

| 参数名称 | 参数含义 | 调优建议 |
| -------- | -------- | -------- |
| max_degree | hgraph图算法的最大度数 | 较小的值可以减少索引的大小，但可能会降低搜索精度；较大的值可以提高搜索精度，但会增加索引的大小和构建时间。 |
| ef_construction | 图构建时的候选数 | 较小的值可以加快索引构建速度，但可能会降低索引质量；较大的值可以提高索引的质量，但会增加索引构建时间。构建时使用较大值以获得更好的索引质量。 |
| ef_search | 图搜索时的候选数 | 较小的值可以加快搜索速度，但可能会降低搜索精度；较大的值可以提高搜索精度，但会增加搜索时间，导致QPS下降。 |

>![表示说明的图片](public_sys-resources/icon-note.gif) **说明：**
> 在type字段设置为`build,search`时进行上述参数调优，确保召回率大于0.95的情况下，寻找QPS最高的参数组合。
> gist数据集的调优参数建议设置为max_degree=192、ef_construction=2000、ef_search=450。

## UT测试

执行以下UT测试脚本。

```bash
cd /path/to/vsag
BUILD_DIR=build-ft bash scripts/coverage/run_ut.sh --no-coverage \
    --filter '[ut][RaBitQuantizer],[ut][RaBitQuantizerParameter],[ut][RaBitQSplitDataCell],[ut][query_computer_pool],[ut][SQ8Quantizer],[ut][SQ8QuantizerParameter],[ut][SQ8UniformQuantizer],[ut][SQ8UniformQuantizerParameter],[ut][simd]~[!benchmark],[ft][hgraph][rabitq],[ut][LabelTable]'
```

UT测试完成后回显如下所示。

```output
==================== TEST RESULTS ====================
  suite  : unittests: PASS
  suite  : functests: PASS
  suite  : tests_mockimpl: PASS
  filter : [ut][RaBitQuantizer],[ut][RaBitQuantizerParameter],[ut][RaBitQSplitDataCell],[ut][query_computer_pool],[ut][SQ8Quantizer],[ut][SQ8QuantizerParameter],[ut][SQ8UniformQuantizer],[ut][SQ8UniformQuantizerParameter],[ut][simd]~[!benchmark],[ft][hgraph][rabitq],[ut][LabelTable]
======================================================
RESULT: PASSED - all test suites passed
```

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| ---- | ---- | ---- |
| 01 | 2026-12-30 | 第一次正式发布。|
