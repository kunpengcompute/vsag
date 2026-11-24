# 常见问题和故障案例

本文档汇总vsag鲲鹏优化版本（基于vsag v1.0.0及优化补丁0001-vsag_1.0.0-optimize-rabitq_sq8.patch）使用过程中的常见问题和故障案例。常见问题主要说明参数、平台和索引的使用方式；故障案例主要分析优化未生效、性能或召回率不符合预期时的排查过程和处理方法。

## 常见问题

### 如何确认rabitq_enable_batch的4bit查表路径是否生效？

#### 问题

设置`"rabitq_enable_batch": true`后，不确定是否真正走到4bit查表路径。

#### 回答

有两类方法可判断4bit查表是否生效，分别是条件判断（理论角度）和perf判断（现象角度）。条件判断回答“按配置和平台应该走哪条路径”，perf判断回答“实际走了哪条路径”。建议先用条件判断确认配置，再用perf确认运行路径。

**方法一：按生效条件判断**

**批量查表是否开启**

满足以下条件时，`GetBatchMode()`返回1，进入4bit批量查表；不满足时走逐码路径。

- `base_quantization_type`为`rabitq`；
- `rabitq_enable_batch`为`true`；
- `rabitq_bits_per_dim_query`为32；
- `rabitq_bits_per_dim_base`为1。

这一层不要求CPU支持SVE2-bitperm。没有SVE2-bitperm时，仍可能使用generic批量实现。

**SVE内核是否运行**

在批量查表已开启的前提下，只有同时满足以下条件才调用SVE内核。

- 编译时包含SVE（`ENABLE_SVE`）；
- 运行时CPU支持SVE2和SVE-bitperm；
- `svcntw() == 8`；
- `dim`为64的倍数。

任一条件不满足时，回退到generic批量实现。

generic和SVE批量实现都使用4bit查找表，结果一致，实现方式的区别如下所示。

- generic：标量循环，逐条code、逐字节查16项float16表并累加，不依赖SVE，兼容所有平台。
- SVE：256位SVE实现，一次处理32条code，使用SVE向量表查找和float16向量累加，满足上述条件时吞吐通常高于generic。

>![表示说明的图片](public_sys-resources/icon-note.gif) **说明：**
> 条件不满足时程序不会报错：批量开关不满足时走逐码路径，SVE内核条件不满足时回退generic批量实现，功能均可正常使用。

**方法二：按perf热点判断**

perf采样命令如下所示，`perf.data`为采样结果文件。

```bash
perf record -o perf.data -p <pid> -- sleep 10
perf report -i perf.data
```

- 逐码路径：热点为`RaBitQFloatBinaryIP`，如`vsag::sve::RaBitQFloatBinaryIP`或`vsag::generic::RaBitQFloatBinaryIP`，同时可见`ComputeDistImpl`等逐码计算函数。说明批量查表未开启。
- generic批量：热点为`vsag::generic::RaBitQFloatBatchBinaryIPBy4BitLookup`。说明批量查表已开启，但未运行SVE内核。
- SVE批量：热点为`vsag::sve::RaBitQFloatBatchBinaryIPBy4BitLookup`。说明批量查表和SVE内核均已生效。
- 查询预处理阶段会出现`RaBitQFloatBuild4BitLookupTable`；SVE表构建函数只在编译包含SVE且运行时支持SVE2-bitperm时选择，否则使用generic表构建函数。

`ef_search`较小时端到端QPS可能没有明显变化，不能仅以QPS是否提升判断路径是否生效。

### x86上需要设置rabitq_enable_batch吗？

#### 问题

x86平台是否需要开启`rabitq_enable_batch`。

#### 回答

x86平台不会获得4bit查表的SVE加速。若将`rabitq_enable_batch`设置为`true`，且`rabitq_bits_per_dim_query`为32、`rabitq_bits_per_dim_base`为1，仍可进入generic批量查表；不满足这些条件时走逐码路径。是否开启取决于是否需要generic批量查表，但不要期待SVE加速，如无特殊需求可保持默认值。

### 只支持SVE1的ARM处理器会不会出现SIGILL？

#### 问题

只支持SVE1、不支持SVE2-bitperm的ARM处理器，运行4bit查表时是否会出现SIGILL。

#### 回答

不会。4bit查表内核按运行时能力分发，只有CPU同时支持SVE2和SVE-bitperm时才调用SVE实现，否则回退到generic批量实现，不会执行SVE2-bitperm指令。回退generic时批量查表仍然开启。若实际运行出现SIGILL，请确认使用的二进制和补丁版本与本文档一致。

### 索引构建完成后立即搜索，为什么没有BFS重排效果？

#### 问题

在同一进程中完成Build后直接调用`KnnSearch`，未观察到BFS重排效果。

#### 回答

BFS图序底库重排在索引加载（反序列化）完成后执行，不是构建期优化。如果在同一进程中完成Build后直接`KnnSearch`，不会触发重排。需要先保存索引，再重新加载后搜索，才能获得该优化效果。该优化仅在ARM SVE环境生效。

### 预处理阶段的Computer池化需要配置吗？

#### 问题

是否需要为Computer池化设置额外参数。

#### 回答

不需要，hgraph搜索时默认生效。将`use_reorder`设置为`true`且reorder使用精码表时，池中需要同时保留basic Computer和precise Computer，同一query的预处理次数由L+2次降为2次；reorder使用base codes时，三个阶段共享basic Computer，次数降为1次；未开启reorder时，由L+1次降为1次。

### 配置了precise_quantization_type，召回率为什么没有提升？

#### 问题

已经配置`precise_quantization_type`，但召回率没有明显提升。

#### 回答

`precise_quantization_type`只指定精排量化类型，精码表要参与重排还需设置`use_reorder`为`true`。如果未开启reorder，或索引实际使用base codes重排，精码表不会参与结果重排，召回率提升有限。建议同时确认`use_reorder`和精排配置，并在目标数据集上对比召回率。

### 内存占用较高，主要来自哪里？

#### 问题

索引加载或搜索时内存占用较高，需要定位主要来源。

#### 回答

- `use_reorder=true`会额外保存精码表，内存占用随向量数量和精排量化类型增加。
- `max_degree`越大，图邻接表占用越高。
- 量化类型和位宽决定粗码表大小；保留原始向量或附加信息也会增加内存。

处理建议如下所示。

- 在召回率允许的前提下关闭`use_reorder`，或调整精排量化配置。
- 适当降低`max_degree`。
- 避免保留不需要的原始向量和附加信息。
- 在相同数据集和参数下对比不同配置的`Memory(search)`，不要只看构建阶段内存。

### 如何一次查询多条向量？

#### 问题

`KnnSearch`一次只能传入一条查询，如何实现多查询。

#### 回答

vsag的`KnnSearch`每次处理一条查询。多查询场景可在应用中循环调用，并使用多线程并发；使用`eval_performance`测试时，通过`num_threads_searching`控制搜索线程数。

### 为什么第一次要设置type为build,search，之后要改成search？

#### 问题

第一次运行和后续运行的`type`配置为什么不同。

#### 回答

第一次运行需要构建索引并保存到`index_path`指定的路径，因此使用`build,search`；索引保存后，后续测试改为`search`直接加载，避免重复建图。若索引尚未构建就使用`search`，将无法加载索引。

### 保存后重新加载，为什么性能和刚构建时不同？

#### 问题

同一份索引保存后重新加载，性能与刚构建时存在差异。

#### 回答

重新加载后可能执行BFS图序底库重排，向量和图数据的物理布局发生变化，缓存局部性随之改变；同时首次搜索还存在cache预热等因素。因此性能测试建议先完成构建和保存，再以`search`模式加载索引进行稳定对比。

### 使用删除或更新后，还能享受BFS重排吗？

#### 问题

索引发生删除或更新后，是否还能获得BFS重排带来的缓存收益。

#### 回答

不能。索引一旦存在删除记录或启用强制删除，加载时会跳过BFS重排。重复标签、属性过滤、附加信息和原始向量等特性也会导致同样的结果。此类索引仍可正常搜索，但不会获得BFS重排带来的缓存收益。

### 属性过滤、附加信息能和这些优化同时使用吗？

#### 问题

属性过滤、附加信息与4bit查表、batch4、Computer池化、BFS重排是否可以同时使用。

#### 回答

可以同时使用，搜索功能正常，但BFS重排会自动跳过。4bit查表、batch4和Computer池化是否生效仍取决于各自的量化类型、平台和参数条件。此类场景建议重点验证召回率和QPS，不要以是否执行BFS重排判断优化是否生效。

## 故障案例

### 开启rabitq_enable_batch后，QPS提升为什么时高时低？

#### 问题现象描述

同样设置`"rabitq_enable_batch": true`，在不同测试中QPS提升幅度差异较大，`ef_search`较小时甚至观察不到明显变化。

#### 关键过程、根本原因分析

4bit查表优化的是RaBitQ粗排阶段的距离计算吞吐，端到端收益取决于距离计算在总耗时中的占比。

- `ef_search`较大时，候选集大，距离计算成为主要瓶颈，收益明显。
- `ef_search`较小时，图遍历、任务调度和访存占主导，收益下降，甚至无法观测。
- 未满足批量开关条件时走逐码路径；满足批量开关但未满足SVE内核条件时回退generic批量实现，均不会报错。

#### 结论、解决方案及效果

在目标召回率下逐步增大`ef_search`，观察QPS和LatencyAvg的变化；固定数据、线程和其他参数，只切换`rabitq_enable_batch`做对比。低延迟、小`ef_search`场景不要以4bit查表作为主要优化手段。先确认批量查表已开启（开关为`true`，且`rabitq_bits_per_dim_query=32`、`rabitq_bits_per_dim_base=1`）；再确认SVE内核条件（编译包含SVE、运行时支持SVE2和SVE-bitperm、`svcntw() == 8`、`dim`为64的倍数）。不满足SVE内核条件时generic批量实现仍可能生效，但加速幅度通常小于SVE内核。收益会随距离计算占比提高而变得明显。

### SQ8和SQ8 Uniform的batch4为什么没有加速？

#### 问题现象描述

索引使用`sq8`或`sq8_uniform`精排，未观察到batch4带来的加速。

#### 关键过程、根本原因分析

`sq8`和`sq8_uniform`的batch4实现不同，需要分开判断。

- `sq8`：`ScalarQuantizer::ComputeDistsBatch4Impl`调用`SQ8ComputeIPBatch4`或`SQ8ComputeL2SqrBatch4`，在SSE、AVX、AVX2、AVX512和NEON上都有真正的4路批量实现，query、lower_bound和diff只加载一次。用`perf`观察，热点应从`SQ8ComputeIP`或`SQ8ComputeL2Sqr`变为对应的`...Batch4`函数。如果热点没有变化，说明batch4路径没有被调用，或`dim`小于SIMD宽度触发了逐码回退。
- `sq8_uniform`：只有SVE实现了真正的4路批量内核，query只加载一次，并同时计算4路结果；其他ISA的`SQ8UniformComputeCodesIPBatch4`只是包装层，内部仍调用4次`SQ8UniformComputeCodesIP`。因此非SVE平台没有加速是预期行为，`perf`中可能只多出包装函数，真实热点仍是`SQ8UniformComputeCodesIP`。

#### 结论、解决方案及效果

对于`sq8`，先用`perf`确认热点是否切换到`SQ8ComputeIPBatch4`或`SQ8ComputeL2SqrBatch4`；如果没有切换，检查batch4调用路径和`dim`是否过小。对于`sq8_uniform`，非SVE平台没有加速属于设计行为，如需验证批量内核，请在支持SVE的ARM平台测试。不要仅凭QPS判断batch4是否生效，应结合perf热点函数变化确认。

### 加载索引后，BFS重排为什么仍未生效？

#### 问题现象描述

索引已经保存并重新加载，但未观察到BFS图序底库重排效果，也未获得邻居ID局部连续性带来的缓存收益。

#### 关键过程、根本原因分析

BFS重排只在满足以下条件时执行，任一条件不满足都会跳过。

- 运行环境不支持ARM SVE。
- 索引存在删除记录或启用了强制删除。
- 索引开启了重复标签。
- 索引使用了属性过滤。
- 索引使用了附加信息（extra info）。
- 索引保留了原始向量（raw vector）。
- 索引使用了mmap、异步I/O、只读I/O等非内存I/O。
- 精码表或路由图未加载到内存。
- 生成的排列与原始顺序一致（恒等排列）。

#### 结论、解决方案及效果

跳过重排时，搜索结果仍然正确，只是不会获得BFS重排带来的缓存收益。如需该优化，请在满足上述条件的索引上测试；存在删除、更新、属性过滤或附加信息的场景，不要以是否执行BFS重排判断其他优化是否生效。

### 相同数据和参数，召回率与《最佳实践》不一致？

#### 问题现象描述

数据集、构建参数和查询集相同，但召回率与《[最佳实践](./best_practices.md)》中的结果存在差异。

#### 关键过程、根本原因分析

HGraph构图过程存在随机性，例如候选边选择、并行构建时的任务调度等，即使数据集、参数和查询集完全相同，多次构图得到的图结构也可能不同，召回率因此会出现波动。此外，查询集、`ef_search`、`topk`、量化配置或索引加载状态不一致，也会造成召回率差异。通常情况下，构图随机性带来的差异较小。

#### 结论、解决方案及效果

确认对比使用同一份查询集和相同的`ef_search`、`topk`；优化前后对比时尽量复用同一个已保存的索引，只切换待验证的参数，避免构图随机性干扰；如果必须重新构图，建议多次构建后取召回率平均值，并保持构建线程数和构建流程一致；若差异明显，再检查参数、索引是否完整加载以及是否使用了不同的量化配置。《最佳实践》中的数据也是在特定数据集和参数下测得的，实际结果以本地运行情况为准。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| ---- | ---- | ---- |
| 01 | 2026-12-30 | 第一次正式发布。|
