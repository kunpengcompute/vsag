# 特性介绍

## 架构介绍

本节介绍vsag主要算法的逻辑结构与所包含的模块含义及作用。

### 总体分层

vsag采用自上而下的分层架构，各层之间通过抽象接口解耦，便于独立替换与优化。一次典型的搜索调用自上而下经过如下各层：

```text
公共 API 层（include/vsag）
    │  Index / Dataset / Factory / Filter / Resource
    ▼
工厂与引擎层（src/factory）
    │  Engine / IndexRegistry：解析索引类型与JSON参数，装配资源
    ▼
索引实现层（src/index）
    │  IndexImpl<T>：实现公共Index接口，负责入参校验与错误转换
    ▼
算法层（src/algorithm）
    │  InnerIndexInterface派生的各索引算法（HGraph、HNSW、IVF等）
    ▼
搜索器层（src/impl/searcher）
    │  BasicSearcher / ParallelSearcher：贪心图遍历与结果堆管理
    ▼
数据单元层（src/datacell）
    │  GraphInterface：图数据    FlattenInterface：向量码表
    ▼
量化层（src/quantization）
    │  Quantizer：训练与编码    Computer：查询侧距离计算
    ▼
SIMD 计算层（src/simd）
    │  按CPU指令集（SSE/AVX2/AVX512/NEON）运行时分发计算核
    ▼
IO 与序列化层（src/io、src/storage）
       BasicIo系列：内存/缓冲/异步/只读IO    序列化框架：流式读写与Footer
```

### 各模块含义及作用

| 模块（源码目录） | 核心类/接口 | 作用 |
| --- | --- | --- |
| 公共 API 层（`include/vsag`） | `Index`、`Dataset`、`Factory` | 对外暴露的稳定接口，用户通过`Factory::CreateIndex`以JSON参数创建索引 |
| 工厂与引擎层（`src/factory`） | `Engine`、`IndexRegistry` | 索引类型注册与创建分发，管理线程池、分配器等共享资源 |
| 索引实现层（`src/index`） | `IndexImpl<T>` | 模板化包装层，将内部索引适配为公共`Index`接口，统一处理入参检查、错误码转换与索引特性开关 |
| 算法层（`src/algorithm`） | `InnerIndexInterface`、`HGraph`、`HNSW`、`IVF` | 各索引算法的实现，负责建图、增删改查入口与索引级策略 |
| 搜索器层（`src/impl/searcher`） | `BasicSearcher`、`ParallelSearcher` | 图上的贪心搜索过程：候选扩展、访问去重、距离堆收敛，支持单线程与并行两种执行模式 |
| 数据单元层（`src/datacell`） | `GraphInterface`、`FlattenInterface` | 数据的物理组织：图邻接表（含稀疏图、压缩图）、向量码表、属性倒排表、附加信息存储等 |
| 量化层（`src/quantization`） | `Quantizer<QuantT>`、`Computer` | 向量压缩与距离计算：训练/编码/解码由`Quantizer`承担；`Computer`持有查询预处理结果，提供单点、批量、batch4等距离计算入口 |
| SIMD 计算层（`src/simd`） | `simd`命名空间各计算核 | 底层距离计算核，按量化类型（SQ8、RaBitQ、PQ等）组织，运行时按CPU能力分发到SSE/AVX2/AVX512/NEON实现 |
| IO 与序列化层（`src/io`、`src/storage`） | `BasicIo`、`StreamWriter/StreamReader` | 索引持久化：内存IO、缓冲IO、mmap、异步IO等后端，以及带Footer的流式序列化格式 |
| 公共基础组件（`src/impl`、`src/utils`、`src/attr`） | `Allocator`、`DistanceHeap`、`VisitedList`、`ResourceObjectPool`等 | 内存分配器、距离堆、访问位图、标签表、线程池、对象池，以及基于ANTLR4语法解析的属性过滤表达式与执行器 |

## 算法原理

HGraph（分层可导航图）是vsag的主力图索引算法，其逻辑结构自上而下分为路由层与数据层：

- **多层图结构**：底层图（`bottom_graph_`）包含全部向量；上层为若干路由图（`route_graphs_`），节点数逐层指数衰减。搜索时从最上层入口点出发，逐层贪心下降至底层，再在底层完成精细搜索。
- **双码表设计**：
  - 粗码表（`basic_flatten_codes_`）：底库向量经量化器（如RaBitQ）压缩后存放，搜索主链路在其上计算量化距离，降低内存占用与访存开销；
  - 精码表（`high_precise_codes_`，可选）：存放高精度编码（如SQ8），用于对候选结果重排（reorder），保证召回率。
- **搜索链路**：`HGraph::KnnSearch` → `BasicSearcher::Search`循环执行“取邻居（`GraphInterface::GetNeighbors`）→ 批量计算量化距离（`FlattenInterface::Query` + `Computer`）→ 更新距离堆”，直至收敛。
- **构建链路**：支持增量插入（NSW式逐点建边）与全量构建（ODescent全局建图）两种方式，并支持基于缓存的增量构建加速。

其他主要算法：HNSW（基于hnswlib的经典分层图）、IVF（倒排文件索引，支持GNO-IMI分区）、DiskANN（磁盘索引）、Pyramid（分段索引）、SINDI（稀疏向量索引）、BruteForce（暴力检索）等，均复用同一套量化、IO与基础组件设施。

## 核心优化特性

### RaBitQ 4bit查表优化

#### 功能描述

RaBitQ 1bit量化距离计算增加4bit查表路径，通过增大向量展开度和指令级并行度，增加ARM SVE向量寄存器的利用率。

#### 算法设计

原生采用逐bit select+float32乘加的精确单点计算，计算效率较低，现增加4bit查表优化，采用4bit nibble预建float16表+SVE tbl查表+float16累加的高吞吐批量近似计算，提高计算效率。

1. RaBitQ 4bit查表的查询流程

    - RaBitQFloatBuild4BitLookupTable —— 对查询向量做预处理，生成一个 float16 格式的 lookup table（每组 4 个 16 位浮点数，共 dim/4 组）。

    - RaBitQFloatBatchBinaryIPBy4BitLookup —— 对一批候选向量的 4bit code 做拆分取 nibble（0-15），通过查表累加得到float32内积分数。

2. SVE路径性能优化思路

    - 利用 SVE 的 svld1uh_u64（64位无符号半字加载）+ svsli_n_u64（移位插入）将分散存储的 4bit code 高效重组为 16bit 索引值。

    - 通过 svzip1/2 交叉解压指令实现 code 转置，将交错布局统一为连续索引以供 TBL 查表。

    - 查表主循环为32路，使用SVE_INNER_LOOP_DOUBLE_BODY批量处理查表。如果余数大于等于16，用原始SVE_INNER_LOOP_BODY 处理一个 16 路 chunk。如果不足16，使用已有的code填充补位，svwhilelt_b32 生成谓词屏蔽无效通道，保证只计算有效向量，使用该尾处理方式避免牺牲SIMD宽度。

### 预处理阶段Computer对象池化复用

#### 功能描述

预处理阶段Computer对象存在冗余创建的情况。引入请求级Computer对象池QueryComputerPool，同一query在同一flatten/quantizer上的Computer只创建一次并缓存于池中，各搜索阶段获取时命中即直接复用，从而减少重复预处理开销。

#### 算法设计

原始流程中一次搜索分三个串行阶段，每个阶段各自对同一个 query 调用 FactoryComputer：

```text
SearchWithRequest(query)
  ├─ Route Graph 搜索   → FactoryComputer(query)   // L次
  ├─ Bottom Graph 搜索  → FactoryComputer(query)   // 1次
  └─ Reorder 重排序     → FactoryComputer(query)   // 1次
```

FactoryComputer 多次重复创建会导致同一个 query 在相同 flatten/quantizer 上重复执行预处理，例如 ROM transform、normalize、lookup build 等。优化后在 HGraph::SearchWithRequest 顶层构造 QueryComputerPool（绑定当前query），并通过 QueryContext.computer_pool 传递给各搜索阶段。下游 BasicSearcher/ParallelSearcher/FlattenReorder 统一改走 AcquireQueryComputer 入口获取Computer：池内按码表身份查找，命中则直接复用缓存的Computer，未命中才调用 FactoryComputer 创建并缓存入池。reorder 阶段若使用 base codes（reorder_by_base_）则命中池中已有的 basic_computer；如果 reorder 使用 high_precise_codes_，则为精码表单独创建一个Computer（同样缓存于池中供后续复用）。在配置文件中指定了precise_quantization_type并把use_reorder设置为true的情况下，该优化将原先的L+2次预处理精简为2次，减少重复预处理开销。

### BFS图序底库重排

#### 功能描述

使用BFS图序重排底库的ID，在保持图拓扑结构不变的前提下，最大化邻居节点ID的局部连续性，可以提高缓存利用率和访问速度。

#### 算法设计

索引构建完成后，底库向量的物理存储顺序取决于添加顺序，与图搜索阶段的实际访问模式不完全一致。搜索阶段从 entry_point 出发沿图边扩展，访问的向量 id 序列通常呈跳跃特征，容易降低 cache 命中率。该优化在索引反序列化完成后，根据 entry_point 和图邻接关系生成类似BFS的排列方式，并对 basic_flatten_codes、bottom_graph、route_graph、label_table 等内部 id 相关存储做一致的物理重排和 id 映射更新。重排后，图上相邻或近层节点在物理存储中更可能落在相近位置，从而改善候选 id 的空间聚集性和 cache 行利用率。

### sq8和sq8_uniform batch4距离计算优化

#### 功能描述

sq8和sq8_uniform距离计算路径实现batch4距离计算，以支持单查询同时与4条底库向量进行批量距离计算。

#### 算法设计

现有的sq8和sq8_uniform距离计算采用1条查询与1条底库向量计算的形式，batch4距离计算一次性加载1条查询与4条底库向量后同时计算，每次迭代只加载一次共享输入（sq8为query/lower_bound/diff，sq8_uniform为query的uint8编码），4个码各自维护独立累加器，最后水平归约得到4个结果，减少重复内存读，提升指令并行度。两条路径的内部计算方式不同：

- sq8：每条码只需一次uint8零扩展为uint32->转换为float32->乘累加/乘累减的流程即可完成共享项拆分后的距离累加。
- sq8_uniform：利用SVE的svdot整数点积指令直接计算uint8×uint8点积并累加到4个独立的uint32累加器，水平归约后统一做度量修正。
