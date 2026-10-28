# vsag介绍

简体中文|[English](./README_en.md)

## 最新消息

- \[2026.12.30\]：vsag发布于Gitcode平台，实现RaBitQ 4bit查表、预处理阶段Computer对象池化复用、BFS图序底库重排、sq8和sq8_uniform batch4距离计算优化。

## 项目介绍

vsag是蚂蚁开源的高性能C++向量索引库，用于近似最近邻搜索。vsag将算法、量化、数据单元、IO等模块进行解耦，支持自由组合，其主推算法为hgraph图算法。鲲鹏优化基于开源vsag代码做侵入式修改，保持原有接口不变。

## 目录结构

代码仓目录结构如下：

```text
vsag/
├─ 0001-vsag_1.0.0-optimize-rabitq_sq8.patch      // 优化补丁
└── docs
   ├── LICENSE
   └── zh
      ├── feature_introduction.md                 // 特性介绍
      ├── best_practices.md                       // 最佳实践
      ├── installation_guide.md                   // 安装指南
      ├── quick_start.md                          // 快速入门
      └── release_notes.md                        // 版本说明书
```

## 版本说明

关于vsag的版本更新情况请参见《[版本说明书](./docs/zh/release_notes.md)》。

## 学习文档

| 学习资源名称 | 学习资源简介 |
| ------------ | ------------ |
| [特性介绍](./docs/zh/feature_introduction.md) | 介绍vsag的主要特性。 |
| [最佳实践](./docs/zh/best_practices.md) | 提供vsag的测试方法。 |
| [安装指南](./docs/zh/installation_guide.md) | 提供vsag源码编译安装方法的详细指导。 |
| [快速入门](./docs/zh/quick_start.md) | 提供vsag的快速入门指导。 |
| [版本说明书](./docs/zh/release_notes.md) | 提供vsag发布版本的基础信息和特性更新信息。 |

## 免责声明

此代码仓计划参与vsag开源组件，编码风格遵照开源软件，继承开源软件安全设计，不破坏开源软件设计及编码风格和方式。软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

## License

vsag采用Apache 2.0 License许可证授权，支持修改代码和再开源。详见[LICENSE](./LICENSE)文件。

本项目的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题或建议，或者需要反馈特性需求和bug报告，可以提交[Issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md)联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 致谢

感谢来自社区的每一个PR，欢迎贡献vsag！
