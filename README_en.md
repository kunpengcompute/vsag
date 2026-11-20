# Introduction to vsag

<!-- md-trans-meta sourceCommit=8085279f6cf821bce464666389b13ac09eee25ce translatedAt=2026-11-13T09:18:54.835Z pushedAt=2026-11-16T09:18:54.144Z -->

This document introduces vsag and provides entries to learning documents and release notes, for users who need to use vsag.

[简体中文](./README.md)|English

## Latest News

\[2026-12-30\]: Based on the HGraph index, vsag implements RaBitQ 4-bit table lookup, Computer object pooling and reuse in the preprocessing stage, BFS graph-order base vector reordering, and batch4 distance computation optimization for sq8 and sq8_uniform.

## Project Introduction

vsag is a high-performance C++ vector indexing library open-sourced by Ant Group, used for approximate nearest neighbor search. vsag decouples modules such as algorithms, quantization, data units, and I/O, and supports free combination. Its flagship algorithm is the HGraph graph algorithm. The Kunpeng optimization makes intrusive modifications based on the open-source vsag code while keeping the original interfaces unchanged.

## Directory Structure

The directory structure of the code repository is as follows:

```text
vsag/
├─ 0001-vsag_1.0.0-optimize-rabitq_sq8.patch      // Optimization patch
├─ README.md                                      // Project introduction
└── docs
   ├── LICENSE
   └── en
      ├── menu_vsag.md                            // Document menu
      ├── feature_introduction.md                 // Feature introduction
      ├── best_practices.md                       // Best practices
      ├── installation_guide.md                   // Installation guide
      ├── quick_start.md                          // Quick start
      └── release_notes.md                        // Release notes
```

## Release Notes

For version updates of vsag, see [Release Notes](./docs/en/release_notes.md).

## Learning Documents

| Resource Name | Description |
| ------------ | ------------ |
| [Feature Introduction](./docs/en/feature_introduction.md) | Describes the main features of vsag. |
| [Best Practices](./docs/en/best_practices.md) | Provides the testing methods of vsag. |
| [Installation Guide](./docs/en/installation_guide.md) | Provides detailed instructions on building vsag from source code. |
| [Quick Start](./docs/en/quick_start.md) | Provides a quick start guide for vsag. |
| [Release Notes](./docs/en/release_notes.md) | Provides basic information and feature updates of released vsag versions. |

## Disclaimer

This code repository is planned to contribute to the vsag open-source project. It strictly adheres to the coding style and methods, as well as the security design, of open-source software. Any vulnerabilities and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. The Kunpeng computing community does not assume any responsibility for software vulnerabilities and security issues.

## License

vsag is licensed under the Apache 2.0 License, which allows code modification and re-open-sourcing. For details, see [LICENSE](./LICENSE).

The documents of this project are licensed under the CC-BY 4.0 License. For details, see [LICENSE](./docs/LICENSE).

## Contribution Statement

Contributions to the community are welcome. If you have any questions or suggestions during use, or need to report feature requests or bugs, submit an [issue](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md) to contact us. For details about how to contribute, see the [contribution guide](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to join the discussions in [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Acknowledgments

Thank you for every PR from the community. Contributions to vsag are welcome!
