# Demo code for Puzzle: Distillation-Based NAS for Inference-Optimized LLMs

[![ICML 2025](https://img.shields.io/badge/ICML-2025-blue)](https://icml.cc/virtual/2025/poster/45275)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-red)](https://openreview.net/pdf?id=RY5MMBHRqo)
[![Video](https://img.shields.io/badge/Video-Summary-green)](https://www.youtube.com/watch?v=YsIv9Kr99C4)

A demonstration of **Puzzle**, a hardware-aware framework that accelerates Large Language Model (LLM) inference while preserving model capabilities through neural architecture search (NAS) and knowledge distillation.

**Note**: This repo focuses on the MIP-based architecture search, and is not a full release of Puzzle. We are working to release a full end-to-end implementation in the future which would contain the full capabilities of Puzzle, including block library construction and Blockwise Local Distillation.

## Paper

**Puzzle: Distillation-Based NAS for Inference-Optimized LLMs**  
*ICML 2025*

- [Full Paper](https://openreview.net/pdf?id=RY5MMBHRqo)
- [Video Summary](https://www.youtube.com/watch?v=YsIv9Kr99C4)
- [ICML 2025 Presentation](https://icml.cc/virtual/2025/poster/45275)

## Interactive Demo

The best way to understand Puzzle is through our Jupyter notebook demonstration:

```bash
jupyter notebook examples/puzzle_demonstration.ipynb
```

**[puzzle_demonstration.ipynb](examples/puzzle_demonstration.ipynb)** - This notebook provides:
- Complete walkthrough of Puzzle's MIP-based architecture search
- Real examples using Llama 3.3-70B-Instruct block library data
- Visualization of the trade-off between accuracy and runtime/memory
- Multiple deployment scenarios (H100 vs Edge devices)
- Interactive exploration of the search space

## Installation
Create a fresh python environment (recommended: python=3.12) using your favorite package manager and install the dependencies in [requirements.txt](requirements.txt).

## Overview

Puzzle consists of three stages:

1. **Block Library Construction** - Train alternative block variants with Blockwise Local Distillation (BLD)
2. **MIP-based Architecture Search** - Find optimal configuration for target hardware *(this repo)*
3. **Global Knowledge Distillation** - Fine-tune the assembled architecture

This repository implements Stage 2, demonstrating how to:
- Load pre-computed block libraries with accuracy/resource measurements
- Define hardware constraints (memory, latency requirements)
- Search for optimal architectures using Mixed-Integer Programming
- Visualize and compare different deployment scenarios


## Repo Structure

```
puzzle/
├── puzzle/
│   └── mip_nas.py                       # Core MIP-based NAS implementation
├── examples/
│   ├── puzzle_demonstration.ipynb       # Main demo
│   ├── data/                            # Pre-computed block library data
│   │    └── Llama-3.3-70B-Instruct/
│   │        ├── block_library.json
│   │        ├── measurement_info.json
│   │        └── parent_block_stats.json
│   └── standalone_mip_nas_example.py    # Standalone MIP NAS example script
└── requirements.txt
```
