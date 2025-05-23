# DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning

This repository contains the official implementation of **DoMIX**, a continual learning framework based on modular LoRA adapters. This work was conducted during the first year of the Samsung-SNU industry-academic collaboration project:  
**"On-device few-shot continual learning with lightweight adapters."**

## Project Overview

The aim of this project is to enable **efficient domain-adaptive pretraining (DAP)** and continual learning on edge devices using lightweight, modular adapters.

During the first year, we proposed **DoMIX**, a novel adapter architecture that:
- Stores domain-specific knowledge in separate LoRA modules
- Mixes knowledge across domains using a trainable bridge matrix
- Maintains performance regardless of domain order
- Reduces memory and training time significantly compared to prior methods

## Key Contributions

- **DoMIX Architecture**: Concatenates domain-specific LoRA modules and learns how to mix them through a bridge matrix.
- **Domain Robustness**: Overcomes domain order sensitivity through independent adapter management.
- **Efficiency**: Reduces memory usage by **87%** and training time by **58%** compared to DAS baseline.
- **Strong Performance**: Achieves higher accuracy and F1-score than prior methods (DAS, NCL, Joint LoRA).

## Experimental Results

- **Benchmarks**: 6-domain DAP with tasks including sentiment classification, citation intent, and biomedical relation extraction.
- **Performance**:
  - Average accuracy: **81.67%**
  - Average F1-score: **77.84%**
- **Efficiency**:
  - Memory usage: **↓87%**
  - Training time: **↓58%**

## LLM Experiments

- Applied DoMIX to **LLaMA3-8B** and **Gemma2-9B** for commonsense reasoning tasks (BoolQ, PIQA, ARC, etc.)
  - Outperformed LoRA (80.88%) and DoRA (85.23%) with average accuracy **85.34%**
  - Demonstrated scalability to larger LLMs with **Gemma2-9B** (89.55%)

## Paper & Acknowledgements

- Paper accepted to **ACL 2025 Main Conference**
- This project was developed by **M.IN.D Lab @ Seoul National University** in collaboration with the **Samsung MX Language AI Team**  
  Contact: Dohoon Kim (dohoon.kim@snu.ac.kr)
