# Solving ARC with Test-Time Training (TTT)

This repository implements a research-oriented pipeline for solving tasks from the **Abstraction and Reasoning Corpus (ARC)** using **Test-Time Training (TTT)**.  
The approach closely follows the methodology described in:

**The Surprising Effectiveness of Test-Time Training for Few-Shot Learning**  
https://arxiv.org/pdf/2411.07279

The core idea is to combine **supervised fine-tuning** with **test-time adaptation**, allowing the model to specialize to each ARC task at inference time.

---

## High-Level Overview

The pipeline consists of four main stages:

1. Dataset augmentation using REARC
2. Supervised Fine-Tuning (SFT) of a base language model
3. Construction of a Test-Time Training (TTT) dataset
4. Per-task test-time adaptation during inference

This design explicitly separates **learning general priors** (SFT) from **task-specific specialization** (TTT).

---

## Base Model

- **Qwen 2.5 1.5B Instruct**

The model is used in instruction-following mode and operates over textual representations of ARC grids.

---

## Stage 1: REARC-Based Data Augmentation

Due to the small size of the original ARC dataset, we first generate an augmented dataset using **REARC-style transformations**.

This stage produces:
- ARC-like synthetic tasks
- Diverse input–output grid transformations
- Variations that preserve underlying abstract rules

The augmented dataset is used **only for supervised training** and never for evaluation.

---

## Stage 2: Supervised Fine-Tuning (SFT) with LoRA

The base model is fine-tuned on the REARC-augmented dataset using **Low-Rank Adaptation (LoRA)** adapters.

Key characteristics:
- LoRA adapters are trained while keeping the base model frozen  
- Instruction-style prompts  
- Input grids mapped to output grids  
- No test-time adaptation  
- No exposure to evaluation tasks  

The purpose of this stage is to instill strong **general inductive priors** over common ARC transformation patterns while keeping fine-tuning parameter-efficient.

---

## Stage 3: Task-Specific TTT Dataset Construction

For each ARC task at inference time, a **task-specific Test-Time Training (TTT) dataset** is constructed using the task’s in-context learning demonstrations.

Specifically:
- Each ARC task provides multiple input–output demo pairs  
- These demo pairs are converted into supervised training examples  
- A separate TTT dataset is constructed **per task**

To increase the amount and diversity of test-time data, we additionally apply:
- Input–output preserving transformations  
- Symmetry-based and structure-preserving augmentations  
- Variations that do not alter the underlying task rule  

Important constraints:
- TTT datasets are **task-local**
- No TTT data is shared across tasks

---

## Stage 4: Test-Time Training via Supervised Loss

During inference, the model undergoes **test-time supervised fine-tuning** using the task-specific TTT dataset.

For each task:

1. The model is initialized from the SFT checkpoint  
2. A small inner-loop optimization is performed  
3. A supervised loss is computed over:
   - All output grids corresponding to the in-context demo pairs  
   - The predicted output grid for the held-out test input  
4. Gradients are applied to the TTT LoRA parameters only  
5. The adapted model produces the final test prediction  
6. TTT LoRA parameters are reset before the next task  


---
