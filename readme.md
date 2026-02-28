<div align="center">

# 🚀 RECAST

## Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data

**Dataset & Evaluation Code for Complex Instruction Following**

<!-- Badges -->
<p>
  <a href="https://huggingface.co/wenhaoliu123">
    <img alt="Model" src="https://img.shields.io/badge/Model-HuggingFace-orange">
  </a>
  <a href="https://huggingface.co/datasets/zk-guo/RECAST-30K/blob/main/RECAST-30K.json">
    <img alt="Dataset" src="https://img.shields.io/badge/Dataset-HuggingFace-blue">
  </a>
  <a href="https://arxiv.org/abs/2505.19030">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-red">
  </a>
</p>

<!-- Navigation Links -->
<p>
  <a href="#quickstart">Quickstart</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#reproducibility">Reproducibility</a> •
  <a href="#citation">Citation</a>
</p>

</div>

---

## 📋 Overview

**RECAST** is an efficient and scalable **framework for expanding LLMs' ability to follow complex instructions with multiple constraints**. As the number of explicitly stated requirements increases (especially beyond 10 constraints), LLMs often struggle to accurately follow such instructions. RECAST addresses this challenge through a comprehensive approach:

- 🔧 **Data Synthesis Framework** - Systematic methodology for generating datasets with far more constraints than existing benchmarks (19+ constraint types)
- 📊 **RECAST-30K Dataset** - High-quality 30K multi-constraint instruction-following training dataset for model fine-tuning
- 🧪 **RECAST-Test Sets** - Four in-domain test sets with progressive difficulty (5, 10, 15, and all constraints)
- 🔍 **Evaluation Pipeline** - Comprehensive code for evaluating models with automatic constraint verification (rule-based for quantitative, LLM-based for qualitative)

### Key Features

🎯 **Beyond 10 Constraints**: Challenges models with instances containing far more constraints than existing datasets  
📊 **Practical Relevance**: Constraints extracted from real-world prompt-response pairs to ensure applicability  
✅ **Automatic Verification**: Rule-based validators for quantitative constraints and LLM-based validators for qualitative ones  
🔄 **RL-Ready Design**: Verifiable constraints enable reward function design for reinforcement learning  
📈 **Reproducible Results**: Models fine-tuned on RECAST-30K substantially improve complex instruction following while maintaining general capabilities  

---

## 📁 Repository Structure

```text
.
├── dataset/
│   ├── RECAST-30K.json                 # 30K training dataset
│   ├── RECAST-Test_5_constraints.json   # 5-constraint test set
│   ├── RECAST-Test_10_constraints.json  # 10-constraint test set
│   ├── RECAST-Test_15_constraints.json  # 15-constraint test set
│   └── RECAST-Test_all_constraints.json # All-constraint test set
├── code/
│   ├── evaluate.py                      # Evaluation pipeline
│   ├── template.py                      # Constraint templates
│   └── util.py                          # Utilities for constraint validation
├── evaluation_example/
│   └── RECAST-Test_results/
│       └── data/
│           ├── test1
│           └── test1_detailed.json
├── requirements.txt
└── README.md
```

---

## <a id="quickstart"></a> 🚀 Quick Start

### 1️⃣ Environment Setup

```bash
# Create a conda environment
conda create --name recast python=3.10.16
conda activate recast

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Evaluate Your Model on RECAST-Test

The RECAST evaluation pipeline assesses your model's ability to follow complex instructions with multiple constraints:

```bash
# Basic evaluation (5 constraints)
python code/evaluate.py \
  --api-key YOUR_API_KEY \
  --api-url YOUR_MODEL_API_URL \
  --test-set dataset/RECAST-Test_5_constraints.json

# Standard evaluation (10 constraints)
python code/evaluate.py \
  --api-key YOUR_API_KEY \
  --api-url YOUR_MODEL_API_URL \
  --test-set dataset/RECAST-Test_10_constraints.json

# Maximum difficulty (all constraints)
python code/evaluate.py \
  --api-key YOUR_API_KEY \
  --api-url YOUR_MODEL_API_URL \
  --test-set dataset/RECAST-Test_all_constraints.json
```

### 3️⃣ Fine-tune on RECAST-30K

To improve your model's complex instruction-following ability, fine-tune on RECAST-30K using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):

```bash
# Prepare dataset in LLaMA-Factory format and configure training
# Refer to LLaMA-Factory documentation for detailed setup

# After training, evaluate your fine-tuned model
python code/evaluate.py \
  --api-key YOUR_KEY \
  --api-url http://your-finetuned-model:8000 \
  --test-set dataset/RECAST-Test_all_constraints.json
```

---

## <a id="dataset"></a> 📊 Dataset Overview

### RECAST-30K: High-Quality Multi-Constraint Training Dataset

**RECAST-30K** is a large-scale, high-quality dataset of **30,000 instances** spanning **19+ constraint types**. Each example incorporates far more constraints than existing benchmarks (often exceeding 10 constraints), designed to systematically improve LLMs' ability to follow complex instructions. Key characteristics:

- 🔬 **Constraint Types**: 19+ types covering format, content, reasoning, safety, and domain-specific requirements
- 🌍 **Real-World Relevance**: Constraints extracted from actual prompt-response pairs ensuring practical applicability
- ✅ **Verifiable Design**: Each constraint is equipped with validators (rule-based for quantitative, LLM-based for qualitative)
- 🎯 **Designed for Fine-tuning**: Optimized for training LLMs to handle complex, multi-constraint scenarios

### RECAST-Test: Progressive Evaluation Benchmarks

The in-domain test sets feature **progressive difficulty levels** to systematically evaluate model performance:

| Dataset | Constraints | Samples | Purpose |
|---------|------------|---------|---------|
| RECAST-30K | Variable (5-20+) | 30,000 | Training dataset for fine-tuning |
| RECAST-Test-5 | 5 | 498 | Basic capability assessment |
| RECAST-Test-10 | 10 | 498 | Standard complexity evaluation |
| RECAST-Test-15 | 15 | 500 | Advanced capability testing |
| RECAST-Test-All | All  | 500 | Maximum difficulty challenge |

### Important Note on RECAST-Test Files

⚠️ **About the `response` field:**
The `response` field in RECAST-Test JSON files contains outputs generated by our fine-tuned model on RECAST-30K. 

**To evaluate your own model:**
1. Replace the `response` field with outputs from your target model
2. Keep all other fields (instruction, constraints, etc.) unchanged
3. Run the evaluation pipeline on the modified test set

Example workflow:
```bash
# 1. Get predictions from your model for each instruction
# 2. Update the response field in RECAST-Test JSON files with your model's outputs
# 3. Run evaluation
python code/evaluate.py \
  --api-key YOUR_KEY \
  --api-url http://your-model:8000 \
  --test-set dataset/RECAST-Test_all_constraints.json
```

---

## <a id="evaluation"></a> 🔧 Code Components & Evaluation

### `evaluate.py` - Main Evaluation Pipeline

Comprehensive evaluation framework for in-domain testing:

```bash
python code/evaluate.py \
  --api-key YOUR_API_KEY \
  --api-url YOUR_API_URL \
  --test-set dataset/RECAST-Test_all_constraints.json \
  --output-dir results/
```


### `template.py` - Constraint Templates

Provides structured templates for defining **rule-based constraints**:

```python
from code.template import ConstraintTemplate

# Define custom constraints
constraint = ConstraintTemplate(
    constraint_type="format",
    description="Output must be in JSON format",
    validation_fn=lambda x: is_valid_json(x)
)
```

### `util.py` - Constraint Utilities

Core utilities for constraint validation:

- **Rule-Based Constraint Validators**: Functions to verify whether responses satisfy specific constraints (format detection, keyword matching, word/sentence length, starting/ending words, case checking, punctuation rules, etc.)
- **Multilingual Support**: Support for detecting and processing text in multiple languages (Chinese, English, French, German, Japanese, Russian)

### Testing Different Complexity Levels

```bash
# Basic: 5 constraints
python code/evaluate.py --api-key KEY --api-url URL --test-set dataset/RECAST-Test_5_constraints.json

# Intermediate: 10 constraints
python code/evaluate.py --api-key KEY --api-url URL --test-set dataset/RECAST-Test_10_constraints.json

# Advanced: 15 constraints
python code/evaluate.py --api-key KEY --api-url URL --test-set dataset/RECAST-Test_15_constraints.json

# Maximum: All constraints
python code/evaluate.py --api-key KEY --api-url URL --test-set dataset/RECAST-Test_all_constraints.json
```

### Automatic Constraint Verification

The evaluation pipeline includes built-in validators for automatic constraint checking:

- **Quantitative Constraints**: Rule-based validators for numerical, format, and structural requirements
- **Qualitative Constraints**: LLM-based validators for semantic and content-based requirements
- **Detailed Reports**: Per-constraint breakdown and aggregate satisfaction metrics

---

## <a id="reproducibility"></a> 🔬 Reproducibility & Model Improvement

### Environment Setup

```bash
# Create conda environment
conda create --name recast python=3.10.16
conda activate recast

# Install dependencies
pip install -r requirements.txt
```

### Training on RECAST-30K

We use **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** for efficient fine-tuning on RECAST-30K:

```bash
# Step 1: Format RECAST-30K for LLaMA-Factory
# (Refer to LLaMA-Factory documentation for dataset format conversion)

# Step 2: Configure training parameters
# Edit config.yaml with your training settings

# Step 3: Launch training
# (Follow LLaMA-Factory setup instructions)

# Step 4: Evaluate improvement
python code/evaluate.py \
  --api-key YOUR_KEY \
  --api-url http://your-finetuned-model:8000 \
  --test-set dataset/RECAST-Test_all_constraints.json
```

### Using Constraint Validators for Reinforcement Learning

RECAST's verifiable constraints enable reward function design:

- Extract constraint validators from evaluation pipeline
- Design reward signals based on constraint satisfaction
- Train or fine-tune models using RL algorithms
- Further boost performance on complex multi-constraint tasks

---

## 📄 Paper & Resources

**Title:** RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data

**Conference:** ICLR 2026 (Accepted)

**Key Contributions:**
- 🔬 Efficient and scalable framework for synthesizing complex multi-constraint datasets
- 📊 RECAST-30K: 30K instances with 19+ constraint types exceeding existing benchmarks
- ✅ Automatic constraint verification system with rule-based and LLM-based validators
- 🎯 Demonstrated substantial improvements in complex instruction following while maintaining general capabilities
- 🔄 RL-ready design enabling reward function integration for further optimization

**Resources:**
- 📝 [Paper](https://arxiv.org/abs/2505.19030) - Full paper on arXiv
- 🤗 [Model Hub](https://huggingface.co/wenhaoliu123) - Pre-trained models on HuggingFace
- 📊 [Dataset Hub](https://huggingface.co/datasets/zk-guo/RECAST-30K/blob/main/RECAST-30K.json) - RECAST-30K and test sets on HuggingFace

---

## 🎯 Best Practices & Tips

### Start Your RECAST Journey

1. **Baseline Evaluation**: Test your model on RECAST-Test-5 to establish baseline
2. **Progressive Difficulty**: Move to Test-10, Test-15, and Test-All as needed
3. **Fine-tune on RECAST-30K**: Improve performance with our training dataset
4. **Measure Improvement**: Re-evaluate to see gains across all difficulty levels
5. **Optimize with RL**: Use constraint validators to design RL reward functions


---

## <a id="citation"></a> 🤝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{guo2025recast,
  title={RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data},
  author={Guo, Zhengkang and Liu, Wenhao and Xie, Mingchen and Xu, Jingwen and Huang, Zisu and Tian, Muzhao and Xu, Jianhan and Shen, Yuanzhe and Qian, Qi and Wu, Muling and others},
  journal={arXiv preprint arXiv:2505.19030},
  year={2025}
}
```

---

## 📝 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

We acknowledge:
- The open-source community for their invaluable contributions
- [HuggingFace](https://huggingface.co/) for model hosting and dataset management
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for comprehensive training infrastructure
- [veRL](https://github.com/volcengine/verl) for reinforcement learning training framework

---

## 📧 Support & Feedback

- 💬 **Issues:** [GitHub Issues](#)
- 💌 **Email:** [zkguo24@m.fudan.edu.cn]
- 📝 **Discussions:** [GitHub Discussions](#)

We welcome contributions, feedback, and discussions! Feel free to open an issue or start a discussion.

---

<div align="center">

**Made with ❤️ for the research community**

**[↑ Back to Top](#-recast)**

</div>