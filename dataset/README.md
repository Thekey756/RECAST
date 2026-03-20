---
dataset_info:
  features:
    - name: instruction
      dtype: string
    - name: constraints
      sequence:
        - name: constraint_type
          dtype: string
        - name: constraint_description
          dtype: string
    - name: response
      dtype: string
  splits:
    - name: train
      num_bytes: 0
      num_examples: 0
    - name: test
      num_bytes: 0
      num_examples: 0
license: apache-2.0
---

# RECAST-30K

High-quality dataset for complex instruction following with 19+ constraint types

## Dataset Structure

### Data Fields

- `instruction`: The instruction given to the model
- `constraints`: List of constraints that the response must satisfy
  - `constraint_type`: Type of the constraint (e.g., "format", "length", "keyword", etc.)
  - `constraint_description`: Detailed description of the constraint
- `response`: The response from the model

### Data Splits

The dataset includes multiple splits with different constraint complexity levels:
- `train`: Training dataset (RECAST-30K)
- `test_5`: Test set with 5 constraints
- `test_10`: Test set with 10 constraints
- `test_15`: Test set with 15 constraints
- `test_all`: Test set with all constraints

## Usage

You can load the dataset using the HuggingFace `datasets` library:

```python
from datasets import load_dataset

# Load training dataset
dataset = load_dataset("zk-guo/RECAST-30K", split="train")

# Load test datasets
test_5 = load_dataset("zk-guo/RECAST-30K", name="test_5", split="test")
test_10 = load_dataset("zk-guo/RECAST-30K", name="test_10", split="test")
test_15 = load_dataset("zk-guo/RECAST-30K", name="test_15", split="test")
test_all = load_dataset("zk-guo/RECAST-30K", name="test_all", split="test")
```

## Dataset Citation

If you use this dataset, please cite:

```bibtex
@article{guo2025recast,
  title={RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data},
  author={Guo, Zhengkang and Liu, Wenhao and Xie, Mingchen and Xu, Jingwen and Huang, Zisu and Tian, Muzhao and Xu, Jianhan and Shen, Yuanzhe and Qian, Qi and Wu, Muling and others},
  journal={arXiv preprint arXiv:2505.19030},
  year={2025}
}
```

## License

This dataset is licensed under the Apache License 2.0.
