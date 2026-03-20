#!/usr/bin/env python3
"""
Script to prepare RECAST datasets for HuggingFace Hub
This converts JSON files to formats compatible with HuggingFace datasets library
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse


def analyze_json_structure(file_path: str, num_samples: int = 5) -> Dict[str, Any]:
    """Analyze the structure of a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        print(f"✓ Format: List with {len(data)} items")
        if len(data) > 0:
            first_item = data[0]
            print(f"✓ First item type: {type(first_item)}")
            if isinstance(first_item, dict):
                print(f"✓ Keys in first item: {list(first_item.keys())}")
                print(f"\n✓ Sample data (first item):")
                # Print first item without full content
                for k, v in first_item.items():
                    if isinstance(v, str):
                        print(f"  {k}: {v[:100]}..." if len(v) > 100 else f"  {k}: {v}")
                    elif isinstance(v, (list, dict)):
                        print(f"  {k}: {type(v).__name__} with {len(v)} items")
                    else:
                        print(f"  {k}: {v}")
        return {'type': 'list', 'length': len(data), 'item_type': type(first_item).__name__ if len(data) > 0 else 'unknown'}
    
    elif isinstance(data, dict):
        print(f"✓ Format: Dict with {len(data)} keys")
        print(f"✓ Keys: {list(data.keys())}")
        return {'type': 'dict', 'keys': list(data.keys())}
    
    return {'type': 'unknown'}


def convert_to_jsonl(input_path: str, output_path: str) -> int:
    """
    Convert JSON list to JSONL (JSON Lines) format
    JSONL is the recommended format for HuggingFace datasets
    """
    print(f"\n📝 Converting {os.path.basename(input_path)} to JSONL format...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Input must be a list of items for JSONL conversion")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ Converted to JSONL: {output_path}")
    print(f"✓ Total lines: {len(data)}")
    return len(data)


def create_dataset_card(dataset_name: str, description: str, output_dir: str = '.'):
    """Create a README.md file for HuggingFace dataset"""
    
    readme_content = f"""---
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

# {dataset_name}

{description}

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
@article{{guo2025recast,
  title={{RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data}},
  author={{Guo, Zhengkang and Liu, Wenhao and Xie, Mingchen and Xu, Jingwen and Huang, Zisu and Tian, Muzhao and Xu, Jianhan and Shen, Yuanzhe and Qian, Qi and Wu, Muling and others}},
  journal={{arXiv preprint arXiv:2505.19030}},
  year={{2025}}
}}
```

## License

This dataset is licensed under the Apache License 2.0.
"""
    
    output_path = os.path.join(output_dir, 'README.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✓ Created dataset card: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare RECAST datasets for HuggingFace Hub")
    parser.add_argument('--analyze', action='store_true', help='Analyze JSON structure')
    parser.add_argument('--convert', action='store_true', help='Convert JSON to JSONL')
    parser.add_argument('--create-card', action='store_true', help='Create dataset card')
    parser.add_argument('--all', action='store_true', help='Do all conversions')
    
    args = parser.parse_args()
    
    dataset_dir = Path('.')
    
    files_to_process = {
        'RECAST-30K.json': 'RECAST-30K.jsonl',
        'RECAST-Test_5_constraints.json': 'RECAST-Test_5_constraints.jsonl',
        'RECAST-Test_10_constraints.json': 'RECAST-Test_10_constraints.jsonl',
        'RECAST-Test_15_constraints.json': 'RECAST-Test_15_constraints.jsonl',
        'RECAST-Test_all_constraints.json': 'RECAST-Test_all_constraints.jsonl',
    }
    
    print("=" * 60)
    print("RECAST Dataset Preparation for HuggingFace Hub")
    print("=" * 60)
    
    if args.analyze or args.all:
        print("\n📊 Analyzing dataset structures...")
        for json_file in files_to_process.keys():
            if (dataset_dir / json_file).exists():
                print(f"\n🔍 Analyzing {json_file}:")
                analyze_json_structure(str(dataset_dir / json_file))
    
    if args.convert or args.all:
        print("\n\n📝 Converting to JSONL format...")
        for json_file, jsonl_file in files_to_process.items():
            if (dataset_dir / json_file).exists():
                try:
                    convert_to_jsonl(str(dataset_dir / json_file), str(dataset_dir / jsonl_file))
                except Exception as e:
                    print(f"✗ Error converting {json_file}: {e}")
    
    if args.create_card or args.all:
        print("\n\n📋 Creating dataset resources...")
        create_dataset_card(
            "RECAST-30K",
            "High-quality dataset for complex instruction following with 19+ constraint types"
        )
    
    print("\n" + "=" * 60)
    print("✓ Preparation complete!")
    print("\n📌 Next steps:")
    print("1. Push the JSONL files to HuggingFace Hub")
    print("2. Create a repository on HuggingFace: https://huggingface.co/new-dataset")
    print("3. Upload JSONL files to the 'data' directory")
    print("4. Users can load with: datasets.load_dataset('your-username/RECAST-30K')")
    print("=" * 60)


if __name__ == "__main__":
    main()
