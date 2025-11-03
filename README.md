# Custom LLM

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif" width="100" />
</p>

<p align="center">
  <a href="https://github.com/SolvyrEryx/custom-llm/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/SolvyrEryx/custom-llm/ci.yml?label=CI&logo=github" /></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" />
  <img alt="License" src="https://img.shields.io/github/license/SolvyrEryx/custom-llm" />
  <img alt="Issues" src="https://img.shields.io/github/issues/SolvyrEryx/custom-llm" />
</p>

A lightweight, end-to-end repository to train, evaluate, and run a custom transformer-based language model locally with PyTorch. No API keys needed.

---

## Highlights
- End-to-end: data prep → training → evaluation → inference
- Pure PyTorch implementation with clean, modular code
- Trainer script for quick experiments and an advanced training pipeline
- Inference script with CLI and Python usage
- Tokenizer, batching, and checkpointing utilities included
- Works fully offline; reproducible with fixed seeds

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257463-4d082cb4-7483-4eaf-bc25-6dde2628aabd.gif" width="100" />
</p>

## Architecture Overview
```
+-------------------------------+
| Dataset / Text Files          |
+-------------------------------+
            | tokenization
            v
+-------------------------------+
| Dataloader (batches)          |
+-------------------------------+
            |
            v
+-------------------------------+
| Transformer Model (PyTorch)   |
| - Embeddings                  |
| - Multi-Head Attention        |
| - FFN + Residuals + LayerNorm |
+-------------------------------+
            |
            v
+-------------------------------+
| Trainer (train_llm_advanced)  |
| - Optim/Sched, Grad Clip      |
| - Checkpointing               |
+-------------------------------+
            |
            v
+-------------------------------+
| Inference (inference.py)      |
| - Greedy/Top-k/Temp sampling  |
+-------------------------------+
```

---

## Quickstart: How to Use

Follow these steps to go from zero to a trained and running model.

### 1) Install
- Clone the repo and install dependencies:
```bash
git clone https://github.com/SolvyrEryx/custom-llm.git
cd custom-llm
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Prepare data (optional if using provided examples)
- Put your text files under data/ (create it if missing). Each line can be a training sample, or a single large corpus.
- Tips:
  - Clean unusual unicode and very long lines
  - Split train/val for evaluation (e.g., 95/5)

### 3) Train
- Quick training with defaults:
```bash
python train_llm_advanced.py \
  --data_dir data \
  --save_dir runs/exp1 \
  --epochs 1 \
  --batch_size 32 \
  --lr 3e-4 \
  --seq_len 256
```
- Common flags (examples):
  - --model_size small|base|custom
  - --num_layers 6 --n_heads 8 --d_model 512 --ffn_dim 2048
  - --grad_clip 1.0 --warmup_steps 2000 --seed 42

### 4) Inference (CLI)
```bash
python inference.py \
  --ckpt runs/exp1/latest.pt \
  --prompt "Hello, I am a custom LLM" \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_k 50
```

### 5) Inference (Python)
```python
from inference import load_model_and_tokenizer, generate

model, tok, device = load_model_and_tokenizer(
    ckpt_path="runs/exp1/latest.pt",
)

text = generate(
    model,
    tok,
    prompt="Once upon a time",
    max_new_tokens=80,
    temperature=0.8,
    top_k=50,
)
print(text)
```

### 6) Evaluate (optional)
- If evaluation utilities are provided in custom_llm_complete.py or train_llm_advanced.py, enable validation and capture perplexity/loss on the val split via flags such as --eval_every or --val_ratio.

---

## Makefile Automation

If a Makefile exists, you can use convenient commands. If it's not present, create a Makefile with the following targets to automate common tasks.

Example Makefile:
```Makefile
.PHONY: setup train infer test clean

PY=python
DATA_DIR?=data
RUN_DIR?=runs/exp1
CKPT?=$(RUN_DIR)/latest.pt

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	$(PY) train_llm_advanced.py --data_dir $(DATA_DIR) --save_dir $(RUN_DIR) --epochs 1 --batch_size 32 --lr 3e-4 --seq_len 256

infer:
	$(PY) inference.py --ckpt $(CKPT) --prompt "Hello from Make" --max_new_tokens 64 --temperature 0.8 --top_k 50

test:
	$(PY) -m pytest -q || echo "No tests yet"

clean:
	rm -rf $(RUN_DIR) __pycache__ .pytest_cache
```

Usage examples:
- make setup
- make train DATA_DIR=data RUN_DIR=runs/exp1
- make infer CKPT=runs/exp1/latest.pt

If you prefer not to add a Makefile, you can run the equivalent python commands shown in the Quickstart.

---

## Examples

- Tiny experiment on a toy dataset:
```bash
mkdir -p data
printf "hello world\nhello there\nhello custom llm\n" > data/toy.txt
python train_llm_advanced.py --data_dir data --save_dir runs/toy --epochs 1 --batch_size 64 --seq_len 64
python inference.py --ckpt runs/toy/latest.pt --prompt "hello" --max_new_tokens 20
```

- Longer-form generation with temperature/top-k:
```bash
python inference.py --ckpt runs/exp1/latest.pt --prompt "In a distant future," --max_new_tokens 200 --temperature 0.9 --top_k 100
```

---

## Requirements
```txt
python>=3.10
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/216644497-1951db19-8f3d-4e44-ac08-8e9d7e0d94a7.gif" width="400" />
</p>

## Contributing
Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License
This project is licensed under the MIT License - see LICENSE for details.

## Acknowledgments
- 🔧 Code Fixes, Stability Improvements & Optimizations: [@SolvyrEryx](https://github.com/SolvyrEryx)
- 🎨 Visual Design & Polish: [@SolvyrEryx](https://github.com/SolvyrEryx)
- 📝 Original Implementation: [@AnishVyapari](https://github.com/AnishVyapari)
- 📚 Inspired by "Attention Is All You Need"
- 🔥 Built with PyTorch and love ❤️

## Contact
For questions or feedback, join our Discord: https://discord.gg/dzsKgWMgjJ or open an issue.

## Connect
- Credits: Special thanks to [@SolvyrEryx](https://github.com/SolvyrEryx) for code fixes, stability, optimizations, and design enhancements
- Connect: https://guns.lol/ineffablebeast

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20,24&height=120&section=footer" width="100%"/>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=00D9FF&center=true&vCenter=true&width=600&lines=Built+with+PyTorch+%F0%9F%94%A5;Code+fixes+by+SolvyrEryx+%F0%9F%94%A7;Designed+by+SolvyrEryx+%F0%9F%8E%A8;100%25+Local+AI+%F0%9F%A4%96;No+API+Keys+Required+%E2%9C%A8" alt="Footer Animation" />
  <strong>Made with ❤️ by the Custom LLM Team</strong>
</p>
