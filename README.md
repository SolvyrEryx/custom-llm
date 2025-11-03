<!-- Neural Network Glassmorphism Banner -->
<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20,24&height=200&section=header&text=Custom%20LLM&fontSize=60&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Neural%20Network%20Powered%20Transformer&descSize=20&descAlignY=55" alt="Neural Network Banner" width="100%"/>
</div>

<!-- Animated Neural Network Loader -->
<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">
  <br>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&multiline=true&repeat=true&width=600&height=100&lines=Spinning+Neural+Nodes+%F0%9F%A7%A0;Glass+Effect+Processing...;AI+Model+Loading..." alt="Neural Loader">
</div>

<div align="center">
  
# Custom LLM (PyTorch)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](./requirements.txt) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)

![Stars](https://img.shields.io/github/stars/SolvyrEryx/custom-llm?style=social) ![Forks](https://img.shields.io/github/forks/SolvyrEryx/custom-llm?style=social) ![Issues](https://img.shields.io/github/issues/SolvyrEryx/custom-llm) ![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

A visually polished, GPT-style transformer model built from scratch in PyTorch with full training, inference, tokenizer, and docs. No API keys. No external LLMs. 100% local and hackable.

Join the community: [Discord invite](https://discord.gg/dzsKgWMgjJ)

</div>

---

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700">
</div>

## 🎨 Visual Design Credits

**Visual Design & Enhancements by [@SolvyrEryx](https://github.com/SolvyrEryx)**

*This fork features enhanced animations, glassmorphism effects, and polished visual elements designed by SolvyrEryx to showcase the technical capabilities of the LLM implementation with stunning presentation.*

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/229223156-0cbdaba9-3128-4d8e-8719-b6b4cf741b67.gif" width="400">
</div>

---

## ✨ Highlights

- 🧠 Multi-Head Self-Attention, GELU FFN, LayerNorm, Residual, Causal masking
- 🔤 Character tokenizer (easy to swap for BPE), temperature + top-k sampling
- 🎯 Clean training loop with AdamW, cosine LR, gradient clipping, checkpoints
- 💬 CLI chat and scriptable inference
- 📚 Fully documented with Quickstart, Setup, and rich README visuals

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284087-bbe7e430-757e-4901-90bf-4cd2ce3e1852.gif" width="100">
</div>

## 🚀 Quickstart

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Train
python train_llm_advanced.py

# 3) Chat / generate
python inference.py "Who created you?"
python inference.py  # interactive
```

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/216120981-b9507c36-0e04-4469-8e27-c99271b45ba5.png" width="30" />
  <img src="https://user-images.githubusercontent.com/74038190/216120974-24a76b31-7f39-41f1-a38f-b3c1377cc612.png" width="30" />
  <img src="https://user-images.githubusercontent.com/74038190/216121919-60befe4d-11c6-4227-8992-35221d12ff54.png" width="30" />
  <img src="https://user-images.githubusercontent.com/74038190/216121986-1a506a75-2381-41c2-baff-eeab94bcec74.png" width="30" />
</div>

## 📁 Repository Structure

```
custom-llm/
├── custom_llm_complete.py    # Core transformer + tokenizer + generation
├── train_llm_advanced.py     # Training pipeline, checkpoints
├── inference.py              # CLI and chat
├── README.md                 # This page (polished)
├── QUICKSTART.md             # 5‑minute guide
├── SETUP_GITHUB.md           # Repo polish + tips
├── requirements.txt          # Dependencies
├── .gitignore                # Clean repo
├── LICENSE                   # MIT
└── FILES_SUMMARY.txt         # Summary of all files
```

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-fcfd7baa4c6b.gif" width="100">
</div>

## 🏗️ Architecture (Animated Flow)

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="100">
</div>

```
Input → [Tokenizer] → Embedding → +Positional → [x N Transformer Blocks]
  ↓                                                         ↓
  └────────────────────────────────────────────────────────> Output Logits → Sample

🔄 Transformer Block:
  Input
   ↓
  [LayerNorm] → [Multi-Head Self-Attention] → [Residual Add]
   ↓
  [LayerNorm] → [FFN (GELU)] → [Residual Add]
   ↓
  Output
```

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257472-08e52665-c503-4bd9-aa20-f5a4dae769b5.gif" width="100">
</div>

## 🎯 Key Features

### 1. 🧠 Transformer Architecture

- **Multi-Head Self-Attention**: Parallel attention heads for richer representations
- **Position-wise FFN**: Two-layer network with GELU activation
- **Layer Normalization**: Pre-norm architecture for stable training
- **Residual Connections**: Skip connections around each sub-layer
- **Causal Masking**: Ensures autoregressive generation

### 2. 🎓 Training Pipeline

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Cosine annealing schedule
- **Gradient Clipping**: Prevents exploding gradients
- **Checkpointing**: Automatic model saving
- **Loss Tracking**: Comprehensive training metrics

### 3. 🚀 Generation

- **Temperature Sampling**: Control randomness
- **Top-k Sampling**: Limit to k most likely tokens
- **Interactive Chat**: CLI interface for conversations
- **Batch Processing**: Efficient inference

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif" width="100">
</div>

## 📊 Training

```python
from train_llm_advanced import train_model

# Train with default parameters
train_model(
    data_path="your_text_data.txt",
    num_epochs=10,
    batch_size=32,
    learning_rate=3e-4
)
```

## 🤖 Inference

```python
from inference import generate_text

# Generate text
generate_text(
    prompt="Once upon a time",
    max_length=100,
    temperature=0.8,
    top_k=40
)
```

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif" width="100">
</div>

## ⚙️ Configuration

Model hyperparameters can be adjusted in `custom_llm_complete.py`:

```python
config = {
    'vocab_size': 65,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 6,
    'ff_dim': 2048,
    'max_seq_len': 512,
    'dropout': 0.1
}
```

## 📈 Performance

<div align="center">

| Metric | Value |
|--------|-------|
| 🏃 Training Speed | ~1000 tokens/sec on GPU |
| 💾 Memory Usage | ~2GB VRAM for base model |
| ⚡ Generation Speed | ~50 tokens/sec |

</div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif" width="100">
</div>

## 📦 Requirements

```txt
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257463-4d082cb4-7483-4eaf-bc25-6dde2628aabd.gif" width="100">
</div>

## 🤝 Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/216644497-1951db19-8f3d-4e44-ac08-8e9d7e0d94a7.gif" width="400">
</div>

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- 🎨 **Visual Design & Polish**: [@SolvyrEryx](https://github.com/SolvyrEryx) - Enhanced animations, glassmorphism effects, and modern UI elements
- 📝 **Original Implementation**: [@AnishVyapari](https://github.com/AnishVyapari) - Core LLM architecture and training pipeline
- 📚 Inspired by the original Transformer paper "Attention Is All You Need"
- 🔥 Built with PyTorch and love ❤️

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257456-8b62abb6-5709-4d07-9ea1-09a4f5e5e8f5.gif" width="100">
</div>

## 📞 Contact

For questions or feedback, join our [Discord](https://discord.gg/dzsKgWMgjJ) or open an issue!

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20,24&height=120&section=footer" width="100%"/>
  
  <br>
  
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=00D9FF&center=true&vCenter=true&width=600&lines=Built+with+PyTorch+%F0%9F%94%A5;Designed+by+SolvyrEryx+%F0%9F%8E%A8;100%25+Local+AI+%F0%9F%A4%96;No+API+Keys+Required+%E2%9C%A8" alt="Footer Animation">
  
  <br><br>
  
  Made with ❤️ by the Custom LLM Team
  
</div>
