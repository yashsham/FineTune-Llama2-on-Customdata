# Fine-Tuning Llama-2 on Custom Data 🦙🚀

This repository contains a comprehensive guide and implementation for fine-tuning the **Llama-2-7b-chat** model using **QLoRA (4-bit Quantized Low-Rank Adaptation)**. This approach enables high-performance fine-tuning on consumer-grade hardware, such as a single NVIDIA T4 GPU.

---

## 📖 Overview
Fine-tuning Large Language Models (LLMs) usually requires massive computational resources. This project utilizes **PEFT (Parameter-Efficient Fine-Tuning)** to train only a small fraction of the model's parameters, making it possible to adapt Llama-2 to specific domains or datasets with minimal VRAM.

### Key Highlights:
* **Memory Efficiency:** Uses 4-bit quantization via `bitsandbytes`.
* **Speed:** Optimized training loops using the `SFTTrainer` from the TRL library.
* **Stability:** Implements Paged AdamW optimizer to handle memory spikes.
* **Flexibility:** Easily swappable for any custom instruction dataset.

---

## 🛠️ Tech Stack
* **Model:** Llama-2-7b-chat-hf
* **Quantization:** NF4 (Normal Float 4)
* **Frameworks:** PyTorch, Hugging Face Transformers
* **PEFT Technique:** QLoRA (Rank = 64, Alpha = 16)
* **Orchestration:** Accelerate & TRL (Transformer Reinforcement Learning)

---

## 📂 Project Structure
```text
.
├── Fine_tune_Llama2.ipynb    # Complete training & inference notebook
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── results/                  # Generated model checkpoints and logs
