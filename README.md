# Speech Emotion Fine-Tuning on RESD

Research project comparing fine-tuning strategies and backbone architectures for **Russian speech emotion recognition** on the [RESD dataset](https://huggingface.co/datasets/Aniemore/resd).

## What's being studied

| Dimension | Options |
|-----------|---------|
| **Backbones** | HuBERT Large, WavLM Base, wav2vec2 XLS-R 300M |
| **Strategies** | Head-only, LoRA, Top-N layers, Full fine-tune, BiLSTM |
| **Dataset** | RESD — 7 emotions, ~1400 Russian speech samples |

All backbones are pre-trained on [Dusha](https://huggingface.co/datasets/salute-developers/dusha) (5 classes). The classification head is replaced with a fresh 7-class head for RESD.

**RESD emotions:** happiness · sadness · anger · fear · disgust · enthusiasm · neutral

---

## Project structure

```
speech_emo_finetune/
├── configs/
│   ├── base.yaml                  # shared defaults
│   ├── hubert_large_head.yaml
│   ├── hubert_large_lora.yaml
│   ├── hubert_large_topn.yaml
│   ├── hubert_large_lstm.yaml
│   ├── wavlm_base_head.yaml
│   ├── wavlm_base_lora.yaml
│   ├── wavlm_base_full.yaml
│   ├── wav2vec2_xlsr_head.yaml
│   └── wav2vec2_xlsr_lora.yaml
├── src/
│   ├── config.py      # ExperimentConfig dataclass + YAML loader
│   ├── dataset.py     # RESD loading, stratified split, DataLoader
│   ├── models.py      # model factory (5 strategies)
│   ├── metrics.py     # accuracy, F1, confusion matrix
│   └── trainer.py     # training loop, checkpointing, JSONL logging
├── notebooks/
│   ├── eda.ipynb          # exploratory data analysis
│   └── train_colab.ipynb  # self-contained Colab / Kaggle notebook
├── train.py           # CLI entry point
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/aibryanov/speech_emo_finetune.git
cd speech_emo_finetune

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running experiments

```bash
# fastest experiment (~8 min on T4)
python train.py configs/wavlm_base_head.yaml

# LoRA on HuBERT Large
python train.py configs/hubert_large_lora.yaml

# override any config value from CLI
python train.py configs/wavlm_base_lora.yaml --override epochs=5 lr=5e-5
```

Results are saved to the `output_dir` specified in the config:

```
outputs/wavlm_base_head/
├── best_model.pt    # best checkpoint (by dev F1 weighted)
└── metrics.jsonl   # per-epoch metrics + final test results
```

---

## Configuration

Each experiment YAML overrides values from `configs/base.yaml`.

```yaml
# configs/wavlm_base_lora.yaml
model_name: xbgoose/wavlm-base-speech-emotion-recognition-russian-dusha-finetuned
fine_tune_strategy: lora   # head_only | lora | top_n | full | lstm
lora_rank: 8
lora_alpha: 16
lora_target_modules:
  - q_proj
  - v_proj
batch_size: 8
grad_accum_steps: 4        # effective batch = batch_size × grad_accum_steps
lr: 1.0e-4
epochs: 10
run_name: wavlm_base_lora
output_dir: outputs/wavlm_base_lora
```

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| `fine_tune_strategy` | Which parts of the model to train |
| `lora_rank` / `lora_alpha` | LoRA hyperparameters |
| `top_n_layers` | Number of top encoder layers to unfreeze (for `top_n` strategy) |
| `lstm_hidden` / `lstm_layers` | BiLSTM head size (for `lstm` strategy) |
| `grad_accum_steps` | Gradient accumulation steps |
| `max_audio_len_s` | Truncate audio longer than this (seconds) |
| `dev_ratio` | Fraction of train set used for validation |

---

## Fine-tuning strategies

| Strategy | Trainable params | Notes |
|----------|-----------------|-------|
| `head_only` | Classifier head only | Fastest; good baseline |
| `lora` | LoRA adapters + head | Efficient; best quality/cost trade-off |
| `top_n` | Last N encoder layers + head | Middle ground |
| `full` | All parameters | Best quality; needs more VRAM |
| `lstm` | Frozen backbone + BiLSTM + head | Sequence modelling on top of features |

---

## Metrics

Each run logs to `metrics.jsonl`. Per epoch:
- `train_loss`, `dev_accuracy`, `dev_weighted_accuracy`
- `dev_f1_macro`, `dev_f1_weighted`

Final test evaluation (best checkpoint):
- All of the above on test split
- Per-class precision / recall / F1
- Confusion matrix

---

## Notebooks

**`notebooks/eda.ipynb`** — run locally to explore RESD:
- Class distribution (train / test)
- Duration histogram and per-class boxplot
- Unique sample rates
- Waveform examples per class
- Mel spectrograms per class

**`notebooks/train_colab.ipynb`** — self-contained for Colab / Kaggle:
- Installs dependencies, clones repo, runs training
- Plots loss / F1 curves
- Shows confusion matrix
- Comparison table across all completed experiments

---

## GPU requirements

Tested on **T4 (16 GB)**. Approximate training time per experiment (10 epochs):

| Config | Time |
|--------|------|
| `wavlm_base_head` | ~8 min |
| `wavlm_base_lora` | ~12 min |
| `wavlm_base_full` | ~20 min |
| `hubert_large_head` | ~15 min |
| `hubert_large_lora` | ~25 min |
| `hubert_large_lstm` | ~10 min |
| `wav2vec2_xlsr_head` | ~15 min |
| `wav2vec2_xlsr_lora` | ~25 min |

---

## Models used

| Model | Architecture | Pretrain data |
|-------|-------------|---------------|
| [xbgoose/hubert-large-dusha](https://huggingface.co/xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned) | HuBERT Large (300M) | Dusha |
| [xbgoose/wavlm-base-dusha](https://huggingface.co/xbgoose/wavlm-base-speech-emotion-recognition-russian-dusha-finetuned) | WavLM Base (94M) | Dusha |
| [KELONMYOSA/wav2vec2-xls-r-300m](https://huggingface.co/KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru) | wav2vec2 XLS-R (300M) | Dusha |
