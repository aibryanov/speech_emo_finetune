# Speech Emotion Recognition: DUSHA + RESD

Код к ВКР «Классификация эмоциональной окраски речи с применением нейронных сетей»
(СПбГУ, 2026, Ибрянов А.С.). Задача — распознавание эмоций по русской речи на
датасетах **DUSHA** (crowd, 5 классов) и **RESD** (актёрская речь, 7 классов):
агрегация крауд-меток → классический ML → fine-tuning SSL-трансформеров →
мультимодальный audio+text fusion.

## Структура

- `src/` — общий код: `config.py`, `dataset.py`, `models.py`, `metrics.py`, `trainer.py`
- `configs/` — YAML-конфиги экспериментов для `train.py`
- `train.py` — CLI для экспериментов на RESD
- `notebooks/` — эксперименты на Kaggle/Colab (см. ниже)

## Ноутбуки — что где искать

**Данные и разметка**
| Ноутбук | Что делает |
|---|---|
| `eda.ipynb` | Разведочный анализ RESD: классы, длительности, спектрограммы |
| `aggregate_labels.ipynb` | Агрегация крауд-меток DUSHA: Majority Vote + Дэвид–Скин (пороги 0.85 / 0.9 / 0.95 / 0.98) |

**Классический ML**
| Ноутбук | Что делает |
|---|---|
| `classical_ml_kaggle.ipynb` | SVM / Random Forest / 2D ResNet на 85 признаках (MFCC, Pitch, ZCR, DWT) — отдельно на RESD и DUSHA |

**WavLM на DUSHA**
| Ноутбук | Что делает |
|---|---|
| `bert_dusha_kaggle.ipynb` | Fine-tune ruBERT (rubert-tiny2) на текстах реплик DUSHA |
| `dusha_smoke_test.ipynb` | Быстрый sanity-check пайплайна (1 батч, 1 эпоха) |
| `dusha_train_kaggle.ipynb` | Full fine-tune WavLM Base на 5 вариантах агрегации DUSHA |
| `dusha_eval_kaggle.ipynb` | Кросс-оценка 5×5: какая модель на какой разметке точнее |

**RESD — обучение и baseline**
| Ноутбук | Что делает |
|---|---|
| `backbone_eval.ipynb` / `eval_wavlm_resd_colab.ipynb` | Оценка готовой модели `Aniemore/wavlm-emotion-russian-resd` на RESD без дообучения (baseline) |
| `train_colab.ipynb` | Запуск экспериментов `train.py` (head_only / LoRA / top-N / full / BiLSTM) в Colab/Kaggle |

**Fusion текст + аудио (DGCA)**
| Ноутбук | Что делает |
|---|---|
| `dgca_dusha_kaggle.ipynb` | Fusion WavLM + ruBERT на DUSHA (бэкбоны заморожены) |
| `dgca_resd_kaggle.ipynb` | То же на RESD (+ Whisper для транскрипции, т.к. текста в RESD нет) |
| `dgca_fusion_kaggle.ipynb` | Более ранняя версия DGCA (бэкбоны дообучаются, не заморожены) |
| `infer_dgca_resd_kaggle.ipynb` | Инференс/оценка уже обученной DGCA-модели на RESD |
| `alpha_analysis_dusha_kaggle.ipynb` / `alpha_analysis_resd_kaggle.ipynb` | Анализ гейтинг-весов α — когда модель доверяет тексту, а когда аудио |

**Fusion аудио-only (без текста, не DGCA)**
| Ноутбук | Что делает |
|---|---|
| `fusion_colab.ipynb` | Двухэтапный fusion: backbone + BiLSTM(MFCC/mel) + статистики → MLP (через `train.py configs/lstm_features.yaml` → `fusion_wavlm.yaml`) |

## Установка и запуск

```bash
git clone https://github.com/aibryanov/speech_emo_finetune.git
cd speech_emo_finetune
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py configs/wavlm_base_head.yaml
```

Ноутбуки самодостаточны — зависимости ставятся в первой ячейке, рассчитаны на запуск
в Kaggle/Colab (там же лежат данные DUSHA/RESD и чекпоинты).

---

Основные результаты работы: на RESD WavLM даёт Accuracy 0.81, DGCA fusion (WavLM+ruBERT) —
0.825; на DUSHA WavLM даёт 0.85, лучшая агрегация меток — Дэвид–Скин с порогом 0.95.
Подробности и вывод формул — в тексте ВКР.
