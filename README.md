---
title: Textile Dept Assistant
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
app_dir: chatbots/website_chatbot
pinned: false
---

# Indian Textile Heritage AI

AI tooling built around traditional Indian textile art forms — **Baluchari, Maheshwari, Negamam, and Phulkari** — developed as part of the DST SHRI textile heritage project (Dept. of Textile & Fibre Engineering, IIT Delhi).

The repository is organized by kind of work:

| Folder | What it is | Tech |
|---|---|---|
| [classification/](classification/) | Training, evaluation, ensembling and TFLite export for the 4-class saree classifier | PyTorch, Ultralytics YOLO11, TensorFlow Lite |
| [chatbots/](chatbots/) | Two RAG chatbot projects: the department website chatbot and the "Ask Textile" course-content platform | FastAPI, Pinecone, Groq, React, Node |
| [app/](app/) | The Tana mobile app with on-device saree classification (bundled TFLite models) | Flutter |

---

## 1. Classification (`classification/`)

Two model families are trained on the same 4-class dataset and combined into an ensemble:

- **YOLO11m (classification head)** — fine-tuned from an earlier 3-class benchmark model. Best validation top-1 accuracy **~93.5%** (full report: [model_report.md](classification/runs/classify/YOLO11m_4class4/model_report.md)).
- **ResNet50** — fine-tuned with torchvision; checkpoints in [classification/models/](classification/models/).

### Layout

- `training/` — all training / evaluation / export scripts (listed below)
- `models/` — tracked ResNet50 checkpoints (plus local TFLite calibration data)
- `results/` — confusion matrices, ensemble comparison and disagreement plots
- `runs/` — Ultralytics training runs; `YOLO11m_4class4` is the deployed one

### Scripts (run from inside `classification/`)

| Script | Purpose |
|---|---|
| [patch.py](classification/training/patch.py) | Extract patches from raw images (`data/` to `data_patched/`) |
| [train_4class.py](classification/training/train_4class.py) | Train YOLO11m on the 4-class patched dataset |
| [train_resnet_finetune.py](classification/training/train_resnet_finetune.py) | Fine-tune ResNet50, writes `models/resnet50_4saree_best.pt` |
| [evaluate_4class.py](classification/training/evaluate_4class.py) | Evaluate YOLO11m on the test split; dumps misclassifications to `wrong_predictions/` |
| [evaluate_resnet.py](classification/training/evaluate_resnet.py) | Evaluate ResNet50; writes confusion matrix to `results/` |
| [compare_models.py](classification/training/compare_models.py) | Benchmark every run under `runs/classify/` on the test split |
| [ensemble_compare.py](classification/training/ensemble_compare.py) | Compare ResNet50 vs YOLO11m vs their ensemble; plots in `results/` |
| [export_to_tflite.py](classification/training/export_to_tflite.py) | Export both models to TFLite (float16/float32) for the mobile app |
| [ensemble_tflite.py](classification/training/ensemble_tflite.py) | Validate the exported TFLite models as an ensemble |
| [diagnose_tflite.py](classification/training/diagnose_tflite.py) | Debug accuracy drift between the .pt and .tflite versions |

### Environment

```bash
conda env create -f classification/environment.yml   # creates the "saree" env
conda activate saree
cd classification
python training/train_4class.py                      # scripts expect classification/ as the working directory
```

The dataset folders (`data/`, `data_patched/`, `random_16_02/`, ...) live under `classification/` on the development machine but are gitignored — see [Local-only folders](#local-only-folders).

---

## 2. Chatbots (`chatbots/`)

### Website chatbot (`chatbots/website_chatbot/`)

FastAPI RAG backend + static frontend answering questions about the department's heritage textile documentation (source PDFs are included in the folder). Retrieval uses **Pinecone**, generation uses **Groq**. The YAML front-matter at the top of this README configures it as a Docker-based **Hugging Face Space** serving on port 7860.

```bash
cd chatbots/website_chatbot
pip install -r requirements.txt
# create .env with PINECONE_API_KEY and GROQ_API_KEY (not committed)
python ingest_v4_new.py               # one-time: index the PDFs
uvicorn app:app --host 0.0.0.0 --port 7860
```

Or via Docker:

```bash
docker build -t textile-chatbot chatbots/website_chatbot
docker run -p 7860:7860 --env-file chatbots/website_chatbot/.env textile-chatbot
```

### Ask Textile (`chatbots/ask_textile/`)

Three-tier RAG application over textile course material (`RAG/textile_courses.json`, with a YouTube-augmented variant):

- `frontend/` — React + TypeScript + Vite + Tailwind UI
- `middleware/` — Node/TypeScript API layer with Prisma (see [SETUP.md](chatbots/ask_textile/middleware/SETUP.md))
- `RAG/` — Python retrieval service: Pinecone ingestion ([ingest_pinecone.py](chatbots/ask_textile/RAG/ingest_pinecone.py)), retriever, LLM prompts, and a **RAGAS** evaluation harness ([ragas_evaluation.py](chatbots/ask_textile/RAG/ragas_evaluation.py), results in `ragas_results.csv`)

Requires the same `PINECONE_API_KEY` / `GROQ_API_KEY` in `chatbots/ask_textile/.env` (not committed).

---

## 3. App (`app/`)

Flutter app ("Tana") for saree recognition and textile heritage exploration. The exported TFLite classifiers (`yolo11m_4class.tflite`, `resnet50_4saree.tflite`) are bundled under `app/tana_app/assets/models/` and run fully on-device.

```bash
cd app/tana_app
flutter pub get
flutter run
```

---

## Repository layout

```
textile_design/
├── classification/          # saree classifier: training, models, results, runs
│   ├── training/            # scripts (run with classification/ as working dir)
│   ├── models/              # tracked ResNet50 checkpoints
│   ├── results/             # evaluation and ensemble plots
│   ├── runs/                # Ultralytics training runs (YOLO11m_4class4 = deployed)
│   └── environment.yml      # conda env ("saree")
├── chatbots/
│   ├── website_chatbot/     # FastAPI RAG chatbot (Hugging Face Space, Docker)
│   └── ask_textile/         # full-stack course-content RAG platform
├── app/
│   └── tana_app/            # Flutter mobile app with bundled TFLite models
├── textile-heritage/        # git submodule reference (separate repo, not vendored here)
└── .gitignore
```

### Local-only folders

These exist on the development machine but are intentionally **not** pushed (datasets and large or derived artifacts, see [.gitignore](.gitignore)):

- `classification/data/` — raw dataset (~7 GB)
- `classification/data_patched/` — patched dataset
- `classification/random_16_02/` — raw image collection
- `classification/tflite_models/` — TFLite export output
- `classification/YOLO11m_benchmark/` — original 3-class benchmark run
- `classification/wrong_predictions/` — misclassified samples
- `app/sareeclassifier_appfile/` — app packaging files
- `app/stitch_designs/` — app UI mockups
- `brain/` — scratch

---

## Secrets

No API keys are committed. Each chatbot component reads `PINECONE_API_KEY` and `GROQ_API_KEY` from its own local `.env` file, which is gitignored.
