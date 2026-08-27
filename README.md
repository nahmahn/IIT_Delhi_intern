---
title: Textile Dept Assistant
emoji: 👗
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
app_dir: website_chatbot
pinned: false
---

# Indian Textile Heritage AI

AI tooling built around traditional Indian textile art forms — **Baluchari, Maheshwari, Negamam, and Phulkari** — developed as part of the DST SHRI textile heritage project (Dept. of Textile & Fibre Engineering, IIT Delhi).

The repository contains four related components:

| Component | What it is | Tech |
|---|---|---|
| [training/](training/) | Training, evaluation, ensembling and TFLite export for the 4-class saree classifier | PyTorch, Ultralytics YOLO11, TensorFlow Lite |
| [website_chatbot/](website_chatbot/) | RAG chatbot for the Textile Dept website, deployed as a Hugging Face Space (Docker) | FastAPI, Pinecone, Groq |
| [ask_textile/](ask_textile/) | Full-stack "Ask Textile" RAG platform for textile course content | React + Vite frontend, Node/Prisma middleware, Python RAG backend |
| [tana_app/](tana_app/) | Mobile app with on-device saree classification (bundled TFLite models) and a heritage chatbot | Flutter |

---

## 1. Saree Classifier (`training/`)

Two model families are trained on the same 4-class dataset and combined into an ensemble:

- **YOLO11m (classification head)** — fine-tuned from an earlier 3-class benchmark model. Best validation top-1 accuracy **~93.5%** (full report: [runs/classify/YOLO11m_4class4/model_report.md](runs/classify/YOLO11m_4class4/model_report.md)).
- **ResNet50** — fine-tuned with torchvision; checkpoints in [models/](models/).

### Scripts (run all of them from the repo root)

| Script | Purpose |
|---|---|
| [patch.py](training/patch.py) | Extract patches from raw images (`data/` → `data_patched/`) |
| [train_4class.py](training/train_4class.py) | Train YOLO11m on the 4-class patched dataset |
| [train_resnet_finetune.py](training/train_resnet_finetune.py) | Fine-tune ResNet50 → `models/resnet50_4saree_best.pt` |
| [evaluate_4class.py](training/evaluate_4class.py) | Evaluate YOLO11m on the test split; dumps misclassifications to `wrong_predictions/` |
| [evaluate_resnet.py](training/evaluate_resnet.py) | Evaluate ResNet50; writes confusion matrix to `results/` |
| [compare_models.py](training/compare_models.py) | Benchmark every run under `runs/classify/` on the test split |
| [ensemble_compare.py](training/ensemble_compare.py) | Compare ResNet50 vs YOLO11m vs their ensemble; plots in `results/` |
| [export_to_tflite.py](training/export_to_tflite.py) | Export both models to TFLite (float16/float32) for the mobile app |
| [ensemble_tflite.py](training/ensemble_tflite.py) | Validate the exported TFLite models as an ensemble |
| [diagnose_tflite.py](training/diagnose_tflite.py) | Debug accuracy drift between the .pt and .tflite versions |

### Key artifacts

- `models/resnet50_4saree_best.pt` — best ResNet50 checkpoint (tracked)
- `runs/classify/YOLO11m_4class4/weights/best.pt` — deployed YOLO11m checkpoint (tracked, with ONNX/TFLite exports alongside)
- `results/` — confusion matrices, ensemble comparison and disagreement plots
- `tana_app/assets/models/` — the TFLite models actually bundled in the mobile app

### Environment

```bash
conda env create -f environment.yml   # creates the "saree" env
conda activate saree
python training/train_4class.py       # always run from the repo root
```

> **Note:** the dataset folders (`data/`, `data_patched/`, `random_16_02/`, …) are local-only and gitignored — see [Local-only folders](#local-only-folders).

---

## 2. Website Chatbot (`website_chatbot/`)

FastAPI RAG backend + static frontend answering questions about the department's heritage textile documentation (source PDFs are included in the folder). Retrieval uses **Pinecone**, generation uses **Groq**. The repo root's YAML front-matter in this README configures it as a Docker-based **Hugging Face Space** serving on port 7860.

```bash
cd website_chatbot
pip install -r requirements.txt
# .env with PINECONE_API_KEY and GROQ_API_KEY (not committed)
python ingest_v4_new.py               # one-time: index the PDFs
uvicorn app:app --host 0.0.0.0 --port 7860
```

Or via Docker: `docker build -t textile-chatbot website_chatbot && docker run -p 7860:7860 --env-file website_chatbot/.env textile-chatbot`

---

## 3. Ask Textile (`ask_textile/`)

Three-tier RAG application over textile course material (`RAG/textile_courses.json`, with YouTube-augmented variant):

- `frontend/` — React + TypeScript + Vite + Tailwind UI
- `middleware/` — Node/TypeScript API layer with Prisma (see [SETUP.md](ask_textile/middleware/SETUP.md))
- `RAG/` — Python retrieval service: Pinecone ingestion ([ingest_pinecone.py](ask_textile/RAG/ingest_pinecone.py)), retriever, LLM prompts, and a **RAGAS** evaluation harness ([ragas_evaluation.py](ask_textile/RAG/ragas_evaluation.py), results in `ragas_results.csv`)

Requires the same `PINECONE_API_KEY` / `GROQ_API_KEY` in `ask_textile/.env` (not committed).

---

## 4. Tana App (`tana_app/`)

Flutter app ("Tana") for saree recognition and textile heritage exploration. The exported TFLite classifiers (`yolo11m_4class.tflite`, `resnet50_4saree.tflite`) are bundled under `tana_app/assets/models/` and run fully on-device.

```bash
cd tana_app
flutter pub get
flutter run
```

---

## Repository layout

```
textile_design/
├── training/            # classifier training / eval / export scripts (run from repo root)
├── models/              # tracked ResNet50 checkpoints (+ local TFLite calibration data)
├── results/             # evaluation & ensemble plots (formerly ensemble_results/)
├── runs/                # Ultralytics training runs; YOLO11m_4class4 is the deployed one
├── website_chatbot/     # FastAPI RAG chatbot (HF Space, Docker)
├── ask_textile/         # full-stack course-content RAG platform
├── tana_app/            # Flutter mobile app with bundled TFLite models
├── textile-heritage/    # git submodule reference (separate repo, not vendored here)
├── environment.yml      # conda env ("saree") for the training scripts
└── .gitignore
```

### Local-only folders

These exist on the development machine but are intentionally **not** pushed (datasets and large/derived artifacts, see [.gitignore](.gitignore)):

`data/` (raw dataset, ~7 GB) · `data_patched/` (patched dataset) · `random_16_02/` (raw collection) · `tflite_models/` (export output) · `YOLO11m_benchmark/` (original 3-class benchmark run) · `sareeclassifier_appfile/` · `stitch_designs/` (app UI mockups) · `wrong_predictions/` (misclassified samples) · `brain/` (scratch)

---

## Secrets

No API keys are committed. Each chatbot component reads `PINECONE_API_KEY` and `GROQ_API_KEY` from its own local `.env` file, which is gitignored.
