## Self-Pruning Neural Network (CIFAR-10) + FastAPI/RAG/SQL demo

### Install

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### Train (lambda sweep)

```bash
python train.py --lambdas 0 1e-5 1e-4 1e-3 --epochs 20 --output_dir outputs
```

This produces:
- `outputs/best_model_lam*.pt`
- `outputs/lambda_sweep_results.json`
- plots in `outputs/`

### Generate the Markdown report

```bash
python report.py --results outputs/lambda_sweep_results.json --out REPORT.md
```

### Run the API (FastAPI + RAG + SQLite)

The API demonstrates:
- **FastAPI** endpoints (`/predict`, `/rag/ask`)
- **RAG** retrieval over `rag_docs/` using embeddings + FAISS
- **SQL** logging to SQLite (requests/results)

```bash
set SELF_PRUNING_CKPT=outputs/best_model_lam0_0.pt
uvicorn api.main:app --reload --port 8000
```

Optional (LLM synthesis for RAG):

```bash
set OPENAI_API_KEY=... 
set OPENAI_MODEL=gpt-4o-mini
```

Endpoints:
- `GET /health`
- `POST /predict` (multipart file upload)
- `POST /rag/ask` (JSON)
- `GET /logs/rag/recent`
- `GET /logs/predictions/recent`

