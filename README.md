\# Customer Support Retrieval System



A hybrid customer-support retrieval system built using Python, FastAPI, Sentence Transformers, BM25, and React. The system retrieves relevant historical support conversations using keyword-based and semantic search.



\## Features



\- \*\*Keyword Search:\*\* BM25-based lexical retrieval.

\- \*\*Semantic Search:\*\* Dense embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).

\- \*\*Hybrid Retrieval:\*\* Combines BM25 and semantic rankings.

\- \*\*Reciprocal Rank Fusion:\*\* Merges rankings using configurable `k=60`.

\- \*\*Retrieval Evaluation:\*\* Precision@5, Mean Reciprocal Rank (MRR), and NDCG@5.

\- \*\*Web Interface:\*\* React frontend with FastAPI backend.

\- \*\*Containerization:\*\* Docker-based backend deployment.



\## Dataset



The project uses a public customer-support conversation dataset containing approximately 2.8 million raw records.



After preprocessing, approximately 21,268 retrieval cases were retained for experimentation.



The raw and processed datasets are not included in this repository.



\## Retrieval Architecture



1\. Receive a customer-support query.

2\. Retrieve candidate conversations using BM25.

3\. Generate query embeddings and retrieve semantically similar conversations.

4\. Combine rankings using Reciprocal Rank Fusion (RRF).

5\. Deduplicate and return the top-ranked historical conversations.



\## Evaluation



The initial BM25 baseline was evaluated using 30 queries and top-5 retrieval results.



| Metric | TF-IDF Baseline | BM25 |

|---|---:|---:|

| Precision@5 | 0.6000 | 0.7000 |

| MRR | 0.7722 | 0.8289 |

| NDCG@5 | 0.8403 | 0.9286 |



These are initial baseline results. Hybrid retrieval evaluation requires validated manual relevance labels and is not reported here.



\## Tech Stack



\*\*Backend:\*\* Python, FastAPI



\*\*Retrieval:\*\* BM25, Sentence Transformers, Scikit-learn, Reciprocal Rank Fusion



\*\*Frontend:\*\* React, Vite



\*\*Deployment:\*\* Docker



\## Project Structure



```text

.

├── src/

│   ├── api/

│   ├── data/

│   ├── models/

│   ├── rag/

│   └── retrieval/

├── frontend/

│   ├── src/

│   ├── package.json

│   └── package-lock.json

├── docs/

├── Dockerfile

├── .dockerignore

├── .gitignore

└── requirements.txt

```



\## Setup



\### Backend



```bash

python -m venv .venv

```



Activate the virtual environment and install dependencies:



```bash

pip install -r requirements.txt

```



Place the preprocessed retrieval dataset at:



```text

data/processed/verizon\_retrieval\_cases\_clean.csv

```



Configure the required environment variables using a local `.env` file.



Start the API:



```bash

uvicorn src.api.main:app --reload

```



\### Frontend



```bash

cd frontend

npm install

npm run dev

```



\## Testing



Run backend tests:



```bash

python -m pytest src -q

```



\## Current Limitations



\- Retrieval quality depends on the relevance and coverage of historical conversations.

\- Hybrid retrieval evaluation is still being validated.

\- The project uses historical public support conversations, not live customer account data.

\- It is a research and engineering prototype, not an official customer-support integration.



\## Future Improvements



\- Complete manual relevance labeling and hybrid retrieval evaluation.

\- Add cross-encoder reranking.

\- Improve intent-aware retrieval.

\- Expand automated retrieval evaluation.

\- Improve production monitoring and error handling.



\## Author



Developed as an AI and backend engineering project.
