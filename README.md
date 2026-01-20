# X-RAG-Framework-for-Trusted-Knowledge-Retrieval


An Explainable Retrieval-Augmented Generation (X-RAG) framework that delivers accurate, source-grounded answers from domain-specific documents. The system enhances trust by providing source attribution, similarity scores, and confidence levels, effectively reducing hallucinations and improving transparency in AI-driven knowledge retrieval.


<img width="1918" height="1079" alt="image" src="https://github.com/user-attachments/assets/dd2d9aa4-3be7-43e0-b2b9-e734ba8dce77" />



---

## ✨ Key Features

- CSV-based data ingestion (drag & drop or file path)
- Automatic row-to-chunk processing
- Adjustable Top-K semantic retrieval
- Optional LLM-based answer generation
- Debug mode to view raw retrieved metadata
- Clean and professional dark-themed UI



---

## 🖥️ User Interface Overview

### Ingest / Settings Panel
- Upload CSV files (up to 200MB)
- Configure default Top-K retrieval
- Select LLM provider
- Add API keys (optional)

### Query Panel
- Ask hospital-related questions
- Modify Top-K retrieval dynamically
- Enable or disable LLM for final answers

### Debug / Raw Output Panel
- View retrieved chunks
- Inspect metadata used for responses

---
## Done by
- Nhowmitha S AI&DS Student
- Contact : @nhowmi05@gmail.com
---

## ⚙️ Configuration Options

| Setting | Description |
|------|------------|
| Top-K Retrieval | Number of relevant chunks retrieved |
| LLM Provider | Google (default) |
| Use LLM | Toggle LLM-based answer generation |
| API Keys | Optional for deployment |

---

## 🚀 How to Run Locally

```bash
1. Clone the repository
2. Navigate to project directory
3. Install dependencies
4. Run the application
