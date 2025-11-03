call-llm-pipeline

> An AI-powered pipeline for analyzing **Turkish-language bank call transcripts** using LangChain and large language models (LLMs).  
> Automatically generates structured insights such as summaries, topics, and named entities from real-world customer service conversations.

[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![langchain](https://img.shields.io/badge/langchain-0.1.x-orange)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Quick Start](#quick-start)
- [Usage Example](#usage-example)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview
**This call-llm-pipeline is a modular NLP pipeline designed to process and analyze **Turkish customer service call transcripts** — particularly for the banking domain.

It leverages **LangChain**, **OpenAI**, and **vector databases (FAISS)** to extract structured insights from unstructured conversation text, including:
- Call summaries  
- Topic and subtopic classification  
- Named Entity Recognition (NER) for products, transactions, and amounts  
- JSON-based structured outputs ready for storage or analytics  

---

## Key Features
✅ LLM-based summarization and classification  
✅ 16 structured output fields — including `intent`, `summary`, `topics`, `sentiment`, and `complaint`  
✅ Context-aware LLM extraction using LangChain chains  
✅ Topic classification enhanced with RAG (FAISS vector search)  
✅ Asynchronous batch processing for large-scale efficiency  
✅ End-to-end persistence with SQLAlchemy and Pydantic models  
✅ Modular design — reusable across domains beyond ba

---

## Architecture
Setup DB → Build FAISS Index → LLM Extraction → RAG Classification → DB Writeback


### Core Steps
1. **Setup:** Initialize database and load transcripts  
2. **Vector Store:** Build FAISS index from `topic_hierarchy.json`  
3. **LLM Extraction:** Generate 14 of 16 fields (intent, summary, etc.)  
4. **RAG Classification:** Fill in the 2 guided topic fields  
5. **Store Results:** Write structured results back to the database  

---

## Data Model

### 📥 Input Table — `calls_input`
| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Primary key |
| `call_id` | string | Unique ID per call |
| `transcript` | text | Raw call transcript |
| `status` | string | One of: `pending`, `processed`, `failed` |
| `created_at` | datetime | Record creation time |

### 📤 Output Table — `calls_output`
| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Primary key |
| `input_call_id` | int | Reference to input table |
| `intent` | string | Initial customer intent |
| `summary` | text | Short call summary |
| `main_topic_free` | string | LLM free-text main topic |
| `main_topic_guided` | string | RAG-guided topic match |
| `sub_topics_free` | string | Free-text subtopics |
| `sub_topics_guided` | string | RAG-guided subtopics |
| `sentiment` | string | Positive / Negative / Neutral |
| `is_complaint` | bool | Complaint flag |
| `complaint_reason` | string | Reason if complaint detected |
| `is_product_offer` | bool | Product offer detected |
| `is_escalation` | bool | Escalation flag |
| `is_regulatory_mention` | bool | Regulatory content flag |
| `is_other_bank_mention` | bool | Mentions other banks |
| `nps_score` | int | Predicted NPS score |
| `nps_rationale` | text | Explanation for NPS score |
| `top_keywords` | string | Comma-separated keyword list |

---

## Pipeline Flow

### 1. One-Time Setup
**Scripts:**  
- `app/setup_db.py` — creates SQLite DB (`bank_calls.db`) with both tables  
- `app/build_vector_store.py` — builds semantic FAISS index from `topic_hierarchy.json`  

### 2. Main Processing Loop
**Script:** `app/main.py`

**Steps:**
1. Load pending calls from `calls_input`  
2. Extract dual context (`transcript_start`, `full_transcript`)  
3. Run **LLM Extraction Chain** → fills 14/16 fields  
4. Run **RAG Classification** → fills guided topic fields  
5. Merge all results and write them into `calls_output`  
6. Mark processed calls as `status='processed'`  

Supports **batch execution** and **async parallelism** for performance.

---

## Technologies

| Component | Library |
|------------|----------|
| Framework | LangChain |
| LLM Provider | OpenAI |
| Vector DB | FAISS |
| Database | SQLite / SQLAlchemy |
| Schema Validation | Pydantic |
| Async Execution | asyncio |
| Logging | Python logging |
| Language | Turkish (domain-specific) |

---

License

This project is licensed under the MIT License.

Contact

Author: Enes Aydın
Contact: enesaydin91@gmail.com

⚠️ Data Privacy Note

This project is designed for research and internal analytics.
Ensure all call transcripts are anonymized and compliant with KVKK / GDPR before processing.

