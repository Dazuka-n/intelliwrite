# AEO Blog Engine

The **AEO Blog Engine** is an intelligent, agent-based pipeline designed to generate high-quality blog content optimized for **Answer Engine Optimization (AEO)** and **Generative Engine Optimization (GEO)**. 

Unlike standard AI writers, this engine follows a strict "Answer-First" methodology, ensuring content is structured for featured snippets, voice search, and AI overviews.

## 🚀 Features

- **Multi-Agent Pipeline**:
  - **Researcher**: Gathers real-time facts and statistics using DuckDuckGo.
  - **Planner**: Structures content with logical H2/H3 hierarchies.
  - **Writer**: Drafts content focusing on clarity, authority, and direct answers.
  - **Optimizer**: Refines answers for conciseness and generates valid JSON-LD Schema (FAQPage, Article).
  - **Finalizer**: Polishes the markdown to production-ready standards.
- **RAG (Retrieval-Augmented Generation)**: Uses a local knowledge base (e.g., AEO Rulebooks) to enforce style and formatting guidelines.
- **Web Search Integration**: robust web search with rate-limit handling.
- **CLI Interface + Flask API**: Generate blogs via command line or HTTP. Both paths share the same Agno-powered pipeline and Postgres persistence layer.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd aeo_blog_engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**:
   Create a `.env` file in the root directory and add your API keys plus database URL:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   DATABASE_URL=postgresql+psycopg2://user:password@host/dbname
   # Optional (if using remote Qdrant)
   # QDRANT_URL=...
   # QDRANT_API_KEY=...
   ```

## 📖 Usage

### 1. Ingest Knowledge Base
Before running the generator, ingest your AEO/GEO guidelines and example docs into the vector database.

```bash
python main.py --ingest
```
*This reads markdown files from `knowledge/docs/` and indexes them.*

### 2. Generate a Blog Post (CLI)
Run the pipeline with a specific topic. To also persist the result into Postgres, include the company metadata.

```bash
# Just print result
python main.py --topic "What Is Answer Engine Optimization?"

# Generate + store in DB
python main.py --topic "What Is Answer Engine Optimization?" \
  --company-url "https://example.com" \
  --email "hello@example.com" \
  --brand "Example Co"
```

### 3. Run the Flask API

```bash
export FLASK_APP=api:app
flask run
```

Endpoints:
- `POST /blogs` – JSON body with `topic`, `company_url`, optional `email_id`, `brand_name`.
- `GET /blogs/<id>` – fetch generated blog entry.

### 4. Workflow

1. Retrieve AEO guidelines.
2. Research the topic online.
3. Plan the outline.
4. Write the draft.
5. Optimize and add Schema markup.
6. Finalize and store the blog in Postgres (when using DB-backed flows).

## 📂 Project Structure

```
aeo_blog_engine/
├── api.py               # Flask app entrypoint
├── database/            # SQLAlchemy models, sessions, repository helpers
├── config/              # Configuration settings
├── knowledge/           # RAG ingestion & retrieval
├── pipeline/            # Core workflow logic (Agents)
├── tools/               # Custom tools (e.g., DuckDuckGo wrapper)
├── services.py          # Orchestrates pipeline + DB persistence
├── main.py              # CLI entry point
└── requirements.txt     # Python dependencies
```

## 🧠 How It Works

The system implements a sequential chain of thought:
1. **Research Phase**: It looks for "key facts statistics questions" related to your topic.
2. **Knowledge Retrieval**: It pulls the "AEO Rulebook" to ensure it knows *how* to write (e.g., "Answer-First" style).
3. **Drafting**: It synthesizes the research + rules into a blog post.
4. **Optimization**: It specifically looks for opportunities to shorten answers (<50 words) and appends technical Schema markup.
5. **Persistence**: When using the DB/Flask paths, the generated content is stored in your Neon-hosted Postgres `blogs` table.


