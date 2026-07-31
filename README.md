# Pain Researcher

Pain Researcher is a LangGraph pipeline that mines Reddit for real,
recurring, monetizable pain points, validates them against actual
evidence, checks whether anyone's already solved them, and outputs a
ranked list of software business opportunities — instead of just
summarizing a topic.

It's built on top of [`local-deep-researcher`](https://github.com/langchain-ai/local-deep-researcher)'s
LangGraph/LangGraph-Studio scaffolding, but the research pipeline itself
is a different shape: a funnel with fan-out validation, not a single
linear research loop. See [Architecture](#architecture) below.

## Data sources

Three sources, each toggleable independently in `settings.yaml`'s
`sources.enabled` — useful since Reddit API access can be pending
approval for reasons outside this project's control, while the other
two need zero credentials and work immediately:

| Source | Access | Credentials |
|---|---|---|
| Reddit | Official API (PRAW) | Required if enabled |
| Hacker News | Algolia Search API | None — free, keyless |
| Stack Exchange (~170 sites) | Official API | Optional (raises quota) |

## What it does

1. **Discover** — finds candidate subreddits, either autonomously, from a
   seed niche, or from a fixed watchlist you provide (Reddit only —
   Hacker News has no sub-communities to discover, Stack Exchange sites
   are a fixed list in `settings.yaml`)
2. **Harvest** — pulls threads from every enabled source, then filters
   deterministically (upvotes, age, complaint-language match) before any
   LLM ever sees the text
3. **Extract & cluster** — an LLM pulls candidate pain points out of the
   surviving threads and merges duplicates reported across different
   threads into single candidates
4. **Validate** — for each candidate that clears an evidence floor
   (minimum distinct authors/threads): searches Reddit for more
   corroborating mentions, searches the web + crawls competitor pages
   (Crawl4AI) to check if it's already solved, and has a stronger LLM
   judge severity, willingness-to-pay signals, and buildability
5. **Score & export** — a deterministic, config-weighted formula (not an
   LLM guess) ranks every validated candidate, and the run produces a
   JSON/CSV of all candidates plus a markdown evidence brief per
   top-ranked one

## Setup

1. `cp .env.example .env` and fill in `GOOGLE_API_KEY` (from
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey) —
   the pipeline runs on Gemma 4 26B/31B) and `REDDIT_CLIENT_ID` /
   `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT` (from a script-type app
   at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)).
2. `pip install -e .` then `playwright install chromium` (needed by
   Crawl4AI for the competitor-discovery step).
3. Tune `src/pain_researcher/settings.yaml` — model profiles/quota
   limits, content budgets, prefilter phrase lists, candidate gating,
   and scoring weights all live there; no code changes needed to retune.
4. Run via LangGraph Studio (`langgraph dev`, then select the
   `pain_researcher` graph) or standalone:
   ```shell
   python -m pain_researcher.graph
   ```
   The standalone entry point checkpoints to SQLite
   (`checkpoint.db_path` in settings.yaml), so a run that spans multiple
   days (a real possibility at 14.4K requests/day on the free tier)
   resumes rather than restarts.

**Before running for real**, project worst-case quota spend for your
current settings with no network calls at all:
```shell
python -m pain_researcher.estimate
```
Retune `settings.yaml` and re-run to see the projected numbers change —
no code edits needed.

## Configuration

Every threshold in the system — model rate limits, how much comment text
gets pulled per thread, the prefilter's complaint/willingness-to-pay
phrase lists, how many candidates get promoted to expensive validation,
and every scoring weight — lives in
[`src/pain_researcher/settings.yaml`](src/pain_researcher/settings.yaml),
not hardcoded in the pipeline. Two Gemma models are used by role, not by
name, so swapping models later is a config edit:

```yaml
roles:
  cheap: gemma-4-26b   # discovery, extraction, clustering
  judge: gemma-4-31b   # severity/WTP/competitor-gap judgment
```

## Architecture

```
select_targets → harvest_threads → extract_pain_points
  → cluster_pain_points → prefilter_candidates
  → [fan-out: one branch per surviving candidate]
       corroborate → competitor_scan → judge
  → [fan-in]
  → score_and_rank → export_results
```

| File | Purpose |
|---|---|
| [`config.py`](src/pain_researcher/config.py) | Pydantic settings, loaded from `settings.yaml`, validated at startup |
| [`quota.py`](src/pain_researcher/quota.py) | Per-model rate limiting/pacing — the binding constraint is tokens/minute, not requests/minute |
| [`providers/llm.py`](src/pain_researcher/providers/llm.py) | Gemma router: role → model, quota-checked, defensive JSON parsing |
| [`providers/reddit.py`](src/pain_researcher/providers/reddit.py) | PRAW wrapper — official API, not scraping |
| [`providers/hackernews.py`](src/pain_researcher/providers/hackernews.py) | Algolia HN Search API — free, keyless |
| [`providers/stackexchange.py`](src/pain_researcher/providers/stackexchange.py) | Official API across ~170 Stack Exchange sites |
| [`providers/crawl.py`](src/pain_researcher/providers/crawl.py) | Crawl4AI, scoped to competitor-page discovery only |
| [`prefilter.py`](src/pain_researcher/prefilter.py) | Deterministic pre-LLM filtering — the biggest token saver in the pipeline |
| [`scoring.py`](src/pain_researcher/scoring.py) | Deterministic, weighted scoring from LLM-emitted signals |
| [`graph.py`](src/pain_researcher/graph.py) | LangGraph wiring, incl. the `Send`-based validation fan-out |
| [`estimate.py`](src/pain_researcher/estimate.py) | Config-only worst-case spend projection, no network calls |

---

## Also in this repo: Local Deep Researcher (the base project)

This is a fork of [langchain-ai/local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher),
kept intact in `src/ollama_deep_researcher/`. It's a fully local web
research assistant that uses any LLM hosted by [Ollama](https://ollama.com/search)
or [LMStudio](https://lmstudio.ai/): give it a topic and it generates a
web search query, gathers results, summarizes them, reflects to find
knowledge gaps, and repeats for a configurable number of cycles, ending
in a markdown summary with sources.

![ollama-deep-research](https://github.com/user-attachments/assets/1c6b28f8-6b64-42ba-a491-1ab2875d50ea)

Short summary video:
<video src="https://github.com/user-attachments/assets/02084902-f067-4658-9683-ff312cab7944" controls></video>

### 📺 Video Tutorials

- [Overview of Local Deep Researcher with R1](https://www.youtube.com/watch?v=sGUjmyfof4Q) - Load and test [DeepSeek R1](https://api-docs.deepseek.com/news/news250120) [distilled models](https://ollama.com/library/deepseek-r1).
- [Building Local Deep Researcher from Scratch](https://www.youtube.com/watch?v=XGuTzHoqlj8) - Overview of how this is built.

### 🚀 Quickstart

Then edit the `.env` file to customize the environment variables according to your needs. These environment variables control the model selection, search tools, and other configuration settings. When you run the application, these values will be automatically loaded via `python-dotenv` (because `langgraph.json` point to the "env" file).
```shell
cp .env.example .env
```

#### Selecting local model with Ollama

1. Download the Ollama app for Mac [here](https://ollama.com/download).

2. Pull a local LLM from [Ollama](https://ollama.com/search). As an [example](https://ollama.com/library/deepseek-r1:8b):
```shell
ollama pull deepseek-r1:8b
```

3. Optionally, update the `.env` file with the following Ollama configuration settings. 

* If set, these values will take precedence over the defaults set in the `Configuration` class in `configuration.py`. 
```shell
LLM_PROVIDER=ollama
OLLAMA_BASE_URL="http://localhost:11434" # Ollama service endpoint, defaults to `http://localhost:11434` 
LOCAL_LLM=model # the model to use, defaults to `llama3.2` if not set
```

#### Selecting local model with LMStudio

1. Download and install LMStudio from [here](https://lmstudio.ai/).

2. In LMStudio:
   - Download and load your preferred model (e.g., qwen_qwq-32b)
   - Go to the "Local Server" tab
   - Start the server with the OpenAI-compatible API
   - Note the server URL (default: http://localhost:1234/v1)

3. Optionally, update the `.env` file with the following LMStudio configuration settings. 

* If set, these values will take precedence over the defaults set in the `Configuration` class in `configuration.py`. 
```shell
LLM_PROVIDER=lmstudio
LOCAL_LLM=qwen_qwq-32b  # Use the exact model name as shown in LMStudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1
```

#### Selecting search tool

By default, it will use [DuckDuckGo](https://duckduckgo.com/) for web search, which does not require an API key. But you can also use [SearXNG](https://docs.searxng.org/), [Tavily](https://tavily.com/) or [Perplexity](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api) by adding their API keys to the environment file. Optionally, update the `.env` file with the following search tool configuration and API keys. If set, these values will take precedence over the defaults set in the `Configuration` class in `configuration.py`. 
```shell
SEARCH_API=xxx # the search API to use, such as `duckduckgo` (default)
TAVILY_API_KEY=xxx # the tavily API key to use
PERPLEXITY_API_KEY=xxx # the perplexity API key to use
MAX_WEB_RESEARCH_LOOPS=xxx # the maximum number of research loop steps, defaults to `3`
FETCH_FULL_PAGE=xxx # fetch the full page content (with `duckduckgo`), defaults to `false`
```

#### Running with LangGraph Studio

##### Mac

1. (Recommended) Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Launch LangGraph server:

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

##### Windows

1. (Recommended) Create a virtual environment: 

* Install `Python 3.11` (and add to PATH during installation). 
* Restart your terminal to ensure Python is available, then create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Launch LangGraph server:

```powershell
# Install dependencies
pip install -e .
pip install -U "langgraph-cli[inmem]"            

# Start the LangGraph server
langgraph dev
```

#### Using the LangGraph Studio UI

When you launch LangGraph server, you should see the following output and Studio will open in your browser:
> Ready!

> API: http://127.0.0.1:2024

> Docs: http://127.0.0.1:2024/docs

> LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

Open `LangGraph Studio Web UI` via the URL above. In the `configuration` tab, you can directly set various assistant configurations. Keep in mind that the priority order for configuration values is:

```
1. Environment variables (highest priority)
2. LangGraph UI configuration
3. Default values in the Configuration class (lowest priority)
```

<img width="1621" alt="Screenshot 2025-01-24 at 10 08 31 PM" src="https://github.com/user-attachments/assets/7cfd0e04-28fd-4cfa-aee5-9a556d74ab21" />

Give the assistant a topic for research, and you can visualize its process!

<img width="1621" alt="Screenshot 2025-01-24 at 10 08 22 PM" src="https://github.com/user-attachments/assets/4de6bd89-4f3b-424c-a9cb-70ebd3d45c5f" />

#### Model Compatibility Note

When selecting a local LLM, set steps use structured JSON output. Some models may have difficulty with this requirement, and the assistant has fallback mechanisms to handle this. As an example, the [DeepSeek R1 (7B)](https://ollama.com/library/deepseek-llm:7b) and [DeepSeek R1 (1.5B)](https://ollama.com/library/deepseek-r1:1.5b) models have difficulty producing required JSON output, and the assistant will use a fallback mechanism to handle this.
  
#### Browser Compatibility Note

When accessing the LangGraph Studio UI:
- Firefox is recommended for the best experience
- Safari users may encounter security warnings due to mixed content (HTTPS/HTTP)
- If you encounter issues, try:
  1. Using Firefox or another browser
  2. Disabling ad-blocking extensions
  3. Checking browser console for specific error messages

### How it works

Local Deep Researcher is inspired by [IterDRAG](https://arxiv.org/html/2410.04343v1#:~:text=To%20tackle%20this%20issue%2C%20we,used%20to%20generate%20intermediate%20answers.). This approach will decompose a query into sub-queries, retrieve documents for each one, answer the sub-query, and then build on the answer by retrieving docs for the second sub-query. Here, we do similar:
- Given a user-provided topic, use a local LLM (via [Ollama](https://ollama.com/search) or [LMStudio](https://lmstudio.ai/)) to generate a web search query
- Uses a search engine / tool to find relevant sources
- Uses LLM to summarize the findings from web search related to the user-provided research topic
- Then, it uses the LLM to reflect on the summary, identifying knowledge gaps
- It generates a new search query to address the knowledge gaps
- The process repeats, with the summary being iteratively updated with new information from web search
- Runs for a configurable number of iterations (see `configuration` tab)

### Outputs

The output of the graph is a markdown file containing the research summary, with citations to the sources used. All sources gathered during research are saved to the graph state. You can visualize them in the graph state, which is visible in LangGraph Studio:

![Screenshot 2024-12-05 at 4 08 59 PM](https://github.com/user-attachments/assets/e8ac1c0b-9acb-4a75-8c15-4e677e92f6cb)

The final summary is saved to the graph state as well:

![Screenshot 2024-12-05 at 4 10 11 PM](https://github.com/user-attachments/assets/f6d997d5-9de5-495f-8556-7d3891f6bc96)

### Deployment Options

There are [various ways](https://langchain-ai.github.io/langgraph/concepts/#deployment-options) to deploy this graph. See [Module 6](https://github.com/langchain-ai/langchain-academy/tree/main/module-6) of LangChain Academy for a detailed walkthrough of deployment options with LangGraph.

### TypeScript Implementation

A TypeScript port of this project (without Perplexity search) is available at:
https://github.com/PacoVK/ollama-deep-researcher-ts

### Running as a Docker container

The included `Dockerfile` only runs LangChain Studio with local-deep-researcher as a service, but does not include Ollama as a dependant service. You must run Ollama separately and configure the `OLLAMA_BASE_URL` environment variable. Optionally you can also specify the Ollama model to use by providing the `LOCAL_LLM` environment variable.

Clone the repo and build an image:
```
$ docker build -t local-deep-researcher .
```

Run the container:
```
$ docker run --rm -it -p 2024:2024 \
  -e SEARCH_API="tavily" \ 
  -e TAVILY_API_KEY="tvly-***YOUR_KEY_HERE***" \
  -e LLM_PROVIDER=ollama \
  -e OLLAMA_BASE_URL="http://host.docker.internal:11434/" \
  -e LOCAL_LLM="llama3.2" \  
  local-deep-researcher
```

NOTE: You will see log message:
```
2025-02-10T13:45:04.784915Z [info     ] 🎨 Opening Studio in your browser... [browser_opener] api_variant=local_dev message=🎨 Opening Studio in your browser...
URL: https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:2024
```
...but the browser will not launch from the container.

Instead, visit this link with the correct baseUrl IP address: [`https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2024`](https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2024)
