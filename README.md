# Maui Gemma 3 — On-Device RAG Chat for Android

A fully on-device RAG (Retrieval-Augmented Generation) chat app built with .NET MAUI and Gemma 3. Pick a PDF, ask questions, get streamed answers — no internet required after the first model download, no API keys, no cloud.

---

## Demo

1. App launches → automatically downloads Gemma 3 270M + MiniLM from HuggingFace (~950MB total)
2. Pick any PDF from your device
3. Ask questions → answers stream token by token, grounded in the document

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    MAUI Android App                  │
│                                                      │
│  Setup         PDF Picker        Chat                │
│  (download)    (index PDF)       (ask questions)     │
└──────────────────────┬──────────────────────────────┘
                       │ uses
┌──────────────────────▼──────────────────────────────┐
│                      RagCore                         │
│                                                      │
│  ModelDownloader   ← downloads models from HF        │
│  PdfTextExtractor  ← Syncfusion PDF → page strings   │
│  TextChunker       ← pages → overlapping chunks      │
│  MiniLmEmbedder    ← chunk/question → 384-dim vector │
│  SqliteChunkRepository ← cache index by PDF hash     │
│  InMemoryVectorStore   ← cosine similarity search    │
│  GemmaAnswerGenerator  ← streams answer tokens       │
└─────────────────────────────────────────────────────┘
```

### RAG Flow

```
PDF
 └─ PdfTextExtractor ──► pages
     └─ TextChunker ──────► chunks
         └─ MiniLmEmbedder ► vectors ──► SqliteChunkRepository (cache)
                                                    │
Question                                            │ load on next run
 └─ MiniLmEmbedder ──► vector                       │
     └─ InMemoryVectorStore.Search ◄────────────────┘
         └─ top 3 chunks + question
             └─ GemmaAnswerGenerator ──► streamed tokens
```

---

## Project Structure

```
Maui-gemma-3/
├── patches/                          ← builder patches (see below)
│   ├── gemma_builder_rope.patch
│   ├── base_builder_rope_theta.patch
│   └── apply.sh
└── src/
    ├── RagCore/                      ← shared class library
    │   ├── Chunking/
    │   │   ├── TextChunk.cs          ← data model: page number + text
    │   │   └── TextChunker.cs        ← splits pages into overlapping chunks
    │   ├── Embedding/
    │   │   ├── IEmbedder.cs
    │   │   └── MiniLmEmbedder.cs     ← ONNX inference → 384-dim vector
    │   ├── Generation/
    │   │   ├── IAnswerGenerator.cs
    │   │   └── GemmaAnswerGenerator.cs ← Gemma 3 streaming via OnnxRuntimeGenAI
    │   ├── Ingestion/
    │   │   └── PdfTextExtractor.cs   ← Syncfusion PDF → page strings
    │   ├── Persistence/
    │   │   ├── IChunkRepository.cs
    │   │   └── SqliteChunkRepository.cs ← SQLite cache keyed by PDF SHA256
    │   ├── Retrieval/
    │   │   ├── IVectorStore.cs
    │   │   ├── InMemoryVectorStore.cs ← cosine similarity search in RAM
    │   │   ├── ScoredChunk.cs
    │   │   └── VectorMath.cs
    │   └── Services/
    │       └── ModelDownloader.cs    ← resumable download from HuggingFace
    ├── RagConsole/                   ← CLI tool for desktop testing
    │   └── Program.cs
    └── MauiApp/                      ← Android app
        ├── MauiProgram.cs
        ├── App.xaml
        ├── AppShell.xaml
        ├── MainPage.xaml             ← 3-section UI: setup → picker → chat
        ├── MainPage.xaml.cs          ← all app logic
        ├── ChatMessage.cs            ← INotifyPropertyChanged for streaming bubbles
        └── Platforms/Android/
```

---

## Models

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| Gemma 3 270M-it (int4 ONNX) | Text generation | ~864 MB | [ihassantariq/gemma-3-270m-it-onnx-int4](https://huggingface.co/ihassantariq/gemma-3-270m-it-onnx-int4) |
| all-MiniLM-L6-v2 (ONNX) | Sentence embeddings | ~90 MB | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

Both are downloaded automatically on first launch. Downloads are resumable — if interrupted, the app picks up from where it left off.

### Why Gemma 3 270M and not 4B?

Google's Gemma 3 comes in two architectures:
- **270M / 1B** — `Gemma3ForCausalLM` (text-only). Model type `"gemma3_text"` in genai_config.
- **4B / 12B / 27B** — `Gemma3ForConditionalGeneration` (multimodal, text + vision). Model type `"gemma3"`.

Our pinned `onnxruntime-genai 0.8.3` runtime loads `"gemma3"` models with the full vision pipeline, adding a ~645MB vision component even when you never send images. The 270M produces a `"gemma3_text"` bundle — no vision overhead, ~864MB total vs ~6.2GB.

---

## NuGet Version Pinning

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI" Version="0.8.3" />
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.22.0" />
```

These two must be pinned together. The GenAI execution provider selection breaks with mismatched versions.

---

## Building the ONNX Model

The `google/gemma-3-270m-it` HuggingFace repo ships PyTorch weights (`.safetensors`), not ONNX. We use Microsoft's `onnxruntime-genai` Python model builder to convert.

### Prerequisites

- Python 3.10 (PyTorch does not support 3.13 on Apple Silicon)
- Miniconda arm64: [docs.anaconda.com/miniconda](https://docs.anaconda.com/miniconda/)
- HuggingFace account with license accepted at [huggingface.co/google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)

### Step 1 — Create Python environment

```bash
/path/to/python3.10 -m venv /tmp/genai-builder-venv310
source /tmp/genai-builder-venv310/bin/activate
pip install onnxruntime-genai==0.11.4 onnxruntime transformers torch accelerate onnx-ir huggingface_hub
```

We pin `onnxruntime-genai==0.11.4` for the builder because:
- It writes `"type": "gemma3_text"` for `Gemma3ForCausalLM` architectures
- This is exactly what our 0.8.3 .NET runtime expects for the text-only code path

### Step 2 — Log in to HuggingFace

```bash
hf auth login --token YOUR_HF_WRITE_TOKEN --force
```

### Step 3 — Apply patches

`transformers 5.12+` reorganized Gemma 3's RoPE (Rotary Position Embedding) config from direct attributes into a nested dict. The builder at v0.11.4 predates this change. The patches add `hasattr` guards so the builder handles both old and new transformers versions.

```bash
bash patches/apply.sh
```

**What each patch does:**

`patches/gemma_builder_rope.patch` — fixes `gemma.py` line ~127:
```python
# Before (breaks with transformers 5.12+):
self.rope_local_theta = config.rope_local_base_freq

# After:
if hasattr(config, "rope_local_base_freq"):
    self.rope_local_theta = config.rope_local_base_freq
elif hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
    self.rope_local_theta = config.rope_parameters.get("sliding_attention", {}).get("rope_theta", 10000.0)
else:
    self.rope_local_theta = 10000.0
```

`patches/base_builder_rope_theta.patch` — fixes `base.py` rope_theta fallback:
```python
# Before:
rope_theta = config.rope_theta if hasattr(config, "rope_theta") else \
             config.rope_embedding_base if hasattr(config, "rope_embedding_base") else 10000

# After (also checks rope_parameters dict):
rope_theta = (
    config.rope_theta if hasattr(config, "rope_theta")
    else config.rope_embedding_base if hasattr(config, "rope_embedding_base")
    else config.rope_parameters.get("full_attention", {}).get("rope_theta", 10000)
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict)
    else 10000
)
```

Relevant builder source (v0.11.4):
- [builders/gemma.py](https://github.com/microsoft/onnxruntime-genai/blob/v0.11.4/src/python/py/models/builders/gemma.py)
- [builders/base.py](https://github.com/microsoft/onnxruntime-genai/blob/v0.11.4/src/python/py/models/builders/base.py)
- [Model builder README](https://github.com/microsoft/onnxruntime-genai/blob/v0.11.4/src/python/py/models/README.md)

### Step 4 — Run the builder

```bash
python -m onnxruntime_genai.models.builder \
  -m google/gemma-3-270m-it \
  -o ~/.cache/maui-gemma-3/models/gemma-3-270m-it \
  -p int4 -e cpu
```

| Flag | Value | Meaning |
|------|-------|---------|
| `-m` | `google/gemma-3-270m-it` | HuggingFace model ID to download and convert |
| `-o` | output path | Where to write ONNX files |
| `-p` | `int4` | 4-bit integer quantization — smallest size, fastest inference |
| `-e` | `cpu` | CPU execution provider (no GPU required) |

Output files:
```
gemma-3-270m-it/
├── genai_config.json       ← runtime config for onnxruntime-genai
├── chat_template.jinja     ← Gemma prompt format template
├── tokenizer.json          ← SentencePiece vocabulary
├── tokenizer_config.json   ← tokenizer settings
├── model.onnx              ← ONNX graph structure (~300KB)
└── model.onnx.data         ← quantized weights (~864MB, LFS)
```

### Step 5 — Fix temperature in genai_config.json

The builder emits `"temperature": null`. The 0.8.3 runtime requires a number:

```bash
sed -i '' 's/"temperature": null/"temperature": 1.0/' \
  ~/.cache/maui-gemma-3/models/gemma-3-270m-it/genai_config.json
```

The app's `ModelDownloader.PatchGenAiConfigTemperature()` applies this automatically at runtime.

### Step 6 — Upload to HuggingFace

```bash
hf repos create gemma-3-270m-it-onnx-int4 --type model
hf upload <your-username>/gemma-3-270m-it-onnx-int4 \
  ~/.cache/maui-gemma-3/models/gemma-3-270m-it .
```

Then update `GemmaBaseUrl` in `src/RagCore/Services/ModelDownloader.cs`:
```csharp
private const string GemmaBaseUrl =
    "https://huggingface.co/<your-username>/gemma-3-270m-it-onnx-int4/resolve/main/";
```

---

## Running the App

### Requirements

- .NET 10 SDK
- Android SDK (API 35) + emulator or physical device (API 24+)
- Visual Studio / VS Code with .NET MAUI extension

### Deploy to emulator

```bash
dotnet build -t:Run -f net10.0-android src/MauiApp/MauiApp.csproj
```

On first launch the app downloads both models (~950MB total). Downloads are resumable.

### Run the console tool (desktop testing)

```bash
dotnet run --project src/RagConsole -- /path/to/your.pdf
```

---

## Roadmap

- [ ] Upgrade to Gemma 3 1B for better answer quality (same architecture, same build process)
- [ ] Add Mac Catalyst target
- [ ] Disk-space and Wi-Fi gating before model download
