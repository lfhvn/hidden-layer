# LLM State Explorer - Quick Start Guide

## 60-Minute MVP Challenge

**Goal**: Get a working activation heatmap in under an hour.

This guide provides the minimal code to see LLM activations visualized in real-time.

---

## Architecture at a Glance

```
┌─────────────────────┐
│   Next.js Frontend  │  ← User types text
│   localhost:3000    │  ← Sees heatmap update
└──────────┬──────────┘
           │ HTTP/WebSocket
┌──────────▼──────────┐
│   FastAPI Backend   │  ← Runs model
│   localhost:8000    │  ← Captures activations
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   GPT-2 (124M)      │  ← Small, fast model
│   via transformers  │  ← Perfect for testing
└─────────────────────┘
```

---

## File Structure

```
representations/state-explorer/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── provider.py      # Model provider abstraction
│   │   │   └── activation.py    # Activation capture
│   │   └── api/
│   │       ├── __init__.py
│   │       └── routes.py        # API endpoints
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx         # Main page
│   │   │   └── layout.tsx       # Layout
│   │   ├── components/
│   │   │   ├── TextInput.tsx    # Text input box
│   │   │   ├── ActivationHeatmap.tsx  # Canvas heatmap
│   │   │   └── ModelSelector.tsx      # Model picker
│   │   ├── lib/
│   │   │   └── api.ts           # API client
│   │   └── types/
│   │       └── index.ts         # TypeScript types
│   ├── package.json
│   ├── next.config.js
│   └── tsconfig.json
│
├── docker-compose.yml
├── README.md
└── .env.example
```

---

## Step 1: Backend Setup (20 min)

### 1.1 Create Backend Structure

```bash
cd representations
mkdir -p state-explorer/backend/app/{models,api}
cd state-explorer/backend
```

### 1.2 Requirements (`requirements.txt`)

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
transformers==4.35.0
torch==2.1.0
numpy==1.24.3
pydantic==2.5.0
python-multipart==0.0.6
```

### 1.3 Activation Capture (`app/models/activation.py`)

```python
"""
Minimal activation capture using PyTorch hooks.
Reused from Lens with simplifications.
"""
from typing import Dict, List, Optional
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

class ActivationCapture:
    """Capture activations from transformer layers."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _make_hook(self, name: str):
        """Create a hook function that captures output."""
        def hook(module, input, output):
            # Store the output activation
            if isinstance(output, tuple):
                output = output[0]  # For transformers, first element is hidden states
            self.activations[name] = output.detach()
        return hook

    def register_hooks(self):
        """Register forward hooks on specified layers."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(self) -> Dict[str, np.ndarray]:
        """Get captured activations as numpy arrays."""
        return {
            name: act.cpu().numpy()
            for name, act in self.activations.items()
        }

    def clear(self):
        """Clear stored activations."""
        self.activations = {}


class SimpleModelProvider:
    """
    Minimal model provider for GPT-2.
    Extend this for other models later.
    """

    def __init__(self, model_name: str = "gpt2"):
        print(f"Loading {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Auto-detect layer names
        self.layer_names = self._detect_layers()
        print(f"Detected {len(self.layer_names)} layers")

    def _detect_layers(self) -> List[str]:
        """Detect transformer layer names."""
        layers = []
        for name, module in self.model.named_modules():
            # GPT-2 pattern: transformer.h.0, transformer.h.1, ...
            if "transformer.h." in name and name.count(".") == 2:
                layers.append(name)
        return sorted(layers)

    def get_activations(self, text: str, layers: Optional[List[str]] = None) -> Dict:
        """
        Get activations for text at specified layers.

        Returns:
            {
                "tokens": List[str],
                "layers": {
                    "transformer.h.0": np.ndarray,  # shape: (seq_len, hidden_dim)
                    "transformer.h.1": np.ndarray,
                    ...
                }
            }
        """
        if layers is None:
            layers = self.layer_names

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Capture activations
        capture = ActivationCapture(self.model, layers)
        capture.register_hooks()

        with torch.no_grad():
            self.model(**inputs)

        activations = capture.get_activations()
        capture.remove_hooks()

        # Process activations (batch_size=1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
        processed = {}
        for layer_name, act in activations.items():
            if len(act.shape) == 3:  # (batch, seq, hidden)
                act = act[0]  # Remove batch dimension
            processed[layer_name] = act

        return {
            "tokens": tokens,
            "layers": processed,
            "model": self.model_name,
            "num_layers": len(processed)
        }

    def get_layer_info(self) -> Dict:
        """Get model architecture info."""
        return {
            "model_name": self.model_name,
            "layer_names": self.layer_names,
            "num_layers": len(self.layer_names),
            "hidden_dim": self.model.config.hidden_size,
        }
```

### 1.4 API Routes (`app/api/routes.py`)

```python
"""
API routes for LLM State Explorer.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

router = APIRouter()

# Global model instance (lazy loaded)
_model_provider = None


class ActivationRequest(BaseModel):
    text: str
    layers: Optional[List[str]] = None


class ActivationResponse(BaseModel):
    tokens: List[str]
    layers: Dict[str, List[List[float]]]  # layer_name -> (seq_len, hidden_dim)
    model: str
    num_layers: int


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/model/info")
async def get_model_info():
    """Get current model information."""
    global _model_provider
    if _model_provider is None:
        from app.models.activation import SimpleModelProvider
        _model_provider = SimpleModelProvider("gpt2")

    return _model_provider.get_layer_info()


@router.post("/activate", response_model=ActivationResponse)
async def get_activations(request: ActivationRequest):
    """
    Get activations for input text.

    Example:
        POST /activate
        {
            "text": "The cat sat on the mat",
            "layers": ["transformer.h.6", "transformer.h.11"]  # optional
        }
    """
    global _model_provider

    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Lazy load model
    if _model_provider is None:
        from app.models.activation import SimpleModelProvider
        _model_provider = SimpleModelProvider("gpt2")

    # Get activations
    result = _model_provider.get_activations(request.text, request.layers)

    # Convert numpy arrays to lists for JSON serialization
    layers_serialized = {
        name: act.tolist()
        for name, act in result["layers"].items()
    }

    return ActivationResponse(
        tokens=result["tokens"],
        layers=layers_serialized,
        model=result["model"],
        num_layers=result["num_layers"]
    )
```

### 1.5 Main App (`app/main.py`)

```python
"""
FastAPI application for LLM State Explorer.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="LLM State Explorer API",
    description="Real-time LLM internal state visualization",
    version="0.1.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "message": "LLM State Explorer API",
        "docs": "/docs"
    }
```

### 1.6 Test Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Test it:**
```bash
# Check health
curl http://localhost:8000/api/health

# Get model info
curl http://localhost:8000/api/model/info

# Get activations
curl -X POST http://localhost:8000/api/activate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "layers": ["transformer.h.0", "transformer.h.5"]}'
```

---

## Step 2: Frontend Setup (30 min)

### 2.1 Create Next.js App

```bash
cd ../  # state-explorer/
npx create-next-app@latest frontend --typescript --tailwind --app --no-src
cd frontend
```

Select:
- TypeScript: Yes
- ESLint: Yes
- Tailwind CSS: Yes
- `src/` directory: Yes
- App Router: Yes
- Import alias: No

### 2.2 Install Dependencies

```bash
npm install zustand  # State management
```

### 2.3 Types (`src/types/index.ts`)

```typescript
export interface ModelInfo {
  model_name: string;
  layer_names: string[];
  num_layers: number;
  hidden_dim: number;
}

export interface ActivationData {
  tokens: string[];
  layers: Record<string, number[][]>;  // layer_name -> (seq_len, hidden_dim)
  model: string;
  num_layers: number;
}

export interface ActivationRequest {
  text: string;
  layers?: string[];
}
```

### 2.4 API Client (`src/lib/api.ts`)

```typescript
import type { ModelInfo, ActivationData, ActivationRequest } from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function getModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE}/api/model/info`);
  if (!response.ok) throw new Error('Failed to fetch model info');
  return response.json();
}

export async function getActivations(
  request: ActivationRequest
): Promise<ActivationData> {
  const response = await fetch(`${API_BASE}/api/activate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get activations');
  }

  return response.json();
}
```

### 2.5 Text Input Component (`src/components/TextInput.tsx`)

```typescript
'use client';

import { useState, useCallback } from 'react';
import { debounce } from '@/lib/utils';

interface TextInputProps {
  onTextChange: (text: string) => void;
  loading?: boolean;
}

export default function TextInput({ onTextChange, loading }: TextInputProps) {
  const [text, setText] = useState('');

  // Debounce API calls
  const debouncedOnChange = useCallback(
    debounce((value: string) => onTextChange(value), 300),
    [onTextChange]
  );

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setText(value);
    debouncedOnChange(value);
  };

  return (
    <div className="w-full">
      <label className="block text-sm font-medium mb-2">
        Input Text
        {loading && <span className="ml-2 text-gray-500">Processing...</span>}
      </label>
      <textarea
        value={text}
        onChange={handleChange}
        placeholder="Type to see internal activations..."
        className="w-full h-32 p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        disabled={loading}
      />
    </div>
  );
}
```

### 2.6 Heatmap Component (`src/components/ActivationHeatmap.tsx`)

```typescript
'use client';

import { useEffect, useRef } from 'react';

interface ActivationHeatmapProps {
  tokens: string[];
  layers: Record<string, number[][]>;  // layer_name -> (seq_len, hidden_dim)
  layerNames: string[];
}

export default function ActivationHeatmap({
  tokens,
  layers,
  layerNames,
}: ActivationHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !tokens.length || !layerNames.length) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Dimensions
    const cellWidth = 60;
    const cellHeight = 30;
    const width = tokens.length * cellWidth;
    const height = layerNames.length * cellHeight;

    canvas.width = width;
    canvas.height = height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Aggregate activations to single value per (token, layer)
    const intensities: number[][] = [];
    let maxIntensity = 0;

    layerNames.forEach((layerName) => {
      const layerAct = layers[layerName];  // (seq_len, hidden_dim)
      const tokenIntensities: number[] = [];

      layerAct.forEach((tokenAct) => {
        // Aggregate: take L2 norm of activation vector
        const norm = Math.sqrt(
          tokenAct.reduce((sum, val) => sum + val * val, 0)
        );
        tokenIntensities.push(norm);
        maxIntensity = Math.max(maxIntensity, norm);
      });

      intensities.push(tokenIntensities);
    });

    // Draw heatmap
    intensities.forEach((tokenIntensities, layerIdx) => {
      tokenIntensities.forEach((intensity, tokenIdx) => {
        // Normalize to [0, 1]
        const normalized = intensity / maxIntensity;

        // Color: blue (low) -> yellow (high)
        const hue = (1 - normalized) * 240;  // 240 = blue, 0 = red/yellow
        const saturation = 70;
        const lightness = 40 + normalized * 40;

        ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
        ctx.fillRect(
          tokenIdx * cellWidth,
          layerIdx * cellHeight,
          cellWidth - 1,
          cellHeight - 1
        );

        // Draw intensity value
        ctx.fillStyle = '#fff';
        ctx.font = '10px monospace';
        ctx.fillText(
          normalized.toFixed(2),
          tokenIdx * cellWidth + 5,
          layerIdx * cellHeight + 20
        );
      });
    });

    // Draw token labels
    ctx.fillStyle = '#000';
    ctx.font = '12px monospace';
    tokens.forEach((token, idx) => {
      ctx.save();
      ctx.translate(
        idx * cellWidth + cellWidth / 2,
        height + 15
      );
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(token, 0, 0);
      ctx.restore();
    });

    // Draw layer labels
    layerNames.forEach((name, idx) => {
      ctx.fillStyle = '#000';
      ctx.font = '11px monospace';
      ctx.fillText(
        name,
        width + 5,
        idx * cellHeight + cellHeight / 2
      );
    });

  }, [tokens, layers, layerNames]);

  if (!tokens.length) {
    return (
      <div className="text-center text-gray-500 py-12">
        Enter text to see activations
      </div>
    );
  }

  return (
    <div className="overflow-auto border rounded-lg p-4">
      <canvas ref={canvasRef} />
    </div>
  );
}
```

### 2.7 Utility Functions (`src/lib/utils.ts`)

```typescript
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return function (...args: Parameters<T>) {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}
```

### 2.8 Main Page (`src/app/page.tsx`)

```typescript
'use client';

import { useState, useEffect } from 'react';
import TextInput from '@/components/TextInput';
import ActivationHeatmap from '@/components/ActivationHeatmap';
import { getModelInfo, getActivations } from '@/lib/api';
import type { ModelInfo, ActivationData } from '@/types';

export default function Home() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [activationData, setActivationData] = useState<ActivationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load model info on mount
  useEffect(() => {
    getModelInfo()
      .then(setModelInfo)
      .catch((err) => setError(err.message));
  }, []);

  const handleTextChange = async (text: string) => {
    if (!text) {
      setActivationData(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Get activations for all layers
      const data = await getActivations({ text });
      setActivationData(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold mb-2">LLM State Explorer</h1>
          <p className="text-gray-600">
            Real-time visualization of internal model activations
          </p>
          {modelInfo && (
            <div className="mt-2 text-sm text-gray-500">
              Model: {modelInfo.model_name} | Layers: {modelInfo.num_layers} | Hidden Dim: {modelInfo.hidden_dim}
            </div>
          )}
        </header>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        <div className="space-y-6">
          <TextInput onTextChange={handleTextChange} loading={loading} />

          {activationData && modelInfo && (
            <div>
              <h2 className="text-2xl font-semibold mb-4">Activation Heatmap</h2>
              <p className="text-sm text-gray-600 mb-4">
                Color intensity shows L2 norm of activation vectors. Blue = low, Yellow = high.
              </p>
              <ActivationHeatmap
                tokens={activationData.tokens}
                layers={activationData.layers}
                layerNames={modelInfo.layer_names}
              />
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
```

### 2.9 Environment Variables (`.env.local`)

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 2.10 Run Frontend

```bash
npm run dev
```

Open http://localhost:3000

---

## Step 3: Test End-to-End (10 min)

### 3.1 Start Both Services

**Terminal 1 (Backend):**
```bash
cd backend
uvicorn app.main:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

### 3.2 Test Flow

1. Open http://localhost:3000
2. Type: "The cat sat on the mat"
3. Watch heatmap appear
4. See activations across all 12 GPT-2 layers
5. Try longer text: "Once upon a time, in a galaxy far away, there lived a small robot who dreamed of becoming a musician."
6. Notice how different layers activate differently

### 3.3 Expected Output

You should see:
- **Tokens** along X-axis (Ġ indicates space, common in GPT-2)
- **Layers** along Y-axis (transformer.h.0 through transformer.h.11)
- **Colors** showing activation intensity
- **Numbers** showing normalized values (0.00 to 1.00)

---

## Next Steps

### Immediate Improvements (Next 1-2 Hours)

1. **Layer Selection**
   - Add checkboxes to select which layers to visualize
   - Reduces computation for large models

2. **Better Aggregation**
   - Add dropdown: "L2 Norm", "Max", "Mean"
   - Let user choose how to aggregate high-dim vectors

3. **Hover Tooltips**
   - Show top-5 neurons on hover
   - Display exact activation values

4. **Color Schemes**
   - Add color picker (Viridis, Plasma, Grayscale)
   - Accessibility considerations

### Short-Term (Next Few Days)

5. **Layer Detail View**
   - Click a cell to see full activation vector
   - Show top-k neurons with highest activation
   - Histogram of activation distribution

6. **Multiple Models**
   - Add model selector dropdown
   - Support BERT, LLaMA, etc.
   - Auto-detect architecture

7. **Performance**
   - Add caching (Redis)
   - Optimize canvas rendering
   - WebSocket for real-time streaming

### Medium-Term (Next Week)

8. **Concept Mapping**
   - Build initial concept library
   - Show which concepts activate
   - Highlight concept-related neurons

9. **Token Playback**
   - Animate token-by-token processing
   - Show activation flow through layers
   - Variable speed control

10. **Sharing**
    - Generate shareable links
    - Export as image/video
    - Static HTML export

---

## Common Issues & Fixes

### Backend Issues

**Problem**: `ModuleNotFoundError: No module named 'app'`
```bash
# Run from backend/ directory, not backend/app/
cd backend
uvicorn app.main:app --reload
```

**Problem**: CUDA out of memory
```python
# In activation.py, force CPU
self.device = torch.device("cpu")
```

**Problem**: Slow first request
- Normal! Model loads lazily on first request
- Subsequent requests are fast

### Frontend Issues

**Problem**: CORS errors
```python
# In backend/app/main.py, check CORS middleware allows localhost:3000
allow_origins=["http://localhost:3000"]
```

**Problem**: Canvas not rendering
- Check browser console for errors
- Ensure tokens/layers data is not empty
- Try smaller text first

**Problem**: Heatmap colors wrong
- Check normalization (should be 0-1)
- Verify L2 norm calculation
- Try different color scheme

---

## Docker Deployment (Optional)

### `docker-compose.yml`

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    environment:
      - PYTHONUNBUFFERED=1
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    command: npm run dev
```

### Run with Docker

```bash
docker-compose up
```

---

## Success Checklist

- [ ] Backend starts without errors
- [ ] Frontend starts without errors
- [ ] Model info loads on page load
- [ ] Typing text triggers activation request
- [ ] Heatmap renders with colors
- [ ] Tokens and layers are labeled
- [ ] Activations update as you type (debounced)
- [ ] No CORS errors in console
- [ ] Works with sentences up to 50 tokens

---

## Resources

**Code References:**
- FastAPI docs: https://fastapi.tiangolo.com/
- Next.js docs: https://nextjs.org/docs
- Transformers docs: https://huggingface.co/docs/transformers
- Canvas API: https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API

**Similar Projects:**
- LM Debugger: https://github.com/mega002/lm-debugger
- exBERT: https://github.com/bhoov/exbert
- BertViz: https://github.com/jessevig/bertviz

**Debugging:**
- Backend logs: Check terminal running uvicorn
- Frontend logs: Check browser console (F12)
- Network requests: Browser DevTools → Network tab

---

## What's Next?

Once you have this working, refer to the full implementation plan in `LLM_STATE_VISUALIZATION_PLAN.md` for:

- Multi-model support (Ollama, MLX)
- Concept mapping
- Token-by-token playback
- Shareable links
- SAE feature integration
- Advanced visualizations

**You now have a foundation to build upon. Ship the MVP, get feedback, iterate!** 🚀
