# LLM Internal State Visualization - Assessment & Plan

## Executive Summary

**Vision**: A highly shareable, visual web app where anyone can:
1. Point at an LLM (local or API)
2. Input text and see internal activations in real-time
3. Explore what concepts and features activate at each layer
4. Share discoveries via links

**Current State**: You have a **production-ready foundation** (Lens) that's 80% of the way there, but it's optimized for SAE research, not general-purpose visualization.

**Recommendation**: Build a new companion app called **"LLM State Explorer"** that repurposes Lens infrastructure into an interactive, shareable visualization tool.

**Timeline**: 2-3 weeks for MVP, 4-6 weeks for polished v1

---

## Current State Assessment

### ✅ What You Have (Production-Ready)

**Lens - SAE Interpretability Platform**
- Full-stack web app (FastAPI + Next.js)
- Working activation capture from transformer layers
- Sparse autoencoder training and feature discovery
- Docker deployment
- Database storage (SQLModel)
- API with OpenAPI spec
- WebSocket support for real-time updates

**Technical Capabilities:**
- Hook into any HuggingFace transformer layer
- Extract activations during forward passes
- Train SAEs to discover interpretable features
- Multi-architecture support (GPT-2, BERT, LLaMA)
- Experiment tracking and reproducibility

### ❌ What's Missing for Your Vision

**1. Real-Time Interactivity**
- Current: Upload text, run experiment, view results later
- Needed: Type text → see activations update instantly

**2. Multi-Model Support**
- Current: Hardcoded model selection in backend
- Needed: Point at any model (local Ollama, MLX, or API)

**3. General-Purpose Visualization**
- Current: Focused on SAE feature discovery workflow
- Needed: Explore activations even without trained SAEs

**4. Concept Mapping**
- Current: No concept vectors (docs exist but no implementation)
- Needed: Map raw activations to human concepts

**5. Shareable Outputs**
- Current: Local dashboard, no sharing mechanism
- Needed: Generate shareable links to specific visualizations

**6. Token-by-Token Playback**
- Current: Static analysis of complete text
- Needed: Animated playback showing processing step-by-step

**7. Cross-Layer Comparison**
- Current: Single-layer focus
- Needed: Simultaneous multi-layer view

### 📊 Gap Analysis

| Capability | Lens Has It | Priority for Vision | Implementation Effort |
|------------|-------------|---------------------|----------------------|
| Activation capture | ✅ Yes | Critical | Done |
| SAE features | ✅ Yes | Nice-to-have | Done |
| Web frontend | ✅ Yes | Critical | Done |
| Real-time updates | 🟡 Partial (WebSocket exists) | Critical | Medium |
| Multi-model support | ❌ No | Critical | Medium |
| Interactive text input | ❌ No | Critical | Low |
| Concept mapping | ❌ No | High | High |
| Token-by-token playback | ❌ No | High | Medium |
| Shareable links | ❌ No | High | Medium |
| Multi-layer view | ❌ No | High | Low |
| Raw activation viz | 🟡 Partial | Critical | Low |

---

## Proposed Solution: LLM State Explorer

### Core Concept

**A real-time, interactive web app that makes LLM internal states visible and explorable.**

Think: "Developer tools for language models"

### Key User Flows

#### Flow 1: Explore Any Model
```
1. User opens app
2. Selects model source (Local Ollama / MLX / API)
3. Picks specific model from dropdown
4. App loads model and displays layer architecture
```

#### Flow 2: Live Activation Exploration
```
1. User types in text input: "The cat sat on the"
2. As they type, activations update in real-time
3. Visualizations show:
   - Heatmap of activations across all layers
   - Top-k active neurons per layer
   - Feature activations (if SAE trained for this model)
   - Concept activations (if concept library exists)
4. User can:
   - Hover over tokens to see local activations
   - Click layers to zoom in
   - Scrub through token-by-token playback
   - Toggle between raw activations and SAE features
```

#### Flow 3: Share Discovery
```
1. User finds interesting activation pattern
2. Clicks "Share"
3. App generates shareable link with:
   - Model configuration
   - Input text
   - Visualization state (selected layers, view mode)
4. Anyone with link can view (read-only)
5. Option to export as static HTML or video
```

### Visual Design Principles

**1. Immediacy**: Activations update as you type (< 100ms latency)

**2. Clarity**: Clear visual hierarchy
- Token level → Neuron level → Layer level → Model level

**3. Exploration**: Easy to:
- Zoom in/out across abstraction levels
- Compare different layers side-by-side
- Switch between visualization modes

**4. Beauty**: Thoughtful use of:
- Color gradients for activation intensity
- Animation for token-by-token processing
- Spatial layout reflecting model architecture

### Core Visualizations

#### 1. **Activation Heatmap** (Primary View)
```
         Token:  "The"  "cat"  "sat"  "on"  "the"
Layer 0   [███] [███] [██░] [█░░] [███]
Layer 1   [██░] [███] [███] [██░] [██░]
Layer 2   [█░░] [██░] [███] [███] [█░░]
...
Layer N   [░░░] [█░░] [███] [███] [██░]
```
- X-axis: Tokens
- Y-axis: Layers
- Color: Activation intensity
- Hover: Show top-k neurons
- Click: Zoom to layer details

#### 2. **Layer Detail View**
```
Layer 6 - Residual Stream (dim=768)

Top Active Neurons:
#342  ████████░░ 0.87  [Concept: Animals]
#128  ██████░░░░ 0.64  [Concept: Furniture]
#891  █████░░░░░ 0.51  [Concept: Spatial-Relations]

Activation Distribution:
[Histogram showing sparsity]

If SAE trained:
Top Active Features:
Feature #23  ████████░░ 0.92  "Cat-like animals"
Feature #156 ██████░░░░ 0.68  "Sitting posture"
```

#### 3. **Token-by-Token Playback**
```
[◄◄] [◄] [▶] [▶▶]  Speed: [1x ▼]

Processing token 3/5: "sat"

[Animated visualization showing]:
- Token embedding
- Flow through layers
- Activation updates cascading
- Final prediction probabilities
```

#### 4. **Concept Space Map** (If concept vectors exist)
```
[2D projection of activation vector]

Current activation mapped to:
- Animals (0.82)
- Furniture (0.67)
- Spatial Relations (0.59)
- Actions (0.45)

[Interactive scatter plot showing concept proximity]
```

#### 5. **Cross-Layer Flow Diagram**
```
Input → [Layer 0] → [Layer 1] → [Layer 2] → ... → [Layer N] → Output
         ↓            ↓            ↓                    ↓
      Residual    Attention    Residual            Attention
       Stream      Heads        Stream               Heads
         ↓            ↓            ↓                    ↓
     [Concepts]  [Relations]  [Concepts]          [Predictions]
```

### Differentiation from Lens

| Feature | Lens (SAE Research) | State Explorer (Viz Tool) |
|---------|--------------------|-----------------------------|
| **Purpose** | Train SAEs, discover features | Explore any model's internals |
| **Workflow** | Experiment-based, batch processing | Interactive, real-time |
| **Models** | Single model per experiment | Multi-model support |
| **Focus** | Feature discovery | Activation exploration |
| **SAE Required** | Yes (core feature) | No (optional enhancement) |
| **Sharing** | Local dashboard | Shareable links |
| **Use Case** | Researchers training SAEs | Anyone curious about LLMs |

---

## Technical Architecture

### High-Level Stack

```
┌─────────────────────────────────────────┐
│         Frontend (Next.js)              │
│  - Interactive text input               │
│  - Real-time visualizations             │
│  - Shareable link generation            │
└─────────────────────────────────────────┘
                    ↕ WebSocket
┌─────────────────────────────────────────┐
│      Backend (FastAPI)                  │
│  - Model management                     │
│  - Activation streaming                 │
│  - Concept vector lookup                │
└─────────────────────────────────────────┘
                    ↕
┌──────────┬──────────┬──────────┬─────────┐
│  Ollama  │   MLX    │ Anthropic│ OpenAI  │
│  (local) │ (local)  │   (API)  │  (API)  │
└──────────┴──────────┴──────────┴─────────┘
```

### Backend Components

#### 1. Model Provider Abstraction
```python
class ModelProvider(ABC):
    @abstractmethod
    async def load_model(self, model_id: str) -> Model:
        """Load model and return metadata"""

    @abstractmethod
    async def get_activations(
        self,
        text: str,
        layers: List[str]
    ) -> Dict[str, Tensor]:
        """Get activations for text at specified layers"""

    @abstractmethod
    def get_architecture_info(self) -> ArchitectureInfo:
        """Return layer names, dimensions, architecture type"""
```

**Implementations:**
- `OllamaProvider` - Interface with local Ollama
- `MLXProvider` - Use MLX for M-series Macs
- `HuggingFaceProvider` - Local HF transformers (reuse from Lens)
- `AnthropicProvider` - Limited (API models don't expose internals)
- `OpenAIProvider` - Limited (API models don't expose internals)

**Key Challenge**: API providers (Claude, GPT) don't expose activations
**Solution**: Focus on local models for full functionality, but allow text generation comparison for API models

#### 2. Activation Streaming Engine
```python
class ActivationStreamer:
    async def stream_activations(
        self,
        model: Model,
        text: str,
        layers: List[str],
        mode: Literal["incremental", "complete"]
    ) -> AsyncIterator[ActivationFrame]:
        """
        Stream activations as they're computed.

        - incremental: Token-by-token (for playback)
        - complete: All at once (for full-text analysis)
        """
```

**Outputs:**
```python
@dataclass
class ActivationFrame:
    token_id: int
    token_text: str
    layer_activations: Dict[str, np.ndarray]  # layer_name -> activation vector
    timestamp_ms: int
    metadata: Dict[str, Any]
```

#### 3. Concept Vector Manager
```python
class ConceptLibrary:
    def __init__(self, library_path: Path):
        """Load pre-computed concept vectors"""

    def map_activation_to_concepts(
        self,
        activation: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Return top-k concepts by cosine similarity.
        Returns: [(concept_name, similarity_score), ...]
        """

    def get_available_concepts(self) -> List[str]:
        """List all concepts in library"""
```

**Concept Storage Format:**
```json
{
  "library_name": "base-concepts-v1",
  "model": "llama-3.2-1b",
  "layer": "transformer.h.6",
  "dimension": 768,
  "concepts": [
    {
      "name": "Animals",
      "vector": [0.12, -0.34, ...],  // 768-dim
      "examples": ["cat", "dog", "bird"],
      "activation_threshold": 0.6
    }
  ]
}
```

#### 4. Feature Enhancer (Optional)
```python
class SAEFeatureEnhancer:
    def __init__(self, sae_checkpoint: Path):
        """Load pre-trained SAE for this model/layer"""

    def get_feature_activations(
        self,
        raw_activation: np.ndarray
    ) -> List[Tuple[int, float, str]]:
        """
        Decode raw activation into SAE features.
        Returns: [(feature_id, activation, description), ...]
        """
```

**Integration with Lens:**
- If SAE exists for (model, layer) → enhance with features
- If no SAE → show raw activations only
- Allows training SAEs in Lens, then using them in Explorer

### Frontend Components

#### 1. Model Selector
```typescript
interface ModelConfig {
  provider: 'ollama' | 'mlx' | 'huggingface' | 'anthropic' | 'openai';
  modelId: string;
  layers: string[];  // Available layer names
  dimension: number;  // Activation dimension
  supportsActivations: boolean;
}

<ModelSelector
  onModelSelect={(config) => loadModel(config)}
  availableModels={models}
/>
```

#### 2. Interactive Text Input
```typescript
<TextInput
  value={text}
  onChange={(text) => {
    setText(text);
    // Debounced activation fetch
    debouncedFetchActivations(text);
  }}
  placeholder="Type to see internal activations..."
  streaming={isProcessing}
/>
```

#### 3. Activation Heatmap
```typescript
<ActivationHeatmap
  tokens={tokens}
  layers={layers}
  activations={activations}  // tokens × layers × dim
  aggregation="max"  // or "mean", "l2_norm"
  colorScheme="viridis"
  onCellClick={(token, layer) => showDetail(token, layer)}
  onCellHover={(token, layer) => showTooltip(token, layer)}
/>
```

**Rendering Strategy:**
- Use Canvas for performance (100s of cells)
- Aggregate high-dim activations to single intensity value
- Configurable aggregation (max, mean, L2 norm)

#### 4. Layer Detail Panel
```typescript
<LayerDetailView
  layer={selectedLayer}
  activation={selectedActivation}  // 768-dim vector
  concepts={mappedConcepts}  // From ConceptLibrary
  features={saeFeatures}  // From SAE (optional)
  showTopK={20}
/>
```

#### 5. Token Playback Controller
```typescript
<TokenPlayback
  tokens={tokens}
  activations={tokenActivations}  // time series
  currentIndex={playbackIndex}
  onSeek={(index) => setPlaybackIndex(index)}
  speed={playbackSpeed}
  playing={isPlaying}
/>
```

#### 6. Share Dialog
```typescript
<ShareDialog
  config={{
    model: currentModel,
    text: inputText,
    layers: visibleLayers,
    viewMode: currentViewMode
  }}
  onGenerateLink={() => {
    const linkId = saveVisualizationState();
    return `https://llm-state-explorer.app/view/${linkId}`;
  }}
  exportFormats={['link', 'html', 'video', 'json']}
/>
```

### Data Flow

#### Real-Time Activation Flow
```
User types "The cat" → Frontend debounces (300ms)
                     → Send to backend via WebSocket

Backend receives text → Tokenize
                      → Run forward pass with hooks
                      → Capture activations per layer
                      → Aggregate to manageable size
                      → Stream back via WebSocket

Frontend receives → Update heatmap
                  → Update detail panels
                  → Trigger concept mapping (async)
                  → Render in < 16ms (60 FPS)
```

#### Token-by-Token Playback Flow
```
User clicks play → Backend runs forward pass with per-token capture
                 → Returns time series: [(token_0, activations_0), ...]

Frontend → Animate at user-selected speed
         → Update visualizations at each step
         → Show token highlighting
         → Update layer flows
```

### Performance Optimizations

**1. Activation Aggregation**
- Don't send full 768-dim vectors to frontend
- Send top-k neurons OR aggregate to single value
- Lazy-load full vectors on detail view

**2. WebSocket Streaming**
- Send incremental updates, not full state
- Use binary format (MessagePack or Protobuf) not JSON
- Debounce user input (300ms)

**3. Frontend Rendering**
- Use Canvas for heatmaps (not SVG)
- Virtualize long token sequences
- Memoize expensive computations
- Use Web Workers for concept mapping

**4. Model Loading**
- Cache loaded models in backend
- Lazy-load layers (only hook requested layers)
- Use model quantization for large models

### Storage & Sharing

#### Shareable State Format
```typescript
interface SharedVisualization {
  id: string;  // UUID
  created_at: string;
  model: ModelConfig;
  input_text: string;
  layers: string[];
  view_mode: 'heatmap' | 'detail' | 'playback';
  activations?: CompressedActivations;  // Optional pre-computed
  concepts?: ConceptMapping[];
  features?: SAEFeatures[];
}
```

**Storage Options:**
1. **Lightweight Links**: Store only config, recompute activations on view
2. **Full Snapshots**: Store pre-computed activations for instant loading
3. **Export Options**:
   - Static HTML with embedded data
   - Video export of playback
   - JSON for programmatic access

### Technology Choices

**Backend:**
- FastAPI (reuse from Lens)
- WebSockets (reuse from Lens)
- PyTorch for activation capture
- NumPy for processing
- Redis for caching (new)

**Frontend:**
- Next.js 14 (reuse from Lens)
- TypeScript
- Zustand for state management (lighter than Redux)
- D3.js for custom visualizations
- Canvas API for heatmaps
- Framer Motion for animations

**Infrastructure:**
- Docker Compose (reuse from Lens)
- PostgreSQL for sharing (reuse from Lens)
- Redis for model caching
- nginx for static exports

---

## Implementation Plan

### Phase 1: MVP (Week 1-2)

**Goal**: Interactive single-model visualization with basic heatmap

**Tasks:**
1. **Backend Foundation** (3 days)
   - [ ] Create `ModelProvider` abstraction
   - [ ] Implement `HuggingFaceProvider` (reuse Lens code)
   - [ ] Create `ActivationStreamer` with WebSocket
   - [ ] API endpoints: `/models`, `/activate`, `/ws`

2. **Frontend Foundation** (3 days)
   - [ ] Model selector component
   - [ ] Interactive text input with debouncing
   - [ ] Basic activation heatmap (Canvas-based)
   - [ ] WebSocket client

3. **Integration** (2 days)
   - [ ] Connect frontend to backend
   - [ ] Test with GPT-2 (small, fast)
   - [ ] Performance testing and optimization

4. **Deployment** (1 day)
   - [ ] Docker Compose setup
   - [ ] Local deployment instructions
   - [ ] Basic documentation

**Deliverable**: Working prototype - type text, see activation heatmap update in real-time

### Phase 2: Enhancement (Week 3-4)

**Goal**: Multi-model support, layer details, concept mapping

**Tasks:**
1. **Multi-Model Support** (3 days)
   - [ ] Implement `OllamaProvider`
   - [ ] Implement `MLXProvider`
   - [ ] Model architecture auto-detection
   - [ ] Model switching UI

2. **Layer Detail View** (3 days)
   - [ ] Top-k neuron display
   - [ ] Activation distribution histogram
   - [ ] Click-to-zoom from heatmap
   - [ ] Export layer data

3. **Concept Mapping** (4 days)
   - [ ] Design concept vector format
   - [ ] Create `ConceptLibrary` class
   - [ ] Build initial concept set (emotions, objects, relations)
   - [ ] UI for concept similarities
   - [ ] Concept highlighting in heatmap

4. **Polish** (2 days)
   - [ ] Loading states
   - [ ] Error handling
   - [ ] Responsive design
   - [ ] Performance optimization

**Deliverable**: Multi-model support with concept mapping

### Phase 3: Sharing & Playback (Week 5-6)

**Goal**: Token-by-token playback and shareable links

**Tasks:**
1. **Token Playback** (4 days)
   - [ ] Per-token activation capture
   - [ ] Playback controller UI
   - [ ] Animation system
   - [ ] Variable speed control
   - [ ] Layer flow visualization

2. **Sharing System** (4 days)
   - [ ] Visualization state serialization
   - [ ] Database storage for shared links
   - [ ] Share dialog UI
   - [ ] Read-only viewer mode
   - [ ] Static HTML export

3. **SAE Integration** (3 days)
   - [ ] Load SAE checkpoints from Lens
   - [ ] `SAEFeatureEnhancer` implementation
   - [ ] Feature activation display
   - [ ] Toggle between raw and features

4. **Final Polish** (3 days)
   - [ ] UI/UX refinement
   - [ ] Documentation
   - [ ] Video export (optional)
   - [ ] Gallery of examples

**Deliverable**: Full-featured v1 with sharing and playback

### Optional Phase 4: Advanced Features (Week 7-8+)

**Stretch Goals:**
- [ ] Attention head visualization
- [ ] Cross-layer comparison view
- [ ] Custom concept vector creation
- [ ] Batch comparison (multiple inputs side-by-side)
- [ ] Integration with Lens (train SAE → use in Explorer)
- [ ] Mobile-responsive design
- [ ] Collaborative annotations
- [ ] API for programmatic access

---

## Success Criteria

### Technical Metrics
- [ ] **Latency**: < 500ms from text input to visualization update
- [ ] **Models**: Support 3+ model sources (HF, Ollama, MLX)
- [ ] **Layers**: Visualize up to 48 layers simultaneously
- [ ] **Performance**: Handle 1000+ token sequences smoothly
- [ ] **Sharing**: Generate shareable links in < 2 seconds

### User Experience Metrics
- [ ] **Intuitive**: New users understand within 30 seconds
- [ ] **Responsive**: Feels real-time (no perceivable lag)
- [ ] **Beautiful**: Visualizations are clear and aesthetically pleasing
- [ ] **Shareable**: Easy to share discoveries (1-click)
- [ ] **Informative**: Users learn something new about LLMs

### Research Impact Metrics
- [ ] **Concept Discovery**: Identify 3+ novel concept activations
- [ ] **Teaching**: Use in educational content
- [ ] **Debugging**: Help debug prompt engineering
- [ ] **Interpretability**: Contribute to SAE feature understanding

---

## Key Design Decisions

### 1. Build New vs. Extend Lens?

**Decision: Build New ("State Explorer")**

**Rationale:**
- Lens is optimized for SAE research workflow (experiment-based)
- State Explorer is optimized for interactive exploration
- Different mental models, different UX patterns
- Shared backend components (reuse activation capture, SAE code)
- Lens → "Train SAEs", State Explorer → "Explore models"

**Shared Infrastructure:**
- Docker setup
- Backend FastAPI patterns
- Database models
- SAE implementation
- Activation capture hooks

### 2. Activation Streaming Strategy?

**Decision: Hybrid (Debounced + On-Demand)**

**Rationale:**
- **Debounced** for live typing (300ms delay, send complete text)
- **On-Demand** for token playback (request time series)
- Avoids overwhelming backend with per-keystroke requests
- Balances responsiveness and efficiency

### 3. API Models Support?

**Decision: Limited Support**

**Rationale:**
- Claude/GPT APIs don't expose internal activations
- Can show text generation comparison
- Document limitation clearly
- Focus on local models for full functionality
- Future: If APIs expose activations, add support

### 4. Concept Vector Source?

**Decision: Pre-Computed + Community-Contributed**

**Rationale:**
- Ship with base concept library (emotions, objects, relations)
- Allow users to contribute custom concepts
- Store in shared/concepts/ for reuse across projects
- Format: JSON + NumPy arrays

**Initial Concepts (50-100):**
- Emotions: joy, sadness, anger, fear, surprise, disgust
- Objects: animals, furniture, tools, vehicles, food
- Relations: spatial, temporal, causal, social
- Abstract: numbers, colors, sizes, shapes
- Actions: motion, communication, cognition

### 5. Visualization Complexity?

**Decision: Progressive Disclosure**

**Rationale:**
- **Default**: Simple heatmap (clear, fast)
- **Click**: Detailed layer view (more info)
- **Toggle**: Advanced modes (features, concepts, flows)
- Avoid overwhelming new users
- Power users can go deep

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Performance** (slow activation capture) | Medium | High | Start with small models, optimize with caching, use quantization |
| **Model compatibility** (layer name variance) | High | Medium | Build robust layer detection, fallback patterns |
| **Frontend rendering** (100+ layers × 1000 tokens) | Medium | Medium | Canvas rendering, virtualization, aggregation |
| **WebSocket stability** | Low | Medium | Fallback to polling, reconnection logic |
| **Memory usage** (large models) | Medium | High | Model caching, lazy loading, clear unused models |

### User Experience Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Complexity** (too much info) | High | High | Progressive disclosure, clear defaults |
| **Setup friction** (hard to get started) | Medium | High | Pre-configured examples, one-click demos |
| **Interpretation** (misunderstand activations) | High | Medium | Educational tooltips, example interpretations |
| **Sharing limits** (large activations) | Medium | Low | Compress data, optional full snapshots |

### Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Scope creep** | High | Medium | Stick to phased plan, defer advanced features |
| **Integration complexity** (Lens + Explorer) | Low | Low | Keep codebases separate, share components |
| **Maintenance burden** | Medium | Medium | Document thoroughly, modular architecture |

---

## Alternative Approaches Considered

### Alternative 1: Extend Lens with "Explorer Mode"

**Pros:**
- Single codebase
- Reuse all Lens infrastructure
- Unified deployment

**Cons:**
- Mental model clash (experiment-based vs. interactive)
- Risk of making Lens too complex
- Harder to optimize for different workflows

**Verdict**: Rejected - better to have focused tools

### Alternative 2: Pure Client-Side (No Backend)

**Pros:**
- No deployment needed
- Easy to share (static site)
- Fast iteration

**Cons:**
- Can't run large models in browser
- Limited to ONNX or TensorFlow.js models
- Sacrifices power for convenience

**Verdict**: Rejected - want to support full-size local models

### Alternative 3: Notebook-Based (Jupyter)

**Pros:**
- Familiar to researchers
- Easy to extend with custom code
- Reproducible

**Cons:**
- Poor UX for non-technical users
- Not shareable as easily
- Harder to make beautiful visualizations

**Verdict**: Rejected - want broader accessibility

### Alternative 4: Browser Extension (Inspect Running Models)

**Pros:**
- Unique positioning
- Could inspect models in ChatGPT, Claude UI
- Novel use case

**Cons:**
- Very hard technically (need API access)
- API providers don't expose activations
- Limited reach

**Verdict**: Interesting but defer to v2

---

## Open Questions

### Technical
1. **Activation compression**: What's the best aggregation for high-dim vectors? (max, mean, PCA?)
2. **Concept vector creation**: How to generate initial concept library? (run many examples, average activations?)
3. **SAE compatibility**: Can we load SAEs trained in Lens without code changes?
4. **MLX integration**: Does MLX support forward hooks like PyTorch?

### Design
5. **Default view**: Heatmap or layer detail first?
6. **Color scheme**: Viridis, Plasma, or custom gradient?
7. **Token granularity**: Character-level, token-level, or word-level?
8. **Sharing limits**: How large can shared activations be before degrading UX?

### Product
9. **Target audience**: Researchers? Educators? Hobbyists? All?
10. **Hosting**: Self-hosted only or provide cloud demo?
11. **Monetization**: Free & open-source or freemium?
12. **Branding**: Part of Hidden Layer or separate tool?

---

## Next Steps

### Immediate (This Week)
1. **Decision**: Review this plan and decide:
   - Build State Explorer as new project? (Recommended)
   - Or extend Lens with Explorer mode?
   - Or notebook-based prototype first?

2. **Setup**: If building State Explorer:
   ```bash
   mkdir -p representations/state-explorer/{backend,frontend,docs}
   ```

3. **Prototype**: Build minimal activation heatmap (2 days)
   - Single model (GPT-2)
   - Text input → activation capture → heatmap
   - No WebSocket yet, just HTTP

4. **Validate**: Show to 2-3 users, get feedback

### Short-Term (Next 2 Weeks)
5. **MVP Development**: Follow Phase 1 plan
6. **Documentation**: Setup guide, architecture doc
7. **Demo**: Record video showing capabilities

### Long-Term (Month 2-3)
8. **Feature Development**: Phases 2-3
9. **Community**: Open-source and share with interpretability community
10. **Integration**: Connect with Lens (train SAEs → use features in Explorer)
11. **Research**: Use tool to explore specific hypotheses (theory of mind, deception, etc.)

---

## Appendix: Example Concepts

### Seed Concept Library (MVP)

**Emotions (10)**
- Joy, Sadness, Anger, Fear, Surprise, Disgust, Love, Hate, Trust, Anticipation

**Objects (15)**
- Animals, Plants, Furniture, Tools, Vehicles, Food, Clothing, Buildings, Weapons, Art, Technology, Nature, Body Parts, Containers, Instruments

**Relations (10)**
- Spatial (above, below, inside, outside), Temporal (before, after, during), Causal (cause, effect), Social (family, friend, authority), Possession

**Abstract (10)**
- Numbers, Colors, Sizes, Shapes, Quantities, Qualities, Modality (must, can, might), Negation, Comparison, Intensity

**Actions (15)**
- Motion (walk, run, fly), Communication (speak, write, read), Cognition (think, know, believe), Perception (see, hear, feel), Creation (make, build, destroy), Social (help, harm, cooperate)

**Total: 60 concepts**

### How to Generate Concept Vectors

```python
# Example: Generate "joy" concept vector

prompts = [
    "I am so happy and joyful!",
    "This brings me great joy.",
    "Feeling joyful and cheerful today.",
    # ... 20 more examples
]

activations = []
for prompt in prompts:
    activation = capture_activation(model, prompt, layer="transformer.h.6")
    activations.append(activation)

# Average to get concept vector
joy_vector = np.mean(activations, axis=0)

# Store
save_concept("joy", joy_vector, layer="transformer.h.6", model="gpt2")
```

---

## References & Inspiration

**Similar Tools:**
- [LM Debugger](https://github.com/mega002/lm-debugger) - Debugging language models
- [Captum](https://captum.ai/) - Model interpretability for PyTorch
- [Neuroscope](https://neuroscope.io/) - Neural network visualization
- [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) - Attention visualization

**Research:**
- Anthropic's [Circuits Thread](https://transformer-circuits.pub/)
- OpenAI's [Microscope](https://microscope.openai.com/)
- [Neuron2Graph](https://neuron2graph.github.io/)
- [LLM Visualization](https://bbycroft.net/llm) - 3D GPT-2 visualization

**Technical Inspiration:**
- Chrome DevTools (inspect internal browser state)
- Weights & Biases (experiment tracking + visualization)
- Observable (interactive notebooks)
- Distill.pub (beautiful ML explanations)

---

## Conclusion

**You have a strong foundation** with Lens. The path forward is clear:

1. **Build "LLM State Explorer"** as a companion tool
2. **Reuse** Lens infrastructure (activation capture, SAE, backend patterns)
3. **Focus** on real-time interactivity and shareability
4. **Start small** with MVP (heatmap + single model)
5. **Iterate** based on user feedback

**Timeline**: 2-3 weeks to something shareable, 6 weeks to something great.

**Impact**: A tool that makes LLM internals accessible to everyone - researchers, educators, and the curious.

Let's build it. 🚀
