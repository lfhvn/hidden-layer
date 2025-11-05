# Repository Organization Alternatives

## Current Proposal (Flat Projects)

```
hidden-layer/
├── harness/                    # Infrastructure
├── shared/                     # Shared resources
├── projects/                   # All projects flat
│   ├── multi-agent/
│   ├── ai-to-ai-comm/
│   ├── selphi/
│   ├── introspection/
│   ├── latent-space/
│   └── steerability/
├── web-tools/                  # Deployment
└── docs/
```

**Pros:**
- Simple, flat structure
- Follows current CLAUDE.md structure
- Easy to navigate

**Cons:**
- No explicit grouping by research theme
- Connections between related projects not obvious from structure
- All projects at same level despite different maturity/scope

---

## Alternative A: Research Areas at Top Level

```
hidden-layer/
├── infrastructure/
│   ├── harness/              # Core infrastructure
│   ├── shared/               # Shared resources (concepts, datasets, utils)
│   └── web-tools/            # Deployment infrastructure
│
├── communication/            # Research Area 1
│   ├── multi-agent/
│   │   ├── code/
│   │   ├── notebooks/
│   │   └── arena/            # Web app (dev version)
│   └── ai-to-ai-comm/
│       └── code/
│
├── theory-of-mind/           # Research Area 2
│   ├── selphi/
│   │   ├── code/
│   │   └── notebooks/
│   └── introspection/
│       └── code/
│
├── representations/          # Research Area 3
│   └── latent-space/
│       ├── lens/             # SAE interpretability (web app)
│       ├── topologies/       # Mobile exploration
│       └── shared/           # Shared latent-space utilities
│
├── alignment/                # Research Area 4
│   └── steerability/
│       ├── code/
│       └── dashboard/        # Web app (dev version)
│
├── docs/                     # Documentation
└── config/
```

**Pros:**
- Research themes explicit in directory structure
- Easy to see which projects are related
- Natural grouping for papers/documentation
- Clearer for newcomers to understand lab focus
- Infrastructure cleanly separated

**Cons:**
- Some projects span multiple themes (e.g., introspection relates to both ToM and representations)
- Deeper nesting for individual projects
- More migration work

---

## Alternative B: Hybrid (Areas + Flat Projects)

```
hidden-layer/
├── infrastructure/
│   ├── harness/
│   ├── shared/
│   └── web-tools/
│
├── research/
│   ├── communication/
│   │   ├── multi-agent/
│   │   └── ai-to-ai-comm/
│   ├── theory-of-mind/
│   │   ├── selphi/
│   │   └── introspection/
│   ├── representations/
│   │   └── latent-space/
│   └── alignment/
│       └── steerability/
│
└── docs/
```

**Pros:**
- Clear separation: infrastructure vs. research
- Research themes visible
- Single `/research/` root for all research projects

**Cons:**
- Still requires decisions about cross-cutting projects
- One extra level of nesting (`/research/`)

---

## Alternative C: Tags/Categories (Keep Flat, Add Metadata)

Keep current flat structure but add:
- `RESEARCH_AREA.md` in each project
- Tags in project README.md headers
- A `/docs/research-map.md` showing connections

**Pros:**
- Minimal disruption
- Projects can belong to multiple themes
- Flexibility

**Cons:**
- Structure doesn't reflect organization
- Relies on documentation discipline

---

## Comparison: Where Projects Live

| Project | Current | Alt A | Alt B |
|---------|---------|-------|-------|
| multi-agent | `projects/multi-agent/` | `communication/multi-agent/` | `research/communication/multi-agent/` |
| ai-to-ai-comm | `projects/ai-to-ai-comm/` | `communication/ai-to-ai-comm/` | `research/communication/ai-to-ai-comm/` |
| selphi | `projects/selphi/` | `theory-of-mind/selphi/` | `research/theory-of-mind/selphi/` |
| introspection | `projects/introspection/` | `theory-of-mind/introspection/` | `research/theory-of-mind/introspection/` |
| latent-space | `projects/latent-space/` | `representations/latent-space/` | `research/representations/latent-space/` |
| steerability | `projects/steerability/` | `alignment/steerability/` | `research/alignment/steerability/` |

---

## Handling Cross-Cutting Projects

Some projects span multiple themes:

**Introspection**:
- Theory of Mind (self-knowledge)
- Representations (what activations correspond to introspection?)
- Alignment (honest reporting)

**Options:**
1. **Primary placement** - Put in main research area, document connections in README
2. **Symlinks** - Create symlinks in multiple areas (confusing for git)
3. **Shared utilities** - Extract shared code to `/shared/`, specific code stays in primary area

**Recommendation**: Option 1 - Primary placement with clear documentation

---

## Infrastructure Organization

All alternatives agree: Infrastructure should be separate and include:

```
infrastructure/  (or keep at root)
├── harness/          # Core LLM/experiment infrastructure
├── shared/           # Research resources (concepts, datasets, utils)
└── web-tools/        # Deployment infrastructure
    ├── multi-agent-arena/
    ├── latent-lens/
    ├── steerability/
    └── shared/backend/
```

---

## Import Path Considerations

### Current (Flat):
```python
from harness import llm_call
from shared.concepts import ConceptLibrary
```

### Alternative A (Research Areas):
```python
from infrastructure.harness import llm_call
from infrastructure.shared.concepts import ConceptLibrary
```

This affects:
- All existing imports (major breaking change)
- Python path setup in scripts
- Documentation

**Mitigation**: Could keep `harness/` and `shared/` at root for backward compatibility:

```
hidden-layer/
├── harness/                  # Keep at root for imports
├── shared/                   # Keep at root for imports
├── communication/            # Research areas
├── theory-of-mind/
├── representations/
├── alignment/
└── web-tools/               # Keep at root (deployment)
```

---

## Recommendation

**Option: Alternative A with root-level infrastructure**

```
hidden-layer/
├── harness/                  # Root-level for easy imports
├── shared/                   # Root-level for easy imports
├── web-tools/                # Root-level (deployment)
│
├── communication/            # Research areas
│   ├── multi-agent/
│   └── ai-to-ai-comm/
├── theory-of-mind/
│   ├── selphi/
│   └── introspection/
├── representations/
│   └── latent-space/
│       ├── lens/
│       └── topologies/
└── alignment/
    └── steerability/
        ├── code/
        └── dashboard/
```

**Why:**
1. ✅ Research themes explicit in structure
2. ✅ No breaking import changes
3. ✅ Infrastructure stays accessible
4. ✅ Clear separation of concerns
5. ✅ Easy to find related projects
6. ✅ Better for papers/documentation organization

**Migration effort:** Moderate
- Move projects to research area folders
- Update CLAUDE.md to reflect new structure
- Add research area README.md files
- No import changes needed

---

## Next Steps

1. Choose structure (Current, A, B, or C)
2. Update CLAUDE.md to reflect choice
3. Execute migration in stages
4. Update all documentation

**Question for user**: Which structure do you prefer?
