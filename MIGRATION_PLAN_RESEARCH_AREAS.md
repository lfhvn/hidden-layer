# Migration Plan: Research Area Organization

**Decision**: Alternative A - Research Areas at Top Level
**Date**: 2025-11-05
**Status**: Ready to Execute

---

## Final Structure

```
hidden-layer/
├── harness/                    # Core infrastructure (✅ at root for easy imports)
├── shared/                     # Research resources (✅ at root for easy imports)
│   ├── concepts/              # Consolidated from /concepts
│   ├── datasets/
│   └── utils/
├── web-tools/                  # Deployment infrastructure (✅ stays at root)
│   ├── multi-agent-arena/
│   ├── latent-lens/
│   ├── steerability/
│   └── shared/backend/
│
├── communication/              # 🔬 Research Area 1: Agent Communication & Coordination
│   ├── multi-agent/
│   │   ├── code/
│   │   ├── notebooks/
│   │   └── arena/             # Dev version (from web-tools? TBD)
│   ├── ai-to-ai-comm/
│   │   └── code/
│   └── README.md              # Research area overview
│
├── theory-of-mind/             # 🔬 Research Area 2: Theory of Mind & Self-Knowledge
│   ├── selphi/
│   │   ├── code/
│   │   ├── notebooks/         # From /notebooks/selphi
│   │   ├── CLAUDE.md
│   │   └── README.md
│   ├── introspection/
│   │   ├── code/
│   │   ├── CLAUDE.md
│   │   └── README.md
│   └── README.md              # Research area overview
│
├── representations/            # 🔬 Research Area 3: Internal Representations & Interpretability
│   ├── latent-space/
│   │   ├── lens/              # From /latent-lens (dev version)
│   │   │   ├── frontend/
│   │   │   ├── backend/
│   │   │   └── README.md
│   │   ├── topologies/        # From /latent-topologies (dev version)
│   │   │   ├── scripts/
│   │   │   ├── data/
│   │   │   └── README.md
│   │   ├── CLAUDE.md
│   │   └── README.md
│   └── README.md              # Research area overview
│
├── alignment/                  # 🔬 Research Area 4: Alignment, Steerability & Deception
│   ├── steerability/
│   │   ├── code/
│   │   ├── dashboard/         # From /steerability-dashboard (dev version)
│   │   │   ├── frontend/
│   │   │   ├── backend/
│   │   │   └── README.md
│   │   ├── CLAUDE.md
│   │   └── README.md
│   └── README.md              # Research area overview
│
├── docs/                       # Lab-wide documentation
│   ├── infrastructure/
│   ├── hardware/
│   ├── workflows/
│   ├── archive/
│   ├── ARCHITECTURE.md        # From root
│   ├── BENCHMARKS.md          # From root
│   ├── MIGRATION.md           # From root
│   ├── INTROSPECTION.md       # From root
│   ├── INTROSPECTION_PLAN.md  # From root
│   ├── MLX_MODELS_2025.md     # From root
│   └── PROVIDER_LIMITATIONS.md # From root
│
├── tests/                      # Lab-wide tests
├── config/                     # Lab-wide config
│
├── README.md                   # Lab overview (keep at root)
├── CLAUDE.md                   # Development guide (keep at root, update for new structure)
├── RESEARCH.md                 # Research overview (keep at root, update)
├── QUICKSTART.md               # Quick start (keep at root)
├── SETUP.md                    # Setup guide (keep at root)
├── LICENSE
├── Makefile
└── setup files...
```

---

## Benefits of This Structure

### 1. Scalability - Room to Grow ✅

**New research directions** = New top-level folders:
```
hidden-layer/
├── communication/
├── theory-of-mind/
├── representations/
├── alignment/
├── emergent-behavior/          # NEW: Future research area
└── safety-testing/             # NEW: Future research area
```

**New projects within areas** = New subdirectories:
```
communication/
├── multi-agent/
├── ai-to-ai-comm/
├── swarm-intelligence/         # NEW: Future project
└── consensus-mechanisms/       # NEW: Future project
```

### 2. Clear Organization
- Research themes explicit in directory structure
- Related projects grouped together
- Easy for newcomers to understand lab focus

### 3. No Breaking Changes
- `harness/`, `shared/`, `web-tools/` stay at root
- All imports continue to work: `from harness import ...`
- Web tools deployment unchanged

### 4. Better for Research Output
- Natural grouping for papers/documentation
- Research area READMEs can track area progress
- Cross-project collaboration clearer

---

## Migration Stages

### Stage 4: Create Research Area Directories ✅ (Quick)

**Create empty structure:**
```bash
mkdir -p communication
mkdir -p theory-of-mind
mkdir -p representations
mkdir -p alignment
```

**Create research area README files** with overviews from RESEARCH.md

---

### Stage 5: Move Communication Projects

**From:**
```
projects/multi-agent/
projects/ai-to-ai-comm/
```

**To:**
```
communication/multi-agent/
communication/ai-to-ai-comm/
```

**Commands:**
```bash
mv projects/multi-agent communication/
mv projects/ai-to-ai-comm communication/
```

**Note:** Multi-agent arena stays in `web-tools/` (deployment version)

---

### Stage 6: Move Theory of Mind Projects

**From:**
```
projects/selphi/
projects/introspection/
notebooks/selphi/
```

**To:**
```
theory-of-mind/selphi/
theory-of-mind/introspection/
theory-of-mind/selphi/notebooks/  (merge with existing)
```

**Commands:**
```bash
mv projects/selphi theory-of-mind/
mv projects/introspection theory-of-mind/

# Merge notebooks
if [ -d "theory-of-mind/selphi/notebooks" ]; then
  cp -r notebooks/selphi/* theory-of-mind/selphi/notebooks/
else
  mv notebooks/selphi theory-of-mind/selphi/notebooks
fi
```

---

### Stage 7: Move Representations Projects

**From:**
```
latent-lens/              (dev version at root)
latent-topologies/        (dev version at root)
projects/latent-space/    (existing structure)
```

**To:**
```
representations/latent-space/lens/        (from /latent-lens)
representations/latent-space/topologies/  (from /latent-topologies)
```

**Commands:**
```bash
# Create latent-space directory if needed
mkdir -p representations/latent-space

# Move lens
mv latent-lens representations/latent-space/lens

# Move topologies
mv latent-topologies representations/latent-space/topologies

# Move existing latent-space structure if it exists
if [ -d "projects/latent-space" ]; then
  # Merge or move additional files from projects/latent-space
  cp -r projects/latent-space/* representations/latent-space/
fi
```

**Note:** `web-tools/latent-lens/` stays (deployment version)

---

### Stage 8: Move Alignment Projects

**From:**
```
steerability-dashboard/    (dev version at root)
projects/steerability/     (existing structure)
```

**To:**
```
alignment/steerability/code/
alignment/steerability/dashboard/  (from /steerability-dashboard)
```

**Commands:**
```bash
mkdir -p alignment/steerability

# Move existing code structure
if [ -d "projects/steerability/code" ]; then
  mv projects/steerability/code alignment/steerability/
fi

# Move dashboard
mv steerability-dashboard alignment/steerability/dashboard

# Move any remaining steerability files
if [ -d "projects/steerability" ]; then
  cp -r projects/steerability/* alignment/steerability/
fi
```

**Note:** `web-tools/steerability/` stays (deployment version)

---

### Stage 9: Consolidate Shared Resources

**From:**
```
concepts/              (orphaned at root)
```

**To:**
```
shared/concepts/      (merge with existing)
```

**Commands:**
```bash
# Check if shared/concepts exists and is empty or has content
if [ -d "shared/concepts" ] && [ "$(ls -A shared/concepts)" ]; then
  # Merge
  cp -r concepts/* shared/concepts/
  rm -rf concepts/
else
  # Move
  mv concepts shared/
fi
```

---

### Stage 10: Organize Notebooks

**From:**
```
notebooks/crit/
```

**To:**
```
communication/multi-agent/notebooks/crit/  (merge with existing notebooks)
```

**Commands:**
```bash
# Create notebooks directory if needed
mkdir -p communication/multi-agent/notebooks

# Move crit notebooks
if [ -d "notebooks/crit" ]; then
  mv notebooks/crit communication/multi-agent/notebooks/
fi

# Remove empty notebooks directory
if [ -z "$(ls -A notebooks)" ]; then
  rm -rf notebooks
fi
```

---

### Stage 11: Clean Up Root Documentation

**Move to /docs/:**
- ARCHITECTURE.md
- BENCHMARKS.md
- MIGRATION.md
- INTROSPECTION.md
- INTROSPECTION_PLAN.md
- MLX_MODELS_2025.md
- PROVIDER_LIMITATIONS.md

**Delete:**
- *.md.old files
- REORGANIZATION_COMPLETE.md (outdated)

**Keep at root:**
- README.md
- CLAUDE.md
- RESEARCH.md
- QUICKSTART.md
- SETUP.md
- LICENSE

**Commands:**
```bash
# Move to docs/
mv ARCHITECTURE.md docs/
mv BENCHMARKS.md docs/
mv MIGRATION.md docs/
mv INTROSPECTION.md docs/
mv INTROSPECTION_PLAN.md docs/
mv MLX_MODELS_2025.md docs/
mv PROVIDER_LIMITATIONS.md docs/

# Delete obsolete
rm -f *.md.old
rm -f REORGANIZATION_COMPLETE.md
```

---

### Stage 12: Create Research Area READMEs

Create `README.md` in each research area with:
- Research area overview
- List of projects
- Research questions
- Cross-project connections
- Recent findings

**Example: `communication/README.md`**
```markdown
# Communication Research Area

Research focus: How do multiple AI agents communicate and coordinate?

## Projects

- **multi-agent/**: Debate, CRIT, consensus, XFN teams
- **ai-to-ai-comm/**: Non-linguistic communication via latent representations

## Research Questions

- When and why do multi-agent systems outperform single agents?
- Can agents communicate more efficiently through latent representations?
- What coordination mechanisms emerge?

## Cross-Area Connections

- **Representations**: Can agents use latent space coordinates for communication?
- **Theory of Mind**: Do agents develop models of each other?
```

---

### Stage 13: Update CLAUDE.md

Update main CLAUDE.md to reflect new structure:

1. Update file organization section
2. Update project paths
3. Update navigation instructions
4. Add note about research area organization
5. Update examples with new paths

---

### Stage 14: Remove Empty /projects/ Directory

**Commands:**
```bash
# After all moves, check if projects/ is empty
if [ -z "$(ls -A projects)" ]; then
  rm -rf projects
fi
```

---

### Stage 15: Final Verification

**Check:**
1. ✅ All imports still work: `python -c "from harness import llm_call; print('OK')"`
2. ✅ Run test suite: `pytest tests/`
3. ✅ Check no broken symlinks: `find . -xtype l`
4. ✅ Verify web-tools deployment scripts still work
5. ✅ Update any absolute paths in configs

---

## Rollback Plan

If issues arise:
1. All changes are in git - can revert commits
2. Original structure preserved in git history
3. No import changes mean most code unaffected

---

## Post-Migration

1. Update documentation
2. Update CI/CD paths if applicable
3. Announce new structure to collaborators
4. Create research area tracking issues

---

## Execution Plan

Execute stages 4-15 one at a time:
- Commit after each stage
- Test between stages
- Can pause/resume at any stage boundary

**Ready to start?** Begin with Stage 4 (creating directories).
