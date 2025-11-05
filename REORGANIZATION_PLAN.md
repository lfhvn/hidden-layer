# Repository Reorganization Plan

**Created**: 2025-11-05
**Status**: Stage 1 Complete - Analysis Done

---

## Stage 1 Findings: Duplicate Analysis ✅

### `/harness` vs `/code/harness`
- **KEEP**: `/harness/` (74KB) - This is the canonical version, all imports use it
- **REMOVE**: `/code/harness/` (174KB) - Old location, project-specific files already moved to:
  - `projects/introspection/code/` (activation_steering, concept_vectors, introspection_*)
  - `projects/multi-agent/code/` (strategies.py, rationale.py)

### `/code/` directory - COMPLETELY REDUNDANT
All contents already migrated to proper locations:
- `/code/crit/` → `/projects/multi-agent/code/crit/` (identical, verified with diff)
- `/code/selphi/` → `/projects/selphi/code/` (identical)
- `/code/harness/` → see above
- `/code/cli.py` → `/projects/multi-agent/code/cli.py` (check this)

**Decision**: Remove entire `/code/` directory after final verification

---

## Remaining Stages

### Stage 2: Check web-tools/ duplication
- `/latent-lens` vs `/web-tools/latent-lens` vs should be in `/projects/latent-space/lens/`
- `/steerability-dashboard` vs `/web-tools/steerability` vs `/projects/steerability/dashboard/`
- `/web-tools/multi-agent-arena` → `/projects/multi-agent/arena/`
- `/web-tools/shared/` vs `/shared/`

### Stage 3: Organize latent-space
- Move `/latent-lens` → `/projects/latent-space/lens/`
- Move `/latent-topologies` → `/projects/latent-space/topologies/`

### Stage 4: Consolidate steerability
- Move `/steerability-dashboard` → `/projects/steerability/dashboard/`

### Stage 5: Move multi-agent arena
- Move `/web-tools/multi-agent-arena` → `/projects/multi-agent/arena/`

### Stage 6: Consolidate shared resources
- Move `/concepts/` → `/shared/concepts/`
- Resolve `/web-tools/shared/` vs `/shared/`

### Stage 7: Documentation cleanup
Root has 13 .md files, many should move to `/docs/`:
- Keep at root: README.md, CLAUDE.md, RESEARCH.md, QUICKSTART.md, LICENSE
- Move to /docs/: ARCHITECTURE.md, MIGRATION.md, BENCHMARKS.md, etc.
- Delete: *.md.old, REORGANIZATION_COMPLETE.md (outdated)

### Stage 8: Remove obsolete directories
- `/code/` (entire directory)
- `/web-tools/` (after moving contents)
- `/notebooks/` (if empty or should be in projects)

### Stage 9: Update all import paths
- Check if any imports need updating
- Update any documentation references

### Stage 10: Final verification
- Run tests
- Check all projects still work
- Commit changes

---

## Expected Final Structure

```
hidden-layer/
├── harness/                    # Core infrastructure (canonical)
├── shared/
│   ├── concepts/              # Moved from /concepts
│   ├── datasets/
│   └── utils/
├── projects/
│   ├── latent-space/
│   │   ├── lens/             # Moved from /latent-lens
│   │   └── topologies/       # Moved from /latent-topologies
│   ├── steerability/
│   │   └── dashboard/        # Moved from /steerability-dashboard
│   ├── multi-agent/
│   │   ├── arena/            # Moved from /web-tools/multi-agent-arena
│   │   └── code/             # Existing
│   ├── selphi/
│   ├── introspection/
│   └── ai-to-ai-comm/
├── docs/                       # Organized documentation
├── tests/
├── config/
├── README.md
├── CLAUDE.md
├── RESEARCH.md
├── QUICKSTART.md
└── setup files...
```

---

## Commands Preview (Not Executed Yet)

```bash
# Stage 1: Remove /code/ directory
rm -rf /home/user/hidden-layer/code/

# Stages 2-8: To be planned after verification
```
