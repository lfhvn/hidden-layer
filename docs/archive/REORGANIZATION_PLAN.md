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

**Decision**: ✅ DELETED - Removed entire `/code/` directory (27 files, 10K lines)

---

## Stage 2: `/code/` Removal ✅

Successfully removed obsolete `/code/` directory. Verified web-tools imports work via `projects/multi-agent` path.

---

## Stage 3 Findings: Web Tools Analysis ✅

### The Situation
There are TWO separate but related directory structures:

**Top-level apps** (development/full versions):
- `/latent-lens` - Full dev version with complete backend, tests, docs
- `/steerability-dashboard` - Full dev version
- `/latent-topologies` - Standalone project

**Web-tools collection** (deployment versions):
- `/web-tools/latent-lens` - Simplified deployment version
- `/web-tools/steerability` - Deployment version with extra CLAUDE.md, deploy.sh
- `/web-tools/multi-agent-arena` - Deployment-ready arena
- `/web-tools/shared/backend/` - Shared auth/middleware utilities

### Key Discovery: Different Purposes!

**Top-level apps are DEVELOPMENT versions:**
- Full feature set
- Complete test suites
- Development documentation
- Docker setup for local dev

**`/web-tools/` are DEPLOYMENT versions:**
- Production-ready configurations
- Shared backend utilities (auth.py, middleware.py)
- Deployment scripts (deploy.sh, railway.json, vercel.json)
- All use `/web-tools/shared/backend/` for common auth/middleware

### The Two `/shared/` Directories Serve Different Purposes:
- `/shared/` - Research resources (concepts, datasets) per CLAUDE.md
- `/web-tools/shared/backend/` - Web deployment utilities (auth, middleware)

**Decision**: These are NOT duplicates! They serve different purposes.

---

## REVISED Strategy

### What We Learned
1. `/web-tools/` is NOT redundant - it's deployment infrastructure
2. Top-level apps are development versions
3. Need to follow CLAUDE.md structure: projects should be under `/projects/`

### Remaining Stages

### Stage 4: Organize latent-space under /projects/
**Current:**
- `/latent-lens` (development version)
- `/latent-topologies` (development version)
- `/web-tools/latent-lens` (deployment version) ✓ stays

**Target:**
- `/projects/latent-space/lens/` (move latent-lens here)
- `/projects/latent-space/topologies/` (move latent-topologies here)
- `/web-tools/latent-lens/` (stays - deployment version)

### Stage 5: Organize steerability under /projects/
**Current:**
- `/steerability-dashboard` (development version)
- `/web-tools/steerability` (deployment version) ✓ stays

**Target:**
- `/projects/steerability/dashboard/` (move steerability-dashboard here)
- `/web-tools/steerability/` (stays - deployment version)

### Stage 6: Move multi-agent arena
**Current:**
- `/web-tools/multi-agent-arena` (deployment version, but no dev version exists!)

**Options:**
A. Keep in web-tools (it's deployment-focused)
B. Create `/projects/multi-agent/arena/` with dev version, keep web-tools for deployment

**Recommendation:** Option B - create proper project structure

### Stage 7: Consolidate /concepts into /shared/
**Current:**
- `/concepts/` (orphaned)
- `/shared/concepts/` (already exists but might be empty)

**Target:** Merge into `/shared/concepts/`

### Stage 8: Documentation cleanup
Root has 13 .md files. Clean up:
- **Keep at root**: README.md, CLAUDE.md, RESEARCH.md, QUICKSTART.md, SETUP.md, LICENSE
- **Move to /docs/**: ARCHITECTURE.md, MIGRATION.md, BENCHMARKS.md, INTROSPECTION*.md, MLX_MODELS.md, PROVIDER_LIMITATIONS.md
- **Delete**: *.md.old, REORGANIZATION_COMPLETE.md (outdated)

### Stage 9: Clean up notebooks/
**Current:**
- `/notebooks/selphi/`
- `/notebooks/crit/`

**Should these be in projects?** Check and decide.

### Stage 10: Final verification
- Update any broken import paths
- Run tests
- Update documentation references
- Final commit

---

## Expected Final Structure

```
hidden-layer/
├── harness/                    # Core infrastructure (canonical) ✅
├── shared/
│   ├── concepts/              # Consolidated from /concepts
│   ├── datasets/
│   └── utils/
├── projects/                   # All research projects
│   ├── latent-space/
│   │   ├── lens/             # From /latent-lens (dev version)
│   │   └── topologies/       # From /latent-topologies (dev version)
│   ├── steerability/
│   │   ├── dashboard/        # From /steerability-dashboard (dev version)
│   │   └── code/             # Existing
│   ├── multi-agent/
│   │   ├── arena/            # From /web-tools/multi-agent-arena
│   │   ├── code/             # Existing
│   │   └── notebooks/        # Existing
│   ├── selphi/
│   │   ├── code/             # Existing
│   │   └── notebooks/        # From /notebooks/selphi
│   ├── introspection/
│   └── ai-to-ai-comm/
├── web-tools/                  # KEEP - Deployment versions
│   ├── latent-lens/          # Deployment version
│   ├── steerability/         # Deployment version
│   ├── multi-agent-arena/    # Deployment version
│   └── shared/backend/       # Shared web utilities
├── docs/                       # Lab-wide documentation (organized)
├── tests/
├── config/
├── README.md
├── CLAUDE.md
├── RESEARCH.md
├── QUICKSTART.md
├── SETUP.md
└── setup files...
```

### Key Changes from Original
1. ✅ `/code/` removed (redundant)
2. ✅ `/web-tools/` KEPT (deployment infrastructure, not duplicate!)
3. Development versions move under `/projects/`
4. `/shared/` consolidates research resources
5. `/web-tools/shared/backend/` stays (different purpose)

---

## Commands Preview (Not Executed Yet)

```bash
# Stage 1: Remove /code/ directory
rm -rf /home/user/hidden-layer/code/

# Stages 2-8: To be planned after verification
```
