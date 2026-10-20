# Development Guidelines

## Python Environment & Tooling
- Use `uv` for environment management
- Launch python programs with `uv run python ...`

## Design Philosophy
- Concise, readable, opinionated design
- Support a few common workflows (not sprawling/configurable)
- **Brevity is paramount**
- Optimized for human learning and readability

## Project Structure
- `volumes/data/` - top-level directory for data
- `volumes/models/` - top-level directory for models
- `planning/` - Write planning documents here for archival. You may not edit old entries.

## Compute Environment
- Local: M4 Apple Air
- Cloud: Modal (NVidia GPUs)
