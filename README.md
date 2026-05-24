# AutoCrew

AI Project Orchestrator — turns project ideas or existing codebases into a scoped CrewAI squad and builds them from A to Z.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
copy .env.example .env
```

**No API keys required** if you use the [Cursor Composer workflow](docs/cursor-workflow.md) (Option A).

## Cursor Composer workflow (Option A — recommended)

Use Cursor as the brain; AutoCrew handles orchestration.

| Step | What you do |
|------|-------------|
| 1 | Describe your idea in Cursor chat **or** run `autocrew scout ./my-project -o output/scout.json` |
| 2 | Ask Cursor to create `context.json` (see `docs/templates/context.example.json`) |
| 3 | `autocrew import-context context.json` |
| 4 | `autocrew plan` |
| 5 | `autocrew debate --root ./my-project` |
| 6 | `autocrew build --yes` → `autocrew track` |

Full guide: **[docs/cursor-workflow.md](docs/cursor-workflow.md)**

## Commands

| Command | API keys? | Description |
|---------|-----------|-------------|
| `autocrew scout ./path` | No | Export codebase snapshot for Cursor |
| `autocrew import-context ctx.json` | No | Import Cursor-generated context + build squad |
| `autocrew plan` | No | Generate product, architecture, task docs |
| `autocrew debate` | No | Squad critiques plan until consensus |
| `autocrew build` | No | Run the crew build |
| `autocrew track` | No | Progress report |
| `autocrew status` | No | Show latest report |
| `autocrew new "idea"` | Yes | Auto-analyze idea via LLM API |
| `autocrew analyze ./path` | Yes | Auto-analyze codebase via LLM API |

## Direct API workflow (Option B)

Add keys to `.env` for fully automated analysis:

```env
OPENAI_API_KEY=sk-...
DEFAULT_LLM=gpt-4o
```

Then: `autocrew new "Build a SaaS CRM"` → `plan` → `debate` → `build` → `track`

## Squad debate

All agents critique the plan in rounds until consensus (or max rounds):

```powershell
autocrew debate --root C:\planity --rounds 3 --yes
```

Output: `output/debate/planity_clone/round-1/`, `round-2/`, `final_plan.md`, plus implementation tasks for `build`.

## Configuration

See `.env.example`:

- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — optional (only for `new` / `analyze`)
- `ENFORCE_SCOPE` — agent write scope enforcement
- `PARALLEL_EXECUTION` — parallel dev agent phase

## Project Structure

```
autocrew/
├── analyzer/     # Idea and codebase analysis
├── squad/        # Agent squad generation
├── tasks/        # Task plan generation
├── crew/         # CrewAI execution
├── tools/        # Scoped file, git, command tools
├── tracker/      # Progress tracking
└── main.py       # CLI entry point
```

## Testing

```bash
pytest
```

## License

MIT
