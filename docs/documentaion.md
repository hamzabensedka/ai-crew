AutoCrew — AI Project Orchestrator
Full Technical Specification

What This System Does
The user provides either:

A new project idea (plain text description)
An existing unfinished project (folder path or uploaded codebase)

The system then:

Analyzes the input
Auto-generates a full CrewAI squad with custom roles tailored to the project
Lets the user review and confirm the squad
Executes the full build from A to Z with one command


Project Structure
autocrew/
├── main.py                    # Entry point — CLI interface
├── config.py                  # Global config (LLM keys, model settings, paths)
├── analyzer/
│   ├── __init__.py
│   ├── idea_analyzer.py       # Analyzes new project ideas
│   ├── codebase_analyzer.py   # Analyzes existing codebases
│   └── project_model.py       # Shared ProjectContext dataclass
├── squad/
│   ├── __init__.py
│   ├── squad_builder.py       # Generates agents from ProjectContext
│   ├── role_templates.py      # Base role definitions (PO, Architect, Dev, etc.)
│   └── squad_model.py         # Squad and Agent dataclasses
├── tasks/
│   ├── __init__.py
│   ├── task_builder.py        # Generates tasks from squad + ProjectContext
│   └── task_model.py          # Task dataclasses
├── crew/
│   ├── __init__.py
│   ├── crew_runner.py         # Assembles and runs the CrewAI Crew
│   └── crew_logger.py         # Real-time output logger
├── tracker/
│   ├── __init__.py
│   ├── progress_tracker.py    # QA agent: compares spec vs codebase
│   └── report_model.py        # ProgressReport dataclass
├── tools/
│   ├── __init__.py
│   ├── file_tools.py          # Read/write files in the project folder
│   ├── git_tools.py           # Git operations (init, commit, branch)
│   ├── search_tools.py        # Web search for docs/packages
│   └── code_tools.py          # Run code, linters, tests
├── output/
│   ├── squads/                # Saved squad configs as JSON
│   ├── reports/               # Progress tracker reports
│   └── logs/                  # Full execution logs
├── docs/
│   └── templates/
│       ├── product.md.j2      # Jinja2 template for product spec
│       ├── architecture.md.j2
│       └── tasks.md.j2
└── tests/
    ├── test_analyzer.py
    ├── test_squad_builder.py
    └── test_tracker.py

Tech Stack
LayerTechnologyLanguagePython 3.11+Agent FrameworkCrewAI 0.28+LLMClaude 3.5 Sonnet via Anthropic SDK (or OpenAI GPT-4o as fallback)CLITyper + RichConfigPydantic Settings + .envTemplatingJinja2File I/OpathlibGitGitPythonTestingpytest

Data Models
ProjectContext — analyzer/project_model.py
pythonfrom dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class ProjectType(str, Enum):
    NEW_IDEA = "new_idea"
    EXISTING_CODE = "existing_code"

class ProjectDomain(str, Enum):
    SAAS = "saas"
    MOBILE_APP = "mobile_app"
    API = "api"
    DATA_PIPELINE = "data_pipeline"
    ECOMMERCE = "ecommerce"
    AI_TOOL = "ai_tool"
    OTHER = "other"

@dataclass
class TechStack:
    frontend: list[str] = field(default_factory=list)    # e.g. ["Next.js", "Tailwind"]
    backend: list[str] = field(default_factory=list)     # e.g. ["FastAPI", "PostgreSQL"]
    devops: list[str] = field(default_factory=list)      # e.g. ["Docker", "GitHub Actions"]
    other: list[str] = field(default_factory=list)

@dataclass
class FeatureItem:
    name: str
    description: str
    status: str  # "done" | "partial" | "not_started"
    priority: str  # "high" | "medium" | "low"

@dataclass
class ProjectContext:
    project_type: ProjectType
    project_name: str
    domain: ProjectDomain
    description: str
    tech_stack: TechStack
    features: list[FeatureItem] = field(default_factory=list)
    existing_files: list[str] = field(default_factory=list)   # relative paths
    missing_parts: list[str] = field(default_factory=list)    # inferred gaps
    special_requirements: list[str] = field(default_factory=list)  # auth, billing, etc.
    raw_idea: Optional[str] = None       # original user input
    codebase_path: Optional[str] = None # path to existing project
Agent Model — squad/squad_model.py
pythonfrom dataclasses import dataclass, field
from enum import Enum

class AgentRole(str, Enum):
    PRODUCT_OWNER = "product_owner"
    ARCHITECT = "architect"
    BACKEND_DEV = "backend_developer"
    FRONTEND_DEV = "frontend_developer"
    FULLSTACK_DEV = "fullstack_developer"
    DEVOPS = "devops_engineer"
    DATA_ENGINEER = "data_engineer"
    AI_ENGINEER = "ai_engineer"
    CODE_REVIEWER = "code_reviewer"
    PROGRESS_TRACKER = "progress_tracker"
    TESTER = "tester"

@dataclass
class AgentConfig:
    role: AgentRole
    name: str                          # display name e.g. "Alex — Backend Dev"
    goal: str                          # specific to THIS project
    backstory: str                     # tailored to the tech stack
    tools: list[str]                   # tool names this agent can use
    can_write_to: list[str]            # folder scopes e.g. ["/backend", "/api"]
    can_read: list[str]                # read-only scopes
    verbose: bool = True
    allow_delegation: bool = False     # only Architect and PO delegate

@dataclass
class Squad:
    project_name: str
    agents: list[AgentConfig]
    execution_order: list[str]         # role names in order
    parallel_groups: list[list[str]]   # roles that run in parallel
    created_at: str
Task Model — tasks/task_model.py
pythonfrom dataclasses import dataclass, field
from typing import Optional

@dataclass
class TaskConfig:
    task_id: str
    title: str
    description: str                   # full instruction for the agent
    assigned_agent_role: str
    depends_on: list[str]              # task_ids this task waits for
    output_format: str                 # "file" | "report" | "code" | "markdown"
    output_path: Optional[str] = None  # where to write the result
    expected_output: str = ""          # description of what success looks like
    context_files: list[str] = field(default_factory=list)  # files to inject as context
Progress Report — tracker/report_model.py
pythonfrom dataclasses import dataclass, field

@dataclass
class FeatureStatus:
    name: str
    status: str          # "done" | "partial" | "missing" | "bug"
    details: str
    files_involved: list[str]

@dataclass
class ProgressReport:
    timestamp: str
    project_name: str
    completion_percentage: float
    done: list[FeatureStatus]
    partial: list[FeatureStatus]
    missing: list[FeatureStatus]
    bugs: list[FeatureStatus]
    next_priorities: list[str]        # ordered list for PO
    raw_summary: str                  # plain text for display

Module Specifications

analyzer/idea_analyzer.py
Purpose: Takes a raw text idea from the user and extracts a full ProjectContext.
How it works:

Calls Claude/GPT with a structured prompt asking it to extract: project name, domain, tech stack (if mentioned or inferred), features list, special requirements
Returns a ProjectContext with project_type = NEW_IDEA
All features start with status = "not_started"

Key function:
pythondef analyze_idea(raw_text: str) -> ProjectContext:
    # 1. Build extraction prompt (see prompt below)
    # 2. Call LLM
    # 3. Parse JSON response
    # 4. Return ProjectContext
LLM Prompt template for idea extraction:
You are a senior software architect.
A user described a project idea. Extract structured information.

User idea:
"""
{raw_text}
"""

Return a JSON object with this exact structure:
{
  "project_name": "...",
  "domain": "saas|mobile_app|api|data_pipeline|ecommerce|ai_tool|other",
  "description": "...",
  "tech_stack": {
    "frontend": [...],
    "backend": [...],
    "devops": [...],
    "other": [...]
  },
  "features": [
    {"name": "...", "description": "...", "priority": "high|medium|low"}
  ],
  "special_requirements": ["auth", "billing", "real-time", ...]
}

If the user didn't specify a tech stack, infer a sensible modern default.
Return only valid JSON. No explanation.

analyzer/codebase_analyzer.py
Purpose: Takes a folder path to an existing project and produces a ProjectContext reflecting what exists vs what is missing.
How it works:

Recursively walks the directory tree (respects .gitignore)
Reads key files: README.md, package.json, pyproject.toml, requirements.txt, existing route files, schema files, env examples
Builds a file map (relative paths + file sizes, skips node_modules, .git, __pycache__, build artifacts)
Sends the file map + key file contents to LLM
LLM returns a ProjectContext with feature statuses (done, partial, not_started) and inferred missing parts

Key functions:
pythondef analyze_codebase(folder_path: str) -> ProjectContext:
    file_map = _build_file_map(folder_path)
    key_contents = _read_key_files(folder_path, file_map)
    return _extract_context_from_llm(file_map, key_contents, folder_path)

def _build_file_map(folder_path: str) -> list[str]:
    # Walk tree, return relative paths
    # Skip: node_modules, .git, __pycache__, dist, build, .next, venv
    # Include: all .py, .ts, .tsx, .js, .json, .md, .sql, .yaml, .env.example

def _read_key_files(folder_path: str, file_map: list[str]) -> dict[str, str]:
    # Read and return content of: README, package.json, main config files
    # Truncate files > 4000 chars, keep first and last 2000

def _extract_context_from_llm(file_map, key_contents, folder_path) -> ProjectContext:
    # Build prompt, call LLM, parse response
LLM Prompt for codebase analysis:
You are a senior software architect reviewing an existing project.

File tree:
{file_map}

Key file contents:
{key_contents}

Analyze this codebase and return a JSON object:
{
  "project_name": "...",
  "domain": "...",
  "description": "...",
  "tech_stack": { ... },
  "features": [
    {
      "name": "...",
      "description": "...",
      "status": "done|partial|not_started",
      "priority": "high|medium|low",
      "evidence": "which files suggest this status"
    }
  ],
  "missing_parts": ["list of clearly absent features or components"],
  "special_requirements": [...]
}

Be conservative: mark as "done" only if you see real implementation, not just a stub or empty file.
Return only valid JSON.

squad/squad_builder.py
Purpose: Takes a ProjectContext and generates the right set of AgentConfig objects for this specific project.
Core Logic:
The builder decides which roles are needed based on the ProjectContext:

Always include: Product Owner, Architect, Code Reviewer, Progress Tracker
Include Frontend Dev if tech_stack.frontend is not empty
Include Backend Dev if tech_stack.backend is not empty
Include Fullstack Dev if both are small/simple (instead of separate)
Include DevOps if tech_stack.devops is not empty or if Docker/CI is needed
Include Data Engineer if domain is data_pipeline or features mention DB migrations, ETL, etc.
Include AI Engineer if domain is ai_tool or features mention LLM, embeddings, fine-tuning
Include Tester if there are more than 5 features

Each agent's goal, backstory, and tools are dynamically generated to be specific to the project's tech stack and features, not generic.
Key function:
pythondef build_squad(context: ProjectContext) -> Squad:
    roles_needed = _determine_roles(context)
    agents = [_build_agent(role, context) for role in roles_needed]
    order, parallel_groups = _determine_execution_plan(roles_needed, context)
    return Squad(
        project_name=context.project_name,
        agents=agents,
        execution_order=order,
        parallel_groups=parallel_groups,
        created_at=datetime.now().isoformat()
    )
Execution Order Logic:
Phase 1 (sequential):
  → Product Owner (creates spec)
  → Architect (designs structure, creates folder scaffold)

Phase 2 (parallel):
  → Backend Dev + Frontend Dev run at the same time
  OR
  → All devs in parallel if more than 2 dev roles

Phase 3 (sequential):
  → Tester (if included)
  → Code Reviewer
  → Progress Tracker (reports back to Product Owner)

Phase 4:
  → Product Owner reviews tracker report and updates priorities

squad/role_templates.py
Purpose: Stores base role definitions that squad_builder.py uses as a starting point, then customizes per project.
Each role template defines:

Default goal template (with {project_name}, {tech_stack} placeholders)
Default backstory template
Default allowed tools list
Default read/write scopes

Role templates to define:
pythonROLE_TEMPLATES = {
    AgentRole.PRODUCT_OWNER: {
        "goal_template": "Define complete feature specifications and acceptance criteria for {project_name}. Ensure all user needs are captured and prioritized.",
        "backstory_template": "You are a senior product manager with 10 years building {domain} products. You think from the user's perspective and translate business goals into clear developer tasks.",
        "tools": ["file_read", "file_write", "web_search"],
        "can_write_to": ["/docs"],
        "can_read": ["*"]
    },
    AgentRole.ARCHITECT: {
        "goal_template": "Design the complete system architecture for {project_name} using {tech_stack}. Create the folder structure, define service boundaries, and produce the architecture doc.",
        "backstory_template": "You are a staff software engineer who has designed {domain} systems at scale. You prioritize clean architecture, clear separation of concerns, and maintainability.",
        "tools": ["file_read", "file_write", "file_create_folder"],
        "can_write_to": ["/docs", "/"],  # can scaffold root structure
        "can_read": ["*"]
    },
    AgentRole.BACKEND_DEV: {
        "goal_template": "Implement all backend features for {project_name} as defined in the spec. Stack: {backend_stack}.",
        "backstory_template": "You are a senior backend engineer expert in {backend_stack}. You write clean, tested, production-ready code with proper error handling and security practices.",
        "tools": ["file_read", "file_write", "run_command", "web_search"],
        "can_write_to": ["/backend", "/api", "/server", "/src/api"],
        "can_read": ["/docs", "/"]
    },
    AgentRole.FRONTEND_DEV: {
        "goal_template": "Build all frontend UI for {project_name} as defined in the spec. Stack: {frontend_stack}.",
        "backstory_template": "You are a senior frontend engineer expert in {frontend_stack}. You build accessible, responsive, well-structured UIs with good UX.",
        "tools": ["file_read", "file_write", "run_command", "web_search"],
        "can_write_to": ["/frontend", "/app", "/src/app", "/components", "/pages"],
        "can_read": ["/docs", "/"]
    },
    AgentRole.CODE_REVIEWER: {
        "goal_template": "Review all code produced for {project_name} for bugs, security issues, bad patterns, and maintainability problems.",
        "backstory_template": "You are a strict staff engineer and security-conscious code reviewer. You catch issues before they reach production.",
        "tools": ["file_read", "run_command"],
        "can_write_to": ["/output/reports"],
        "can_read": ["*"]
    },
    AgentRole.PROGRESS_TRACKER: {
        "goal_template": "Scan the entire {project_name} codebase, compare it against the product spec, and produce a detailed completion report for the Product Owner.",
        "backstory_template": "You are an engineering manager and QA lead. You know how to assess real implementation versus spec promises. You report honestly and with precision.",
        "tools": ["file_read", "run_command"],
        "can_write_to": ["/output/reports"],
        "can_read": ["*"]
    },
    # ... DevOps, DataEngineer, AIEngineer, Tester follow the same pattern
}

tasks/task_builder.py
Purpose: Takes a Squad + ProjectContext and generates all TaskConfig objects, in dependency order.
Key logic:

Product Owner's first task always generates /docs/product.md from the Jinja2 template
Architect's first task reads product.md and generates /docs/architecture.md + creates the folder scaffold
Each dev agent gets one task per major feature group (not one task per feature — grouped logically)
Reviewer gets one task: read all modified files, write review report
Tracker gets one task: compare spec vs codebase, write progress report
All tasks have expected_output filled in so CrewAI knows what done looks like

Task generation is LLM-assisted:
After the squad is built, the system calls the LLM once more with the ProjectContext + squad definition and asks it to generate the full task list as JSON. This ensures the tasks are contextual, not generic.
pythondef build_tasks(squad: Squad, context: ProjectContext) -> list[TaskConfig]:
    raw = _call_llm_for_tasks(squad, context)
    tasks = _parse_task_list(raw)
    tasks = _inject_standard_tasks(tasks, squad, context)  # reviewer + tracker always added
    tasks = _resolve_dependencies(tasks)
    return tasks

tracker/progress_tracker.py
Purpose: The Progress Tracker agent — runs as a CrewAI agent with read-only file access. Compares the product spec against actual code files and produces a ProgressReport.
What it checks:

Does each feature from product.md have a corresponding implementation?
Are API endpoints defined and implemented?
Are DB schemas created and migrated?
Are UI pages/components built for each feature?
Are there obvious bugs (empty try/catch, TODO comments, stub functions)?
What percentage of features are complete?

Output: A ProgressReport object that is:

Saved to /output/reports/progress_{timestamp}.json
Converted to a readable markdown summary at /output/reports/progress_{timestamp}.md
Fed back to the Product Owner agent as context for the next iteration


crew/crew_runner.py
Purpose: Assembles the CrewAI Crew from the built squad and tasks, then runs it.
Key responsibilities:

Converts AgentConfig → crewai.Agent objects
Converts TaskConfig → crewai.Task objects
Handles parallel execution using CrewAI's Process.hierarchical or custom grouping
Streams real-time logs via crew_logger.py
Catches errors per-agent and retries up to 2 times before marking task failed
Saves full output to /output/logs/run_{timestamp}.log

pythondef run_crew(squad: Squad, tasks: list[TaskConfig], context: ProjectContext):
    agents = [_build_crewai_agent(a) for a in squad.agents]
    crewai_tasks = [_build_crewai_task(t, agents) for t in tasks]
    
    crew = Crew(
        agents=agents,
        tasks=crewai_tasks,
        process=Process.sequential,  # override for parallel groups
        verbose=True,
        memory=True,                 # agents share memory across tasks
    )
    
    result = crew.kickoff()
    return result
Parallel execution: CrewAI doesn't natively support arbitrary parallel groups. Implement parallel groups by running multiple Crew instances concurrently using asyncio.gather() for Phase 2 (dev agents). Each parallel crew gets the shared docs as context injection.

tools/ — Custom Tools
All tools are wrapped as CrewAI BaseTool subclasses.
file_tools.py:

FileReadTool(path) — reads a file, respects agent's can_read scope
FileWriteTool(path, content) — writes a file, enforces can_write_to scope, raises PermissionError if out of scope
ListDirectoryTool(path) — lists directory contents
CreateFolderTool(path) — creates folder structure

git_tools.py:

GitInitTool() — initializes git repo if not exists
GitCommitTool(message) — stages all and commits with agent name in message
GitBranchTool(branch_name) — creates and checks out a branch

code_tools.py:

RunCommandTool(command) — runs shell command in project directory, returns stdout + stderr, has timeout of 30s
RunTestsTool() — runs the project's test command (detected from package.json or pyproject.toml)
LintTool() — runs the project's linter

search_tools.py:

WebSearchTool(query) — searches for documentation, packages, solutions

Scope enforcement is a critical safety layer. Every write tool checks the agent's allowed scopes before writing. This prevents the backend agent from accidentally modifying frontend files and vice versa.

main.py — CLI Interface
Built with Typer + Rich for a clean terminal experience.
Commands:
autocrew new "my project idea"

Analyzes idea
Shows extracted ProjectContext for confirmation
Builds squad, shows squad summary for confirmation
On confirmation: saves squad to /output/squads/ and shows "Ready. Run autocrew build to start."

autocrew analyze ./my-existing-project

Analyzes codebase
Shows ProjectContext with feature statuses (color-coded: green=done, yellow=partial, red=missing)
Builds squad tailored to what's missing
On confirmation: saves squad

autocrew build

Loads the last saved squad and context
Runs the full crew
Shows real-time progress (Rich live display: agent name, current task, status)

autocrew build --squad ./output/squads/my_squad.json

Loads a specific saved squad

autocrew track

Runs ONLY the Progress Tracker agent on the current codebase
Outputs a fresh progress report without running the full crew

autocrew status

Shows the last progress report in a readable format


config.py
pythonfrom pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    default_llm: str = "claude-3-5-sonnet-20241022"  # or "gpt-4o"
    fallback_llm: str = "gpt-4o"
    
    # Paths
    output_dir: str = "./output"
    squads_dir: str = "./output/squads"
    reports_dir: str = "./output/reports"
    logs_dir: str = "./output/logs"
    
    # Execution
    max_retries_per_task: int = 2
    task_timeout_seconds: int = 300
    parallel_execution: bool = True
    
    # Safety
    enforce_scope: bool = True        # enforce agent write scopes
    require_confirmation: bool = True # ask user to confirm squad before build
    
    class Config:
        env_file = ".env"

settings = Settings()

.env Template
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
DEFAULT_LLM=claude-3-5-sonnet-20241022
PARALLEL_EXECUTION=true
ENFORCE_SCOPE=true

Dependencies — pyproject.toml
toml[tool.poetry.dependencies]
python = "^3.11"
crewai = "^0.28"
crewai-tools = "^0.1"
anthropic = "^0.25"
openai = "^1.30"
typer = {extras = ["all"], version = "^0.12"}
rich = "^13.7"
pydantic = "^2.7"
pydantic-settings = "^2.2"
jinja2 = "^3.1"
gitpython = "^3.1"
pathspec = "^0.12"   # for .gitignore parsing
python-dotenv = "^1.0"

[tool.poetry.dev-dependencies]
pytest = "^8.0"
pytest-asyncio = "^0.23"

Key Implementation Rules

All LLM calls return JSON. Every prompt that extracts structured data ends with "Return only valid JSON. No explanation." Parse with json.loads() inside a try/except. On failure, retry once with a stricter prompt.
Scope enforcement is mandatory. Before any file write, check the agent's can_write_to list. Raise a descriptive error if violated. Log the violation. Do not silently skip.
Every agent gets injected context. When a task is built, the relevant docs (product.md, architecture.md) are read from disk and injected into the task description as context. Agents do not rely on memory alone.
Parallel groups use asyncio. Phase 2 dev agents run via asyncio.gather(), each in their own Crew instance. They share read access to /docs but write only to their scoped folders.
Progress Tracker always runs last. It is always the final agent in execution order, before the Product Owner's review step. It is always read-only.
Squads are serializable. Squad and all child objects must be JSON-serializable (use dataclasses.asdict() or Pydantic models). A saved squad file is a complete snapshot that can recreate the full crew without re-analyzing.
Rich live display during execution. Use rich.live.Live with a table showing each agent, their current task, and status (waiting / running / done / failed) updating in real time.
Git commits after each agent completes. After each agent finishes their task, auto-commit with message: [autocrew] {agent_role}: {task_title}. This creates a recoverable history.


First Sprint — What to Build First
Build in this exact order:

config.py + .env setup
analyzer/project_model.py (dataclasses only, no logic)
analyzer/idea_analyzer.py (LLM call + JSON parsing)
analyzer/codebase_analyzer.py (file walking + LLM call)
squad/squad_model.py (dataclasses)
squad/role_templates.py (static dict)
squad/squad_builder.py (role selection + agent config generation)
main.py with just autocrew new and autocrew analyze commands (no build yet)
Confirm the analysis → squad flow works end to end
Then build tasks/, crew/, tools/, tracker/ in that order