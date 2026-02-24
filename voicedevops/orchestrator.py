"""
Agent router, self-heal loop, multi-agent chaining, and auto-setup flow.

Responsibilities:
  1. Classify user intent → assign to correct agent(s)
  2. Inject repo context into every generate_hybrid call so all commands are repo-aware
  3. Route tool calls: regex text fallback first, then generate_hybrid (on-device → cloud)
  4. Execute tools via executor.py
  5. Self-heal: catch errors → diagnose via cloud → apply fix → retry
  6. Chain multi-agent sequences (Builder → Runner → Tester)
  7. Auto-setup: full discover → plan → execute → health-check → docs flow
"""

import sys
import os
import re
import json
import time
from contextlib import contextmanager
from io import StringIO

@contextmanager
def _suppress_stdout():
    """
    Mute Python-level sys.stdout during model calls.
    Raw fd 1 is already permanently redirected to /dev/null by cli.py at startup,
    so C-level cactus writes are silenced regardless; we only need to swap the
    Python layer here.
    """
    old_py = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))

from tools import (
    EXPLORER_TOOLS, BUILDER_TOOLS, RUNNER_TOOLS, TESTER_TOOLS, DOCUMENTATION_TOOLS,
    TOOL_PLAN_SETUP_COMMANDS, ALL_TOOLS, TOOL_MAP, strip_routing_metadata,
    TOOL_DIAGNOSE_ERROR,
)
from executor import execute_tool, get_workspace, set_workspace, exec_clone_repo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import generate_hybrid, generate_cloud


# ─── Agent definitions ────────────────────────────────────────────────────────

AGENTS = {
    "explorer": {
        "tools": EXPLORER_TOOLS,
        "label": "Explorer",
        "keywords": ["list", "show", "read", "find", "tree", "disk", "view", "what files", "ls", "cat", "grep"],
    },
    "builder": {
        "tools": BUILDER_TOOLS,
        "label": "Builder",
        "keywords": ["create", "write", "generate", "dockerfile", "docker-compose", "compose", "install", "edit", "delete", "make file", "setup file"],
    },
    "runner": {
        "tools": RUNNER_TOOLS,
        "label": "Runner",
        "keywords": ["run", "start", "stop", "execute", "docker", "port", "logs", "command", "launch", "bring up"],
    },
    "tester": {
        "tools": TESTER_TOOLS,
        "label": "Tester",
        "keywords": ["check", "test", "health", "validate", "verify", "diagnose", "is it running", "working", "ping"],
    },
    "documentation": {
        "tools": DOCUMENTATION_TOOLS,
        "label": "Documentation",
        "keywords": ["docs", "instructions", "how to run", "readme", "guide", "how do i", "how to use"],
    },
}

# "auto_setup" is a sentinel — not a real agent entry, handled directly in process()
# "clone_setup" is a sentinel — clone repo first, then auto_setup
_MULTI_AGENT_PATTERNS = [
    # Clone + set up: "clone this repo and set it up", "clone and run", etc.
    (re.compile(
        r"clone\s+.+\s+(?:and\s+)?(?:set\s*up|setup|run|install|build|start)"
        r"|clone\s+.*(?:repo|repository|project)"
        r"|git\s+clone"
        r"|clone\s+(?:this|it|a\s+repo)",
        re.I,
    ), ["clone_setup"]),
    # Generic repo-level setup (no specific service named) — must be checked FIRST
    (re.compile(
        # "set up this/the app/repo/project [for me]"
        r"set\s*up\s+(?:for\s+)?(?:this|the)\s+(?:repo|app|project|codebase|code)"
        # "set this/the app/repo/project up"
        r"|set\s+(?:this|the)\s+(?:repo|app|project)\s+up"
        # "set this/the app/repo/project [for me]" (no "up" keyword)
        r"|set\s+(?:this|the)\s+(?:app|repo|project|codebase|code)\b"
        # "set me up", "set it up [for me]"
        r"|set\s+(?:me|it)\s+up\b"
        # "get this/the app running/started/set up"
        r"|get\s+(?:this|the)\s+(?:app|repo|project)\s+(?:set\s+up|running|started)"
        # "make this/the app work/run/start"
        r"|make\s+(?:this|the)\s+(?:app|repo|project)\s+(?:work|run|start)"
        # "install everything / all deps"
        r"|install\s+(?:everything|all\s+(?:the\s+)?(?:deps|dependencies))"
        # "bootstrap [this/the app]"
        r"|bootstrap\b"
        # "initialize/init this/the app"
        r"|init(?:ializ[ei])?\s+(?:this|the)\s+(?:app|repo|project)"
        # bare keyword
        r"|auto[- ]?setup",
        re.I,
    ), ["auto_setup"]),
    # Specific Docker-service setup → Builder → Runner → Tester
    (re.compile(r"set up|setup|deploy|provision", re.I), ["builder", "runner", "tester"]),
    (re.compile(r"install and run|install.*then run", re.I), ["builder", "runner", "tester"]),
]

# Regex to extract a git URL from user text
_GIT_URL_RE = re.compile(
    r"((?:https?://|git@)[\w.:/-]+?(?:\.git)?)\b"
)

# Question patterns — triggers Q&A mode instead of tool-calling
_QUESTION_RE = re.compile(
    r"^(?:what|how|why|where|which|who|when|can|does|is\s+there|are\s+there|"
    r"tell\s+me|explain|describe|show\s+me\s+(?:the|how)|"
    r"what's|what\s+is|what\s+are|what\s+does|how\s+does|how\s+do\s+i|"
    r"do\s+i\s+need|should\s+i|could\s+you)\b",
    re.I,
)

# Commands that look like questions but should still be tool calls
_NOT_QUESTION_RE = re.compile(
    r"^(?:show\s+(?:tree|files|logs|disk)|check\s+(?:port|health|disk)|"
    r"run\s|execute\s|install\s|create\s|delete\s|edit\s|start\s|stop\s|"
    r"set\s*up|setup|clone\s|deploy\s|build\s)",
    re.I,
)


def classify_agents(user_text: str) -> list:
    """Determine which agent(s) should handle this request. Returns list in execution order."""
    text_lower = user_text.lower()

    for pattern, agents in _MULTI_AGENT_PATTERNS:
        if pattern.search(text_lower):
            return agents

    # Detect questions (but not imperative commands that start with question words)
    if _QUESTION_RE.search(text_lower) and not _NOT_QUESTION_RE.search(text_lower):
        return ["chat"]

    scores = {}
    for name, agent in AGENTS.items():
        score = sum(1 for kw in agent["keywords"] if kw in text_lower)
        if score > 0:
            scores[name] = score

    if not scores:
        return ["explorer"]

    best = max(scores, key=scores.get)
    return [best]


# ─── Repo context cache ───────────────────────────────────────────────────────

_repo_summary_cache: dict = {}


def _get_repo_summary(ws) -> str:
    """
    Fast, cached scan of workspace root.
    Returns a one-line project description injected into every generate_hybrid call.
    Runs once per workspace path per session.
    """
    key = str(ws)
    if key in _repo_summary_cache:
        return _repo_summary_cache[key]

    try:
        root_files = sorted(f.name for f in ws.iterdir() if not f.name.startswith("."))
    except Exception:
        root_files = []

    type_map = [
        ("package.json",     "Node.js"),
        ("requirements.txt", "Python"),
        ("pyproject.toml",   "Python"),
        ("Pipfile",          "Python"),
        ("go.mod",           "Go"),
        ("Cargo.toml",       "Rust"),
        ("pom.xml",          "Java/Maven"),
        ("build.gradle",     "Java/Gradle"),
    ]
    detected = "unknown"
    for filename, label in type_map:
        if (ws / filename).exists():
            detected = label
            break

    extra = ""
    manifest = ws / "package.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            scripts = list(data.get("scripts", {}).keys())
            if scripts:
                extra = f", scripts: {', '.join(scripts[:6])}"
        except Exception:
            pass

    summary = f"Project type: {detected}. Root files: {', '.join(root_files[:15])}{extra}."
    _repo_summary_cache[key] = summary
    return summary


# ─── DevOps text fallback (regex-based, no LLM needed) ───────────────────────

def _devops_text_fallback(user_text: str, tool_names: set) -> list:
    """
    Pattern-match natural-language DevOps commands without calling any model.
    Covers common operations and demo steps. Returns tool calls or empty list.
    """
    text = user_text.lower()
    calls = []

    # Canonical "current directory" tokens — also includes workspace name so
    # "show me files in fastapi-todo" works when the user is already inside it.
    _ws = get_workspace()
    _cwd_tokens = {"the", "a", "an", "workspace", "workdir", "current", "here", "dir",
                   _ws.name.lower(), str(_ws).lower()}

    # Explorer
    if "show_tree" in tool_names and re.search(r"\b(tree|show|list|files|workspace|ls)\b", text):
        dir_m = re.search(r"\bin\s+(?:the\s+|a\s+|an\s+)?([\w./\\-]+)\s*(?:directory|folder|workspace)?", user_text, re.I)
        raw_dir = dir_m.group(1) if dir_m else "."
        directory = "." if raw_dir.lower() in _cwd_tokens else raw_dir
        calls.append({"name": "show_tree", "arguments": {"directory": directory}})
        return calls

    if "list_files" in tool_names and re.search(r"\b(list|show|files|workspace|ls|what\s+files)\b", text):
        dir_m = re.search(r"\bin\s+(?:the\s+|a\s+|an\s+)?([\w./\\-]+)\s*(?:directory|folder|workspace)?", user_text, re.I)
        raw_dir = dir_m.group(1) if dir_m else "."
        directory = "." if raw_dir.lower() in _cwd_tokens else raw_dir
        calls.append({"name": "list_files", "arguments": {"directory": directory}})
        return calls

    if "read_file" in tool_names and re.search(r"\b(read|show|cat|view|open|display)\b", text) and not re.search(r"\blog", text):
        path_m = re.search(r"\b([\w./\\-]+\.\w+)\b", user_text)
        filepath = path_m.group(1) if path_m else "."
        calls.append({"name": "read_file", "arguments": {"filepath": filepath}})
        return calls

    # Builder
    if "create_docker_compose" in tool_names and re.search(
        r"\b(create|write|generate|make|build)\b.*\b(docker.?compose|compose)\b"
        r"|\b(docker.?compose|compose)\b.*\b(for|with)\b",
        text,
    ):
        svc_m = re.search(r"\bfor\s+([\w\s,]+?)(?:\s+(?:and|with|on|compose)\b|\s*$)", text)
        services = svc_m.group(1).strip() if svc_m else "postgres"
        calls.append({"name": "create_docker_compose", "arguments": {"services": services}})
        return calls

    if "create_dockerfile" in tool_names and re.search(r"\b(create|write|generate|make)\b.*\bdockerfile\b", text):
        svc_m = re.search(r"\bfor\s+([\w-]+)", text)
        service = svc_m.group(1) if svc_m else "app"
        calls.append({"name": "create_dockerfile", "arguments": {"service_name": service, "requirements": service}})
        return calls

    if "edit_file" in tool_names and re.search(r"\b(edit|modify|update|change)\b", text):
        path_m = re.search(r"\b([\w./\\-]+\.\w+)\b", user_text)
        filepath = path_m.group(1) if path_m else "file"
        calls.append({"name": "edit_file", "arguments": {"filepath": filepath, "old_text": "", "new_text": ""}})
        return calls

    if "delete_file" in tool_names and re.search(r"\b(delete|remove|rm)\b", text):
        path_m = re.search(r"\b([\w./\\-]+\.\w+)\b", user_text)
        filepath = path_m.group(1) if path_m else "file"
        calls.append({"name": "delete_file", "arguments": {"filepath": filepath}})
        return calls

    # Tester
    if "validate_config" in tool_names and re.search(r"\b(validate|check|lint|verify)\b.*\b(config|yaml|yml|json)\b", text):
        path_m = re.search(r"\b([\w./\\-]+\.(?:yml|yaml|json|toml))\b", user_text)
        filepath = path_m.group(1) if path_m else "docker-compose.yml"
        fmt_m = re.search(r"\b(yaml|yml|json|toml)\b", text)
        fmt = "yaml" if not fmt_m or fmt_m.group(1) in ("yaml", "yml") else fmt_m.group(1)
        calls.append({"name": "validate_config", "arguments": {"filepath": filepath, "format": fmt}})
        return calls

    if "check_health" in tool_names and re.search(r"\b(health|healthy|reachable|alive)\b", text):
        port_m = re.search(r"\bport\s+(\d+)\b", text)
        port_str = port_m.group(1) if port_m else "80"
        _non_http = {"5432", "5433", "3306", "27017", "6379", "5672", "9200", "9042", "6443"}
        if port_str in _non_http and "check_port" in tool_names:
            calls.append({"name": "check_port", "arguments": {"port": int(port_str)}})
            return calls
        host_m = re.search(r"\bon\s+((?!port\b)[\w.-]+)\b", user_text, re.I)
        host = host_m.group(1) if host_m else "localhost"
        url = f"http://{host}:{port_str}"
        calls.append({"name": "check_health", "arguments": {"url": url}})
        return calls

    if "check_port" in tool_names and re.search(r"\bport\s+(\d+)\b", text):
        port_m = re.search(r"\bport\s+(\d+)\b", text)
        calls.append({"name": "check_port", "arguments": {"port": int(port_m.group(1))}})
        return calls

    if "run_test" in tool_names and re.search(r"\b(test|run\s+tests?|pytest|jest)\b", text):
        calls.append({"name": "run_test", "arguments": {"test_command": "pytest"}})
        return calls

    # Runner
    if "start_service" in tool_names and re.search(r"\b(start|bring\s+up|spin\s+up|launch)\b", text):
        path_m = re.search(r"\b([\w./\\-]*docker.?compose[\w./\\-]*\.yml)\b", user_text, re.I)
        compose = path_m.group(1) if path_m else "docker-compose.yml"
        calls.append({"name": "start_service", "arguments": {"compose_file": compose}})
        return calls

    if "stop_service" in tool_names and re.search(r"\b(stop|shut\s+down|bring\s+down|down|kill)\b", text):
        path_m = re.search(r"\b([\w./\\-]*docker.?compose[\w./\\-]*\.yml)\b", user_text, re.I)
        compose = path_m.group(1) if path_m else "docker-compose.yml"
        calls.append({"name": "stop_service", "arguments": {"compose_file": compose}})
        return calls

    if "read_logs" in tool_names and re.search(r"\b(log|logs)\b", text):
        svc_m = re.search(r"\bof\s+([\w-]+)\s+log", text) or re.search(r"\b([\w-]+)\s+log", text)
        service = svc_m.group(1) if svc_m else "postgres"
        lines_m = re.search(r"\b(\d+)\s+line", text)
        lines = int(lines_m.group(1)) if lines_m else 50
        calls.append({"name": "read_logs", "arguments": {"service_name": service, "lines": lines}})
        return calls

    if "run_command" in tool_names and re.search(r"\b(run|execute|exec)\b", text):
        cmd_m = re.search(r"\b(?:run|execute|exec)\s+['\"]?([^'\"]+)['\"]?", user_text, re.I)
        cmd = cmd_m.group(1).strip() if cmd_m else "echo hello"
        calls.append({"name": "run_command", "arguments": {"command": cmd}})
        return calls

    if "check_disk_space" in tool_names and re.search(r"\b(disk|space|storage|free\s+space)\b", text):
        calls.append({"name": "check_disk_space", "arguments": {"path": "."}})
        return calls

    return calls


# ─── Cloud generation helpers ─────────────────────────────────────────────────

def _strip_markdown_fence(text: str) -> str:
    """Remove markdown code-block fences that LLMs sometimes wrap output in."""
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _generate_text_via_gemini(prompt: str) -> str:
    """
    Call Gemini directly for plain-text generation (no function-calling schema).
    Used by _handle_cloud_generation for Dockerfile, docker-compose, run instructions,
    and setup-command planning — cases where we want free-form text, not a structured call.
    Returns raw text, or "" on any error.
    """
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Export it before starting the CLI:\n"
            "  export GEMINI_API_KEY=your_key_here"
        )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
    )
    text = ""
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "text"):
                text += part.text
    return text.strip()


def _diagnose_via_cloud(tool_name: str, arguments: dict, error: str, context: str = "") -> str:
    """Send error to Gemini for diagnosis. Returns human-readable fix suggestion."""
    prompt = (
        f"This DevOps command failed:\n"
        f"Tool: {tool_name}\n"
        f"Arguments: {json.dumps(arguments, indent=2)}\n"
        f"Error: {error}\n"
        f"Context: {context}\n\n"
        f"Diagnose the error and provide the EXACT fix. "
        f"Be specific: which file to edit, which line, what to change."
    )
    tools_model = strip_routing_metadata([TOOL_DIAGNOSE_ERROR])
    try:
        with _suppress_stdout():
            result = generate_cloud(
                messages=[{"role": "user", "content": prompt}],
                tools=tools_model,
            )
        calls = result.get("function_calls", [])
        if calls and calls[0]["name"] == "diagnose_error":
            return calls[0].get("arguments", {}).get("error_message", "No diagnosis available")
    except Exception:
        pass
    return f"Cloud diagnosis requested for: {error[:200]}"


def _handle_cloud_generation(tool_name: str, arguments: dict) -> dict:
    """
    For tools that require AI content generation (Dockerfile, docker-compose,
    setup plans, run instructions), call Gemini and pass generated content to executor.
    """
    if tool_name == "create_dockerfile":
        service = arguments.get("service_name", "app")
        reqs = arguments.get("requirements", "")
        prompt = (
            f"Write a production-ready Dockerfile for a {service} service. "
            f"Requirements: {reqs}. "
            f"Output ONLY the Dockerfile content, no explanation."
        )
        try:
            content = _strip_markdown_fence(_generate_text_via_gemini(prompt))
        except Exception:
            content = ""
        if not content:
            content = f"FROM {service}:latest\nWORKDIR /app\nEXPOSE 8080\nCMD [\"{service}\"]\n"
        arguments["content"] = content
        return execute_tool("create_dockerfile", arguments)

    if tool_name == "create_docker_compose":
        services = arguments.get("services", "")
        prompt = (
            f"Write a docker-compose.yml for these services: {services}. "
            f"Include proper networking, volumes, and environment variables. "
            f"Output ONLY the docker-compose.yml content, no explanation."
        )
        try:
            content = _strip_markdown_fence(_generate_text_via_gemini(prompt))
        except Exception:
            content = ""
        if not content:
            content = _default_docker_compose(services)
        arguments["content"] = content
        return execute_tool("create_docker_compose", arguments)

    if tool_name == "plan_setup_commands":
        context = arguments.get("context", "")
        prompt = (
            "You are a DevOps setup assistant. Analyze this repository and output "
            "an ordered list of shell commands to set it up FOR LOCAL DEVELOPMENT.\n\n"
            f"Repo context:\n{context}\n\n"
            "Rules:\n"
            "- Output ONLY shell commands, one per line, no explanations or markdown\n"
            "- Start with dependency installation (npm install / pip install -r requirements.txt / etc.)\n"
            "- If .env.example exists, add: cp .env.example .env\n"
            "- Include a build step only if needed (npm run build, go build, etc.)\n"
            "- End with the start command (uvicorn main:app --reload, npm start, etc.)\n"
            "- PREFER running the app directly (python, node, uvicorn, etc.) over Docker\n"
            "- If a docker-compose.yml is marked PRODUCTION-ONLY, IGNORE it completely\n"
            "- Only use docker-compose if it is clearly a local dev setup (no external networks, no Traefik/deploy labels)\n"
            "- If requirements.txt or package.json exists, always install deps directly even if Dockerfile exists\n"
            "- Do NOT include sudo or system-level package installs\n"
            "- Output 2 to 6 commands maximum"
        )
        # Let exceptions propagate so run_auto_setup can surface the real error message
        raw = _generate_text_via_gemini(prompt)
        commands = _parse_setup_commands(raw)
        return {
            "success": bool(commands),
            "output": raw,
            "commands": commands,
            "error": "" if commands else "No commands generated",
        }

    if tool_name == "generate_run_instructions":
        context = arguments.get("context", "")
        docker_present = arguments.get("docker_present", "False").lower() == "true"

        docker_rule = (
            "- Docker / docker-compose IS present. "
            "Containerise only the backend and database services if identifiable. "
            "Do NOT containerise the frontend — provide native run instructions for it."
            if docker_present else
            "- There are NO Docker or docker-compose files in this repo. "
            "Do NOT mention Docker or containerisation at all."
        )

        prompt = (
            "You are a developer onboarding assistant. "
            "Using the repo context below, write a concise '## How to Run' guide.\n\n"
            f"{context}\n\n"
            "Rules:\n"
            "- Identify the app type (frontend, backend, fullstack, CLI, etc.).\n"
            "- List prerequisites (runtime version, package manager, etc.).\n"
            "- Show how to install dependencies.\n"
            "- Show how to run the app locally in development mode.\n"
            "- Show how to run tests if a test command is evident.\n"
            "- List key URLs / ports the app is accessible on.\n"
            "- Include important environment variables from .env.example if present.\n"
            f"{docker_rule}\n"
            "Format as short numbered steps with exact shell commands. Be concise."
        )
        instructions = _generate_text_via_gemini(prompt)
        if not instructions:
            instructions = (
                "## How to Run\n\n"
                "1. Install dependencies (see README for package manager)\n"
                "2. Copy `.env.example` to `.env` and fill in values\n"
                "3. Start the app in development mode\n"
                "4. Check the README for port and URL details"
            )
        return {"success": True, "output": instructions, "error": ""}

    return {"success": False, "output": "", "error": f"Unknown cloud-gen tool: {tool_name}"}


def _default_docker_compose(services_str: str) -> str:
    """Minimal fallback docker-compose.yml."""
    services = [s.strip() for s in services_str.split(",")]
    lines = ["version: '3.8'", "services:"]
    for svc in services:
        lines += [f"  {svc}:", f"    image: {svc}:latest", f"    restart: unless-stopped"]
        if svc == "postgres":
            lines += [
                "    environment:",
                "      POSTGRES_PASSWORD: postgres",
                "      POSTGRES_USER: postgres",
                "      POSTGRES_DB: postgres",
                "    ports:",
                "      - '5432:5432'",
            ]
        elif svc == "redis":
            lines += ["    ports:", "      - '6379:6379'"]
        elif svc in ("mysql", "mariadb"):
            lines += [
                "    environment:",
                "      MYSQL_ROOT_PASSWORD: root",
                "    ports:",
                "      - '3306:3306'",
            ]
        elif svc in ("nginx", "web"):
            lines += ["    ports:", "      - '80:80'"]
    return "\n".join(lines) + "\n"


# ─── Fix applicator ───────────────────────────────────────────────────────────

_PIP_FAIL_RE = re.compile(
    r"(?:error.*?building.*?(?:wheel|package)\s+for\s+|"
    r"Failed to build\s+|"
    r"Could not build wheels for\s+)"
    r"([a-zA-Z0-9_-]+)",
    re.I,
)


def _try_pip_self_heal(command: str, error: str, working_dir: str) -> dict | None:
    """
    If a pip install -r requirements.txt fails because one package can't build,
    retry without the version pin for that package, or skip it and install the rest.
    Returns the result dict if healed, None otherwise.
    """
    if "pip install" not in command or "requirements" not in command:
        return None

    match = _PIP_FAIL_RE.search(error)
    if not match:
        return None

    failing_pkg = match.group(1).lower().strip()

    # Strategy 1: try installing the failing package without version pin
    r1 = execute_tool("run_command", {
        "command": f"pip install {failing_pkg}",
        "working_dir": working_dir,
    })

    # Strategy 2: install from requirements but skip the failing package
    r2 = execute_tool("run_command", {
        "command": (
            f"pip install -r requirements.txt "
            f"$(grep -iv '^{failing_pkg}' requirements.txt | tr '\\n' ' ') 2>/dev/null || "
            f"grep -iv '^{failing_pkg}' requirements.txt | xargs pip install"
        ),
        "working_dir": working_dir,
    })

    # If at least the skip strategy worked, report success
    if r1.get("success") or r2.get("success"):
        output_parts = []
        if r1.get("success"):
            output_parts.append(f"Installed {failing_pkg} without version pin")
        if r2.get("success"):
            output_parts.append(f"Installed remaining packages (skipped pinned {failing_pkg})")
        return {"success": True, "output": "; ".join(output_parts), "error": ""}

    return None


def _extract_and_apply_fix(diagnosis: str, failed_tool: str, failed_args: dict) -> bool:
    """
    Parse diagnosis text and attempt to apply a concrete fix.
    Returns True if a fix was applied.
    """
    port_match = re.search(r"port[:\s]+(\d+)\s+to\s+(\d+)", diagnosis, re.I)
    if port_match:
        old_port, new_port = port_match.group(1), port_match.group(2)
        compose = get_workspace() / "docker-compose.yml"
        if compose.exists():
            content = compose.read_text()
            if old_port in content:
                compose.write_text(content.replace(f":{old_port}", f":{new_port}"))
                return True

    edit_match = re.search(
        r"edit\s+([^\s:]+)[:\s]+replace\s+['\"](.+?)['\"]\s+with\s+['\"](.+?)['\"]",
        diagnosis, re.I,
    )
    if edit_match:
        filepath, old_text, new_text = edit_match.groups()
        result = execute_tool("edit_file", {"filepath": filepath, "old_text": old_text, "new_text": new_text})
        return result["success"]

    cmd_match = re.search(r"run[:\s]+['\"]?([^'\"\n]+)['\"]?", diagnosis, re.I)
    if cmd_match:
        cmd = cmd_match.group(1).strip()
        result = execute_tool("run_command", {"command": cmd})
        return result["success"]

    return False


# ─── Self-heal loop ───────────────────────────────────────────────────────────

# Tools where failures are not fixable by editing files — skip Gemini diagnosis.
_NO_HEAL_TOOLS = {
    "check_health", "check_port",       # network errors: app simply isn't running
    "show_tree", "list_files",           # path errors: wrong dir, nothing to patch
    "read_file",                         # file-not-found: user typo, not patchable
    "read_logs",                         # docker not available
}

class StepResult:
    def __init__(self, tool_name, arguments, result, routed_to, latency_ms, attempt=1):
        self.tool_name = tool_name
        self.arguments = arguments
        self.result = result
        self.routed_to = routed_to
        self.latency_ms = latency_ms
        self.attempt = attempt
        self.success = result.get("success", False)


def execute_with_self_heal(tool_name: str, arguments: dict, max_retries: int = 3,
                           progress_callback=None) -> StepResult:
    """
    Execute a tool with automatic error recovery.
    On failure: ask cloud for diagnosis → apply fix → retry.
    """
    last_result = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        result = execute_tool(tool_name, arguments)
        latency = (time.time() - t0) * 1000

        if result["success"]:
            return StepResult(tool_name, arguments, result, "local", latency, attempt)

        last_result = result
        error_msg = result.get("error", "") or result.get("stderr", "")

        if progress_callback:
            progress_callback("error", tool_name, error_msg, attempt)

        if attempt >= max_retries:
            break

        # Skip healing for tools where Gemini can't write a meaningful fix
        if tool_name in _NO_HEAL_TOOLS:
            break

        # Fast pip-specific self-heal (no cloud call needed)
        if tool_name == "run_command" and "pip install" in arguments.get("command", ""):
            pip_fix = _try_pip_self_heal(
                arguments["command"], error_msg,
                arguments.get("working_dir", ""),
            )
            if pip_fix:
                if progress_callback:
                    progress_callback("fix_applied", tool_name, pip_fix["output"], attempt)
                return StepResult(tool_name, arguments, pip_fix, "local", latency, attempt)

        if progress_callback:
            progress_callback("healing", tool_name, error_msg, attempt)

        diagnosis = _diagnose_via_cloud(
            tool_name, arguments, error_msg,
            context=f"Attempt {attempt} of {max_retries}",
        )

        if progress_callback:
            progress_callback("diagnosis", tool_name, diagnosis, attempt)

        fixed = _extract_and_apply_fix(diagnosis, tool_name, arguments)
        if not fixed:
            if progress_callback:
                progress_callback("no_fix", tool_name, diagnosis, attempt)
            break

        if progress_callback:
            progress_callback("fix_applied", tool_name, diagnosis, attempt)

    return StepResult(
        tool_name, arguments,
        last_result or {"success": False, "output": "", "error": "No result"},
        "local", 0, max_retries,
    )


# ─── Auto-setup helpers ───────────────────────────────────────────────────────

_SETUP_SCAN_FILES = [
    "package.json",
    "requirements.txt", "pyproject.toml", "setup.py", "Pipfile",
    "go.mod", "Cargo.toml", "pom.xml", "build.gradle",
    "Makefile", ".env.example", "config.yml", "config.yaml",
    "README.md", "README.rst",
    "docker-compose.yml", "docker-compose.yaml", "Dockerfile",
]

_USAGE_COMMENT_RE = re.compile(
    r"(?:#|//)\s*(?:to\s+run|usage|how\s+to|run\s+with|start(?:up)?)\s*:?\s*(.+)",
    re.I,
)
_SOURCE_EXTS = {".py", ".js", ".ts", ".go", ".rb", ".sh", ".bash"}


def _discover_repo(ws) -> tuple:
    """
    Scan workspace for config files and inline usage comments.
    Returns (context_string, docker_present, hints_count, files_count).
    """
    context_parts = []

    try:
        root_files = sorted(f.name for f in ws.iterdir() if not f.name.startswith("."))
        if root_files:
            context_parts.append(f"Root files: {', '.join(root_files)}")
    except Exception:
        pass

    read_keys = set()
    docker_present = False

    for filename in _SETUP_SCAN_FILES:
        for match in sorted(ws.rglob(filename)):
            try:
                rel = match.relative_to(ws)
            except ValueError:
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            if filename in read_keys:
                continue
            read_keys.add(filename)
            try:
                content = match.read_text(errors="replace")
                limit = 300 if filename.startswith("README") else 800
                # Detect production-only docker-compose files
                if filename in ("docker-compose.yml", "docker-compose.yaml"):
                    docker_present = True
                    is_prod = any(kw in content.lower() for kw in [
                        "external: true", "traefik", "certresolver",
                        "letsencrypt", "nginx-proxy", "swarm",
                        "deploy:", "replicas:",
                    ])
                    if is_prod:
                        context_parts.append(
                            f"{rel}: [PRODUCTION-ONLY docker-compose — uses external networks/Traefik/deploy config. "
                            f"DO NOT use for local setup.]\n{content[:limit]}"
                        )
                        continue
                elif filename == "Dockerfile":
                    docker_present = True
                context_parts.append(f"{rel}:\n{content[:limit]}")
            except Exception:
                pass

    hints = []
    for f in sorted(ws.rglob("*")):
        if f.suffix in _SOURCE_EXTS and f.is_file():
            try:
                if f.stat().st_size > 100_000:
                    continue
                head = "\n".join(f.read_text(errors="replace").splitlines()[:30])
                for m in _USAGE_COMMENT_RE.finditer(head):
                    hints.append(f"Usage hint in {f.relative_to(ws)}: {m.group(1).strip()}")
            except Exception:
                pass

    if hints:
        context_parts.append("Inline usage comments:\n" + "\n".join(hints[:10]))

    return "\n\n".join(context_parts), docker_present, len(hints), len(read_keys)


def _detect_port_from_context(context: str):
    """Extract the most likely app port from repo context. Returns int or None."""
    for pattern in [
        r'"port"\s*:\s*(\d{4,5})',
        r"PORT\s*=\s*(\d{4,5})",
        r"port\s*=\s*(\d{4,5})",
        r"localhost:(\d{4,5})",
    ]:
        m = re.search(pattern, context, re.I)
        if m:
            port = int(m.group(1))
            if 1024 < port < 65535:
                return port
    return None


def _prepend_network_creates(commands: list, ws) -> list:
    """
    If docker-compose up is planned and the compose file declares external networks,
    prepend 'docker network create <name> 2>/dev/null || true' for each one so the
    run doesn't fail due to missing external networks.
    """
    has_compose_up = any(re.search(r"docker.?compose\s+up", c, re.I) for c in commands)
    if not has_compose_up:
        return commands

    compose_file = ws / "docker-compose.yml"
    if not compose_file.exists():
        return commands

    try:
        import yaml as _yaml
        data = _yaml.safe_load(compose_file.read_text())
        networks = (data or {}).get("networks", {}) or {}
        external_nets = [
            name for name, cfg in networks.items()
            if isinstance(cfg, dict) and cfg.get("external")
        ]
    except Exception:
        return commands

    if not external_nets:
        return commands

    create_cmds = [f"docker network create {n} 2>/dev/null || true" for n in external_nets]
    # Insert before the first docker-compose command
    for i, cmd in enumerate(commands):
        if re.search(r"docker.?compose", cmd, re.I):
            return commands[:i] + create_cmds + commands[i:]
    return create_cmds + commands


def _parse_setup_commands(raw: str) -> list:
    """Strip markdown formatting from a newline-separated command list."""
    commands = []
    for line in raw.splitlines():
        line = line.strip().lstrip("0123456789.-) ").strip("`").strip()
        if line and not line.startswith("#") and len(line) > 2:
            commands.append(line)
    return commands


# ─── Main orchestrator ────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(self, progress_callback=None):
        self.cb = progress_callback or (lambda *a, **k: None)
        self.history = []

    def _emit(self, event, tool="", msg="", **extra):
        self.cb(event, tool, msg, **extra)

    def process(self, user_text: str, url_prompt_fn=None) -> list:
        """
        Main entry point: process a user command end-to-end.

        url_prompt_fn: optional callback(prompt_text) -> str  that the CLI
        provides so we can ask the user for a repo URL interactively.
        """
        agents = classify_agents(user_text)
        self._emit("plan", msg=f"Agents: {' -> '.join(a.upper() for a in agents)}")

        if agents == ["chat"]:
            return self.ask(user_text)

        if agents == ["clone_setup"]:
            return self.run_clone_and_setup(user_text, url_prompt_fn)

        if agents == ["auto_setup"]:
            return self.run_auto_setup(user_text)

        all_results = []
        for agent_name in agents:
            results = self._run_agent(agent_name, user_text, all_results)
            all_results.extend(results)

        self.history.extend(all_results)
        return all_results

    def run_clone_and_setup(self, user_text: str, url_prompt_fn=None) -> list:
        """
        Clone a repo, switch workspace to it, then auto-setup.

        Extracts URL from user text if present, otherwise prompts interactively.
        """
        all_results = []

        # Phase 0: Get the repo URL
        self._emit("phase_start", msg="0 / 6   CLONE REPOSITORY", phase=0)

        url_match = _GIT_URL_RE.search(user_text)
        repo_url = url_match.group(1) if url_match else None

        if not repo_url and url_prompt_fn:
            repo_url = url_prompt_fn("Paste the git repo URL")
            if repo_url:
                repo_url = repo_url.strip()

        if not repo_url:
            self._emit("error", "", "No repo URL provided. Say 'clone <URL>' or paste a URL.")
            return all_results

        self._emit("start", "clone_repo", f"Cloning {repo_url}")
        t0 = time.time()
        result = exec_clone_repo(repo_url)
        elapsed = (time.time() - t0) * 1000

        sr = StepResult("clone_repo", {"repo_url": repo_url}, result, "local", elapsed)
        all_results.append(sr)

        if result["success"]:
            self._emit("success", "clone_repo", result["output"])
            self._emit("phase_done", msg=f"Cloned into {get_workspace()}", phase=0)
        else:
            self._emit("error", "clone_repo", result.get("error", "Clone failed"))
            # Try self-healing: maybe git isn't installed or URL is wrong
            self._emit("healing", "clone_repo", result.get("error", ""))
            diagnosis = _diagnose_via_cloud(
                "clone_repo",
                {"repo_url": repo_url},
                result.get("error", ""),
                context="Git clone failed during auto-setup",
            )
            self._emit("diagnosis", "clone_repo", diagnosis)
            return all_results

        # Now run the full auto_setup on the cloned repo
        setup_results = self.run_auto_setup(user_text)
        all_results.extend(setup_results)
        return all_results

    def _run_agent(self, agent_name: str, user_text: str, prior_results: list) -> list:
        t0 = time.time()
        agent = AGENTS[agent_name]
        self._emit("agent_start", msg=f"[{agent['label']}]")

        tools = agent["tools"]
        tools_for_model = strip_routing_metadata(tools)

        context = ""
        if prior_results:
            last = prior_results[-1]
            context = f"\nPrevious step result: {last.result.get('output', '')[:200]}"

        # Always inject repo context so the model knows what kind of project it's in
        repo_hint = _get_repo_summary(get_workspace())
        messages = [{"role": "user", "content": user_text + context + f"\n\n[Workspace: {repo_hint}]"}]

        # 1. Try regex text fallback first (0ms, no model call)
        calls = _devops_text_fallback(user_text, set(TOOL_MAP.keys()))
        routing_latency = (time.time() - t0) * 1000
        if calls:
            self._emit("routed_local", msg=f"text-fallback ({routing_latency:.0f}ms)", latency=routing_latency)
        else:
            # 2. Fall back to generate_hybrid (on-device FunctionGemma -> cloud)
            t0 = time.time()
            with _suppress_stdout():
                hybrid_result = generate_hybrid(messages, tools_for_model)
            routing_latency = (time.time() - t0) * 1000

            source = hybrid_result.get("source", "")
            routed_to = "local" if "on-device" in source else "cloud"
            routing_reason = hybrid_result.get("routing_reason", "")
            self._emit(
                f"routed_{routed_to}",
                msg=f"{routing_reason} ({routing_latency:.0f}ms)",
                latency=routing_latency,
            )

            calls = hybrid_result.get("function_calls", [])
            if not calls:
                self._emit("no_calls", msg="No tool calls generated for this input")
                return []

        step_results = []
        for call in calls:
            tool_name = call.get("name", "")
            arguments = call.get("arguments", {})

            self._emit("start", tool_name, f"Args: {json.dumps(arguments)[:120]}")

            tool_def = TOOL_MAP.get(tool_name, {})
            if tool_def.get("_routing") == "cloud":
                result = _handle_cloud_generation(tool_name, arguments)
                sr = StepResult(tool_name, arguments, result, "cloud", routing_latency)
            else:
                sr = execute_with_self_heal(
                    tool_name, arguments,
                    max_retries=3,
                    progress_callback=lambda evt, tn, msg, att: self._emit(evt, tn, msg, attempt=att),
                )

            step_results.append(sr)

            if sr.success:
                self._emit("success", tool_name, sr.result.get("output", "")[:300])
            else:
                self._emit("error", tool_name, sr.result.get("error", "unknown error"))

        self._emit("agent_done", msg=f"Done ({len(step_results)} tool calls)")
        return step_results

    def run_auto_setup(self, user_text: str) -> list:
        """
        Full discover -> plan -> execute -> health-check -> docs flow.
        Triggered when the user gives a vague repo-level setup command.
        """
        ws = get_workspace()
        all_results = []

        # Phase 1: Discovery
        self._emit("phase_start", msg="1 / 5   DISCOVERY", phase=1)
        context, docker_present, hint_count, file_count = _discover_repo(ws)
        if not context.strip():
            self._emit("error", "", "Workspace appears empty — nothing to set up")
            return all_results
        self._emit("phase_done", msg=f"{file_count} config file(s) scanned, {hint_count} usage hint(s) found", phase=1)

        # Phase 2: Plan via generate_hybrid (FunctionGemma first, Gemini fallback)
        self._emit("phase_start", msg="2 / 5   PLANNING", phase=2)
        t0 = time.time()
        messages = [{"role": "user", "content": (
            "Set up this repository. Return the exact ordered shell commands needed "
            "to install dependencies, configure the environment, and start the app.\n\n"
            f"Repo context:\n{context}"
        )}]
        tools_for_model = strip_routing_metadata([TOOL_PLAN_SETUP_COMMANDS])
        with _suppress_stdout():
            hybrid_result = generate_hybrid(messages, tools_for_model)
        routing_latency = (time.time() - t0) * 1000

        source = hybrid_result.get("source", "")
        routed_to = "local" if "on-device" in source else "cloud"
        self._emit(
            f"routed_{routed_to}",
            msg=f"plan_setup_commands ({routing_latency:.0f}ms)",
            latency=routing_latency,
        )

        calls = hybrid_result.get("function_calls", [])
        commands = []
        detected_type = "unknown"
        if calls and calls[0]["name"] == "plan_setup_commands":
            raw_commands = calls[0]["arguments"].get("commands", "")
            detected_type = calls[0]["arguments"].get("detected_type", "unknown")
            commands = _parse_setup_commands(raw_commands)
            # Validate: FunctionGemma sometimes fills the parameter *description* as its value
            # (e.g. "install dependencies, configure environment, build if needed, start the app").
            # Reject any command list where no entry starts with a recognisable shell token.
            _SHELL_PREFIX_RE = re.compile(
                r"^(pip3?|npm|npx|yarn|pnpm|python3?|uvicorn|gunicorn|node|go\s|"
                r"cargo|make|cp\s|mv\s|touch\s|mkdir|export\s|source\s|\./|bash\s|sh\s|"
                r"docker|git\s|brew\s|apt|curl\s|wget\s|ruby|bundle|composer|php\s|"
                r"java\s|mvn|gradle|flask|fastapi|django|rails|mix\s|dotnet|"
                r"rustup|rustc|echo\s|cat\s|cd\s)",
                re.I,
            )
            if commands and not any(_SHELL_PREFIX_RE.match(c) for c in commands):
                self._emit(
                    "routed_cloud",
                    msg="on-device returned description text, not real commands — falling back to Gemini",
                )
                commands = []

        # generate_hybrid returned no function calls (FunctionGemma didn't recognise the tool,
        # or Gemini replied with text instead of a structured call) — fall back to a direct
        # plain-text cloud generation prompt so Gemini can answer in free-form.
        if not commands:
            self._emit("routed_cloud", msg="on-device returned no plan — asking Gemini directly")
            try:
                fallback = _handle_cloud_generation("plan_setup_commands", {"context": context})
            except Exception as exc:
                self._emit("error", "plan_setup_commands", f"Gemini call failed: {exc}")
                return all_results
            commands = fallback.get("commands", [])
            if not commands:
                self._emit("error", "plan_setup_commands", "No setup commands generated — try a more specific command")
                return all_results

        # Post-process: if docker-compose up is planned but compose file has
        # external networks, prepend 'docker network create' for each one.
        commands = _prepend_network_creates(commands, ws)

        self._emit("setup_plan", msg=detected_type, commands=commands, phase=2)
        self._emit("phase_done", msg=f"{len(commands)} step(s) planned, detected: {detected_type}", phase=2)

        # Phase 3: Execute each command with self-heal
        self._emit("phase_start", msg=f"3 / 5   EXECUTING  ({len(commands)} step(s))", phase=3)
        step_records = []
        for i, cmd in enumerate(commands, 1):
            self._emit("setup_step", msg=cmd, step=i, total=len(commands))
            t_step = time.time()
            sr = execute_with_self_heal(
                "run_command",
                {"command": cmd, "working_dir": str(ws)},
                max_retries=2,
                progress_callback=lambda evt, tn, msg, att: self._emit(evt, tn, msg, attempt=att),
            )
            elapsed = (time.time() - t_step) * 1000
            all_results.append(sr)
            step_records.append((cmd, sr.success, elapsed, sr.result.get("output", "")[:120]))
            self._emit(
                "setup_step_done",
                msg=sr.result.get("output", "") or sr.result.get("error", ""),
                success=sr.success,
                elapsed_ms=elapsed,
                step=i,
            )

        succeeded = sum(1 for _, ok, _, _ in step_records if ok)
        self._emit("phase_done", msg=f"{succeeded} / {len(commands)} step(s) succeeded", phase=3)

        # Phase 4: Health check
        self._emit("phase_start", msg="4 / 5   HEALTH CHECK", phase=4)
        port = _detect_port_from_context(context)
        running = False
        if port:
            result = execute_tool("check_port", {"port": port})
            running = result.get("in_use", False)
            sr = StepResult("check_port", {"port": port}, result, "local", 0)
            all_results.append(sr)
            self._emit("health_result", msg=str(port), running=running, port=port)
        else:
            self._emit("phase_done", msg="No port detected in repo context", phase=4)
        if port:
            self._emit("phase_done", msg=f"Port {port} — {'running' if running else 'not responding'}", phase=4)

        # Phase 5: Run instructions
        self._emit("phase_start", msg="5 / 5   RUN INSTRUCTIONS", phase=5)
        instructions = self.run_documentation_agent()
        if instructions:
            self._emit("success", "generate_run_instructions", instructions)
        self._emit("phase_done", msg="", phase=5)

        # Summary
        self._emit(
            "setup_summary",
            msg="",
            steps=step_records,
            port=port,
            running=running,
        )

        self.history.extend(all_results)
        return all_results

    def _gather_project_context(self) -> tuple[str, bool]:
        """
        Scan the workspace for project files and return (context_string, docker_present).
        Shared by run_documentation_agent() and ask().
        """
        ws = get_workspace()
        context_parts = []

        try:
            root_files = sorted(
                f.name for f in ws.iterdir()
                if not f.name.startswith(".") and f.name != ".gitkeep"
            )
            if root_files:
                context_parts.append(f"Root-level files: {', '.join(root_files)}")
        except Exception:
            pass

        if not context_parts:
            return "", False

        SCAN_FILES = [
            "package.json",
            "requirements.txt", "pyproject.toml", "setup.py", "Pipfile",
            "go.mod", "Cargo.toml", "pom.xml", "build.gradle",
            "Makefile", ".env.example", "config.yml", "config.yaml",
            "README.md", "README.rst", "README.txt",
            "docker-compose.yml", "docker-compose.yaml", "Dockerfile",
            "main.py", "app.py", "server.py", "index.js", "index.ts",
        ]
        docker_present = False
        read_files = set()

        for filename in SCAN_FILES:
            for match in sorted(ws.rglob(filename)):
                try:
                    rel = match.relative_to(ws)
                except ValueError:
                    continue
                if any(part.startswith(".") for part in rel.parts):
                    continue
                if filename in read_files:
                    continue
                read_files.add(filename)
                try:
                    content = match.read_text(errors="replace")
                    limit = 300 if filename.startswith("README") else 600
                    context_parts.append(f"{rel}:\n{content[:limit]}")
                    if filename in ("docker-compose.yml", "docker-compose.yaml", "Dockerfile"):
                        docker_present = True
                except Exception:
                    pass

        if self.history:
            history_items = []
            for r in self.history:
                status = "OK" if r.success else "FAIL"
                history_items.append(f"{r.tool_name}({status})")
            if history_items:
                context_parts.append(f"Session history: {', '.join(history_items)}")

        return "\n\n".join(context_parts), docker_present

    def run_documentation_agent(self) -> str:
        """
        Scan the repo for project files and ask Gemini to generate 'How to Run' docs.
        Called automatically at session exit and after auto_setup.
        Returns the instructions string, or "" if workspace is empty.
        """
        context, docker_present = self._gather_project_context()
        if not context:
            return ""

        self._emit("agent_start", msg="[Documentation]")

        try:
            result = _handle_cloud_generation(
                "generate_run_instructions",
                {"context": context, "docker_present": str(docker_present)},
            )
        except Exception as exc:
            self._emit("error", "generate_run_instructions", f"Gemini call failed: {exc}")
            self._emit("agent_done", msg="Skipped")
            return ""

        self._emit("agent_done", msg="Run instructions generated")
        return result.get("output", "")

    def ask(self, question: str) -> str:
        """
        Answer a free-form question about the project using Gemini.
        Gathers project context and sends it with the question for a natural language answer.
        """
        context, _ = self._gather_project_context()
        if not context:
            return "No project files found in the workspace. Clone or cd into a repo first."

        prompt = (
            "You are a helpful DevOps assistant. The user has a project they are working with.\n"
            "Here is the project context (file listings and contents):\n\n"
            f"{context}\n\n"
            "Answer the user's question naturally, concisely, and helpfully. "
            "Use markdown formatting. If the question is about how to run the app, "
            "give specific commands. If it's about the codebase, reference specific files.\n\n"
            f"User's question: {question}"
        )

        self._emit("agent_start", msg="[Q&A]")
        try:
            answer = _generate_text_via_gemini(prompt)
        except Exception as exc:
            self._emit("error", "chat", f"Gemini call failed: {exc}")
            self._emit("agent_done", msg="Failed")
            return f"Could not get an answer: {exc}"

        self._emit("agent_done", msg="Answered")
        return answer

    def plan(self, user_text: str) -> dict:
        """Return execution plan without running anything."""
        agents = classify_agents(user_text)
        if agents == ["clone_setup"]:
            return {
                "user_text": user_text,
                "agents": [{
                    "agent": "clone_setup",
                    "label": "Clone + Setup",
                    "tools_available": ["clone_repo", "plan_setup_commands", "run_command", "check_port", "generate_run_instructions"],
                    "routing_hints": {
                        "clone_repo": "local",
                        "plan_setup_commands": "cloud",
                        "run_command": "local",
                        "check_port": "local",
                        "generate_run_instructions": "cloud",
                    },
                }],
            }
        if agents == ["auto_setup"]:
            return {
                "user_text": user_text,
                "agents": [{
                    "agent": "auto_setup",
                    "label": "Auto Setup",
                    "emoji": "",
                    "tools_available": ["plan_setup_commands", "run_command", "check_port", "generate_run_instructions"],
                    "routing_hints": {
                        "plan_setup_commands": "cloud",
                        "run_command": "local",
                        "check_port": "local",
                        "generate_run_instructions": "cloud",
                    },
                }],
            }
        plan = []
        for agent_name in agents:
            agent = AGENTS[agent_name]
            tools = agent["tools"]
            plan.append({
                "agent": agent_name,
                "label": agent["label"],
                "tools_available": [t["name"] for t in tools],
                "routing_hints": {
                    t["name"]: TOOL_MAP[t["name"]].get("_routing", "local")
                    for t in tools
                },
            })
        return {"user_text": user_text, "agents": plan}
