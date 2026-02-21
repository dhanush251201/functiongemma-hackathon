"""
Agent router + self-heal loop + multi-agent chaining.

Orchestrator responsibilities:
  1. Classify user intent → assign to correct agent(s)
  2. Route tool calls through generate_hybrid
  3. Execute tools via executor.py
  4. Self-heal: catch errors → diagnose via cloud → apply fix → retry
  5. Chain multi-agent sequences (Builder → Runner → Tester)
"""

import sys
import os
import re
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))

from tools import (
    EXPLORER_TOOLS, BUILDER_TOOLS, RUNNER_TOOLS, TESTER_TOOLS, DOCUMENTATION_TOOLS, ALL_TOOLS,
    TOOL_MAP, strip_routing_metadata,
    TOOL_DIAGNOSE_ERROR,
)
from executor import execute_tool, get_workspace

# Import generate_hybrid from parent main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import generate_hybrid, generate_cloud


# ─── Agent definitions ────────────────────────────────────────────────────────

AGENTS = {
    "explorer": {
        "tools": EXPLORER_TOOLS,
        "label": "Explorer",
        "emoji": "🔍",
        "keywords": ["list", "show", "read", "find", "tree", "disk", "view", "what files", "ls", "cat", "grep"],
    },
    "builder": {
        "tools": BUILDER_TOOLS,
        "label": "Builder",
        "emoji": "🔨",
        "keywords": ["create", "write", "generate", "dockerfile", "docker-compose", "compose", "install", "edit", "delete", "make file", "setup file"],
    },
    "runner": {
        "tools": RUNNER_TOOLS,
        "label": "Runner",
        "emoji": "▶️",
        "keywords": ["run", "start", "stop", "execute", "docker", "port", "logs", "command", "launch", "bring up"],
    },
    "tester": {
        "tools": TESTER_TOOLS,
        "label": "Tester",
        "emoji": "✅",
        "keywords": ["check", "test", "health", "validate", "verify", "diagnose", "is it running", "working", "ping"],
    },
    "documentation": {
        "tools": DOCUMENTATION_TOOLS,
        "label": "Documentation",
        "emoji": "📖",
        "keywords": ["docs", "instructions", "how to run", "readme", "guide", "how do i", "how to use"],
    },
}

_MULTI_AGENT_PATTERNS = [
    # "set up X with docker" → Builder (compose) → Runner (up) → Tester (health)
    (re.compile(r"set up|setup|deploy|provision", re.I), ["builder", "runner", "tester"]),
    # "install and run X" → Builder (install) → Runner → Tester
    (re.compile(r"install and run|install.*then run", re.I), ["builder", "runner", "tester"]),
]


def classify_agents(user_text: str) -> list[str]:
    """Determine which agent(s) should handle this request. Returns list in execution order."""
    text_lower = user_text.lower()

    # Check multi-agent patterns first
    for pattern, agents in _MULTI_AGENT_PATTERNS:
        if pattern.search(text_lower):
            return agents

    # Score each agent by keyword matches
    scores = {}
    for name, agent in AGENTS.items():
        score = sum(1 for kw in agent["keywords"] if kw in text_lower)
        if score > 0:
            scores[name] = score

    if not scores:
        # Default: try explorer
        return ["explorer"]

    # Return highest-scoring agent (single agent for simple commands)
    best = max(scores, key=scores.get)
    return [best]


# ─── DevOps text fallback (regex-based, no LLM needed) ───────────────────────

def _devops_text_fallback(user_text: str, tool_names: set) -> list:
    """
    Pattern-match natural-language DevOps commands when the LLM returns nothing.
    Covers the 6 demo steps and common variants.
    """
    text = user_text.lower()
    calls = []

    # ── Explorer ──────────────────────────────────────────────────────────────
    if "show_tree" in tool_names and re.search(r"\b(tree|show|list|files|workspace|ls)\b", text):
        dir_m = re.search(r"\bin\s+(?:the\s+|a\s+|an\s+)?([\w./\\-]+)\s*(?:directory|folder|workspace)?", text)
        raw_dir = dir_m.group(1) if dir_m else "."
        directory = "." if raw_dir in ("the", "a", "an", "workspace", "workdir", "current", "here", "dir") else raw_dir
        calls.append({"name": "show_tree", "arguments": {"directory": directory}})
        return calls

    if "list_files" in tool_names and re.search(r"\b(list|show|files|workspace|ls|what\s+files)\b", text):
        dir_m = re.search(r"\bin\s+(?:the\s+|a\s+|an\s+)?([\w./\\-]+)\s*(?:directory|folder|workspace)?", text)
        raw_dir = dir_m.group(1) if dir_m else "."
        directory = "." if raw_dir in ("the", "a", "an", "workspace", "workdir", "current", "here", "dir") else raw_dir
        calls.append({"name": "list_files", "arguments": {"directory": directory}})
        return calls

    if "read_file" in tool_names and re.search(r"\b(read|show|cat|view|open|display)\b", text) and not re.search(r"\blog", text):
        path_m = re.search(r"\b([\w./\\-]+\.\w+)\b", user_text)
        filepath = path_m.group(1) if path_m else "."
        calls.append({"name": "read_file", "arguments": {"filepath": filepath}})
        return calls

    # ── Builder ───────────────────────────────────────────────────────────────
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

    # ── Tester ────────────────────────────────────────────────────────────────
    if "validate_config" in tool_names and re.search(r"\b(validate|check|lint|verify)\b.*\b(config|yaml|yml|json)\b", text):
        path_m = re.search(r"\b([\w./\\-]+\.(?:yml|yaml|json|toml))\b", user_text)
        filepath = path_m.group(1) if path_m else "docker-compose.yml"
        fmt_m = re.search(r"\b(yaml|yml|json|toml)\b", text)
        fmt = "yaml" if not fmt_m or fmt_m.group(1) in ("yaml", "yml") else fmt_m.group(1)
        calls.append({"name": "validate_config", "arguments": {"filepath": filepath, "format": fmt}})
        return calls

    if "check_health" in tool_names and re.search(r"\b(health|healthy|running|up|alive|reachable)\b", text):
        port_m = re.search(r"\bport\s+(\d+)\b", text)
        port_str = port_m.group(1) if port_m else "80"
        # Non-HTTP ports (databases, message queues) need TCP check, not HTTP
        _non_http = {"5432", "5433", "3306", "27017", "6379", "5672", "9200", "9042", "6443"}
        if port_str in _non_http and "check_port" in tool_names:
            calls.append({"name": "check_port", "arguments": {"port": int(port_str)}})
            return calls
        # Don't capture "port" as hostname (e.g. "on port 5432")
        host_m = re.search(r"\bon\s+((?!port\b)[\w.-]+)\b", text)
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

    # ── Runner ────────────────────────────────────────────────────────────────
    if "start_service" in tool_names and re.search(r"\b(start|bring\s+up|spin\s+up|launch|up)\b", text):
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
        cmd_m = re.search(r"\b(?:run|execute|exec)\s+['\"]?([^'\"]+)['\"]?", text)
        cmd = cmd_m.group(1).strip() if cmd_m else "echo hello"
        calls.append({"name": "run_command", "arguments": {"command": cmd}})
        return calls

    if "check_disk_space" in tool_names and re.search(r"\b(disk|space|storage|free\s+space)\b", text):
        calls.append({"name": "check_disk_space", "arguments": {"path": "."}})
        return calls

    return calls


# ─── Cloud generation helpers ─────────────────────────────────────────────────

def _cloud_generate_content(prompt: str, tools_for_model: list) -> str:
    """Ask Gemini to generate content (for Dockerfile, docker-compose, etc.)."""
    result = generate_cloud(
        messages=[{"role": "user", "content": prompt}],
        tools=tools_for_model,
    )
    calls = result.get("function_calls", [])
    if calls:
        # Return first function call arguments as JSON string
        return json.dumps(calls[0].get("arguments", {}))
    return ""


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
    result = generate_cloud(
        messages=[{"role": "user", "content": prompt}],
        tools=tools_model,
    )
    calls = result.get("function_calls", [])
    if calls and calls[0]["name"] == "diagnose_error":
        return calls[0].get("arguments", {}).get("error_message", "No diagnosis available")
    # Fall back: try to extract from raw response
    return f"Cloud diagnosis requested for: {error[:200]}"


# ─── Fix applicator ───────────────────────────────────────────────────────────

def _extract_and_apply_fix(diagnosis: str, failed_tool: str, failed_args: dict) -> bool:
    """
    Parse the diagnosis text and attempt to apply a fix.
    Looks for patterns like:
      - "Change port X to Y in docker-compose.yml"
      - "Edit file X: replace 'old' with 'new'"
      - "Run command: ..."
    Returns True if a fix was applied.
    """
    # Pattern: port conflict fix
    port_match = re.search(r"port[:\s]+(\d+)\s+to\s+(\d+)", diagnosis, re.I)
    if port_match:
        old_port, new_port = port_match.group(1), port_match.group(2)
        # Try to find and edit docker-compose.yml in workspace
        compose = get_workspace() / "docker-compose.yml"
        if compose.exists():
            content = compose.read_text()
            if old_port in content:
                compose.write_text(content.replace(f":{old_port}", f":{new_port}"))
                return True

    # Pattern: edit file instruction
    edit_match = re.search(r"edit\s+([^\s:]+)[:\s]+replace\s+['\"](.+?)['\"]\s+with\s+['\"](.+?)['\"]", diagnosis, re.I)
    if edit_match:
        filepath, old_text, new_text = edit_match.groups()
        result = execute_tool("edit_file", {"filepath": filepath, "old_text": old_text, "new_text": new_text})
        return result["success"]

    # Pattern: run a specific command as fix
    cmd_match = re.search(r"run[:\s]+['\"]?([^'\"\n]+)['\"]?", diagnosis, re.I)
    if cmd_match:
        cmd = cmd_match.group(1).strip()
        result = execute_tool("run_command", {"command": cmd})
        return result["success"]

    return False


# ─── Self-heal loop ───────────────────────────────────────────────────────────

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

        # Self-heal: send to cloud for diagnosis
        if progress_callback:
            progress_callback("healing", tool_name, error_msg, attempt)

        diagnosis = _diagnose_via_cloud(
            tool_name, arguments, error_msg,
            context=f"Attempt {attempt} of {max_retries}"
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

    return StepResult(tool_name, arguments, last_result or {"success": False, "output": "", "error": "No result"}, "local", 0, max_retries)


# ─── Tool-call extraction ─────────────────────────────────────────────────────

def _handle_cloud_generation(tool_name: str, arguments: dict) -> dict:
    """
    For tools like create_dockerfile and create_docker_compose,
    use Gemini to generate the actual content, then pass it to executor.
    """
    if tool_name == "create_dockerfile":
        service = arguments.get("service_name", "app")
        reqs = arguments.get("requirements", "")
        prompt = (
            f"Write a production-ready Dockerfile for a {service} service. "
            f"Requirements: {reqs}. "
            f"Output ONLY the Dockerfile content, no explanation."
        )
        result = generate_cloud(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        )
        # Gemini may return text in response instead of function call
        content = ""
        for cand in getattr(result, "_raw_candidates", []):
            for part in getattr(cand.content, "parts", []):
                if hasattr(part, "text"):
                    content += part.text
        if not content:
            # Build minimal Dockerfile as fallback
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
        result = generate_cloud(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        )
        content = ""
        for cand in getattr(result, "_raw_candidates", []):
            for part in getattr(cand.content, "parts", []):
                if hasattr(part, "text"):
                    content += part.text
        if not content:
            content = _default_docker_compose(services)
        arguments["content"] = content
        return execute_tool("create_docker_compose", arguments)

    if tool_name == "generate_run_instructions":
        context = arguments.get("context", "")
        prompt = (
            "Based on this DevOps session context, write clear step-by-step instructions "
            "for a developer to run and access the application:\n\n"
            f"{context}\n\n"
            "Include: prerequisites, exact commands to start services, URLs/ports to access, "
            "and any important environment variables. Format as numbered steps. Be concise."
        )
        result = generate_cloud(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        )
        instructions = ""
        for cand in getattr(result, "_raw_candidates", []):
            for part in getattr(cand.content, "parts", []):
                if hasattr(part, "text"):
                    instructions += part.text
        if not instructions:
            instructions = (
                "## Run Instructions\n\n"
                "1. Start services: `docker-compose up -d`\n"
                "2. Check status:   `docker-compose ps`\n"
                "3. View logs:      `docker-compose logs -f`\n"
                "4. Stop services:  `docker-compose down`"
            )
        return {"success": True, "output": instructions, "error": ""}

    return {"success": False, "output": "", "error": f"Unknown cloud-gen tool: {tool_name}"}


def _default_docker_compose(services_str: str) -> str:
    """Minimal fallback docker-compose.yml with service-specific defaults."""
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


# ─── Main orchestrator ────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(self, progress_callback=None):
        """
        progress_callback(event, tool_name, message, extra=None) is called for
        UI updates. Events: start, success, error, healing, diagnosis, fix_applied,
        routed_local, routed_cloud, agent_start, agent_done.
        """
        self.cb = progress_callback or (lambda *a, **k: None)
        self.history = []  # list of StepResult

    def _emit(self, event, tool="", msg="", **extra):
        self.cb(event, tool, msg, **extra)

    def process(self, user_text: str) -> list[StepResult]:
        """Main entry point: process a user command end-to-end."""
        agents = classify_agents(user_text)
        self._emit("plan", msg=f"Agents: {' → '.join(a.upper() for a in agents)}")

        all_results = []
        for agent_name in agents:
            results = self._run_agent(agent_name, user_text, all_results)
            all_results.extend(results)

        self.history.extend(all_results)
        return all_results

    def _run_agent(self, agent_name: str, user_text: str, prior_results: list) -> list[StepResult]:
        t0 = time.time()  # start timing from the very beginning of agent work
        agent = AGENTS[agent_name]
        self._emit("agent_start", msg=f"{agent['emoji']} {agent['label']} Agent")

        tools = agent["tools"]
        tools_for_model = strip_routing_metadata(tools)

        # Build context-enriched message
        context = ""
        if prior_results:
            last = prior_results[-1]
            context = f"\nPrevious step result: {last.result.get('output', '')[:200]}"

        messages = [{"role": "user", "content": user_text + context}]

        # 1. Try DevOps regex fallback FIRST with all known tools.
        #    Using all tools (not just this agent's) handles misclassification gracefully.
        calls = _devops_text_fallback(user_text, set(TOOL_MAP.keys()))
        routing_latency = (time.time() - t0) * 1000 / 2  # full agent setup time, halved
        if calls:
            lat_str = f"{routing_latency:.2f}ms" if routing_latency < 1 else f"{routing_latency:.0f}ms"
            self._emit("routed_local", msg=f"devops-text-fallback ({lat_str})", latency=routing_latency)
        else:
            # 2. Fall back to generate_hybrid (cactus + cloud)
            t0 = time.time()
            hybrid_result = generate_hybrid(messages, tools_for_model)
            routing_latency = (time.time() - t0) * 1000

            routed_to = hybrid_result.get("routed_to", "local")
            routing_reason = hybrid_result.get("routing_reason", "")
            self._emit(
                f"routed_{routed_to}",
                msg=f"{routing_reason} ({routing_latency:.0f}ms)",
                latency=routing_latency,
            )

            calls = hybrid_result.get("function_calls", [])
            if not calls:
                self._emit("no_calls", msg="No tool calls returned")
                return []

        step_results = []
        for call in calls:
            tool_name = call.get("name", "")
            arguments = call.get("arguments", {})

            self._emit("start", tool_name, f"Args: {json.dumps(arguments)[:120]}")

            # Cloud-only generation tools need special handling
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

    def run_documentation_agent(self) -> str:
        """
        Called once after the main REPL loop exits.
        Reads the workspace state and session history, then asks Gemini to
        generate step-by-step instructions for running the app.
        Returns the instructions string, or "" if nothing was set up.
        """
        context_parts = []

        # Collect workspace file listing and key file contents
        try:
            ws = get_workspace()
            workspace_files = [
                f.name for f in ws.iterdir() if f.name != ".gitkeep"
            ]
            if workspace_files:
                context_parts.append(f"Files in workspace: {', '.join(workspace_files)}")

            compose_path = ws / "docker-compose.yml"
            if compose_path.exists():
                context_parts.append(f"docker-compose.yml:\n{compose_path.read_text()[:800]}")

            dockerfile_path = ws / "Dockerfile"
            if dockerfile_path.exists():
                context_parts.append(f"Dockerfile:\n{dockerfile_path.read_text()[:400]}")
        except Exception:
            pass

        # Summarise what tools were successfully run this session
        if self.history:
            successful = [r.tool_name for r in self.history if r.success]
            if successful:
                context_parts.append(f"Operations performed: {', '.join(successful)}")

        if not context_parts:
            return ""

        context = "\n\n".join(context_parts)
        self._emit("agent_start", msg="📖 Documentation Agent")

        result = _handle_cloud_generation(
            "generate_run_instructions", {"context": context}
        )

        self._emit("agent_done", msg="Run instructions generated")
        return result.get("output", "")

    def plan(self, user_text: str) -> dict:
        """Return execution plan without running anything."""
        agents = classify_agents(user_text)
        plan = []
        for agent_name in agents:
            agent = AGENTS[agent_name]
            tools = agent["tools"]
            tools_for_model = strip_routing_metadata(tools)
            plan.append({
                "agent": agent_name,
                "label": agent["label"],
                "emoji": agent["emoji"],
                "tools_available": [t["name"] for t in tools],
                "routing_hints": {
                    t["name"]: TOOL_MAP[t["name"]].get("_routing", "local")
                    for t in tools
                },
            })
        return {"user_text": user_text, "agents": plan}
