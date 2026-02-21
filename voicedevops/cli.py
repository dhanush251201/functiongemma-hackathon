"""
Terminal UI for the Voice DevOps Agent.

Commands:
  <text>        Natural language command (keyboard)
  voice         Start voice recording (5 seconds)
  stats         Show routing statistics
  plan <task>   Show execution plan without running
  reset         Clean workspace
  demo          Run pre-scripted demo sequence
  help          Show help
  quit/exit     Exit
"""

import os
import sys
import pathlib
import time

# ── Permanent fd 1 → /dev/null ────────────────────────────────────────────────
# Cactus (and generate_hybrid/generate_cloud) write JSON noise (e.g. PGRST102)
# to the raw file descriptor 1 asynchronously, even after the call returns.
# Redirecting fd 1 to /dev/null HERE — before any cactus import — means those
# writes are discarded forever regardless of timing.  Python-level sys.stdout is
# re-pointed to the saved real terminal fd so rich / print still work normally.
_TERMINAL_FD = os.dup(1)                        # save real terminal
_dn = os.open(os.devnull, os.O_WRONLY)
os.dup2(_dn, 1)                                 # fd 1 → /dev/null permanently
os.close(_dn)
# Keep a module-level reference so it is never garbage-collected
_TERMINAL_FILE = open(_TERMINAL_FD, "w", buffering=1, closefd=True)
sys.stdout = _TERMINAL_FILE
# ──────────────────────────────────────────────────────────────────────────────

# Load .env from the voicedevops directory (sibling of this file)
_env_file = pathlib.Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich import box

from orchestrator import Orchestrator, AGENTS, classify_agents
from voice_pipeline import record_and_transcribe, voice_status, is_voice_available
from executor import get_workspace, set_workspace

try:
    from main import get_routing_stats
except ImportError:
    def get_routing_stats():
        return {}

console = Console()

# ─── Agent label styles ───────────────────────────────────────────────────────

AGENT_STYLES = {
    "explorer":      "cyan",
    "builder":       "yellow",
    "runner":        "green",
    "tester":        "blue",
    "documentation": "magenta",
    "auto_setup":    "white",
}

AGENT_LABELS = {
    "explorer":      "EXPLORER",
    "builder":       "BUILDER",
    "runner":        "RUNNER",
    "tester":        "TESTER",
    "documentation": "DOCUMENTATION",
    "auto_setup":    "AUTO SETUP",
}

# ─── Progress callback for Orchestrator ──────────────────────────────────────

class RichProgressHandler:
    """
    Translates orchestrator events into Rich terminal output.

    Standard events (existing agents):
      plan, agent_start, agent_done, routed_local, routed_cloud,
      start, success, error, healing, diagnosis, fix_applied, no_fix, no_calls

    Auto-setup events:
      phase_start, phase_done, setup_plan, setup_step, setup_step_done,
      health_result, setup_summary
    """

    def __init__(self):
        self._heal_count = 0

    def __call__(self, event: str, tool: str = "", msg: str = "", **extra):

        # ── Routing source ────────────────────────────────────────────────
        if event == "routed_local":
            console.print(
                f"  [bold green]LOCAL[/bold green]  [dim]{msg}[/dim]"
            )

        elif event == "routed_cloud":
            console.print(
                f"  [bold blue]CLOUD[/bold blue]  [dim]{msg}[/dim]"
            )

        # ── Standard agent lifecycle ──────────────────────────────────────
        elif event == "plan":
            console.print(f"  [dim]plan  {msg}[/dim]")

        elif event == "agent_start":
            console.print(Rule(msg, style="dim white"))

        elif event == "agent_done":
            if msg:
                console.print(f"  [dim]{msg}[/dim]")

        # ── Tool execution ────────────────────────────────────────────────
        elif event == "start":
            console.print(
                f"  [dim]running[/dim]  [bold]{tool}[/bold]  [dim]{msg}[/dim]"
            )

        elif event == "success":
            lines = msg.splitlines()
            preview = lines[0][:120] if lines else ""
            console.print(f"  [green]ok[/green]  {preview}")
            for line in lines[1:3]:
                console.print(f"     [dim]{line[:120]}[/dim]")

        elif event == "error":
            console.print(f"  [red]fail[/red]  {msg[:200]}")

        # ── Self-heal ─────────────────────────────────────────────────────
        elif event == "healing":
            self._heal_count += 1
            console.print(
                f"  [yellow]healing[/yellow]  "
                f"[dim]sending error to Gemini for diagnosis[/dim]"
            )

        elif event == "diagnosis":
            console.print(
                Panel(
                    msg[:400],
                    title="[blue]Gemini diagnosis[/blue]",
                    border_style="blue",
                    expand=False,
                    padding=(0, 1),
                )
            )

        elif event == "fix_applied":
            console.print(f"  [yellow]fix applied[/yellow]  [dim]{msg[:150]}[/dim]")

        elif event == "no_fix":
            console.print(f"  [red]no fix found[/red]  [dim]{msg[:150]}[/dim]")

        elif event == "no_calls":
            console.print(f"  [dim]no tool calls generated for this input[/dim]")

        # ── Auto-setup: phase headers ─────────────────────────────────────
        elif event == "phase_start":
            console.print()
            console.print(Rule(f"[bold cyan]{msg}[/bold cyan]", style="cyan"))

        elif event == "phase_done":
            if msg:
                console.print(f"  [dim]{msg}[/dim]")

        # ── Auto-setup: plan table ────────────────────────────────────────
        elif event == "setup_plan":
            commands = extra.get("commands", [])
            detected = msg  # detected_type passed as msg
            console.print(
                f"\n  Detected project type:  [bold]{detected}[/bold]"
            )
            console.print()
            table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            table.add_column("step", style="dim", width=4, justify="right")
            table.add_column("command", style="bold")
            for i, cmd in enumerate(commands, 1):
                table.add_row(str(i), cmd)
            console.print(table)

        # ── Auto-setup: step execution ────────────────────────────────────
        elif event == "setup_step":
            step = extra.get("step", "?")
            total = extra.get("total", "?")
            console.print(
                f"\n  [dim][{step}/{total}][/dim]  "
                f"[bold cyan]$[/bold cyan] [bold]{msg}[/bold]"
            )

        elif event == "setup_step_done":
            elapsed_ms = extra.get("elapsed_ms", 0)
            success = extra.get("success", False)
            t_str = (
                f"{elapsed_ms / 1000:.1f}s" if elapsed_ms > 1000
                else f"{elapsed_ms:.0f}ms"
            )
            output = msg.strip()
            preview = output.splitlines()[0][:100] if output else ""
            if success:
                console.print(
                    f"       [green]ok[/green]  [dim]{preview}[/dim]  "
                    f"[dim bright_black][{t_str}][/dim bright_black]"
                )
            else:
                console.print(
                    f"       [red]fail[/red]  [dim]{preview}[/dim]"
                )

        # ── Auto-setup: health check result ───────────────────────────────
        elif event == "health_result":
            port = extra.get("port")
            running = extra.get("running", False)
            console.print()
            if running:
                console.print(
                    f"  [bold green]RUNNING[/bold green]  "
                    f"port {port}  ->  http://localhost:{port}"
                )
            else:
                console.print(
                    f"  [bold red]NOT RUNNING[/bold red]  "
                    f"port {port} is not in use — app may not have started"
                )

        # ── Auto-setup: final summary table ──────────────────────────────
        elif event == "setup_summary":
            steps = extra.get("steps", [])
            port = extra.get("port")
            running = extra.get("running", False)

            console.print()
            console.print(Rule("[bold]Setup Summary[/bold]", style="cyan"))

            table = Table(box=box.SIMPLE, padding=(0, 2))
            table.add_column("#", style="dim", width=3, justify="right")
            table.add_column("Command", style="bold", min_width=30)
            table.add_column("Status", width=10)
            table.add_column("Time", justify="right", style="dim")
            for i, (cmd, ok, elapsed_ms, _) in enumerate(steps, 1):
                status = "[green]ok[/green]" if ok else "[red]fail[/red]"
                t_str = (
                    f"{elapsed_ms / 1000:.1f}s" if elapsed_ms > 1000
                    else f"{elapsed_ms:.0f}ms"
                )
                table.add_row(str(i), cmd, status, t_str)
            console.print(table)

            if port and running:
                console.print(
                    f"\n  [bold green]App is running[/bold green]  "
                    f"->  http://localhost:{port}"
                )
            elif port:
                console.print(
                    f"\n  [yellow]App did not start on port {port}[/yellow]  "
                    f"Check the output above for errors."
                )


# ─── Stats display ────────────────────────────────────────────────────────────

def show_stats():
    stats = get_routing_stats()
    table = Table(title="Routing Statistics", box=box.ROUNDED, border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    total = stats.get("total_calls", 0)
    local = stats.get("local_calls", 0)
    cloud = stats.get("cloud_calls", 0)
    edge_ratio = stats.get("edge_ratio", 0.0)

    table.add_row("Total calls", str(total))
    table.add_row("Local (on-device)", f"[green]{local}[/green]")
    table.add_row("Cloud (Gemini)", f"[blue]{cloud}[/blue]")
    table.add_row("Edge ratio", f"[bold]{edge_ratio * 100:.1f}%[/bold]")
    table.add_row("Avg local latency", f"{stats.get('avg_local_latency_ms', 0):.1f}ms")
    table.add_row("Avg cloud latency", f"{stats.get('avg_cloud_latency_ms', 0):.1f}ms")
    console.print(table)


# ─── Plan display ─────────────────────────────────────────────────────────────

# Human-readable descriptions for every tool, shown in `plan` output.
_TOOL_DESCRIPTIONS = {
    # Explorer
    "list_files":               "List files and folders at the given path",
    "read_file":                "Read and display the contents of a file",
    "find_pattern":             "Search for a text pattern across all files (like grep)",
    "show_tree":                "Show the directory tree structure",
    "check_disk_space":         "Check how much disk space is available",
    # Builder
    "create_file":              "Create a new file with the specified content",
    "edit_file":                "Find and replace text inside an existing file",
    "delete_file":              "Delete a file from the workspace",
    "create_dockerfile":        "Generate a production-ready Dockerfile (uses Gemini)",
    "create_docker_compose":    "Generate a docker-compose.yml for the requested services (uses Gemini)",
    "install_dependency":       "Install a package via pip, npm, or brew",
    "plan_setup_commands":      "Analyse the repo and generate the exact setup commands (uses Gemini)",
    # Runner
    "run_command":              "Execute a shell command directly",
    "check_port":               "Check whether a port is currently in use",
    "start_service":            "Start Docker services defined in a compose file",
    "stop_service":             "Stop Docker services",
    "read_logs":                "Read the last N lines from a container's logs",
    # Tester
    "check_health":             "Make an HTTP request and verify the response status",
    "validate_config":          "Parse a config file and check for syntax errors",
    "run_test":                 "Run the test suite and report pass / fail",
    "diagnose_error":           "Send an error to Gemini for diagnosis and a suggested fix",
    # Documentation
    "generate_run_instructions": "Generate a 'How to Run' guide for this project (uses Gemini)",
}

# Auto-setup phases shown in plain English.
_AUTO_SETUP_PHASES = [
    ("on-device", "Scan the repo — read config files, manifests, and any inline usage comments"),
    ("Gemini",    "Ask Gemini to figure out the exact setup commands for this project type"),
    ("on-device", "Run each command one by one, with automatic error recovery if anything fails"),
    ("on-device", "Check whether the app actually started on the expected port"),
    ("Gemini",    "Generate a clear 'How to Run' guide based on everything that was set up"),
]


def show_plan(user_text: str, orch: Orchestrator):
    plan = orch.plan(user_text)

    console.print()
    console.print(Panel(
        f"[bold]Here is what I will do for:[/bold]\n[dim]{user_text}[/dim]",
        border_style="yellow",
        padding=(0, 1),
    ))

    for agent_plan in plan["agents"]:
        agent = agent_plan["agent"]

        if agent == "auto_setup":
            console.print(
                "\n  I will run the [bold]Auto Setup[/bold] flow — "
                "a 5-step sequence to get this repo running:\n"
            )
            for i, (where, description) in enumerate(_AUTO_SETUP_PHASES, 1):
                where_color = "blue" if where == "Gemini" else "green"
                console.print(
                    f"  [dim]{i}.[/dim]  {description}\n"
                    f"       [dim]runs via[/dim] [{where_color}]{where}[/{where_color}]"
                )
        else:
            label = agent_plan["label"]
            tools = agent_plan["tools_available"]
            hints = agent_plan["routing_hints"]

            console.print(f"\n  I will use the [bold]{label}[/bold] agent:\n")
            for t in tools:
                route = hints.get(t, "local")
                description = _TOOL_DESCRIPTIONS.get(t, t)
                where = "Gemini" if route == "cloud" else "on-device"
                where_color = "blue" if route == "cloud" else "green"
                console.print(
                    f"  [dim]-[/dim]  {description}\n"
                    f"       [dim]runs via[/dim] [{where_color}]{where}[/{where_color}]"
                )


# ─── Voice input ──────────────────────────────────────────────────────────────

def do_voice_input() -> str:
    if not is_voice_available():
        status = voice_status()
        console.print("[red]Voice unavailable:[/red]")
        for k, v in status.items():
            color = "green" if v else "red"
            mark = "ok" if v else "fail"
            console.print(f"  [{color}]{mark}[/{color}]  {k}")
        return ""

    console.print("[bold yellow]Recording for 5 seconds — speak now[/bold yellow]")
    try:
        text = record_and_transcribe(duration=5.0)
        console.print(f"  [dim]transcribed:[/dim]  [bold]{text}[/bold]")
        return text
    except Exception as e:
        console.print(f"[red]Voice error:[/red]  {e}")
        return ""


# ─── Workspace reset ──────────────────────────────────────────────────────────

def do_reset():
    import shutil
    count = 0
    for item in get_workspace().iterdir():
        if item.name != ".gitkeep":
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            count += 1
    console.print(f"[green]workspace cleaned[/green]  ({count} item(s) removed)")


# ─── Help ─────────────────────────────────────────────────────────────────────

HELP_TEXT = """
[bold cyan]Voice DevOps Agent[/bold cyan]

  [green]<text>[/green]         Natural language command
                 Examples: "list files", "set up this app", "reinstall req.txt"

  [yellow]voice[/yellow]          Record 5 seconds of audio and transcribe
  [yellow]stats[/yellow]          Show local vs cloud routing statistics
  [yellow]plan <task>[/yellow]    Preview execution plan without running anything
  [yellow]reset[/yellow]          Clean the workspace directory
  [yellow]demo[/yellow]           Run the pre-scripted demo sequence
  [yellow]help[/yellow]           Show this message
  [yellow]quit[/yellow]           Exit (also generates run instructions)

[dim]Agents: EXPLORER  BUILDER  RUNNER  TESTER  AUTO SETUP[/dim]

[dim]Auto-setup triggers:
  "set up this app / repo / project"
  "get this project running"
  "install everything"
  "auto setup"[/dim]
"""


# ─── Header banner ────────────────────────────────────────────────────────────

def show_banner():
    banner = Text()
    banner.append("Voice", style="bold cyan")
    banner.append("DevOps", style="bold white")
    banner.append("Agent", style="bold green")
    banner.append("  —  FunctionGemma + Gemini Flash", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))

    vs = voice_status()
    input_label = "voice ready" if vs["voice_ready"] else "keyboard mode"
    console.print(
        f"  input: [bold]{input_label}[/bold]   "
        f"workspace: [dim]{get_workspace()}[/dim]\n"
    )


# ─── Stats bar (shown after each command) ────────────────────────────────────

def print_stats_bar():
    stats = get_routing_stats()
    total = stats.get("total_calls", 0)
    if total == 0:
        return
    local = stats.get("local_calls", 0)
    cloud = stats.get("cloud_calls", 0)
    edge = stats.get("edge_ratio", 0.0)
    console.print(
        f"\n  [dim]calls: {total}  "
        f"local: {local}  "
        f"cloud: {cloud}  "
        f"edge: {edge * 100:.0f}%[/dim]"
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_demo(orch: Orchestrator):
    from demo_script import DEMO_STEPS
    console.print(Panel("[bold]Running Demo Sequence[/bold]", border_style="magenta"))
    for step in DEMO_STEPS:
        console.print(f"\n  [bold magenta]demo:[/bold magenta]  {step}")
        time.sleep(0.5)
        run_command_text(step, orch)
        time.sleep(1.0)
    console.print("\n[bold green]Demo complete[/bold green]")


# ─── Command dispatcher ───────────────────────────────────────────────────────

def run_command_text(text: str, orch: Orchestrator):
    text = text.strip()
    if not text:
        return

    console.print()

    agents = classify_agents(text)
    for a in agents:
        style = AGENT_STYLES.get(a, "white")
        label = AGENT_LABELS.get(a, a.upper())
        console.print(f"  [{style}][{label}][/{style}]", highlight=False)

    results = orch.process(text)
    print_stats_bar()
    console.print()
    return results


# ─── Post-session documentation ──────────────────────────────────────────────

def _show_final_docs(orch: Orchestrator):
    """Run the documentation agent and display run instructions at session end."""
    console.print()
    console.print(Rule("[bold magenta]Documentation[/bold magenta]", style="magenta"))
    console.print("  [dim]Generating run instructions based on this session...[/dim]")

    instructions = orch.run_documentation_agent()

    if instructions:
        console.print(
            Panel(
                instructions,
                title="[bold magenta]How to Run This App[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )
    else:
        console.print(
            "  [dim]Nothing was set up in this session — no instructions to generate.[/dim]"
        )


# ─── Main REPL ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voice DevOps Agent")
    parser.add_argument(
        "--repo", metavar="PATH", default=None,
        help="Path to a repo to operate on (defaults to current directory if it looks like a repo)",
    )
    args, _ = parser.parse_known_args()

    if args.repo:
        set_workspace(args.repo)
        console.print(f"  [dim]workspace: {args.repo}[/dim]")
    else:
        # Auto-detect CWD if it looks like a real repo (not the built-in workspace/)
        cwd = pathlib.Path.cwd()
        builtin_ws = (pathlib.Path(__file__).parent / "workspace").resolve()
        if cwd.resolve() != builtin_ws:
            try:
                non_hidden = [f for f in cwd.iterdir() if not f.name.startswith(".")]
                if non_hidden:
                    set_workspace(str(cwd))
            except Exception:
                pass

    show_banner()
    progress = RichProgressHandler()
    orch = Orchestrator(progress_callback=progress)

    while True:
        try:
            raw = console.input("[bold cyan]>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]exiting[/dim]")
            _show_final_docs(orch)
            break

        if not raw:
            continue

        lower = raw.lower()

        if lower in ("quit", "exit", "q"):
            _show_final_docs(orch)
            break

        elif lower == "help":
            console.print(HELP_TEXT)

        elif lower == "stats":
            show_stats()

        elif lower == "reset":
            do_reset()

        elif lower == "voice":
            text = do_voice_input()
            if text:
                run_command_text(text, orch)

        elif lower == "demo":
            run_demo(orch)

        elif lower.startswith("plan "):
            task = raw[5:].strip()
            if task:
                show_plan(task, orch)
            else:
                console.print("  [dim]usage: plan <task description>[/dim]")

        else:
            run_command_text(raw, orch)


if __name__ == "__main__":
    main()
