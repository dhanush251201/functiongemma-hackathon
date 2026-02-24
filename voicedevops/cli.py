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
from rich.tree import Tree
from rich.align import Align
from rich.style import Style
from rich.markdown import Markdown
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

# ─── Cyberpunk Color Palette ─────────────────────────────────────────────────

C_CYAN    = "#00fff2"
C_MAGENTA = "#ff00ff"
C_PINK    = "#ff0055"
C_GREEN   = "#00ff88"
C_YELLOW  = "#ffdd00"
C_ORANGE  = "#ff8800"
C_BLUE    = "#0088ff"
C_RED     = "#ff2244"
C_DIM     = "#555577"
C_WHITE   = "#eeeeff"

# Gradient arrays for banner
_GRADIENT_BANNER = [
    "#ff0055", "#ff0077", "#ff0099", "#ff00bb",
    "#ff00dd", "#ff00ff", "#dd00ff", "#bb00ff",
    "#9900ff", "#7700ff", "#5500ff", "#3300ff",
    "#0033ff", "#0055ff", "#0077ff", "#0099ff",
    "#00bbff", "#00ddff", "#00fff2",
]

_GRADIENT_PHASE = [
    "#00fff2", "#00ddff", "#00bbff", "#0099ff",
    "#0077ff", "#0055ff", "#5500ff", "#9900ff",
    "#bb00ff", "#ff00ff",
]


def _gradient_text(text: str, colors: list) -> Text:
    """Apply a diagonal color gradient to a string."""
    result = Text()
    n = len(colors)
    for i, ch in enumerate(text):
        if ch == "\n":
            result.append("\n")
        elif ch == " ":
            result.append(" ")
        else:
            result.append(ch, style=Style(color=colors[i % n], bold=True))
    return result


def _badge_simple(label: str, color: str) -> str:
    """Return a simpler badge with brackets."""
    return f"[bold {color}][{label}][/bold {color}]"


# ─── Agent label styles ───────────────────────────────────────────────────────

AGENT_STYLES = {
    "explorer":      C_CYAN,
    "builder":       C_YELLOW,
    "runner":        C_GREEN,
    "tester":        C_BLUE,
    "documentation": C_MAGENTA,
    "auto_setup":    C_WHITE,
    "clone_setup":   C_GREEN,
    "chat":          C_MAGENTA,
}

AGENT_LABELS = {
    "explorer":      "EXPLORER",
    "builder":       "BUILDER",
    "runner":        "RUNNER",
    "tester":        "TESTER",
    "documentation": "DOCS",
    "auto_setup":    "AUTO SETUP",
    "clone_setup":   "CLONE+SETUP",
    "chat":          "Q&A",
}


# ─── ASCII Art Banner ─────────────────────────────────────────────────────────

_BANNER_ART = r"""
 ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗
 ██║   ██║██╔═══██╗██║██╔════╝██╔════╝
 ██║   ██║██║   ██║██║██║     █████╗
 ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝
  ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗
   ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝
 ██████╗ ███████╗██╗   ██╗ ██████╗ ██████╗ ███████╗
 ██╔══██╗██╔════╝██║   ██║██╔═══██╗██╔══██╗██╔════╝
 ██║  ██║█████╗  ██║   ██║██║   ██║██████╔╝███████╗
 ██║  ██║██╔══╝  ╚██╗ ██╔╝██║   ██║██╔═══╝ ╚════██║
 ██████╔╝███████╗ ╚████╔╝ ╚██████╔╝██║     ███████║
 ╚═════╝ ╚══════╝  ╚═══╝   ╚═════╝ ╚═╝     ╚══════╝
"""


def show_banner():
    banner_text = _gradient_text(_BANNER_ART, _GRADIENT_BANNER)

    subtitle = Text()
    subtitle.append("  FunctionGemma ", style=f"bold {C_CYAN}")
    subtitle.append("+ ", style=f"dim {C_DIM}")
    subtitle.append("Gemini Flash  ", style=f"bold {C_MAGENTA}")

    inner = Text()
    inner.append_text(banner_text)
    inner.append("\n")
    inner.append_text(subtitle)

    console.print(Panel(
        Align.center(inner),
        border_style=Style(color=C_CYAN),
        box=box.DOUBLE,
        padding=(1, 4),
    ))

    # Status badges row
    vs = voice_status()
    mode_badge = (
        _badge_simple("MIC READY", C_GREEN)
        if vs.get("voice_ready")
        else _badge_simple("KEYBOARD", C_YELLOW)
    )
    ws_path = str(get_workspace())
    if len(ws_path) > 50:
        ws_path = "..." + ws_path[-47:]

    console.print(
        f"  {mode_badge}  "
        f"[{C_DIM}]workspace:[/{C_DIM}] [{C_CYAN}]{ws_path}[/{C_CYAN}]"
    )
    console.print()


# ─── Progress callback for Orchestrator ──────────────────────────────────────

class RichProgressHandler:
    """
    Translates orchestrator events into cyberpunk-styled terminal output.
    """

    def __init__(self):
        self._heal_count = 0
        self._step_count = 0
        self._step_total = 0

    def __call__(self, event: str, tool: str = "", msg: str = "", **extra):

        # ── Routing source ────────────────────────────────────────────────
        if event == "routed_local":
            console.print(
                f"  {_badge_simple('ON-DEVICE', C_GREEN)}  [{C_DIM}]{msg}[/{C_DIM}]"
            )

        elif event == "routed_cloud":
            console.print(
                f"  {_badge_simple('CLOUD', C_BLUE)}  [{C_DIM}]{msg}[/{C_DIM}]"
            )

        # ── Standard agent lifecycle ──────────────────────────────────────
        elif event == "plan":
            console.print(f"  [{C_DIM}]plan  {msg}[/{C_DIM}]")

        elif event == "agent_start":
            phase_text = _gradient_text(f"  {msg}  ", _GRADIENT_PHASE)
            console.print()
            console.print(Rule(phase_text, style=Style(color=C_CYAN)))

        elif event == "agent_done":
            if msg:
                console.print(f"  [{C_DIM}]{msg}[/{C_DIM}]")

        # ── Tool execution ────────────────────────────────────────────────
        elif event == "start":
            console.print(
                f"  {_badge_simple('RUN', C_YELLOW)}  "
                f"[bold {C_WHITE}]{tool}[/bold {C_WHITE}]  "
                f"[{C_DIM}]{msg}[/{C_DIM}]"
            )

        elif event == "success":
            lines = msg.splitlines()
            preview = lines[0][:120] if lines else ""
            console.print(
                f"  {_badge_simple('OK', C_GREEN)}   {preview}"
            )
            for line in lines[1:3]:
                console.print(f"         [{C_DIM}]{line[:120]}[/{C_DIM}]")

        elif event == "error":
            console.print(
                f"  {_badge_simple('FAIL', C_RED)}  {msg[:200]}"
            )

        # ── Self-heal ─────────────────────────────────────────────────────
        elif event == "healing":
            self._heal_count += 1
            console.print()
            console.print(
                f"  {_badge_simple('SELF-HEAL', C_ORANGE)}  "
                f"[{C_DIM}]sending error to Gemini for diagnosis...[/{C_DIM}]"
            )

        elif event == "diagnosis":
            console.print(
                Panel(
                    msg[:400],
                    title=f"[bold {C_BLUE}]DIAGNOSIS[/bold {C_BLUE}]",
                    border_style=Style(color=C_BLUE),
                    expand=False,
                    padding=(0, 2),
                )
            )

        elif event == "fix_applied":
            console.print(
                f"  {_badge_simple('FIX APPLIED', C_GREEN)}  "
                f"[{C_DIM}]{msg[:150]}[/{C_DIM}]"
            )

        elif event == "no_fix":
            console.print(
                f"  {_badge_simple('NO FIX', C_RED)}  [{C_DIM}]{msg[:150]}[/{C_DIM}]"
            )

        elif event == "no_calls":
            console.print(
                f"  [{C_DIM}]no tool calls generated for this input[/{C_DIM}]"
            )

        # ── Auto-setup: phase headers ─────────────────────────────────────
        elif event == "phase_start":
            console.print()
            phase_title = _gradient_text(f"  {msg}  ", _GRADIENT_PHASE)
            console.print(Panel(
                Align.center(phase_title),
                border_style=Style(color=C_CYAN),
                box=box.HEAVY,
                padding=(0, 2),
            ))

        elif event == "phase_done":
            if msg:
                console.print(f"  [{C_DIM}]{msg}[/{C_DIM}]")

        # ── Auto-setup: plan table ────────────────────────────────────────
        elif event == "setup_plan":
            commands = extra.get("commands", [])
            detected = msg
            self._step_total = len(commands)

            console.print()
            console.print(
                f"  [{C_DIM}]detected:[/{C_DIM}]  "
                f"[bold {C_CYAN}]{detected}[/bold {C_CYAN}]"
            )
            console.print()

            tree = Tree(
                f"[bold {C_YELLOW}]Setup Commands[/bold {C_YELLOW}]",
                guide_style=Style(color=C_DIM),
            )
            for i, cmd in enumerate(commands, 1):
                tree.add(f"[{C_DIM}]{i}.[/{C_DIM}] [bold {C_WHITE}]{cmd}[/bold {C_WHITE}]")
            console.print(tree)
            console.print()

        # ── Auto-setup: step execution ────────────────────────────────────
        elif event == "setup_step":
            step = extra.get("step", "?")
            total = extra.get("total", "?")
            self._step_count = int(step) if isinstance(step, (int, str)) and str(step).isdigit() else 0

            # Progress bar
            if isinstance(step, int) and isinstance(total, int) and total > 0:
                filled = int((step / total) * 20)
                bar = f"[{C_GREEN}]{'█' * filled}[/{C_GREEN}][{C_DIM}]{'░' * (20 - filled)}[/{C_DIM}]"
                console.print(f"\n  {bar}  [{C_DIM}]{step}/{total}[/{C_DIM}]")

            console.print(
                f"  {_badge_simple('EXEC', C_CYAN)}  "
                f"[bold {C_WHITE}]$ {msg}[/bold {C_WHITE}]"
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
                    f"         {_badge_simple('OK', C_GREEN)}  "
                    f"[{C_DIM}]{preview}[/{C_DIM}]  "
                    f"[{C_DIM}]{t_str}[/{C_DIM}]"
                )
            else:
                console.print(
                    f"         {_badge_simple('FAIL', C_RED)}  "
                    f"[{C_DIM}]{preview}[/{C_DIM}]"
                )

        # ── Auto-setup: health check result ───────────────────────────────
        elif event == "health_result":
            port = extra.get("port")
            running = extra.get("running", False)
            console.print()
            if running:
                console.print(Panel(
                    f"[bold {C_GREEN}]APP IS LIVE[/bold {C_GREEN}]  "
                    f"[{C_WHITE}]http://localhost:{port}[/{C_WHITE}]",
                    border_style=Style(color=C_GREEN),
                    box=box.DOUBLE,
                    padding=(0, 2),
                ))
            else:
                console.print(
                    f"  {_badge_simple('NOT RUNNING', C_RED)}  "
                    f"[{C_DIM}]port {port} is not in use[/{C_DIM}]"
                )

        # ── Auto-setup: final summary table ──────────────────────────────
        elif event == "setup_summary":
            steps = extra.get("steps", [])
            port = extra.get("port")
            running = extra.get("running", False)

            console.print()
            title_text = _gradient_text("  SETUP COMPLETE  ", _GRADIENT_BANNER)
            console.print(Rule(title_text, style=Style(color=C_CYAN)))
            console.print()

            table = Table(
                box=box.DOUBLE,
                border_style=Style(color=C_CYAN),
                header_style=Style(color=C_CYAN, bold=True),
                padding=(0, 1),
            )
            table.add_column("#", style=f"dim {C_DIM}", width=3, justify="right")
            table.add_column("Command", style=f"bold {C_WHITE}", min_width=30)
            table.add_column("Status", width=10, justify="center")
            table.add_column("Time", justify="right", style=C_DIM)

            total_ms = 0
            pass_count = 0
            for i, (cmd, ok, elapsed_ms, _) in enumerate(steps, 1):
                total_ms += elapsed_ms
                if ok:
                    pass_count += 1
                status = (
                    f"[bold {C_GREEN}]PASS[/bold {C_GREEN}]"
                    if ok
                    else f"[bold {C_RED}]FAIL[/bold {C_RED}]"
                )
                t_str = (
                    f"{elapsed_ms / 1000:.1f}s" if elapsed_ms > 1000
                    else f"{elapsed_ms:.0f}ms"
                )
                table.add_row(str(i), cmd, status, t_str)

            console.print(table)

            # Summary stats row
            total_str = f"{total_ms / 1000:.1f}s" if total_ms > 1000 else f"{total_ms:.0f}ms"
            console.print(
                f"\n  [{C_DIM}]steps:[/{C_DIM}] [bold {C_WHITE}]{pass_count}/{len(steps)} passed[/bold {C_WHITE}]"
                f"   [{C_DIM}]total time:[/{C_DIM}] [bold {C_WHITE}]{total_str}[/bold {C_WHITE}]"
            )

            if port and running:
                console.print()
                console.print(Panel(
                    Align.center(Text.from_markup(
                        f"[bold {C_GREEN}]APP IS LIVE[/bold {C_GREEN}]\n"
                        f"[{C_WHITE}]http://localhost:{port}[/{C_WHITE}]"
                    )),
                    border_style=Style(color=C_GREEN),
                    box=box.DOUBLE,
                    padding=(1, 4),
                ))
            elif port:
                console.print(
                    f"\n  [{C_YELLOW}]App did not start on port {port}[/{C_YELLOW}]  "
                    f"[{C_DIM}]Check output above for errors.[/{C_DIM}]"
                )


# ─── Stats display ────────────────────────────────────────────────────────────

def show_stats():
    stats = get_routing_stats()
    total = stats.get("total_calls", 0)
    local = stats.get("local_calls", 0)
    cloud = stats.get("cloud_calls", 0)
    edge_ratio = stats.get("edge_ratio", 0.0)

    console.print()

    # Visual ratio bar
    if total > 0:
        local_blocks = round(edge_ratio * 30)
        cloud_blocks = 30 - local_blocks
        bar = (
            f"[bold {C_GREEN}]{'█' * local_blocks}[/bold {C_GREEN}]"
            f"[bold {C_BLUE}]{'█' * cloud_blocks}[/bold {C_BLUE}]"
        )
        console.print(f"  {bar}")
        console.print(
            f"  [{C_GREEN}]on-device[/{C_GREEN}] {local_blocks * 100 // 30}%"
            f"     [{C_BLUE}]cloud[/{C_BLUE}] {cloud_blocks * 100 // 30}%"
        )
        console.print()

    table = Table(
        title=f"[bold {C_CYAN}]Routing Statistics[/bold {C_CYAN}]",
        box=box.DOUBLE,
        border_style=Style(color=C_CYAN),
        header_style=Style(color=C_CYAN, bold=True),
        padding=(0, 2),
    )
    table.add_column("Metric", style=f"bold {C_WHITE}")
    table.add_column("Value", justify="right")

    table.add_row("Total calls", f"[bold {C_WHITE}]{total}[/bold {C_WHITE}]")
    table.add_row("On-device", f"[bold {C_GREEN}]{local}[/bold {C_GREEN}]")
    table.add_row("Cloud (Gemini)", f"[bold {C_BLUE}]{cloud}[/bold {C_BLUE}]")
    table.add_row("Edge ratio", f"[bold {C_MAGENTA}]{edge_ratio * 100:.1f}%[/bold {C_MAGENTA}]")
    table.add_row(
        "Avg local latency",
        f"[{C_GREEN}]{stats.get('avg_local_latency_ms', 0):.1f}ms[/{C_GREEN}]",
    )
    table.add_row(
        "Avg cloud latency",
        f"[{C_BLUE}]{stats.get('avg_cloud_latency_ms', 0):.1f}ms[/{C_BLUE}]",
    )
    console.print(table)


# ─── Plan display ─────────────────────────────────────────────────────────────

_TOOL_DESCRIPTIONS = {
    "list_files":               "List files and folders at the given path",
    "read_file":                "Read and display the contents of a file",
    "find_pattern":             "Search for a text pattern across all files (like grep)",
    "show_tree":                "Show the directory tree structure",
    "check_disk_space":         "Check how much disk space is available",
    "create_file":              "Create a new file with the specified content",
    "edit_file":                "Find and replace text inside an existing file",
    "delete_file":              "Delete a file from the workspace",
    "create_dockerfile":        "Generate a production-ready Dockerfile (uses Gemini)",
    "create_docker_compose":    "Generate a docker-compose.yml for the requested services (uses Gemini)",
    "install_dependency":       "Install a package via pip, npm, or brew",
    "plan_setup_commands":      "Analyse the repo and generate the exact setup commands (uses Gemini)",
    "run_command":              "Execute a shell command directly",
    "check_port":               "Check whether a port is currently in use",
    "start_service":            "Start Docker services defined in a compose file",
    "stop_service":             "Stop Docker services",
    "read_logs":                "Read the last N lines from a container's logs",
    "check_health":             "Make an HTTP request and verify the response status",
    "validate_config":          "Parse a config file and check for syntax errors",
    "run_test":                 "Run the test suite and report pass / fail",
    "diagnose_error":           "Send an error to Gemini for diagnosis and a suggested fix",
    "generate_run_instructions": "Generate a 'How to Run' guide for this project (uses Gemini)",
}

_AUTO_SETUP_PHASES = [
    ("on-device", "Scan the repo — read config files, manifests, and any inline usage comments"),
    ("Gemini",    "Ask Gemini to figure out the exact setup commands for this project type"),
    ("on-device", "Run each command one by one, with automatic error recovery if anything fails"),
    ("on-device", "Check whether the app actually started on the expected port"),
    ("Gemini",    "Generate a clear 'How to Run' guide based on everything that was set up"),
]

_CLONE_SETUP_PHASES = [
    ("on-device", "Clone the git repository into the workspace"),
] + _AUTO_SETUP_PHASES


def show_plan(user_text: str, orch: Orchestrator):
    plan = orch.plan(user_text)

    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold {C_CYAN}]Execution Plan[/bold {C_CYAN}]\n"
            f"[{C_DIM}]{user_text}[/{C_DIM}]"
        ),
        border_style=Style(color=C_YELLOW),
        box=box.DOUBLE,
        padding=(0, 2),
    ))

    for agent_plan in plan["agents"]:
        agent = agent_plan["agent"]

        if agent in ("auto_setup", "clone_setup"):
            phases = _CLONE_SETUP_PHASES if agent == "clone_setup" else _AUTO_SETUP_PHASES
            label = "Clone + Setup" if agent == "clone_setup" else "Auto Setup"

            tree = Tree(
                f"[bold {C_MAGENTA}]{label} Pipeline[/bold {C_MAGENTA}]",
                guide_style=Style(color=C_DIM),
            )
            for i, (where, description) in enumerate(phases, 1):
                if where == "Gemini":
                    badge = _badge_simple("GEMINI", C_BLUE)
                else:
                    badge = _badge_simple("ON-DEVICE", C_GREEN)
                tree.add(
                    f"[{C_DIM}]{i}.[/{C_DIM}]  {badge}  "
                    f"[{C_WHITE}]{description}[/{C_WHITE}]"
                )
            console.print()
            console.print(tree)
        else:
            label = agent_plan["label"]
            tools = agent_plan["tools_available"]
            hints = agent_plan["routing_hints"]

            tree = Tree(
                f"[bold {AGENT_STYLES.get(agent, C_WHITE)}]{label}[/bold {AGENT_STYLES.get(agent, C_WHITE)}]",
                guide_style=Style(color=C_DIM),
            )
            for t in tools:
                route = hints.get(t, "local")
                description = _TOOL_DESCRIPTIONS.get(t, t)
                if route == "cloud":
                    badge = _badge_simple("GEMINI", C_BLUE)
                else:
                    badge = _badge_simple("ON-DEVICE", C_GREEN)
                tree.add(f"{badge}  [{C_WHITE}]{description}[/{C_WHITE}]")
            console.print()
            console.print(tree)

    console.print()


# ─── Voice input ──────────────────────────────────────────────────────────────

def do_voice_input() -> str:
    if not is_voice_available():
        status = voice_status()
        console.print(f"  {_badge_simple('MIC ERROR', C_RED)}  Voice unavailable")
        for k, v in status.items():
            icon = f"[{C_GREEN}]●[/{C_GREEN}]" if v else f"[{C_RED}]●[/{C_RED}]"
            console.print(f"    {icon}  {k}")
        return ""

    console.print()
    with console.status(
        f"[bold {C_PINK}]  Recording — speak now  [/bold {C_PINK}]",
        spinner="dots12",
        spinner_style=Style(color=C_PINK),
    ):
        try:
            text = record_and_transcribe(duration=5.0)
        except Exception as e:
            console.print(f"  {_badge_simple('ERROR', C_RED)}  {e}")
            return ""

    console.print(
        f"  {_badge_simple('HEARD', C_CYAN)}  [bold {C_WHITE}]{text}[/bold {C_WHITE}]"
    )
    return text


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
    console.print(
        f"  {_badge_simple('RESET', C_GREEN)}  "
        f"[{C_DIM}]{count} item(s) removed[/{C_DIM}]"
    )


# ─── Help ─────────────────────────────────────────────────────────────────────

def show_help():
    console.print()

    # Commands table
    table = Table(
        title=f"[bold {C_CYAN}]Commands[/bold {C_CYAN}]",
        box=box.DOUBLE,
        border_style=Style(color=C_CYAN),
        header_style=Style(color=C_CYAN, bold=True),
        padding=(0, 2),
        show_header=False,
    )
    table.add_column("Command", style=f"bold {C_YELLOW}", min_width=16)
    table.add_column("Description", style=C_WHITE)

    table.add_row("<text>", "Natural language command — just type what you want")
    table.add_row("<question>", "Ask anything about the project — answered by Gemini")
    table.add_row("voice", "Record 5 seconds of audio and transcribe")
    table.add_row("docs", "Generate full project documentation on demand")
    table.add_row("stats", "Show on-device vs cloud routing statistics")
    table.add_row("plan <task>", "Preview execution plan without running")
    table.add_row("reset", "Clean the workspace directory")
    table.add_row("demo", "Run the clone + auto-setup demo")
    table.add_row("help", "Show this message")
    table.add_row("quit", "Exit (generates run instructions)")
    console.print(table)
    console.print()

    # Agent badges
    agents_text = Text("  Agents:  ")
    for name, label in AGENT_LABELS.items():
        if name == "chat":
            continue
        color = AGENT_STYLES.get(name, C_WHITE)
        agents_text.append(f" [{label}] ", style=Style(color=color, bold=True))
        agents_text.append(" ", style="default")
    console.print(agents_text)
    console.print()

    # Example triggers
    console.print(f"  [{C_DIM}]Questions:       \"what does this app do?\"  |  \"how do I run it?\"  |  \"what port?\"[/{C_DIM}]")
    console.print(f"  [{C_DIM}]Clone triggers:  \"clone https://... and set it up\"  |  \"clone a repo\"[/{C_DIM}]")
    console.print(f"  [{C_DIM}]Setup triggers:  \"set up this app\"  |  \"install everything\"  |  \"auto setup\"[/{C_DIM}]")
    console.print()


# ─── Stats bar (shown after each command) ────────────────────────────────────

def print_stats_bar():
    stats = get_routing_stats()
    total = stats.get("total_calls", 0)
    if total == 0:
        return
    local = stats.get("local_calls", 0)
    cloud = stats.get("cloud_calls", 0)
    edge = stats.get("edge_ratio", 0.0)

    # Mini bar
    local_blocks = round(edge * 10)
    cloud_blocks = 10 - local_blocks
    bar = f"[{C_GREEN}]{'█' * local_blocks}[/{C_GREEN}][{C_BLUE}]{'█' * cloud_blocks}[/{C_BLUE}]"

    console.print(
        f"\n  {bar}  "
        f"[{C_DIM}]calls:{total}  local:{local}  cloud:{cloud}  "
        f"edge:{edge * 100:.0f}%[/{C_DIM}]"
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_demo(orch: Orchestrator):
    from demo_script import CLONE_DEMO_STEPS, CLONE_FOLLOWUP_STEPS, DEFAULT_CLONE_URL

    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold {C_MAGENTA}]Clone + Auto-Setup Demo[/bold {C_MAGENTA}]\n"
            f"[{C_DIM}]Repo: {DEFAULT_CLONE_URL}[/{C_DIM}]"
        ),
        border_style=Style(color=C_MAGENTA),
        box=box.DOUBLE,
        padding=(0, 2),
    ))

    steps = [s.format(repo_url=DEFAULT_CLONE_URL) for s in CLONE_DEMO_STEPS]
    steps += CLONE_FOLLOWUP_STEPS

    for i, step in enumerate(steps, 1):
        console.print(
            f"\n  {_badge_simple(f'DEMO {i}/{len(steps)}', C_MAGENTA)}  "
            f"[bold {C_WHITE}]{step}[/bold {C_WHITE}]"
        )
        time.sleep(0.5)
        run_command_text(step, orch)
        time.sleep(1.0)

    console.print()
    console.print(Panel(
        Align.center(Text.from_markup(
            f"[bold {C_GREEN}]DEMO COMPLETE[/bold {C_GREEN}]"
        )),
        border_style=Style(color=C_GREEN),
        box=box.DOUBLE,
        padding=(0, 4),
    ))


# ─── Command dispatcher ───────────────────────────────────────────────────────

def _prompt_for_url(prompt_text: str) -> str:
    """Interactive URL prompt — user can type/paste a URL or say 'voice' to speak it."""
    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold {C_YELLOW}]{prompt_text}[/bold {C_YELLOW}]\n"
            f"[{C_DIM}]Paste a URL below, or type 'voice' to speak it[/{C_DIM}]"
        ),
        border_style=Style(color=C_YELLOW),
        padding=(0, 2),
    ))

    try:
        raw = console.input(f"  [bold {C_CYAN}]URL>[/bold {C_CYAN}] ").strip()
    except (KeyboardInterrupt, EOFError):
        return ""

    if raw.lower() == "voice":
        with console.status(
            f"[bold {C_PINK}]  Listening for URL — speak now  [/bold {C_PINK}]",
            spinner="dots12",
            spinner_style=Style(color=C_PINK),
        ):
            try:
                raw = record_and_transcribe(duration=5.0)
            except Exception as e:
                console.print(f"  {_badge_simple('ERROR', C_RED)}  {e}")
                return ""
        console.print(
            f"  {_badge_simple('HEARD', C_CYAN)}  [bold {C_WHITE}]{raw}[/bold {C_WHITE}]"
        )

    return raw


def run_command_text(text: str, orch: Orchestrator):
    text = text.strip()
    if not text:
        return

    console.print()

    agents = classify_agents(text)
    agent_badges = Text("  ")
    for a in agents:
        color = AGENT_STYLES.get(a, C_WHITE)
        label = AGENT_LABELS.get(a, a.upper())
        agent_badges.append(f" [{label}] ", style=Style(color=color, bold=True))
        agent_badges.append(" ", style="default")
    console.print(agent_badges)

    results = orch.process(text, url_prompt_fn=_prompt_for_url)

    # Q&A mode returns a string instead of list[StepResult]
    if isinstance(results, str):
        console.print()
        console.print(Panel(
            Markdown(results),
            border_style=Style(color=C_CYAN),
            box=box.ROUNDED,
            padding=(1, 2),
        ))
    else:
        print_stats_bar()

    console.print()
    return results


# ─── Post-session documentation ──────────────────────────────────────────────

def _show_final_docs(orch: Orchestrator):
    """Run the documentation agent and display run instructions at session end."""
    console.print()
    console.print(Rule(
        _gradient_text("  Documentation  ", _GRADIENT_PHASE),
        style=Style(color=C_MAGENTA),
    ))

    with console.status(
        f"[bold {C_MAGENTA}]  Generating run instructions...  [/bold {C_MAGENTA}]",
        spinner="dots12",
        spinner_style=Style(color=C_MAGENTA),
    ):
        instructions = orch.run_documentation_agent()

    if instructions:
        console.print(Panel(
            Markdown(instructions),
            title=f"[bold {C_MAGENTA}]How to Run This App[/bold {C_MAGENTA}]",
            border_style=Style(color=C_MAGENTA),
            box=box.DOUBLE,
            padding=(1, 2),
        ))
    else:
        console.print(
            f"  [{C_DIM}]Nothing was set up — no instructions to generate.[/{C_DIM}]"
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
        console.print(f"  [{C_DIM}]workspace: {args.repo}[/{C_DIM}]")
    else:
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
            prompt = Text()
            prompt.append("VOICEDEVOPS", style=Style(color=C_CYAN, bold=True))
            prompt.append(" > ", style=Style(color=C_GREEN, bold=True))
            raw = console.input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n  [{C_DIM}]shutting down...[/{C_DIM}]")
            _show_final_docs(orch)
            break

        if not raw:
            continue

        lower = raw.lower()

        if lower in ("quit", "exit", "q"):
            _show_final_docs(orch)
            break

        elif lower == "help":
            show_help()

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

        elif lower == "docs":
            _show_final_docs(orch)

        elif lower.startswith("plan "):
            task = raw[5:].strip()
            if task:
                show_plan(task, orch)
            else:
                console.print(f"  [{C_DIM}]usage: plan <task description>[/{C_DIM}]")

        else:
            run_command_text(raw, orch)


if __name__ == "__main__":
    main()
