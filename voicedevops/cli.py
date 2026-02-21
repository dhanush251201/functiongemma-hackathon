"""
Rich terminal UI for the Voice DevOps Agent.

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
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.rule import Rule
from rich.columns import Columns
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

# ─── Colour / style constants ─────────────────────────────────────────────────

AGENT_STYLES = {
    "explorer":      ("cyan", "🔍"),
    "builder":       ("yellow", "🔨"),
    "runner":        ("green", "▶️"),
    "tester":        ("blue", "✅"),
    "documentation": ("magenta", "📖"),
}

# ─── Progress callback for Orchestrator ──────────────────────────────────────

class RichProgressHandler:
    def __init__(self):
        self._heal_count = 0

    def __call__(self, event: str, tool: str = "", msg: str = "", **extra):
        if event == "plan":
            console.print(f"  [dim]Plan:[/dim] {msg}")

        elif event == "agent_start":
            console.print(Rule(msg, style="bold white"))

        elif event == "agent_done":
            console.print(f"  [dim]{msg}[/dim]")

        elif event == "routed_local":
            latency = extra.get("latency", 0)
            lat_str = f"{latency:.2f}ms" if latency < 1 else f"{latency:.0f}ms"
            console.print(f"  [bold green]⚡ LOCAL[/bold green] [dim]{lat_str} — {msg}[/dim]")

        elif event == "routed_cloud":
            latency = extra.get("latency", 0)
            lat_str = f"{latency:.2f}ms" if latency < 1 else f"{latency:.0f}ms"
            console.print(f"  [bold blue]☁️  CLOUD[/bold blue] [dim]{lat_str} — {msg}[/dim]")

        elif event == "start":
            console.print(f"  [white]→[/white] [bold]{tool}[/bold]  [dim]{msg}[/dim]")

        elif event == "success":
            lines = msg.splitlines()
            preview = lines[0][:120] if lines else ""
            console.print(f"  [green]✓[/green] {preview}")
            if len(lines) > 1:
                for line in lines[1:4]:
                    console.print(f"    [dim]{line[:120]}[/dim]")

        elif event == "error":
            console.print(f"  [red]✗ Error:[/red] {msg[:200]}")

        elif event == "healing":
            self._heal_count += 1
            console.print(f"  [yellow]🔄 Self-healing...[/yellow] [dim]sending to Gemini for diagnosis[/dim]")

        elif event == "diagnosis":
            console.print(
                Panel(msg[:400], title="[blue]☁️  Diagnosis[/blue]", border_style="blue", expand=False)
            )

        elif event == "fix_applied":
            console.print(f"  [yellow]🔨 Applying fix:[/yellow] {msg[:150]}")

        elif event == "no_fix":
            console.print(f"  [red]⚠ Could not auto-fix:[/red] {msg[:150]}")

        elif event == "no_calls":
            console.print(f"  [dim]No tool calls generated for this input.[/dim]")


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
    table.add_row("Edge ratio", f"[bold]{edge_ratio*100:.1f}%[/bold]")
    table.add_row("Avg local latency", f"{stats.get('avg_local_latency_ms', 0):.1f}ms")
    table.add_row("Avg cloud latency", f"{stats.get('avg_cloud_latency_ms', 0):.1f}ms")
    console.print(table)


# ─── Plan display ─────────────────────────────────────────────────────────────

def show_plan(user_text: str, orch: Orchestrator):
    plan = orch.plan(user_text)
    console.print(Panel(f"[bold]Plan for:[/bold] {user_text}", border_style="yellow"))
    for i, agent_plan in enumerate(plan["agents"], 1):
        emoji = agent_plan["emoji"]
        label = agent_plan["label"]
        tools = agent_plan["tools_available"]
        hints = agent_plan["routing_hints"]
        console.print(f"  [bold]{i}. {emoji} {label} Agent[/bold]")
        for t in tools:
            route = hints.get(t, "local")
            color = "green" if route == "local" else "blue"
            console.print(f"     [{color}]{route.upper():6s}[/{color}]  {t}")


# ─── Voice input ──────────────────────────────────────────────────────────────

def do_voice_input() -> str:
    if not is_voice_available():
        status = voice_status()
        console.print("[red]Voice unavailable:[/red]")
        for k, v in status.items():
            icon = "✓" if v else "✗"
            color = "green" if v else "red"
            console.print(f"  [{color}]{icon}[/{color}] {k}")
        return ""

    console.print("[bold yellow]🎤 Recording for 5 seconds... speak now![/bold yellow]")
    try:
        text = record_and_transcribe(duration=5.0)
        console.print(f"[dim]Transcribed:[/dim] [bold]{text}[/bold]")
        return text
    except Exception as e:
        console.print(f"[red]Voice error:[/red] {e}")
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
    console.print(f"[green]✓ Workspace cleaned[/green] ({count} items removed)")


# ─── Help ─────────────────────────────────────────────────────────────────────

HELP_TEXT = """
[bold cyan]Voice DevOps Agent[/bold cyan] — Commands:

  [green]<text>[/green]       Natural language command  (e.g. "list the files", "set up postgres")
  [yellow]voice[/yellow]       Record 5 seconds of audio and transcribe
  [yellow]stats[/yellow]       Show local vs cloud routing statistics
  [yellow]plan <task>[/yellow] Preview execution plan without running
  [yellow]reset[/yellow]       Clean the workspace directory
  [yellow]demo[/yellow]        Run the pre-scripted self-heal demo
  [yellow]help[/yellow]        Show this message
  [yellow]quit[/yellow]        Exit

[dim]Agents: 🔍 Explorer  🔨 Builder  ▶️ Runner  ✅ Tester[/dim]
"""

# ─── Header banner ────────────────────────────────────────────────────────────

def show_banner():
    banner = Text()
    banner.append("  Voice", style="bold cyan")
    banner.append("DevOps", style="bold white")
    banner.append("Agent", style="bold green")
    banner.append("  ⚡ FunctionGemma + ☁️  Gemini Flash", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))

    vs = voice_status()
    voice_icon = "🎤" if vs["voice_ready"] else "⌨️ "
    voice_label = "Voice ready" if vs["voice_ready"] else "Keyboard mode"
    console.print(
        f"  {voice_icon} [bold]{voice_label}[/bold]   "
        f"[dim]workspace: {get_workspace()}[/dim]\n"
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
        f"  [dim]calls: {total}  ⚡local: {local}  ☁cloud: {cloud}  edge: {edge*100:.0f}%[/dim]"
    )


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_demo(orch: Orchestrator):
    from demo_script import DEMO_STEPS
    console.print(Panel("[bold]Running Demo Sequence[/bold]", border_style="magenta"))
    for step in DEMO_STEPS:
        console.print(f"\n[bold magenta]▶ Demo:[/bold magenta] {step}")
        time.sleep(0.5)
        run_command_text(step, orch)
        time.sleep(1.0)
    console.print("\n[bold green]✅ Demo complete![/bold green]")


# ─── Command dispatcher ───────────────────────────────────────────────────────

def run_command_text(text: str, orch: Orchestrator):
    text = text.strip()
    if not text:
        return

    console.print()
    agents = classify_agents(text)
    for a in agents:
        style, emoji = AGENT_STYLES.get(a, ("white", "🤖"))
        console.print(
            f"[{style}]{emoji} [{style.upper()}] {AGENTS[a]['label']} Agent[/{style}]",
            highlight=False,
        )

    results = orch.process(text)
    print_stats_bar()
    console.print()
    return results


# ─── Post-session documentation ──────────────────────────────────────────────

def _show_final_docs(orch: Orchestrator):
    """Run the documentation agent and display run instructions after the session ends."""
    console.print(Rule("[bold magenta]📖 Documentation Agent[/bold magenta]", style="magenta"))
    console.print("  [dim]Generating run instructions based on this session...[/dim]")

    instructions = orch.run_documentation_agent()

    if instructions:
        console.print(
            Panel(
                instructions,
                title="[bold magenta]📖 How to Run Your App[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )
    else:
        console.print("  [dim]Nothing was set up in this session — no instructions to generate.[/dim]")


# ─── Main REPL ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voice DevOps Agent")
    parser.add_argument(
        "--repo", metavar="PATH", default=None,
        help="Path to a repo to operate on (defaults to built-in workspace/)",
    )
    args, _ = parser.parse_known_args()
    if args.repo:
        set_workspace(args.repo)
        console.print(f"[dim]Repo: {args.repo}[/dim]")

    show_banner()
    progress = RichProgressHandler()
    orch = Orchestrator(progress_callback=progress)

    while True:
        try:
            raw = console.input("[bold cyan]>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            _show_final_docs(orch)
            break

        if not raw:
            continue

        lower = raw.lower()

        if lower in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
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
                console.print("[dim]Usage: plan <task description>[/dim]")

        else:
            run_command_text(raw, orch)


if __name__ == "__main__":
    main()
