"""
Pre-scripted demo sequences for the Voice DevOps Agent.

Two modes:
  1. clone_demo  — Clone a repo from GitHub, auto-setup, self-heal, verify
  2. local_demo  — Original workspace demo (Docker/Postgres)

Run with: python demo_script.py [--mode clone|local] [--repo URL_OR_PATH]
Or trigger via CLI: demo
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(2, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))

# Default demo repo (FastAPI todo app included in this project)
DEFAULT_CLONE_URL = "https://github.com/renceInbox/fastapi-todo"

# ─── Clone demo: git clone → auto-setup → verify ─────────────────────────────

CLONE_DEMO_STEPS = [
    # Step 1: Clone the repo (the URL is injected at runtime)
    "clone {repo_url} and set it up",
]

CLONE_FOLLOWUP_STEPS = [
    # These run after clone+setup finishes
    "show me the files",
    "read the requirements.txt",
]

# ─── Local demo: workspace-based Docker flow ─────────────────────────────────

LOCAL_DEMO_STEPS = [
    "show me the files in the workspace",
    "create a docker-compose for postgres",
    "validate docker-compose.yml config",
    "start the service with docker-compose.yml",
    "check if postgres is healthy on port 5432",
    "read the last 20 lines of postgres logs",
]


def run_demo_standalone(repo_path=None, mode="clone", repo_url=None):
    """Run demo as standalone script with rich UI."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule

    console = Console()

    if mode == "clone":
        url = repo_url or DEFAULT_CLONE_URL
        subtitle = f"[dim]Cloning: {url}[/dim]"
        console.print(Panel(
            f"[bold magenta]Voice DevOps Agent — Clone + Auto-Setup Demo[/bold magenta]\n{subtitle}",
            border_style="magenta",
        ))
    else:
        if repo_path:
            import executor
            executor.set_workspace(repo_path)
            subtitle = f"[dim]Repo: {repo_path}[/dim]"
        else:
            subtitle = "[dim]Scenario: Set up PostgreSQL with Docker[/dim]"
        console.print(Panel(
            f"[bold magenta]Voice DevOps Agent — Self-Heal Demo[/bold magenta]\n{subtitle}",
            border_style="magenta",
        ))

    try:
        from cli import RichProgressHandler, run_command_text, print_stats_bar
        from orchestrator import Orchestrator

        progress = RichProgressHandler()
        orch = Orchestrator(progress_callback=progress)

        if mode == "clone":
            url = repo_url or DEFAULT_CLONE_URL
            steps = [s.format(repo_url=url) for s in CLONE_DEMO_STEPS]
            steps += CLONE_FOLLOWUP_STEPS
        else:
            steps = LOCAL_DEMO_STEPS

        for i, step in enumerate(steps, 1):
            console.print(Rule(f"Step {i}/{len(steps)}: {step}", style="magenta"))
            time.sleep(0.3)
            run_command_text(step, orch)
            print_stats_bar()
            time.sleep(0.8)

        console.print("\n")
        console.print(Panel(
            "[bold green]Demo Complete![/bold green]\n"
            "The agent cloned, detected the project type, installed dependencies,\n"
            "and self-healed any failures automatically.",
            border_style="green",
        ))

        # Show final stats
        try:
            from main import get_routing_stats
        except ImportError:
            def get_routing_stats():
                return {}
        stats = get_routing_stats()
        console.print(
            f"\n[bold]Final Stats:[/bold]  "
            f"Total: {stats.get('total_calls', 0)}  "
            f"Local: {stats.get('local_calls', 0)}  "
            f"Cloud: {stats.get('cloud_calls', 0)}  "
            f"Edge ratio: {stats.get('edge_ratio', 0)*100:.0f}%"
        )

        # Documentation agent — generate run instructions
        console.print("\n")
        console.print(Rule(f"Step {len(steps) + 1}: Documentation Agent", style="magenta"))
        console.print("  [dim]Generating run instructions based on the session...[/dim]")
        time.sleep(0.3)

        instructions = orch.run_documentation_agent()
        if instructions:
            console.print(Panel(
                instructions,
                title="[bold magenta]How to Run Your App[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            ))
        else:
            console.print("  [dim]No instructions generated (workspace may be empty).[/dim]")

    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        console.print("[dim]Make sure rich is installed: pip install rich[/dim]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voice DevOps Agent — Demo")
    parser.add_argument(
        "--mode", choices=["clone", "local"], default="clone",
        help="Demo mode: 'clone' (default) clones a repo and sets up; 'local' uses workspace",
    )
    parser.add_argument(
        "--repo", metavar="URL_OR_PATH", default=None,
        help="Git URL to clone (clone mode) or local path (local mode)",
    )
    args = parser.parse_args()

    if args.mode == "clone":
        run_demo_standalone(mode="clone", repo_url=args.repo)
    else:
        run_demo_standalone(repo_path=args.repo, mode="local")
