"""
Pre-scripted demo sequence that showcases:
1. Explorer: list files in workspace
2. Builder: create a docker-compose.yml (cloud) + a config file (local)
3. Runner: start service (will fail — port conflict or missing docker)
4. Self-heal loop: error → cloud diagnosis → fix → retry
5. Tester: verify health
6. Documentation: generate step-by-step run instructions

Run with: python demo_script.py
Or trigger via CLI: demo
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(2, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))

DEMO_STEPS = [
    # Step 1: Explore workspace
    "show me the files in the workspace",

    # Step 2: Build docker-compose for postgres
    "create a docker-compose for postgres",

    # Step 3: Validate the config
    "validate docker-compose.yml config",

    # Step 4: Try to start the service (may trigger self-heal)
    "start the service with docker-compose.yml",

    # Step 5: Check postgres health
    "check if postgres is healthy on port 5432",

    # Step 6: Read logs
    "read the last 20 lines of postgres logs",
]


def run_demo_standalone(repo_path=None):
    """Run demo as standalone script with rich UI."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule

    console = Console()

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

        for i, step in enumerate(DEMO_STEPS, 1):
            console.print(Rule(f"Step {i}/{len(DEMO_STEPS)}: {step}", style="magenta"))
            time.sleep(0.3)
            run_command_text(step, orch)
            print_stats_bar()
            time.sleep(0.8)

        console.print("\n")
        console.print(Panel(
            "[bold green]✅ Demo Complete![/bold green]\n"
            "The self-heal loop caught failures and automatically applied fixes.",
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
            f"⚡Local: {stats.get('local_calls', 0)}  "
            f"☁Cloud: {stats.get('cloud_calls', 0)}  "
            f"Edge ratio: {stats.get('edge_ratio', 0)*100:.0f}%"
        )

        # Step 7: Documentation agent — generate run instructions
        console.print("\n")
        console.print(Rule(f"Step {len(DEMO_STEPS) + 1}: Documentation Agent", style="magenta"))
        console.print("  [dim]Generating run instructions based on the session...[/dim]")
        time.sleep(0.3)

        instructions = orch.run_documentation_agent()
        if instructions:
            console.print(Panel(
                instructions,
                title="[bold magenta]📖 How to Run Your App[/bold magenta]",
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
    parser = argparse.ArgumentParser(description="Voice DevOps Agent — Self-Heal Demo")
    parser.add_argument(
        "--repo", metavar="PATH", default=None,
        help="Path to a repo to operate on (defaults to built-in workspace/)",
    )
    args = parser.parse_args()
    run_demo_standalone(repo_path=args.repo)
