"""
Tool executors — real file/shell operations.

All shell commands are sandboxed to WORKSPACE_DIR.
Every executor returns a dict with at minimum:
  { "success": bool, "output": str, "error": str }
"""

import os
import json
import glob
import shutil
import subprocess
import pathlib
import urllib.request
import urllib.error

# ─── Workspace root (mutable — call set_workspace() to point at a real repo) ──

WORKSPACE_DIR = pathlib.Path(__file__).parent / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def set_workspace(path) -> None:
    """Point all tools at a new root directory (e.g. a repo passed via --repo)."""
    global WORKSPACE_DIR
    WORKSPACE_DIR = pathlib.Path(path).resolve()
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def get_workspace() -> pathlib.Path:
    """Return the current workspace / repo root directory."""
    return WORKSPACE_DIR


def _safe_path(path: str) -> pathlib.Path:
    """Resolve a path, forcing it inside WORKSPACE_DIR if relative or absolute outside."""
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = WORKSPACE_DIR / p
    try:
        p = p.resolve()
        # Allow paths that are inside workspace or are system paths (for docker, etc.)
        if not str(p).startswith(str(WORKSPACE_DIR.resolve())):
            # Redirect to workspace
            p = WORKSPACE_DIR / pathlib.Path(path).name
    except Exception:
        p = WORKSPACE_DIR / pathlib.Path(path).name
    return p


def _smart_find(filename: str) -> pathlib.Path:
    """
    Like _safe_path but falls back to a recursive search under WORKSPACE_DIR when
    the file isn't found at the expected location.  Useful when operating on a real
    repo where config files may live in subdirectories.
    """
    p = _safe_path(filename)
    if p.exists():
        return p
    # Only recurse for bare filenames (no explicit directory component)
    if pathlib.Path(filename).name == pathlib.Path(filename).as_posix():
        for match in sorted(WORKSPACE_DIR.rglob(filename)):
            rel = match.relative_to(WORKSPACE_DIR)
            if not any(part.startswith(".") for part in rel.parts):
                return match
    return p


def _result(success, output="", error="", **extra):
    return {"success": success, "output": str(output), "error": str(error), **extra}


# ─── EXPLORER TOOLS ───────────────────────────────────────────────────────────

def exec_list_files(directory: str) -> dict:
    try:
        p = _safe_path(directory)
        if not p.exists():
            return _result(False, error=f"Directory not found: {directory}")
        entries = []
        for item in sorted(p.iterdir()):
            stat = item.stat()
            entries.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": stat.st_size,
            })
        output = "\n".join(
            f"{'d' if e['type'] == 'dir' else 'f'}  {e['name']:40s}  {e['size']:>10} bytes"
            for e in entries
        )
        return _result(True, output=output or "(empty)", entries=entries)
    except Exception as e:
        return _result(False, error=str(e))


def exec_read_file(filepath: str) -> dict:
    try:
        p = _smart_find(filepath)
        if not p.exists():
            return _result(False, error=f"File not found: {filepath}")
        content = p.read_text(errors="replace")
        return _result(True, output=content)
    except Exception as e:
        return _result(False, error=str(e))


def exec_find_pattern(pattern: str, directory: str) -> dict:
    try:
        p = _safe_path(directory)
        if not p.exists():
            return _result(False, error=f"Directory not found: {directory}")
        matches = []
        for fpath in p.rglob("*"):
            if fpath.is_file():
                try:
                    for i, line in enumerate(fpath.read_text(errors="replace").splitlines(), 1):
                        if pattern.lower() in line.lower():
                            matches.append(f"{fpath}:{i}: {line.strip()}")
                except Exception:
                    pass
        output = "\n".join(matches) if matches else f"No matches for '{pattern}'"
        return _result(True, output=output, match_count=len(matches))
    except Exception as e:
        return _result(False, error=str(e))


def exec_show_tree(directory: str, depth: int = 3) -> dict:
    try:
        p = _safe_path(directory)
        if not p.exists():
            return _result(False, error=f"Directory not found: {directory}")

        lines = [str(p)]

        def _walk(path, prefix, current_depth):
            if current_depth > depth:
                return
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            for i, item in enumerate(items):
                connector = "└── " if i == len(items) - 1 else "├── "
                lines.append(f"{prefix}{connector}{item.name}")
                if item.is_dir() and current_depth < depth:
                    extension = "    " if i == len(items) - 1 else "│   "
                    _walk(item, prefix + extension, current_depth + 1)

        _walk(p, "", 1)
        return _result(True, output="\n".join(lines))
    except Exception as e:
        return _result(False, error=str(e))


def exec_check_disk_space(path: str) -> dict:
    try:
        p = _safe_path(path)
        usage = shutil.disk_usage(str(p))
        output = (
            f"Total:  {usage.total / 1e9:.2f} GB\n"
            f"Used:   {usage.used / 1e9:.2f} GB\n"
            f"Free:   {usage.free / 1e9:.2f} GB\n"
            f"Usage:  {usage.used / usage.total * 100:.1f}%"
        )
        return _result(True, output=output, free_bytes=usage.free, total_bytes=usage.total)
    except Exception as e:
        return _result(False, error=str(e))


# ─── BUILDER TOOLS ────────────────────────────────────────────────────────────

def exec_create_file(filepath: str, content: str) -> dict:
    try:
        p = _safe_path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return _result(True, output=f"Created {p}")
    except Exception as e:
        return _result(False, error=str(e))


def exec_edit_file(filepath: str, old_text: str, new_text: str) -> dict:
    try:
        p = _safe_path(filepath)
        if not p.exists():
            return _result(False, error=f"File not found: {filepath}")
        original = p.read_text()
        if old_text not in original:
            return _result(False, error=f"Text not found in {filepath}: {old_text[:80]!r}")
        updated = original.replace(old_text, new_text, 1)
        p.write_text(updated)
        return _result(True, output=f"Edited {p}: replaced {len(old_text)} chars")
    except Exception as e:
        return _result(False, error=str(e))


def exec_delete_file(filepath: str) -> dict:
    try:
        p = _safe_path(filepath)
        if not p.exists():
            return _result(False, error=f"File not found: {filepath}")
        p.unlink()
        return _result(True, output=f"Deleted {p}")
    except Exception as e:
        return _result(False, error=str(e))


def exec_create_dockerfile(service_name: str, requirements: str, content: str = "") -> dict:
    """Content is generated by cloud (Gemini) and passed in. Write it to workspace."""
    try:
        if not content:
            return _result(False, error="No Dockerfile content provided (expected from cloud generation)")
        filename = f"Dockerfile.{service_name}" if service_name else "Dockerfile"
        p = WORKSPACE_DIR / filename
        p.write_text(content)
        return _result(True, output=f"Wrote {p}\n\n{content}")
    except Exception as e:
        return _result(False, error=str(e))


def exec_create_docker_compose(services: str, content: str = "") -> dict:
    """Content is generated by cloud (Gemini) and passed in."""
    try:
        if not content:
            return _result(False, error="No docker-compose content provided (expected from cloud generation)")
        p = WORKSPACE_DIR / "docker-compose.yml"
        p.write_text(content)
        return _result(True, output=f"Wrote {p}\n\n{content}")
    except Exception as e:
        return _result(False, error=str(e))



def exec_install_dependency(package_manager: str, package: str) -> dict:
    pm = package_manager.lower().strip()
    if pm == "pip":
        cmd = ["pip", "install", package]
    elif pm == "npm":
        cmd = ["npm", "install", package]
    elif pm == "brew":
        cmd = ["brew", "install", package]
    else:
        return _result(False, error=f"Unknown package manager: {package_manager}")
    return _run_subprocess(cmd, cwd=str(WORKSPACE_DIR))


# ─── RUNNER TOOLS ─────────────────────────────────────────────────────────────

def _run_subprocess(cmd, cwd=None, timeout=30):
    cwd = cwd or str(WORKSPACE_DIR)
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = proc.returncode == 0
        return _result(
            success,
            output=proc.stdout,
            error=proc.stderr,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired:
        return _result(False, error=f"Command timed out after {timeout}s")
    except FileNotFoundError as e:
        return _result(False, error=f"Command not found: {e}")
    except Exception as e:
        return _result(False, error=str(e))


_SLOW_CMD_PREFIXES = (
    "pip", "npm", "yarn", "pnpm", "poetry",
    "docker", "docker-compose",
    "cargo", "go build", "go get",
    "mvn", "gradle", "./gradlew", "./mvnw",
    "bundle install", "gem install",
    "composer install",
)
_SLOW_TIMEOUT = 300  # 5 minutes for package installs / docker pulls


def exec_run_command(command: str, working_dir: str = "", timeout: int = 0) -> dict:
    cwd = str(_safe_path(working_dir)) if working_dir else str(WORKSPACE_DIR)
    if timeout <= 0:
        cmd_lower = command.lstrip().lower()
        timeout = _SLOW_TIMEOUT if any(cmd_lower.startswith(p) for p in _SLOW_CMD_PREFIXES) else 60
    return _run_subprocess(["bash", "-c", command], cwd=cwd, timeout=timeout)


def exec_check_port(port: int) -> dict:
    try:
        result = _run_subprocess(["lsof", "-i", f":{port}", "-P", "-n"])
        in_use = bool(result["output"].strip())
        output = result["output"] if in_use else f"Port {port} is free."
        return _result(True, output=output, in_use=in_use, port=port)
    except Exception as e:
        return _result(False, error=str(e))


def exec_start_service(compose_file: str) -> dict:
    p = _safe_path(compose_file)
    if not p.exists():
        return _result(False, error=f"Compose file not found: {compose_file}")
    return _run_subprocess(["docker-compose", "-f", str(p), "up", "-d"], cwd=str(p.parent))


def exec_stop_service(compose_file: str) -> dict:
    p = _safe_path(compose_file)
    if not p.exists():
        return _result(False, error=f"Compose file not found: {compose_file}")
    return _run_subprocess(["docker-compose", "-f", str(p), "down"], cwd=str(p.parent))


def exec_read_logs(service_name: str, lines: int = 50) -> dict:
    result = _run_subprocess(["docker", "logs", "--tail", str(lines), service_name])
    if not result["success"] and "No such container" in result.get("error", ""):
        # Try docker-compose project naming: workspace-<service>-1
        for candidate in [f"workspace-{service_name}-1", f"{service_name}-1", f"{service_name}_1"]:
            r2 = _run_subprocess(["docker", "logs", "--tail", str(lines), candidate])
            if r2["success"]:
                return r2
        # Last resort: docker-compose logs from workspace
        compose_file = WORKSPACE_DIR / "docker-compose.yml"
        if compose_file.exists():
            r3 = _run_subprocess(
                ["docker-compose", "-f", str(compose_file), "logs", "--tail", str(lines), service_name],
                cwd=str(WORKSPACE_DIR),
            )
            if r3["success"]:
                return r3
    return result


# ─── TESTER TOOLS ─────────────────────────────────────────────────────────────

def exec_check_health(url: str, expected_status: int = 200) -> dict:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            actual = resp.status
            body = resp.read(512).decode(errors="replace")
            ok = actual == expected_status
            return _result(
                ok,
                output=f"HTTP {actual} — {'OK' if ok else f'expected {expected_status}'}",
                status_code=actual,
                body_preview=body,
            )
    except urllib.error.HTTPError as e:
        ok = e.code == expected_status
        return _result(ok, output=f"HTTP {e.code}", status_code=e.code)
    except Exception as e:
        return _result(False, error=str(e))


def exec_validate_config(filepath: str, format: str) -> dict:
    try:
        p = _smart_find(filepath)
        if not p.exists():
            return _result(False, error=f"File not found: {filepath}")
        text = p.read_text()
        fmt = format.lower().strip()
        if fmt in ("yaml", "yml"):
            import yaml
            yaml.safe_load(text)
        elif fmt == "json":
            json.loads(text)
        elif fmt == "toml":
            try:
                import tomllib
                tomllib.loads(text)
            except ImportError:
                import tomli  # type: ignore
                tomli.loads(text)
        else:
            return _result(False, error=f"Unknown format: {format}. Use yaml, json, or toml.")
        return _result(True, output=f"{filepath} is valid {fmt.upper()}")
    except Exception as e:
        return _result(False, error=f"Parse error: {e}")


def exec_run_test(test_command: str, working_dir: str = "") -> dict:
    cwd = str(_safe_path(working_dir)) if working_dir else str(WORKSPACE_DIR)
    result = _run_subprocess(["bash", "-c", test_command], cwd=cwd)
    # Try to parse basic pass/fail from output
    output = result["output"] + result["error"]
    passed = result["success"]
    summary = "PASSED" if passed else "FAILED"
    result["output"] = f"[{summary}]\n{result['output']}"
    return result


def exec_diagnose_error(error_message: str, context: str) -> dict:
    """This tool is handled at the orchestrator level via cloud call; placeholder here."""
    return _result(
        False,
        error="diagnose_error must be routed through cloud — call generate_cloud directly",
    )


# ─── Dispatcher ───────────────────────────────────────────────────────────────

EXECUTORS = {
    "list_files": lambda args: exec_list_files(**args),
    "read_file": lambda args: exec_read_file(**args),
    "find_pattern": lambda args: exec_find_pattern(**args),
    "show_tree": lambda args: exec_show_tree(**args),
    "check_disk_space": lambda args: exec_check_disk_space(**args),
    "create_file": lambda args: exec_create_file(**args),
    "edit_file": lambda args: exec_edit_file(**args),
    "delete_file": lambda args: exec_delete_file(**args),
    "create_dockerfile": lambda args: exec_create_dockerfile(**args),
    "create_docker_compose": lambda args: exec_create_docker_compose(**args),
    "install_dependency": lambda args: exec_install_dependency(**args),
    "run_command": lambda args: exec_run_command(**args),
    "check_port": lambda args: exec_check_port(**args),
    "start_service": lambda args: exec_start_service(**args),
    "stop_service": lambda args: exec_stop_service(**args),
    "read_logs": lambda args: exec_read_logs(**args),
    "check_health": lambda args: exec_check_health(**args),
    "validate_config": lambda args: exec_validate_config(**args),
    "run_test": lambda args: exec_run_test(**args),
    "diagnose_error": lambda args: exec_diagnose_error(**args),
}


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Dispatch a tool call to its executor. Returns standardized result dict."""
    if tool_name not in EXECUTORS:
        return _result(False, error=f"Unknown tool: {tool_name}")
    try:
        return EXECUTORS[tool_name](arguments)
    except TypeError as e:
        return _result(False, error=f"Invalid arguments for {tool_name}: {e}")
    except Exception as e:
        return _result(False, error=f"Executor error for {tool_name}: {e}")
