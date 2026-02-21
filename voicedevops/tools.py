"""
Tool definitions across all agents, with routing metadata.

Each tool dict is OpenAI/Cactus compatible (name, description, parameters).
The `_routing` key is our metadata — stripped before passing to models.
"""

# ─── EXPLORER AGENT (all LOCAL) ──────────────────────────────────────────────

TOOL_LIST_FILES = {
    "name": "list_files",
    "description": "List files and directories at the given path. Returns names, sizes, and types.",
    "parameters": {
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "Directory path to list (default: current workspace)"},
        },
        "required": ["directory"],
    },
    "_routing": "local",
    "_agent": "explorer",
}

TOOL_READ_FILE = {
    "name": "read_file",
    "description": "Read and return the contents of a file.",
    "parameters": {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to read"},
        },
        "required": ["filepath"],
    },
    "_routing": "local",
    "_agent": "explorer",
}

TOOL_FIND_PATTERN = {
    "name": "find_pattern",
    "description": "Search for a text pattern (like grep) across files in a directory. Returns matching lines with file paths.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Text pattern or regex to search for"},
            "directory": {"type": "string", "description": "Directory to search in"},
        },
        "required": ["pattern", "directory"],
    },
    "_routing": "local",
    "_agent": "explorer",
}

TOOL_SHOW_TREE = {
    "name": "show_tree",
    "description": "Show directory tree structure up to a given depth.",
    "parameters": {
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "Root directory for the tree"},
            "depth": {"type": "integer", "description": "Maximum depth to display (default: 3)"},
        },
        "required": ["directory"],
    },
    "_routing": "local",
    "_agent": "explorer",
}

TOOL_CHECK_DISK_SPACE = {
    "name": "check_disk_space",
    "description": "Check available disk space at a path.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Filesystem path to check"},
        },
        "required": ["path"],
    },
    "_routing": "local",
    "_agent": "explorer",
}

# ─── BUILDER AGENT (simple ops LOCAL, generation CLOUD) ──────────────────────

TOOL_CREATE_FILE = {
    "name": "create_file",
    "description": "Create a new file with the given content.",
    "parameters": {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to create"},
            "content": {"type": "string", "description": "Content to write into the file"},
        },
        "required": ["filepath", "content"],
    },
    "_routing": "local",
    "_agent": "builder",
}

TOOL_EDIT_FILE = {
    "name": "edit_file",
    "description": "Find and replace text in an existing file.",
    "parameters": {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to edit"},
            "old_text": {"type": "string", "description": "Text to find and replace"},
            "new_text": {"type": "string", "description": "Replacement text"},
        },
        "required": ["filepath", "old_text", "new_text"],
    },
    "_routing": "local",
    "_agent": "builder",
}

TOOL_DELETE_FILE = {
    "name": "delete_file",
    "description": "Delete a file at the given path.",
    "parameters": {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the file to delete"},
        },
        "required": ["filepath"],
    },
    "_routing": "local",
    "_agent": "builder",
}

TOOL_CREATE_DOCKERFILE = {
    "name": "create_dockerfile",
    "description": "Generate a Dockerfile for a service. Requires cloud reasoning about base image, dependencies, and ports.",
    "parameters": {
        "type": "object",
        "properties": {
            "service_name": {"type": "string", "description": "Name of the service (e.g. postgres, nginx, fastapi)"},
            "requirements": {"type": "string", "description": "Description of requirements, dependencies, and ports"},
        },
        "required": ["service_name", "requirements"],
    },
    "_routing": "cloud",
    "_agent": "builder",
}

TOOL_CREATE_DOCKER_COMPOSE = {
    "name": "create_docker_compose",
    "description": "Generate a docker-compose.yml with specified services. Requires cloud reasoning about networking, volumes, and ports.",
    "parameters": {
        "type": "object",
        "properties": {
            "services": {"type": "string", "description": "Comma-separated list of services to include (e.g. 'postgres, redis, nginx')"},
        },
        "required": ["services"],
    },
    "_routing": "cloud",
    "_agent": "builder",
}

TOOL_INSTALL_DEPENDENCY = {
    "name": "install_dependency",
    "description": "Install a package via pip, npm, or brew.",
    "parameters": {
        "type": "object",
        "properties": {
            "package_manager": {"type": "string", "description": "Package manager to use: pip, npm, or brew"},
            "package": {"type": "string", "description": "Package name to install"},
        },
        "required": ["package_manager", "package"],
    },
    "_routing": "local",
    "_agent": "builder",
}

TOOL_PLAN_SETUP_COMMANDS = {
    "name": "plan_setup_commands",
    "description": (
        "Analyze a software repository and return an ordered list of shell commands "
        "to fully set it up from scratch: install dependencies, configure environment, "
        "build if needed, and start the app."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "commands": {
                "type": "string",
                "description": (
                    "Ordered shell commands to run, one per line. "
                    "Example: 'npm install\\ncp .env.example .env\\nnpm start'"
                ),
            },
            "detected_type": {
                "type": "string",
                "description": "Detected project type: node, python, go, rust, java, etc.",
            },
        },
        "required": ["commands"],
    },
    "_routing": "cloud",
    "_agent": "builder",
}

# ─── RUNNER AGENT (all LOCAL) ─────────────────────────────────────────────────

TOOL_RUN_COMMAND = {
    "name": "run_command",
    "description": "Execute a shell command and return stdout, stderr, and exit code.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "working_dir": {"type": "string", "description": "Working directory for the command (defaults to workspace)"},
        },
        "required": ["command"],
    },
    "_routing": "local",
    "_agent": "runner",
}

TOOL_CHECK_PORT = {
    "name": "check_port",
    "description": "Check if a port is currently in use. Returns boolean and process info.",
    "parameters": {
        "type": "object",
        "properties": {
            "port": {"type": "integer", "description": "Port number to check"},
        },
        "required": ["port"],
    },
    "_routing": "local",
    "_agent": "runner",
}

TOOL_START_SERVICE = {
    "name": "start_service",
    "description": "Run docker-compose up -d to start services defined in a compose file.",
    "parameters": {
        "type": "object",
        "properties": {
            "compose_file": {"type": "string", "description": "Path to the docker-compose.yml file"},
        },
        "required": ["compose_file"],
    },
    "_routing": "local",
    "_agent": "runner",
}

TOOL_STOP_SERVICE = {
    "name": "stop_service",
    "description": "Run docker-compose down to stop services.",
    "parameters": {
        "type": "object",
        "properties": {
            "compose_file": {"type": "string", "description": "Path to the docker-compose.yml file"},
        },
        "required": ["compose_file"],
    },
    "_routing": "local",
    "_agent": "runner",
}

TOOL_READ_LOGS = {
    "name": "read_logs",
    "description": "Read the last N lines of a Docker container's logs.",
    "parameters": {
        "type": "object",
        "properties": {
            "service_name": {"type": "string", "description": "Name of the Docker service/container"},
            "lines": {"type": "integer", "description": "Number of log lines to retrieve (default: 50)"},
        },
        "required": ["service_name"],
    },
    "_routing": "local",
    "_agent": "runner",
}

# ─── TESTER AGENT (checks LOCAL, diagnosis CLOUD) ────────────────────────────

TOOL_CHECK_HEALTH = {
    "name": "check_health",
    "description": "Make an HTTP request to a URL and check if the status matches the expected value.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to health-check (e.g. http://localhost:5432)"},
            "expected_status": {"type": "integer", "description": "Expected HTTP status code (default: 200)"},
        },
        "required": ["url"],
    },
    "_routing": "local",
    "_agent": "tester",
}

TOOL_VALIDATE_CONFIG = {
    "name": "validate_config",
    "description": "Parse a config file (yaml, json, or toml) and check for syntax errors.",
    "parameters": {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the config file"},
            "format": {"type": "string", "description": "Config format: yaml, json, or toml"},
        },
        "required": ["filepath", "format"],
    },
    "_routing": "local",
    "_agent": "tester",
}

TOOL_RUN_TEST = {
    "name": "run_test",
    "description": "Execute a test script or command and parse results for pass/fail.",
    "parameters": {
        "type": "object",
        "properties": {
            "test_command": {"type": "string", "description": "Test command to run (e.g. pytest, npm test)"},
            "working_dir": {"type": "string", "description": "Working directory for the test"},
        },
        "required": ["test_command"],
    },
    "_routing": "local",
    "_agent": "tester",
}

TOOL_DIAGNOSE_ERROR = {
    "name": "diagnose_error",
    "description": "Analyze an error message with surrounding context and suggest a specific fix. Uses cloud AI for deep diagnosis.",
    "parameters": {
        "type": "object",
        "properties": {
            "error_message": {"type": "string", "description": "The error message or output to diagnose"},
            "context": {"type": "string", "description": "Additional context: what command was run, what was expected"},
        },
        "required": ["error_message", "context"],
    },
    "_routing": "cloud",
    "_agent": "tester",
}

# ─── Grouped by agent ─────────────────────────────────────────────────────────

EXPLORER_TOOLS = [
    TOOL_LIST_FILES,
    TOOL_READ_FILE,
    TOOL_FIND_PATTERN,
    TOOL_SHOW_TREE,
    TOOL_CHECK_DISK_SPACE,
]

BUILDER_TOOLS = [
    TOOL_CREATE_FILE,
    TOOL_EDIT_FILE,
    TOOL_DELETE_FILE,
    TOOL_CREATE_DOCKERFILE,
    TOOL_CREATE_DOCKER_COMPOSE,
    TOOL_INSTALL_DEPENDENCY,
    TOOL_PLAN_SETUP_COMMANDS,
]

RUNNER_TOOLS = [
    TOOL_RUN_COMMAND,
    TOOL_CHECK_PORT,
    TOOL_START_SERVICE,
    TOOL_STOP_SERVICE,
    TOOL_READ_LOGS,
]

TESTER_TOOLS = [
    TOOL_CHECK_HEALTH,
    TOOL_VALIDATE_CONFIG,
    TOOL_RUN_TEST,
    TOOL_DIAGNOSE_ERROR,
]

# ─── DOCUMENTATION AGENT (CLOUD) ─────────────────────────────────────────────

TOOL_GENERATE_RUN_INSTRUCTIONS = {
    "name": "generate_run_instructions",
    "description": (
        "Generate clear step-by-step instructions for running and accessing the app "
        "based on the workspace files and what was set up during the session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "context": {
                "type": "string",
                "description": (
                    "Summary of the session: services created, files written, "
                    "ports used, and docker-compose or Dockerfile contents"
                ),
            },
        },
        "required": ["context"],
    },
    "_routing": "cloud",
    "_agent": "documentation",
}

DOCUMENTATION_TOOLS = [TOOL_GENERATE_RUN_INSTRUCTIONS]

ALL_TOOLS = EXPLORER_TOOLS + BUILDER_TOOLS + RUNNER_TOOLS + TESTER_TOOLS + DOCUMENTATION_TOOLS

# Map name → definition for fast lookup
TOOL_MAP = {t["name"]: t for t in ALL_TOOLS}


def strip_routing_metadata(tools):
    """Return tool defs without the _routing/_agent keys (for passing to models)."""
    return [{k: v for k, v in t.items() if not k.startswith("_")} for t in tools]
