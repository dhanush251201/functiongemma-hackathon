# VoiceDevOps

Voice-controlled DevOps agent that clones any GitHub repo and auto-sets it up -- using FunctionGemma (on-device) + Gemini Flash (cloud) with self-healing error recovery.

## What It Does

1. **Clone any repo** -- paste a URL or speak it
2. **Auto-detect project type** -- scans package.json, requirements.txt, go.mod, Cargo.toml, etc.
3. **Generate setup commands** -- Gemini Flash analyzes the repo and plans the exact install/build/run steps
4. **Execute with self-healing** -- runs each command; if one fails (e.g. a pip package won't build), it automatically diagnoses and retries with a fix
5. **Health check** -- detects the expected port and verifies the app is actually running
6. **Q&A** -- ask any question about the project ("what does this app do?", "what port?") and get an answer
7. **Generate docs** -- produces a "How to Run" guide based on what was set up

Everything is controlled via natural language -- type commands or use voice input (5-second recordings via Whisper on-device).

## Architecture

```
+-------------------------------------------------------+
|                  User (voice / text)                   |
+-------------------------------------------------------+
                          |
+-------------------------------------------------------+
|    classify_agents()  --  regex-based intent match     |
+-------------------------------------------------------+
         |          |          |          |          |
  +----------+ +--------+ +--------+ +--------+ +------+
  | Explorer | | Builder| | Runner | | Tester | |  Q&A |
  +----------+ +--------+ +--------+ +--------+ +------+
         |          |          |          |          |
         +----------+-----+----+----------+          |
                          |                          |
              +-----------+----------+    +----------+----------+
              |  generate_hybrid()   |    | _generate_text_via  |
              +----------------------+    |      _gemini()      |
                |                |        +---------------------+
  +-------------+--+  +--+-------------+
  | FunctionGemma  |  |  Gemini Flash  |
  |  (on-device)   |  |    (cloud)     |
  +----------------+  +----------------+
```

### Routing Strategy

Every tool call goes through a 3-tier routing pipeline:

1. **Text fallback** (regex, 0ms) -- common DevOps commands matched without any model call
2. **FunctionGemma on-device** (~200ms) -- 270M parameter function-calling model via Cactus
3. **Gemini Flash cloud** (~500ms) -- fallback for complex reasoning (setup planning, Dockerfile generation, error diagnosis)

Tools that require deep reasoning (e.g. `plan_setup_commands`, `create_dockerfile`, `diagnose_error`) are marked `_routing: "cloud"` and always go to Gemini. Simple operations (file I/O, shell commands, port checks) stay on-device.

### Auto-Setup Pipeline (5 Phases)

| Phase | What Happens | Runs Via |
|-------|-------------|----------|
| 1. Discovery | Scan config files, manifests, README, source code comments | On-device |
| 2. Planning | Ask Gemini to generate exact setup commands for this project type | Cloud |
| 3. Execution | Run each command with self-heal loop (diagnose + fix + retry on failure) | On-device + Cloud |
| 4. Health Check | Detect expected port from configs, check if app is listening | On-device |
| 5. Documentation | Generate a "How to Run" guide for the project | Cloud |

### Self-Healing

When a command fails during setup:

1. **Pip-specific fix** (no cloud call) -- if `pip install` fails on a specific package, retry without version pin or skip it and install the rest
2. **Gemini diagnosis** -- send the error + context to Gemini, get a specific fix (port change, file edit, alternative command)
3. **Apply and retry** -- extract the fix from Gemini's response, apply it, re-run the command

### Q&A Mode

Questions (detected by regex: starts with "what", "how", "why", etc.) bypass tool-calling entirely. The system gathers project context (file listings, config contents, session history) and sends it to Gemini for a natural language answer.

## Project Structure

```
voicedevops/
  cli.py              -- Terminal UI (cyberpunk-styled Rich output, REPL loop)
  orchestrator.py      -- Agent routing, auto-setup pipeline, self-heal, Q&A
  executor.py          -- Tool implementations (file I/O, shell, Docker, git clone)
  tools.py             -- Tool definitions with routing metadata (_routing: local/cloud)
  voice_pipeline.py    -- Sox recording + Whisper transcription via Cactus
  demo_script.py       -- Pre-scripted demo sequence
  .env                 -- API keys (not committed)
  workspace/           -- Default working directory for cloned repos
main.py                -- Hybrid routing: text_fallback -> FunctionGemma -> Gemini Flash
benchmark.py           -- Local eval suite (30 cases across easy/medium/hard)
```

## Setup

### Prerequisites

- macOS with Apple Silicon (for Cactus/FunctionGemma on-device inference)
- Python 3.10+
- [Cactus](https://github.com/cactus-compute/cactus) installed and built with `--python`
- FunctionGemma model downloaded: `cactus download google/functiongemma-270m-it --reconvert`
- Whisper model (for voice): `cactus download whisper-small`
- Sox (for voice recording): `brew install sox`
- Gemini API key from [Google AI Studio](https://aistudio.google.com/api-keys)

### Install

```bash
git clone https://github.com/cactus-compute/cactus
cd cactus && source ./setup && cd ..
cactus build --python
cactus download google/functiongemma-270m-it --reconvert

pip install google-genai rich
export GEMINI_API_KEY="your-key"
```

Or create `voicedevops/.env`:
```
GEMINI_API_KEY=your-key
CACTUS_API_KEY=your-key
```

### Run

```bash
# Clone any repo, then point the agent at it:
git clone https://github.com/some-user/some-app
cd some-app
python /path/to/voicedevops/cli.py

# Or use --repo flag:
python voicedevops/cli.py --repo /path/to/some-app
```

## Usage

```
VOICEDEVOPS > set up this app          # auto-detect + install + run
VOICEDEVOPS > what does this app do?   # Q&A about the project
VOICEDEVOPS > how do I run this?       # natural language answer
VOICEDEVOPS > list files               # explorer agent
VOICEDEVOPS > run npm test             # runner agent
VOICEDEVOPS > check port 8000          # tester agent
VOICEDEVOPS > voice                    # 5-second voice recording
VOICEDEVOPS > docs                     # generate project documentation
VOICEDEVOPS > plan set up this app     # preview without executing
VOICEDEVOPS > stats                    # on-device vs cloud routing stats
VOICEDEVOPS > help                     # full command reference
```

### Agents

| Agent | Tools | Routing |
|-------|-------|---------|
| Explorer | list_files, read_file, find_pattern, show_tree, check_disk_space | All on-device |
| Builder | create_file, edit_file, delete_file, create_dockerfile, create_docker_compose, install_dependency, plan_setup_commands | Simple ops on-device, generation via cloud |
| Runner | run_command, check_port, start_service, stop_service, read_logs | All on-device |
| Tester | check_health, validate_config, run_test, diagnose_error | Checks on-device, diagnosis via cloud |
| Documentation | generate_run_instructions | Cloud |
| Auto Setup | 5-phase pipeline (discovery, planning, execution, health, docs) | Hybrid |
| Q&A | Free-form Gemini text generation with project context | Cloud |

## Tech Stack

- **FunctionGemma** (270M) -- on-device function calling via Cactus native bindings
- **Gemini 2.0 Flash** -- cloud fallback for complex reasoning
- **Whisper** (via Cactus) -- on-device speech-to-text
- **Rich** -- terminal UI (panels, tables, trees, spinners, progress bars)
- **Sox** -- audio recording
