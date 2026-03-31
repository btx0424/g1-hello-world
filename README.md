# g1-hello-world

Prototype viewer for a Unitree G1 robot model with MuJoCo, Viser, Unitree DDS state updates, and a RealSense color stream.

## Project layout

```text
.
├── cfg/                  # runtime configuration stubs
├── main.py               # compatibility entrypoint
├── robot_model/          # URDF/XML assets and meshes
├── src/g1_hello_world/
│   ├── app.py            # runtime orchestration
│   ├── cli.py            # command-line entrypoint
│   ├── constants.py      # shared geometry constants
│   ├── robot_model.py    # MuJoCo robot loading and queries
│   ├── timing.py         # callback frequency helper
│   └── visualization.py  # Viser scene adapters
└── pyproject.toml
```

## Environment

Use the **project-local** uv environment (`.venv` next to `pyproject.toml`):

```bash
cd g1-hello-world
uv sync
```

## Run

From **`g1-hello-world`** (so `uv` resolves this project and its `.venv`):

```bash
uv run app.py --iface eth0
```

Or use the helper script (changes into this directory first, then `uv run app.py`):

```bash
./run.sh --iface eth0
```

## Why this structure

- `main.py` stays tiny, so runtime code is importable and testable.
- `src/g1_hello_world/` separates orchestration from model and visualization logic.
- asset files stay at the repo root because they are large static inputs, not Python modules.
- paths are resolved from the package, so the app does not depend on the current working directory.

## Next cleanup targets

- move RealSense setup into its own adapter module if camera logic grows
- replace `cfg/default.yaml` with real runtime settings or remove `cfg/`
- add smoke tests around path resolution and CLI parsing

