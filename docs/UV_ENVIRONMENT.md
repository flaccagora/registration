# UV Environment Notes

No `uv` commands were run while creating this prototype. This file documents an
optional future workflow only.

If the project later standardizes on `uv`, keep the same separation of source,
external submodules, checkpoints, and generated outputs:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Model-specific dependencies should still be installed from the upstream
submodule documentation:

```bash
cd external/vggt-omega
uv pip install -r requirements.txt
uv pip install -e .
```

```bash
cd external/Medical-SAM3
uv pip install -r requirements.txt
uv pip install -e ".[train]"
```

Do not commit `.venv/`, `uv.lock` churn from unrelated experiments, model
checkpoints, or generated outputs unless the team explicitly decides to track a
lockfile.

