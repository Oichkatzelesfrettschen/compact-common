# Installation Requirements (compact-common)

`compact-common` is a Python library submodule (`src/compact_common`) for compact
object physics utilities (EOS/spacetime/structure).

## Prerequisites

- Python `>=3.9`

## Install (dev)

```bash
cd compact-common
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

## Tests and gates

- Tests: `pytest`
- Strict gates (warnings-as-errors on the contract surface): `scripts/audit/run_tiers.sh`

