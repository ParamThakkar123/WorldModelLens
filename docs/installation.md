# Installation and Local Docs Build

## Install docs dependencies

```bash
pip install -e .[docs]
```

## Build docs locally

From the repository root:

```bash
make docs
```

Windows PowerShell alternative:

```powershell
python -m sphinx -b html docs docs/_build/html
```

## Clean generated docs

```bash
make docs-clean
```
