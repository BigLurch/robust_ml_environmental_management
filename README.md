## Miljösetup - Robust och reproducerbar ML-miljö

Detta projekt använder `uv` för att skapa en isolerad och reproducerbar utvecklingsmiljö för maskininlärning, enligt labbens krav.

### Python-version

Projektet är låst till **Python 3.12**, då detta är en stabil version med fullt stöd i det moderna ML-ekosystemet (PyTorch, Jupyter, scikit-learn m.fl.).

```text
requires-python = ">=3.12,<3.13"
```

projektets Python-version har pinnats med:

```bash
uv python pin 3.12
```

### Beroendehantering

Alla beroenden installeras via `uv` och deklareras i `pyproject.toml`. Exakta versioner låses i `uv.lock` för full reproducerbarhet.

Installerade huvudberoenden:

- pandas - datahantering
- scikit-learn - klassisk maskininlärning
- notebook - Jupyter Notebook-miljö
- torch / torchvision / torchaudio - PyTorch (CPU-version)

### PyTorch-installation (CPU-only)

Datorn saknar dedikerad GPU (CUDA/MPS). Därför installerades CPU-versionen av PyTorch med korrekt index.

För att hantera konflikter mellan PyPI och PyTorchs egna wheels användes en explicit index-strategi:

```bash
uv add torch torchvision torchaudio --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

Detta säkerställer att:

- PyTorch hämtas från PyTorchs officiella CPU-index
- övriga beroenden (t.ex. `requests`) hämtas från PyPI i kompatibla versioner

### Verifiering av installation

PyTorch verifierades via:

```bash
uv run python -c "import torch; print(torch.__version__); print('CUDA', torch.cuda.is_available())"
```

Resultar:

```text
2.10.0+cpu
CUDA False
```

Detta bekräftar att:

- PyTorch är korrekt installerat
- ingen GPU/accelerator finns tillgänglig
- miljön kör korrekt i CPU-läge

### Verifieringsscript (`check_env.py`)

För att verifiera att miljön är korrekt installerad och fungerar som avsett används ett verifieringsscript: `check_env.py`.

Den virtuella miljön (`.venv`) är inte inkluderad i repot.  
Miljön återskapas lokalt genom:

```bash
uv sync
```

Scriptet är avsett att köras med:

```bash
uv run check_env.py
```
