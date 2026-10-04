"""Point-in-time NHL forecasting, pricing and evaluation (no legacy model imports)."""
from pathlib import Path

VERSION = 'nhl-v2.2'
FEATURE_SCHEMA = 'nhl-pit-2'
# Evidence folder for each model version. Code for one version never writes into another
# version's folder, so published results are not relabeled as evidence for a later model.
EVIDENCE = {'nhl-v2.1': 'reports/nhl-rebuild', 'nhl-v2.2': 'reports/nhl-v2.2'}
_ROOT = Path(__file__).resolve().parents[3]


def evidence_dir(path=None):
    """This version's report folder by default; refuses a folder that holds another version's evidence."""
    path = Path(path) if path else _ROOT / EVIDENCE[VERSION]
    for version, folder in EVIDENCE.items():
        if version != VERSION and path.resolve() == (_ROOT / folder).resolve():
            raise ValueError(f'{folder} holds {version} evidence; write {VERSION} results to {EVIDENCE[VERSION]}')
    return path
