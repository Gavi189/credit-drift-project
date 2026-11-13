"""Executa as Etapas 1 a 3 em sequencia (contrato -> drift -> dashboard)."""

import subprocess
import sys
from pathlib import Path

SCRIPTS = ["01_validate_and_train.py", "02_detect_drift.py", "03_build_dashboard.py"]


def main() -> None:
    """Roda cada script e interrompe se algum falhar."""
    here = Path(__file__).parent
    for name in SCRIPTS:
        print(f"\n>>> {name}")
        result = subprocess.run([sys.executable, str(here / name)], check=False)
        if result.returncode != 0:
            sys.exit(result.returncode)
    print("\nPipeline completo. Abra reports/dashboard.html")


if __name__ == "__main__":
    main()
