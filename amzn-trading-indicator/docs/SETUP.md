# Setup (Windows PowerShell)
1. Extract the scaffold.
2. Copy existing CSVs into `data/raw/` (do not delete the originals yet).
3. Create an environment: `py -3.14 -m venv .venv`, then activate it with `\.venv\Scripts\Activate.ps1`.
4. Install dependencies: `python -m pip install -r requirements.txt`. If a dependency does not support Python 3.14, use a supported Python version in a separate environment.
5. Copy `.env.example` to `.env` and add Alpaca credentials. Never commit `.env`.
6. Keep prepared/split CSVs local under `data/processed/` until leakage checks are complete.
7. Run `pytest`.
This scaffold contains no training or live prediction implementation yet.
