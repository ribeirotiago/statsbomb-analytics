# Project Preferences (GEMINI.md)

Guidelines for AI agents working in the **`statsbomb-analytics`** repository.

## 1. Critical Rules

### 1.1 Data
- The source is **StatsBomb Open Data** (license: see `LICENSE.pdf` in their repository). Use the local clone at `STATSBOMB_DATA_DIR` (see `.env.example`), **read-only**.
- **NEVER** copy the clone (~12 GB) into this repository or version raw/heavy derived data.
- Only use the network to download data from the official GitHub `statsbomb/open-data`, if the user asks. Do not access other sources without asking the user.
- Always credit the source ("Data: StatsBomb") in the README and in the app interface.

### 1.2 Git
- **DO NOT run `git commit` or `git push`.** Present a summary and the exact commands:
  ```bash
  git add <files>
  git commit -m "feat/fix(scope): concise description"
  ```

## 2. Code and Quality
- Python 3.11+, `from __future__ import annotations`, type hints on all signatures.
- Ruff (line length ≤ 100; rules `E, F, I, UP, B`). Validate before finishing:
  ```bash
  ./.venv/bin/ruff check app/ src/ tests/ scripts/
  ./.venv/bin/ruff format --check app/ src/ tests/ scripts/
  ./.venv/bin/pytest
  ```
- Every metric, ETL or utility function must have a test in `tests/`.

## 3. Frontend (Streamlit)
- Pages in `app/` and `app/pages/`. Dark theme, palette compatible with `plotly_dark`.
- Prefer native components (`st.dataframe` with `column_config`, `st.metric`, etc.).
- General league tables: `st.dataframe(..., height=600)` (global sorting).

## 4. Academic Nature
- **Academic and portfolio** project, non-commercial. Document **assumptions, limitations and risks** (small samples, team style bias, single 15/16 season, uneven Champions League coverage).
- Start with simple, explainable baselines (e.g., Euclidean k-NN) before ML (PCA, K-Means), and justify the choice.
- Methodological changes come with tests and a methodology note in the interface.

## 5. Language
- **All repository artifacts are in English**: code (identifiers, comments, docstrings), documentation (README, docs, ARCHITECTURE, STATUS), reports, commit messages, generated files and app UI text.
- Future UI localization (English/Portuguese) will be an app feature; English remains the default.
- Chat explanations with the user may be in pt-BR, technical and concise.
- Use clickable markdown links for cited files and symbols.
