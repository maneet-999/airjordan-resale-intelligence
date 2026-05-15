# Contributing to Air Jordan Resale Intelligence Dashboard

Thank you for considering contributing to this project. Any improvement — bug fixes, new statistical tests, UI enhancements, or documentation — is welcome.

---

## How to Contribute

### 1. Fork the Repository
Click the **Fork** button at the top right of the repository page.

### 2. Clone Your Fork
```bash
git clone https://github.com/your-username/airjordan-resale-intelligence.git
cd airjordan-resale-intelligence
```

### 3. Create a Branch
Use a descriptive branch name:
```bash
git checkout -b feature/add-arima-forecasting
git checkout -b fix/regression-coefficient-display
git checkout -b docs/update-readme
```

### 4. Make Your Changes
- Follow existing code style
- Add comments explaining statistical decisions
- Test locally with `streamlit run air_jordan_dashboard.py` before committing

### 5. Commit with a Clear Message
```bash
git add .
git commit -m "Add: ARIMA time series forecasting for monthly price prediction"
```

Commit message format:
- `Add:` new feature
- `Fix:` bug fix
- `Update:` modification to existing feature
- `Docs:` documentation only
- `Refactor:` code restructure, no feature change

### 6. Push and Open a Pull Request
```bash
git push origin feature/your-branch-name
```
Then open a Pull Request on GitHub with a clear description of what you changed and why.

---

## What to Contribute

| Area | Examples |
|------|---------|
| New statistical tests | Levene's test, Mann-Whitney U, regression diagnostics |
| New visualisations | Q-Q plot, residual plot, violin plot |
| ML improvements | Random Forest, XGBoost, hyperparameter tuning |
| UI improvements | Better filters, colour themes, mobile layout |
| Data pipeline | API connections, real-time data feeds |
| Documentation | Better explanations, usage examples |

---

## Code Style Guidelines

- Use 4 spaces for indentation (no tabs)
- Add a comment above every statistical test explaining why it was chosen
- Keep functions under 50 lines where possible
- Use descriptive variable names (`profit_margin_pct` not `pmp`)

---

## Reporting Bugs

Open an issue with:
1. What you expected to happen
2. What actually happened
3. Steps to reproduce
4. Your Python version and OS

---

## Questions

Open a GitHub Discussion or contact via the email in README.md.
