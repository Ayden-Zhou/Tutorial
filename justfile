# Format code
fmt:
    .venv/bin/ruff format src/ .tests/

# Lint and check code
check:
    .venv/bin/ruff check src/ .tests/

commit:
    git add .
    git commit -m "Log: 一次空提交"
