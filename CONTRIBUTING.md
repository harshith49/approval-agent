# Contributing

Contributions are welcome: bug reports, clearer documentation, tests, and small
improvements that keep the project easy for students to understand.

## Local setup

Use Python 3.11 or newer. No API key is needed for development or tests.

```bash
git clone https://github.com/harshith49/approval-agent.git
cd approval-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python examples/demo.py
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.
Fork the repository and create a branch for your change before opening a pull
request against `main`.

## Making a change

Keep changes focused, typed, and documented. Prefer the standard library and
existing dependencies. Add a regression test for a bug or changed behavior, and
run the full test suite and demo before submitting.

Approval behavior is the central invariant: a sensitive tool must never execute
without a prior audit approval matching its thread, request ID, tool, and effective
arguments. Preserve the mandatory approval rules for refunds and external email.
Test edited arguments, rejection, and SQLite restart/resume when changing the
workflow. New tools must stay fake and pass argument validation and the policy
gate; do not introduce real payments or email delivery into this example.

Tests must run with the mock model, without secrets or network access. Never
commit `.env`, API keys, customer data, SQLite databases, or audit logs. If you
change dependencies, update their pinned versions in `requirements.txt` and
check compatibility with the Python versions in the GitHub Actions workflow.

## Pull requests and issues

Describe the problem, resulting behavior, and how you verified the change.
Include a small reproduction for bugs and update documentation when CLI or
policy behavior changes. For larger ideas, open an issue first to discuss scope.

Use synthetic data in issue reports. Follow [SECURITY.md](SECURITY.md) for security
concerns; do not post credentials or sensitive exploit details publicly.

Contributions are licensed under the project's [MIT license](LICENSE).
