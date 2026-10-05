# Security policy

This project is a teaching example with fake tools, not a production security
boundary. The supported version is the current source with the pinned dependency
set. Keep dependencies reviewed and updated; passing tests is not a security audit.

## Boundary

The model can propose tool names and arguments, but only the Python policy and
executor authorize their execution. Refunds and external email require an exact
matching durable approve/edit event. Every call, including a call in a batch,
passes the gate. Edited arguments are validated before approval is recorded.
Tool IDs supplied by the model are not used as authorization IDs.

The person controlling the terminal, source code, policy, environment, checkpoint
DB, execution DB, and audit JSONL is trusted. There is no user authentication,
audit signature, tamper detection, or protection against a malicious local
operator. Use one process per data directory. Malformed audit data fails closed;
recover damaged files carefully rather than deleting audit records to continue.

## Data and integrations

Never enter real secrets or customer data in a demo. SQLite checkpoints and
JSONL audit records contain plaintext inputs and tool arguments. Protect their
permissions and retention. Ollama/Gemini receive the message history; third-party
tracing enabled through environment configuration can also export information.
All order, refund, and email functions are fictional and contact no services.

Automatic approval in examples/demo.py is solely for fake demonstrations. Before
connecting real tools, add authenticated approvers, durable transactional audit
storage, idempotent service operations, concurrency controls, and a reviewed
policy. Human review can still fail, and a company-domain allowlist does not
prevent sensitive content from being sent to an internal recipient.

## Reporting

After publishing, enable GitHub private vulnerability reporting and report
security issues through the repository's Security tab. Until a private reporting
channel exists, do not post secrets or exploit payloads publicly. Maintainers
should acknowledge reports, reproduce against pinned dependencies, and add a
regression test before releasing a fix.
