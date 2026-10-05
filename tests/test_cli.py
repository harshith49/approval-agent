import json
import subprocess
import sys
from typer.testing import CliRunner
from app.cli import app


def test_auto_reject(tmp_path):
    result = CliRunner().invoke(app, ['run', 'Refund #123', '--data-dir', str(tmp_path), '--auto-reject'])
    assert result.exit_code == 0, result.output
    decisions = [json.loads(line) for line in (tmp_path / 'audit.jsonl').read_text().splitlines()]
    assert any(e['decision'] == 'reject' for e in decisions)
    assert not any(e['tool'] == 'refund_order' and e['event'] == 'execution' for e in decisions)


def test_real_process_restart(tmp_path):
    args = [sys.executable, '-m', 'app.cli']
    first = subprocess.run([*args, 'run', 'Refund #123', '--thread-id', 'restart', '--data-dir', str(tmp_path), '--pause'],
                           capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    second = subprocess.run([*args, 'resume', '--thread-id', 'restart', '--data-dir', str(tmp_path)],
                            input='a\na\n', capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    events = [json.loads(line) for line in (tmp_path / 'audit.jsonl').read_text().splitlines()]
    assert [e['tool'] for e in events if e['event'] == 'execution'] == ['lookup_order', 'refund_order', 'send_email']


def test_missing_thread_and_duplicate_run(tmp_path):
    runner = CliRunner()
    assert runner.invoke(app, ['resume', '--thread-id', 'missing', '--data-dir', str(tmp_path)]).exit_code != 0
    command = ['run', 'Refund #123', '--thread-id', 'same', '--data-dir', str(tmp_path), '--pause']
    assert runner.invoke(app, command).exit_code == 0
    assert runner.invoke(app, command).exit_code != 0
