"""Append-only JSONL events, flushed before sensitive tool execution."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def entries(self) -> list[dict[str, Any]]:
        # ponytail: scan a small local log; use an indexed audit store at scale.
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def append(self, event: str, thread_id: str, call: dict[str, Any],
               decision: str | None = None, edited_args: dict[str, Any] | None = None) -> None:
        record = {'timestamp': datetime.now(timezone.utc).isoformat(), 'event': event,
                  'thread_id': thread_id, 'request_id': call['request_id'], 'tool': call['name'],
                  'args': call['args'], 'decision': decision, 'edited_args': edited_args}
        # Interrupt nodes replay. Identical events may be retried, not duplicated.
        for prior in self.entries():
            if all(prior.get(k) == v for k, v in record.items() if k != 'timestamp'):
                return
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def approved(self, thread_id: str, request_id: str, tool: str, args: dict[str, Any]) -> bool:
        decisions = [e for e in self.entries() if e['event'] == 'decision'
                     and e['thread_id'] == thread_id and e['request_id'] == request_id]
        if not decisions:
            return False
        event = decisions[-1]
        effective = event['edited_args'] if event['decision'] == 'edit' else event['args']
        return event['decision'] in {'approve', 'edit'} and event['tool'] == tool and effective == args
