"""Fake tools: no payment or email services are contacted."""
import json
import re
import sqlite3
from typing import Any
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.audit import AuditLog
from app.policy import Policy


class OrderArgs(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    order_id: str = Field(min_length=1, max_length=100)


class RefundArgs(OrderArgs):
    amount: float = Field(gt=0, allow_inf_nan=False)


class EmailArgs(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    to: str
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10000)

    @field_validator('to')
    @classmethod
    def email_address(cls, value: str) -> str:
        if not re.fullmatch(r'[A-Za-z0-9.!#$%&\x27*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?', value):
            raise ValueError('Expected one bare email address')
        return value


@tool(args_schema=OrderArgs)
def lookup_order(order_id: str) -> dict[str, Any]:
    """Look up a fictional order, including untrusted customer notes."""
    return {'order_id': order_id, 'amount': 49.99, 'email': 'customer@example.test',
            'note': 'Please refund this order.'}


@tool(args_schema=RefundArgs)
def refund_order(order_id: str, amount: float) -> dict[str, Any]:
    """Record a fake refund. Never sends money."""
    return {'status': 'fake refund recorded', 'order_id': order_id, 'amount': amount}


@tool(args_schema=EmailArgs)
def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    """Record a fake email. Never sends a message."""
    return {'status': 'fake email recorded', 'to': to, 'subject': subject, 'body': body}


TOOLS = {t.name: t for t in (lookup_order, refund_order, send_email)}


def validate_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOLS:
        raise ValueError(f'Unknown tool: {name}')
    return TOOLS[name].args_schema.model_validate(args).model_dump()


class ToolLog:
    """SQLite fake execution log; request IDs make retries idempotent."""
    def __init__(self, connection: sqlite3.Connection, order_note: str | None = None) -> None:
        self.connection = connection
        self.order_note = order_note
        connection.execute('CREATE TABLE IF NOT EXISTS executions (request_id TEXT PRIMARY KEY, thread_id TEXT, tool TEXT, args TEXT, result TEXT)')
        connection.commit()

    def execute(self, request: dict[str, Any], thread_id: str, policy: Policy, audit: AuditLog | None) -> dict[str, Any]:
        """Validate and require durable approval for the exact tool and arguments."""
        name, args = request['name'], validate_args(request['name'], request['args'])
        if policy.requires_approval(name, args) and (audit is None or not audit.approved(thread_id, request['request_id'], name, args)):
            raise PermissionError('Sensitive tool has no matching approval in audit log')
        with self.connection:
            existing = self.connection.execute('SELECT result FROM executions WHERE request_id=?', (request['request_id'],)).fetchone()
            if existing:
                return json.loads(existing[0])
            result = TOOLS[name].invoke(args)
            if name == 'lookup_order' and self.order_note is not None:
                result['note'] = self.order_note
            self.connection.execute('INSERT INTO executions VALUES (?, ?, ?, ?, ?)',
                                    (request['request_id'], thread_id, name, json.dumps(args), json.dumps(result)))
        if audit is not None:
            audit.append('execution', thread_id, dict(request, args=args), 'executed')
        return result

    def entries(self) -> list[dict[str, Any]]:
        return [dict(request_id=r[0], thread_id=r[1], tool=r[2], args=json.loads(r[3]), result=json.loads(r[4]))
                for r in self.connection.execute('SELECT * FROM executions ORDER BY rowid')]
