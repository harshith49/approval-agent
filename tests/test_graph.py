import sqlite3
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from app.graph import build_graph
from app.llm import MockModel
from app.policy import Policy
from app.tools import ToolLog


def test_safe_tools_complete_without_interrupt(tmp_path):
    with sqlite3.connect(tmp_path / 'tools.db') as conn, SqliteSaver.from_conn_string(str(tmp_path / 'state.db')) as saver:
        log = ToolLog(conn)
        graph = build_graph(saver, Policy.load('policy.yaml'), log, None, MockModel(safe_only=True))
        result = graph.invoke({'messages': [HumanMessage(content='Look up order #123')], 'llm': 'mock'},
                              {'configurable': {'thread_id': 'safe'}})
        assert not result.get('__interrupt__')
        assert [r['tool'] for r in log.entries()] == ['lookup_order']
        assert not result['messages'][-1].tool_calls

from contextlib import contextmanager
import pytest
from langgraph.types import Command
from app.audit import AuditLog


@contextmanager
def workflow(path, model=None, note=None):
    with sqlite3.connect(path / 'tools.db') as conn, SqliteSaver.from_conn_string(str(path / 'state.db')) as saver:
        log, audit = ToolLog(conn, note), AuditLog(path / 'audit.jsonl')
        yield build_graph(saver, Policy.load('policy.yaml'), log, audit, model), log, audit


def start(graph, thread='test'):
    return graph.invoke({'messages': [HumanMessage(content='Refund order #123 and email the customer')], 'llm': 'mock'},
                        {'configurable': {'thread_id': thread}})


@pytest.mark.parametrize('action', ['approve', 'edit', 'reject'])
def test_decisions(tmp_path, action):
    with workflow(tmp_path) as (graph, log, audit):
        result = start(graph)
        assert result['__interrupt__'][0].value['tool'] == 'refund_order'
        assert len(log.entries()) == 1
        answer = {'action': action}
        if action == 'edit':
            answer['args'] = {'order_id': '123', 'amount': 12.0}
        result = graph.invoke(Command(resume=answer), {'configurable': {'thread_id': 'test'}})
        if action == 'reject':
            assert len(log.entries()) == 1
            assert any(m.content == 'denied by human' for m in result['messages'])
        else:
            assert log.entries()[1]['args']['amount'] == (12.0 if action == 'edit' else 49.99)
            assert result['__interrupt__'][0].value['tool'] == 'send_email'
        assert any(e['decision'] == action for e in audit.entries())


def test_pause_restart_resume(tmp_path):
    with workflow(tmp_path) as (graph, log, audit):
        start(graph, 'restart')
    with workflow(tmp_path) as (graph, log, audit):
        config = {'configurable': {'thread_id': 'restart'}}
        assert graph.get_state(config).tasks[0].interrupts
        result = graph.invoke(Command(resume={'action': 'approve'}), config)
        assert result['__interrupt__'][0].value['tool'] == 'send_email'
        graph.invoke(Command(resume={'action': 'approve'}), config)
        assert [r['tool'] for r in log.entries()] == ['lookup_order', 'refund_order', 'send_email']
        assert len([e for e in audit.entries() if e['event'] == 'request']) == 3


def test_invalid_edit_does_not_execute(tmp_path):
    with workflow(tmp_path) as (graph, log, audit):
        start(graph)
        with pytest.raises(ValueError):
            graph.invoke(Command(resume={'action': 'edit', 'args': {'order_id': '123', 'amount': -1}}),
                         {'configurable': {'thread_id': 'test'}})
        assert len(log.entries()) == 1
        assert not any(e['decision'] == 'edit' for e in audit.entries())
