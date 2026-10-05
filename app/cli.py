"""Terminal approval UI. EOF/Ctrl-C leaves the checkpoint paused."""
from enum import Enum
import json
from pathlib import Path
from typing import Any
from uuid import uuid4
import typer
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from app.graph import open_workflow

app = typer.Typer(help='Human approval for fictional agent actions.', no_args_is_help=True)


class LLM(str, Enum):
    mock = 'mock'
    ollama = 'ollama'
    gemini = 'gemini'


def human_decision(auto_reject: bool) -> dict[str, Any]:
    if auto_reject:
        return {'action': 'reject'}
    while True:
        choice = typer.prompt('[a]pprove / [e]dit / [r]eject').strip().lower()
        if choice in {'a', 'approve'}:
            return {'action': 'approve'}
        if choice in {'r', 'reject'}:
            return {'action': 'reject'}
        if choice in {'e', 'edit'}:
            try:
                args = json.loads(typer.prompt('Replacement args (complete JSON object)'))
                if not isinstance(args, dict):
                    raise ValueError('Args must be a JSON object')
                return {'action': 'edit', 'args': args}
            except ValueError as error:
                typer.echo(f'Invalid JSON: {error}')
        else:
            typer.echo('Choose a, e, or r.')


def drive(graph: Any, result: dict[str, Any], config: dict[str, Any], auto_reject: bool, pause: bool) -> None:
    while result.get('__interrupt__'):
        pending = result['__interrupt__'][0].value
        typer.echo(json.dumps(pending, indent=2))
        if pause:
            typer.echo('Paused. Resume with the same thread ID and data directory.')
            return
        decision = human_decision(auto_reject)
        # Validate before giving LangGraph a resume value. Invalid edits stay paused.
        from app.tools import validate_args
        if decision['action'] == 'edit':
            try:
                validate_args(pending['tool'], decision['args'])
            except ValueError as error:
                typer.echo(f'Invalid edit: {error}')
                continue
        result = graph.invoke(Command(resume=decision), config)
    typer.echo(str(result['messages'][-1].content))


@app.command()
def run(
    prompt: str,
    llm: LLM = typer.Option(LLM.mock, help='mock needs no API key.'),
    thread_id: str | None = typer.Option(None),
    data_dir: Path = typer.Option(Path('.'), help='Checkpoint, execution and audit directory.'),
    policy: Path | None = typer.Option(None, help='Trusted YAML policy file.'),
    auto_reject: bool = typer.Option(False),
    pause: bool = typer.Option(False, help='Save the first pending action and exit.'),
) -> None:
    """Start a new run; refuse to overwrite an existing thread."""
    thread = thread_id or str(uuid4())
    config = {'configurable': {'thread_id': thread}, 'recursion_limit': 100}
    typer.echo(f'Thread ID: {thread}')
    with open_workflow(data_dir, policy) as (graph, _, _):
        if graph.get_state(config).values:
            raise typer.BadParameter('Thread already exists; use resume or a new thread ID')
        result = graph.invoke({'messages': [HumanMessage(content=prompt)], 'llm': llm.value}, config)
        drive(graph, result, config, auto_reject, pause)


@app.command()
def resume(
    thread_id: str = typer.Option(...),
    data_dir: Path = typer.Option(Path('.')),
    policy: Path | None = typer.Option(None),
    auto_reject: bool = typer.Option(False),
) -> None:
    """Resume a saved approval; the original LLM mode comes from the checkpoint."""
    config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 100}
    with open_workflow(data_dir, policy) as (graph, _, _):
        snapshot = graph.get_state(config)
        pending = [i for task in snapshot.tasks for i in task.interrupts]
        if not pending:
            raise typer.BadParameter('No pending approval for this thread')
        drive(graph, dict(snapshot.values, __interrupt__=pending), config, auto_reject, False)


def main() -> None:
    try:
        app()
    except (ValueError, PermissionError, OSError) as error:
        typer.echo(f'Error: {error}', err=True)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
