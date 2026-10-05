"""Auto-approve fake tools in a temporary directory; never use this UI for real tools."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# Allow the documented `python examples/demo.py` invocation from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from app.graph import open_workflow


def main() -> None:
    with TemporaryDirectory(prefix='approval-agent-demo-') as directory:
        with open_workflow(directory) as (graph, log, audit):
            config = {'configurable': {'thread_id': 'demo'}}
            result = graph.invoke({'messages': [HumanMessage(content='Refund order #123 and email the customer')],
                                   'llm': 'mock'}, config)
            while result.get('__interrupt__'):
                print('DEMO APPROVAL:', json.dumps(result['__interrupt__'][0].value))
                result = graph.invoke(Command(resume={'action': 'approve'}), config)
            print('FAKE EXECUTIONS:', json.dumps(log.entries(), indent=2))
            print('AUDIT LOG:')
            print(audit.path.read_text(), end='')
            assert [row['tool'] for row in log.entries()] == ['lookup_order', 'refund_order', 'send_email']
            for row in log.entries()[1:]:
                assert audit.approved('demo', row['request_id'], row['tool'], row['args'])


if __name__ == '__main__':
    main()
