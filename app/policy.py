"""Trusted configuration and mandatory safety floors, independent of prompts."""
from pathlib import Path
from typing import Any
import math
import yaml


class Policy:
    def __init__(self, rules: dict[str, dict[str, Any]]) -> None:
        if not isinstance(rules, dict):
            raise ValueError('Policy tools must be a mapping')
        for rule in rules.values():
            if not isinstance(rule, dict) or rule.get('rule') not in {'always', 'never', 'conditional'}:
                raise ValueError('Policy rule must be always, never, or conditional')
            if rule['rule'] == 'conditional':
                conditions = set(rule) - {'rule'}
                if not conditions or conditions - {'external_domain', 'amount_above'}:
                    raise ValueError('Conditional rule needs a supported condition')
                if 'external_domain' in rule and not isinstance(rule['external_domain'], str):
                    raise ValueError('external_domain must be a string')
                if 'amount_above' in rule:
                    value = rule['amount_above']
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError('amount_above must be finite and nonnegative')
            elif set(rule) != {'rule'}:
                raise ValueError('Unexpected policy fields')
        self.rules = rules

    @classmethod
    def load(cls, path: str | Path) -> 'Policy':
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict) or set(data) != {'tools'}:
            raise ValueError('Policy must contain only a tools mapping')
        return cls(data['tools'])

    @staticmethod
    def rule_matches(rule: dict[str, Any], args: dict[str, Any]) -> bool:
        """Multiple conditional predicates are ORed; missing values fail closed."""
        if rule['rule'] != 'conditional':
            return rule['rule'] == 'always'
        if 'external_domain' in rule:
            recipient = args.get('to', '')
            if not isinstance(recipient, str) or recipient.count('@') != 1 or recipient.rsplit('@', 1)[-1].lower() != rule['external_domain'].lower():
                return True
        if 'amount_above' in rule:
            amount = args.get('amount')
            if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount > rule['amount_above']:
                return True
        return False

    def requires_approval(self, tool: str, args: dict[str, Any]) -> bool:
        """YAML may tighten, but cannot weaken refund/external-email approval."""
        if tool == 'refund_order':
            return True
        if tool == 'send_email' and self.rule_matches(
            {'rule': 'conditional', 'external_domain': 'company.com'}, args
        ):
            return True
        rule = self.rules.get(tool)
        return rule is None or self.rule_matches(rule, args)
