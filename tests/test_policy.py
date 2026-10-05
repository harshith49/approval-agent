import pytest
from app.policy import Policy


def test_default_policy():
    policy = Policy.load('policy.yaml')
    assert not policy.requires_approval('lookup_order', {'order_id': '123'})
    assert policy.requires_approval('refund_order', {'order_id': '123', 'amount': 1})
    for to, expected in [('a@company.com', False), ('a@evil.test', True),
                         ('a@company.com.evil.test', True), ('a@COMPANY.COM', False)]:
        assert policy.requires_approval('send_email', {'to': to}) is expected


def test_rules_and_fail_closed():
    policy = Policy({'lookup_order': {'rule': 'always'},
                     'refund_order': {'rule': 'conditional', 'amount_above': 50},
                     'send_email': {'rule': 'never'}})
    assert policy.requires_approval('lookup_order', {})
    # Mandatory safety floors cannot be weakened by configuration.
    assert policy.requires_approval('refund_order', {'amount': 1})
    assert policy.requires_approval('send_email', {'to': 'a@evil.test'})
    assert not policy.requires_approval('send_email', {'to': 'a@company.com'})
    assert policy.requires_approval('unknown', {})
    assert policy.rule_matches({'rule': 'conditional', 'amount_above': 50}, {'amount': 51})
    assert not policy.rule_matches({'rule': 'conditional', 'amount_above': 50}, {'amount': 50})
    with pytest.raises(ValueError):
        Policy({'lookup_order': {'rule': 'typo'}})
    with pytest.raises(ValueError):
        Policy({'lookup_order': {'rule': 'conditional'}})
