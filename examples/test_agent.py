import pytest
from agent import run_agent

@pytest.mark.parametrize("ticket,expected", [
    ("Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box...", "APPROVED"),
    ("Order ORD-1002. The espresso machine is dented and leaking...", "APPROVED"),
    ("I ordered a backpack back at the end of May (ORD-1003)...", "APPROVED"),
])
def test_agent_decisions(ticket, expected):
    result = run_agent(ticket)
    action_taken = result["final_response"].action_taken
    assert expected in str(action_taken), f"Expected {expected} but got {action_taken}"