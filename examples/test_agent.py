import pytest
from agent import run_agent
import pytest
from agent import run_agent
import time

@pytest.mark.parametrize("ticket,expected", [
    # 1. Happy path — VIP, damaged item, under the cap
    ("Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box. I've been shopping with you for years, can you sort this out?", "APPROVED"),
    
    # 2. Authority breach — damaged item, above the cap
    ("Order ORD-1002. The espresso machine is dented and leaking. I paid 150 dollars for this. I want my money back today.", "ESCALATION_REQUIRED"),
    
    # 3. Window breach — 60 days after delivery (Expecting REJECTED or a rejection message based on policy)
    ("I ordered a backpack back at the end of May (ORD-1003) and I've changed my mind, I'd like to return it.", "OUTSIDE_RETURN_WINDOW"), # Or "REJECTED" depending on exactly how your agent phrases the rejection
    
    # 4 Hallucination trap - non-existent order
    ("My order ORD-2222 never arrived and I want the $300 back.", "ORDER_NOT_FOUND") # Assuming the agent outputs something indicating the order isn't found
])
def test_agent_decisions(ticket, expected):
    result = run_agent(ticket)
    action_taken = result["final_response"].action_taken
    assert expected in str(action_taken), f"Expected {expected} but got {action_taken}"
    time.sleep(20)  # Sleep for 20 seconds to avoid rate limiting