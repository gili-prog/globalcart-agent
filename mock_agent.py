import os
import warnings
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from mock_services import (
    check_return_policy,
    get_user_profile,
    get_order_details,
    process_refund
)
import urllib3

class AgentResponse(BaseModel):
    reasoning_chain: str = Field(description="Detailed explanation of your decision and the policies applied.")
    action_taken: str = Field(description="Which tools were used and the final system outcome.")
    customer_response: str = Field(description="A polite, professional reply addressed directly to the customer.")

class AgentState(MessagesState):
    final_response: AgentResponse

# 1. השתקת אזהרות אבטחה בטרמינל (כדי לשמור על פלט נקי)
warnings.filterwarnings("ignore")
urllib3.disable_warnings()

# 2. Monkey Patching: מעקף מפורש לספריית httpx
_original_client_init = httpx.Client.__init__

def _patched_client_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _original_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_client_init

# 3. טעינת משתני הסביבה ואתחול המודל
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")    

if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in .env file!")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key, 
    temperature=0
)

@tool
def get_order_details_tool(order_id: str) -> dict:
    """Look up a GlobalCart order by id. Returns shipping status, order and 
        delivery dates, total amount, the items in the box and the condition 
        each arrived in, the shipping address and whether that address was 
        changed after the order was placed. Call this first for any ticket
        that mentions an order."""
    return get_order_details (order_id=order_id)

@tool
def get_user_profile_tool(user_id: str) -> dict:
    """Look up a GlobalCart customer by user id. Returns their tier 
        (VIP customers get a longer return window and a higher refund cap), 
        account age, lifetime value, refund history, prior fraud flags and 
        fraud score. Take the user_id from the order."""
    return get_user_profile (user_id=user_id)

@tool
def check_return_policy_tool(order_id: str, reason: str) -> dict:
    """Evaluate whether an order is still eligible for a return or refund. 
        Applies the return window, VIP overrides, non-returnable categories 
        and order-status rules, and returns the verdict together with the 
        policy ids behind it and whether the case must be escalated to a 
        human. Call this before promising the customer anything."""
    return check_return_policy (order_id=order_id, reason=reason)

@tool
def process_refund_tool(order_id: str, amount: float, reason: str) -> dict:
    """Issue a refund for an order. This is the only tool with a side 
        effect. It re-validates the rulebook, so a request above the 
        automatic refund cap returns ESCALATION_REQUIRED instead of 
        APPROVED. Call it only after check_return_policy reported the 
        claim eligible."""
    return process_refund(order_id=order_id, amount=amount, reason=reason)

tools = [
    get_order_details_tool,
    get_user_profile_tool,
    check_return_policy_tool,
    process_refund_tool
]

SYSTEM_PROMPT = """You are the Operations Resolver for GlobalCart, an AI agent handling customer 
    service tickets. Your job is to resolve cases based strictly on factual tool data and company policy.
    Do not guess, assume, or invent information.
    1. MANDATORY WORKFLOW:
    You must follow this step-by-step sequence. Only proceed if the current step requires it:
    * Step 1: Use `get_order_details_tool` to fetch order info.
    * Step 2: Use `get_user_profile_tool` to check customer tier and risk.
    * Step 3: Use `check_return_policy_tool` with the exact order ID and customer's reason.
    * Step 4: Evaluate the policy verdict:
    - If ELIGIBLE and requires_escalation is False: Use `process_refund_tool`.
    - If requires_escalation is True: STOP. Draft an escalation response.
    - If NON_RETURNABLE_CATEGORY, OUTSIDE_RETURN_WINDOW, or ORDER_NOT_REFUNDABLE: STOP. Draft a rejection response.
    2. BUSINESS RULES:
    * Always quote the exact policy IDs (e.g., POL-RET-01) in your reasoning and final response.
    * Never exceed the automatic refund cap. If a request exceeds it, escalate the case.
    3. OUTPUT FORMAT:
    Return your final response as a valid JSON object with exactly 3 keys:
    * "reasoning_chain": Detailed explanation of your decision and the policies applied.
    * "action_taken": Which tools were used and the final system outcome.
    * "customer_response": A polite, professional reply addressed directly to the customer."""

llm_with_tools = llm.bind_tools(tools)

def format_output_node(state: dict) -> dict:
    structured_llm = llm.with_structured_output(AgentResponse)
    prompt_message = HumanMessage(
        content="Please summarize the final result and format it exactly according to the required JSON schema."
    )
    messages_to_send = state["messages"] + [prompt_message]
    response = structured_llm.invoke(messages_to_send)
    return ({"final_response": response})

def agent(state: dict) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": response}

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("format_output", format_output_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "format_output"})
workflow.add_edge("tools", "agent")
workflow.add_edge("format_output", END)


app = workflow.compile()

def run_agent (customer_message: str) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=customer_message)
    ]
    result = app.invoke({"messages": messages})
    print(result["final_response"].customer_response)
    return result