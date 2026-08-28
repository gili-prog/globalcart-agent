import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import trim_messages
from mock_services import (
    check_return_policy,
    get_user_profile,
    get_order_details,
    process_refund
)

class AgentResponse(BaseModel):
    reasoning_chain: str = Field(description="Detailed explanation of your decision and the policies applied.")
    action_taken: str = Field(description="Which tools were used and the final system outcome.")
    customer_response: str = Field(description="A polite, professional reply addressed directly to the customer.")

class AgentState(MessagesState):
    final_response: AgentResponse

SYSTEM_PROMPT = """You are the Operations Resolver for GlobalCart, an AI agent handling customer 
        service tickets. Your job is to resolve cases based strictly on factual tool data and company policy.
        Do not guess, assume, or invent information.
        1. MANDATORY WORKFLOW:
        You must follow this step-by-step sequence. Only proceed if the current step requires it:
        * Step 1: Use `get_order_details` to fetch order info.
        * Step 2: Use `get_user_profile` to check customer tier and risk.
        * Step 3: Use `check_return_policy` with the exact order ID and customer's reason.
        * Step 4: Evaluate the policy verdict:
        - If ELIGIBLE and requires_escalation is False: Use `process_refund`.
        - If requires_escalation is True: STOP. Draft an escalation response.
        - If NON_RETURNABLE_CATEGORY, OUTSIDE_RETURN_WINDOW, or ORDER_NOT_REFUNDABLE: STOP. Draft a rejection response.
        2. BUSINESS RULES:
        * Always quote the exact policy IDs (e.g., POL-RET-01) in your reasoning and final response.
        * Never exceed the automatic refund cap. If a request exceeds it, escalate the case.
        *The action_taken field MUST start with one of these exact statuses: APPROVED, ESCALATION_REQUIRED, or REJECTED."""


def build_app():

    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")    

    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env file!")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key, 
        temperature=0
    )

    tools = [
        tool(get_order_details),
        tool(get_user_profile),
        tool(check_return_policy),
        tool(process_refund)
    ]

    llm_with_tools = llm.bind_tools(tools)

    def format_output_node(state: dict) -> dict:
        structured_llm = llm.with_structured_output(AgentResponse)
        prompt_message = HumanMessage(
            content="Please summarize the final result and format it exactly according to the required JSON schema."
        )
        messages_to_send = state["messages"] + [prompt_message]
        response = structured_llm.invoke(messages_to_send)
        return ({"final_response": response})
    
    trimmer = trim_messages(
        max_tokens=1500, # המקסימום טוקנים שאנחנו מוכנים לשלוח
        strategy="last", # שומר רק את ההודעות האחרונות והרלוונטיות
        token_counter=llm, # משתמש במודל עצמו כדי לספור טוקנים במדויק
        include_system=True, # שומר את פרומפט המערכת לעולם לא יימחק
        allow_partial=False # לא חותך הודעות באמצע משפט
    )

    def agent(state: dict) -> dict:
        trimmed_messages = trimmer.invoke(state["messages"])
        response = llm_with_tools.invoke(trimmed_messages)
        return {"messages": response}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("format_output", format_output_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "format_output"})
    workflow.add_edge("tools", "agent")
    workflow.add_edge("format_output", END)

    return workflow.compile()



def run_agent (customer_message: str) -> dict:
    app = build_app()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=customer_message)
    ]
    result = app.invoke({"messages": messages})
    return result