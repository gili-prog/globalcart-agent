# GlobalCart AI Customer Support Agent

An autonomous refund-resolution agent for the GlobalCart e-commerce platform. Given a natural-language support ticket, the agent fetches live order and customer data, applies the company policy rulebook, and either approves a refund, rejects it with a policy citation, or escalates to a human — all without human intervention on the happy path.

---

## 1. Architecture & Framework

### Graph Topology

The agent is implemented as a compiled **LangGraph** `StateGraph`. The execution graph has three nodes and a conditional routing edge:

```
START → [agent] → (tools_condition) → [tools] → [agent]  (loop)
                                    ↘ [format_output] → END
```

| Node | Responsibility |
|---|---|
| `agent` | Invokes the LLM with the current message history and bound tools. Decides whether to call a tool or terminate. |
| `tools` | LangGraph's built-in `ToolNode`. Dispatches whichever tool calls the LLM emitted, executes them, and appends the results as `ToolMessage` objects back into state. |
| `format_output` | Switches the LLM into structured-output mode (via `.with_structured_output(AgentResponse)`) and produces a validated Pydantic object with three fields: `reasoning_chain`, `action_taken`, and `customer_response`. |

State is typed as `AgentState(MessagesState)`, which inherits LangGraph's built-in `messages: list[BaseMessage]` reducer and adds a `final_response: AgentResponse` field populated only at the terminal node.

### Why LangGraph

| Requirement | How LangGraph addresses it |
|---|---|
| **Cyclic tool-call loop** | `add_conditional_edges` with `tools_condition` routes back to the `agent` node after every tool execution — a cycle that plain chain frameworks cannot express. |
| **Shared mutable state** | `MessagesState` appends messages atomically; every node reads the full conversation history. No manual message threading. |
| **Deterministic termination** | The LLM signals it is done by emitting no tool calls. `tools_condition` detects this and routes to `format_output` instead of `tools`. |
| **Structured final output** | The `format_output` node uses `llm.with_structured_output(AgentResponse)` with a Pydantic schema, guaranteeing the response is machine-parseable regardless of what the LLM generated during reasoning. |
| **Production compilation** | `workflow.compile()` returns a reusable, inspectable `CompiledGraph` that can be checkpointed, streamed, or deployed behind a LangGraph Cloud endpoint without any code changes. |

### LLM Integration

The agent uses **Google Gemini** (`gemini-3.6-flash`) via `langchain-google-genai`, configured at `temperature=0` to enforce deterministic, policy-grounded decisions. The LLM is bound to the four tools via `.bind_tools(tools)` before graph construction so tool schemas are transmitted in every request.

A `trim_messages` trimmer (max 1,500 tokens, `strategy="last"`, `include_system=True`) is applied inside the `agent` node on every invocation. This prevents the context window from overflowing across long multi-tool reasoning chains while always preserving the system prompt.

---

## 2. Defined Tools & Run Instructions

### Tool Definitions

All tools are implemented in `mock_services.py` as pure functions over local JSON fixtures. They are registered with LangGraph via `langchain_core.tools.tool`. Every function returns a plain `dict`; business failures are returned as error dicts, never raised as exceptions (see §3).

#### `get_order_details(order_id: str)`
Fetches a single order record by ID. Returns shipping status (`delivered` / `shipped` / `processing` / `delayed` / `cancelled`), delivery date, order total, line items with per-item condition (`new` / `damaged_on_arrival` / `wrong_item` / `missing`), and whether the shipping address was changed post-order. **Always the first tool called** — every downstream decision depends on data read here.

#### `get_user_profile(user_id: str)`
Fetches the customer record using the `user_id` extracted from the order. Returns tier (`VIP` or `Standard`), fraud score (0–100), prior fraud flag count, and full refund history with dates. Tier determines which return window and refund cap policies apply (POL-RET-01/02, POL-REF-01/02).

#### `check_return_policy(order_id: str, reason: str)`
The policy engine. Applies the full GlobalCart rulebook in priority order — non-returnable category check (POL-REF-03), order status check (POL-REF-04), return window check (POL-RET-01/02), then escalation checks (POL-ESC-01/02). Returns a structured verdict (`ELIGIBLE` / `OUTSIDE_RETURN_WINDOW` / `NON_RETURNABLE_CATEGORY` / `ORDER_NOT_REFUNDABLE`), the applicable policy IDs, the `requires_escalation` boolean, and a one-sentence `explanation` the agent can quote verbatim. **The agent must call this before taking any action.**

#### `process_refund(order_id: str, amount: float, reason: str)`
The only tool with a side effect (simulated — no writes to disk). Re-validates the rulebook internally before acting, meaning the agent cannot bypass policy by skipping `check_return_policy`. Returns one of three statuses: `APPROVED` (with a deterministic `refund_id`), `REJECTED`, or `ESCALATION_REQUIRED`. The cap is mechanically enforced: a request above the automatic refund authority (`$50` standard / `$75` VIP) returns `ESCALATION_REQUIRED` regardless of what the LLM requested.

### Local Setup

**Prerequisites:** Python 3.10+, a Google Gemini API key.

```bash
# 1. Clone the repository
git clone <repo-url>
cd "PlaceIL quest#4"

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and set your key:
#   GOOGLE_API_KEY=your_gemini_api_key_here
```

### Running the Agent

```python
# Interactive use from a Python REPL or script
from agent import run_agent

result = run_agent("My order ORD-1001 arrived damaged. I'd like a refund.")
print(result["final_response"].action_taken)
print(result["final_response"].customer_response)
```

### Running the Evaluation Suite

```bash
# From the repository root
python -m pytest examples/test_agent.py -v
```

The test file lives at `examples/test_agent.py`. pytest will discover it automatically via the default collection path. The suite covers four scenarios: approved refund, escalation-required, rejected (window breach), and hallucination trap (non-existent order).

### CI/CD

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the full test suite on every push and pull request to `main` against Python 3.10. The `GOOGLE_API_KEY` is injected via a GitHub Actions secret at runtime.

---

## 3. Reasoning Chain & Edge Case Handling

### Four-Step Reasoning Chain

The system prompt encodes a strict, sequential decision procedure that the agent cannot reorder:

```
Step 1  get_order_details    → establishes facts: status, amount, items, dates
Step 2  get_user_profile     → establishes context: tier, fraud score, history
Step 3  check_return_policy  → applies the rulebook, returns verdict + policy IDs
Step 4  process_refund       → called only if verdict == ELIGIBLE and
                               requires_escalation == False
```

This chain is deliberate. The agent cannot skip to `process_refund` without the policy check because `check_return_policy` is the sole source of truth for eligibility. The system prompt mandates that policy IDs (e.g., `POL-RET-01`) be quoted in the `reasoning_chain` field, producing an auditable trace of every decision. The `action_taken` field must begin with one of three canonical statuses (`APPROVED`, `ESCALATION_REQUIRED`, or `REJECTED`), which the test suite asserts against programmatically.

### Error Dict Pattern — No Silent Failures

Every tool in `mock_services.py` follows a single contract: **business failures are data, not exceptions.** A missing order, an invalid reason code, or a non-positive refund amount all return a structured error dict:

```python
{"error": "ORDER_NOT_FOUND", "message": "No order found with id 'ORD-2222'."}
```

These are appended to the message history as `ToolMessage` objects. The LLM reads them on the next `agent` node invocation and drafts a rejection response grounded in the error code — it never has an opportunity to fabricate a plausible-sounding order record because the tool returned a definitive negative.

### Hallucination Trap

Test case 4 (`ORD-2222`) is a purpose-built hallucination trap. The order does not exist in the fixture. Without error dict discipline, an LLM might invent a delivery date and approve a refund. The contract prevents this: `get_order_details("ORD-2222")` returns `{"error": "ORDER_NOT_FOUND", ...}`, the agent reads the failure, and its `action_taken` field contains `ORDER_NOT_FOUND` — the test asserts exactly this. No fabricated data enters the reasoning chain.

The same pattern applies to `process_refund`, which re-validates the full policy check internally. Even if the LLM were somehow prompted to call `process_refund` on an ineligible order, the tool would return `REJECTED` mechanically before any approval logic could run.

### API Rate Limit Handling (HTTP 429)

The Gemini API enforces per-minute request quotas. Running four LLM-heavy test cases sequentially without throttling triggers `429 Too Many Requests` errors in CI, causing flaky test failures unrelated to agent correctness.

The test suite mitigates this with a deterministic inter-test sleep:

```python
# examples/test_agent.py
time.sleep(20)  # inserted at the end of each parametrized test case
```

`@pytest.mark.parametrize` executes each scenario as a separate test function invocation. The 20-second sleep after each case provides a fixed cooldown window between API call bursts, keeping the per-minute token and request counts within the free-tier limits used in CI. This is surfaced explicitly as a known operational constraint rather than hidden behind a retry decorator, making the throttling behaviour observable in test output timing.
