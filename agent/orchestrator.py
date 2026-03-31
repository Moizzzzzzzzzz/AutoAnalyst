import operator
import logging
import traceback
from typing import TypedDict, Annotated, List, Dict, Any
from functools import wraps
from langgraph.graph import StateGraph, END
from agent.eda_agent import run_eda

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# 1. Graph State Definition
# ------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    TypedDict representing the shared state across all LangGraph nodes.
    Annotated fields use operator.add to append to lists rather than overwrite.
    """
    session_id: str
    dataset_path: str
    user_prompt: str
    dataset_summary: Dict[str, Any]
    
    # Outputs
    eda_results: Dict[str, Any]
    hypotheses: List[Dict[str, Any]]
    model_results: Dict[str, Any]
    visualizations: Annotated[List[str], operator.add]
    
    # Tracking
    agent_reasoning: Annotated[List[str], operator.add]
    errors: Annotated[List[Dict[str, str]], operator.add]
    
    # Routing
    next_steps: List[str]  # e.g., ["EDA", "MODELING", "REPORT"]
    current_step_index: int

# ------------------------------------------------------------------------
# 2. Failure Handler (Decorator)
# ------------------------------------------------------------------------
def with_retry(agent_name: str, max_retries: int = 2):
    """
    Wraps node functions to catch exceptions, retry, and gracefully fail
    by appending to the state's error log rather than crashing the pipeline.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state: AgentState, *args, **kwargs) -> dict:
            attempts = 0
            while attempts <= max_retries:
                try:
                    logger.info(f"[{agent_name}] Attempt {attempts + 1}")
                    return func(state, *args, **kwargs)
                except Exception as e:
                    attempts += 1
                    error_msg = f"Error in {agent_name}: {str(e)}\n{traceback.format_exc()}"
                    logger.warning(error_msg)
                    
                    if attempts > max_retries:
                        logger.error(f"[{agent_name}] Max retries reached. Skipping step.")
                        return {
                            "errors": [{"agent": agent_name, "error": str(e)}],
                            "agent_reasoning": [f"[{agent_name}] Failed after {max_retries} attempts. Skipped."]
                        }
        return wrapper
    return decorator

# ------------------------------------------------------------------------
# 3. Core Nodes
# ------------------------------------------------------------------------
@with_retry("PLANNER", max_retries=1)
def plan_node(state: AgentState) -> dict:
    """
    Analyzes the user prompt and dataset summary to determine the required steps.
    (In a full implementation, this calls an LLM to generate the execution plan).
    """
    prompt = state.get("user_prompt", "").lower()
    plan = ["EDA"] # Always run EDA as a baseline
    
    # Simple heuristic routing (to be replaced by LLM structured output)
    if "predict" in prompt or "model" in prompt or "classify" in prompt:
        plan.append("MODELING")
    if "test" in prompt or "significant" in prompt or "hypothesis" in prompt:
        plan.append("HYPOTHESIS")
        
    plan.append("REPORT")
    
    return {
        "next_steps": plan,
        "current_step_index": 0,
        "agent_reasoning": [f"[PLANNER] Created execution plan: {plan}"]
    }

@with_retry("EDA_AGENT", max_retries=2)
def eda_node(state: AgentState) -> dict:
    """Executes the specialized EDA agent."""
    dataset_path = state.get("dataset_path")
    if not dataset_path:
        raise ValueError("Dataset path is missing from state.")
    
    results, log = run_eda(dataset_path)
    return {
        "eda_results": results,
        "agent_reasoning": [f"[EDA_AGENT] {log}"]
    }

# Placeholder nodes for upcoming implementation
def modeling_node(state: AgentState) -> dict: return {}
def hypothesis_node(state: AgentState) -> dict: return {}
def report_node(state: AgentState) -> dict: return {}

# ------------------------------------------------------------------------
# 4. Routing Logic
# ------------------------------------------------------------------------
def route_next_step(state: AgentState) -> str:
    """Reads the plan and routes to the next required agent."""
    steps = state.get("next_steps", [])
    idx = state.get("current_step_index", 0)
    
    if idx >= len(steps):
        return "END"
        
    return steps[idx]

def advance_step(state: AgentState) -> dict:
    """Pass-through node that simply increments the step index."""
    return {"current_step_index": state.get("current_step_index", 0) + 1}

# ------------------------------------------------------------------------
# 5. Graph Compilation
# ------------------------------------------------------------------------
def build_orchestrator() -> Any:
    """Constructs and compiles the LangGraph state machine."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("PLAN", plan_node)
    workflow.add_node("EDA", eda_node)
    workflow.add_node("MODELING", modeling_node)
    workflow.add_node("HYPOTHESIS", hypothesis_node)
    workflow.add_node("REPORT", report_node)
    workflow.add_node("ADVANCE", advance_step)
    
    # Set entry point
    workflow.set_entry_point("PLAN")
    
    # After PLAN, advance the index
    workflow.add_edge("PLAN", "ADVANCE")
    
    # Conditional routing from ADVANCE based on the plan array
    workflow.add_conditional_edges(
        "ADVANCE",
        route_next_step,
        {
            "EDA": "EDA",
            "MODELING": "MODELING",
            "HYPOTHESIS": "HYPOTHESIS",
            "REPORT": "REPORT",
            "END": END
        }
    )
    
    # All task nodes loop back to ADVANCE to get the next instruction
    for node in ["EDA", "MODELING", "HYPOTHESIS", "REPORT"]:
        workflow.add_edge(node, "ADVANCE")
        
    return workflow.compile()

app = build_orchestrator()