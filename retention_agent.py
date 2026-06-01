# retention_agent.py
import pandas as pd
from typing import Dict, TypedDict
import json
import urllib.request
from langgraph.graph import StateGraph, END

# 1. Define shared dictionary structure to track operational workflow state
class RetentionStateSchema(TypedDict):
    customer_id: int
    churn_risk: float
    top_negative_reason: str
    tenure: int
    derived_retention_strategy: str
    execution_dispatch_log: str

# 2. Node Agent A: Strategy Formulation Logic (Hybrid LLM + Fallback Engine)
def retention_strategist_node(state: RetentionStateSchema) -> Dict:
    """
    Connects to a local open-weights model engine (Ollama/vLLM) to dynamically 
    analyze complaints and write custom recovery emails. Falls back to a deterministic
    rule engine if the local server is offline.
    """
    reason = state['top_negative_reason']
    risk = state['churn_risk']
    tenure = state['tenure']
    cust_id = state['customer_id']
    
    # Construct a descriptive prompt engineering profile for the agent
    system_prompt = (
        f"You are an expert customer save specialist. Write a highly tailored email to Customer #{cust_id}. "
        f"They have an active churn risk profile score of {risk*100:.1f}%, and have been subscribed for {tenure} months. "
        f"Their explicit grievance log is: '{reason}'. Write a professional, empathetic message under 4 sentences. "
        f"Include a concrete, custom remedy matching their exact technical or financial issue."
    )
    
    # Payload layout for a standard local Ollama endpoint configuration (running llama3.1 / mistral)
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1",  # Change to your locally deployed model tag index
        "prompt": system_prompt,
        "stream": False
    }
    
    try:
        # Establish low-overhead HTTP connection vectors without needing external massive client packages
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            ai_generated_copy = res_data.get('response', '')
            strategy = f"Autonomous LLM Engine Strategy Formulation: {ai_generated_copy}"
            
    except Exception as network_or_model_error:
        # Graceful operational fallback rule block if your local LLM server is offline or not installed
        reason_lower = reason.lower()
        if "billing" in reason_lower or "charged" in reason_lower:
            strategy = (
                f"Financial Rescue Loop Active. Target Risk: {risk:.2f}. Length: {tenure} months. "
                f"Action: Apply 1-month statement compensation credit and initiate immediate automated billing audit."
            )
        elif "downtime" in reason_lower or "broken" in reason_lower or "slow" in reason_lower:
            strategy = (
                f"SLA Mitigation Safeguard Active. Target Risk: {risk:.2f}. "
                f"Action: Migrate account configurations to premium high-throughput network nodes and provide a direct engineering point of contact."
            )
        else:
            strategy = (
                f"Standard Customer Loyalty Retention Active. Target Risk: {risk:.2f}. "
                f"Action: Distribute automated account health check invite bundled with a 20% membership renewal coupon."
            )
        
    return {"derived_retention_strategy": strategy}

# 3. Node Agent B: Automated Dispatch Log Emulator
def marketing_outbound_dispatcher_node(state: RetentionStateSchema) -> Dict:
    """Simulates API updates or CRM database outbound webhook entries."""
    strategy = state['derived_retention_strategy']
    cust_id = state['customer_id']
    
    dispatch_message = (
        f"CRM Webhook Outreach Fired Successfully -> Target Customer ID Key: #{cust_id}. "
        f"Executed operational playbook logic: [{strategy}]"
    )
    return {"execution_dispatch_log": dispatch_message}

# 4. Conditional State Routing Verification Gate
def risk_threshold_gate_rule(state: RetentionStateSchema) -> str:
    """Validates risk severity before passing records down the graph edges."""
    if state['churn_risk'] >= 0.60:
        return "route_to_remediation"
    return "route_to_exit"

# 5. Assemble and Compile the Complete LangGraph Pipeline
graph_builder = StateGraph(RetentionStateSchema)

# Register functional system nodes to the graph topology
graph_builder.add_node("RetentionStrategist", retention_strategist_node)
graph_builder.add_node("OutboundDispatcher", marketing_outbound_dispatcher_node)

# Map edge connection conditions and conditional routing paths
graph_builder.set_conditional_entry_point(
    risk_threshold_gate_rule,
    {
        "route_to_remediation": "RetentionStrategist",
        "route_to_exit": END
    }
)

graph_builder.add_edge("RetentionStrategist", "OutboundDispatcher")
graph_builder.add_edge("OutboundDispatcher", END)

# Final state machine graph engine compilation
compiled_retention_graph = graph_builder.compile()

# 6. Public Interface Hook called by main.py
def deploy_retention_campaign(at_risk_dataframe: pd.DataFrame):
    """Loops isolated EvoBoost cohorts through the compiled LangGraph orchestration architecture."""
    print(f"\n=== Triggering Automated LangGraph Recovery Campaigns ({len(at_risk_dataframe)} Users) ===")
    
    for _, row in at_risk_dataframe.iterrows():
        input_payload = {
            "customer_id": int(row['customer_id']),
            "churn_risk": float(row['churn_risk']),
            "top_negative_reason": str(row['top_negative_reason']),
            "tenure": int(row['tenure']),
            "derived_retention_strategy": "",
            "execution_dispatch_log": ""
        }
        
        execution_trace = compiled_retention_graph.invoke(input_payload)
        print(f"\n[Execution Track Result for Customer Account #{input_payload['customer_id']}]")
        print(json.dumps(execution_trace, indent=2))

if __name__ == "__main__":
    # Test dataset mock for independent standalone module verification
    mock_high_risk_data = pd.DataFrame({
        'customer_id': [1105, 1242],
        'churn_risk': [0.87, 0.64],
        'top_negative_reason': [
            "Your service has constant downtime. My team can't access our dashboard during market hours.",
            "Billing is broken. Charged twice this month and nobody in support is answering my chats."
        ],
        'tenure': [22, 5]
    })
    deploy_retention_campaign(mock_high_risk_data)