def event_gate(state):
    
    state = state.lower()
    
    trigger_agent = state in ["anomaly", "drift"]

    return {
        "trigger_agent": trigger_agent
    }