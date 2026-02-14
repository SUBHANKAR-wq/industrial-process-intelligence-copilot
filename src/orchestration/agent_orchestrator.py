def agent_orchestrator(row):
    """
    Orchestrates the full agent pipeline for a single event.
    Uses structured outputs (Pydantic) for LLM agents.
    """

    # Shared Context (in-memory)
    
    shared_context = {
        "timestamp": row["timestamp"],
        "state": row["state"],
        "reconstruction_error": row["reconstruction_error"],
    }

   
    # 1. Sensor Analyst Agent (NON-LLM)
    
    from src.agents.sensor_analyst_agent import sensor_analyst_agent

    sensor_analysis = sensor_analyst_agent(row)
    shared_context["sensor_analysis"] = sensor_analysis

    # Early exit if system is normal
    if row["state"] == "normal":
        shared_context["status"] = "System operating normally"
        return shared_context

    # 2. Root Cause Agent (LLM + Structured Output)
   
    from src.agents.root_cause_agent import root_cause_agent

    root_cause = root_cause_agent(
        sensor_analysis=sensor_analysis,
        state=row["state"]
    )
    shared_context["root_cause"] = root_cause

 
    # 3. Optimization Agent (LLM + Structured Output)

    from src.agents.optimization_agent import optimization_agent

    decision = optimization_agent(
        sensor_analysis=sensor_analysis,
        root_cause=root_cause
    )
    shared_context["decision"] = decision

 
    # 4. Report Generator Agent (LLM)
 
    from src.agents.report_generator_agent import report_generator_agent

    report = report_generator_agent(shared_context)
    shared_context["report"] = report


    return shared_context
