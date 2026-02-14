import json
from pipeline.data_generator import data_generator
from pipeline.state import classify_state   
from pipeline.event_gate import event_gate
from pipeline.preprocessing import preprocess
from pipeline.inference import run_inference
from src.orchestration.agent_orchestrator import agent_orchestrator
with open("models/thresholds.json", "r") as f:
    thresholds = json.load(f)
def main_pipeline(data):

    #  Generate dataframe
    df = data_generator(data)

    #  Preprocess
    X_scaled = preprocess(df)

    #  Inference 
    inference_df = run_inference(df,X_scaled)

    #  State classification
    inference_df["state"] = (
        inference_df["reconstruction_error"]
        .apply(classify_state)
    )

    #  Event decision
    events = event_gate(
        inference_df["state"].iloc[-1]
    )


    agent_response = None

    if events["trigger_agent"]:
        agent_response = agent_orchestrator(inference_df.iloc[-1])

    return {
        "anomaly_score": inference_df["reconstruction_error"].iloc[-1],
        "state": inference_df["state"].iloc[-1],
        "normal_threshold": thresholds["normal_threshold"],
        "anomaly_threshold" : thresholds["anomaly_threshold"],
        "sensor_analysis": agent_response["sensor_analysis"] if agent_response else None,
        "root_cause": agent_response["root_cause"] if agent_response else None,
        "optimization_decision": agent_response["decision"] if agent_response else None,
        "report": agent_response["report"] if agent_response else None
    }