from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from src.schema.report_schema import ReportOutput  # if you use schema

load_dotenv()

def report_generator_agent(context):

    root_cause = context["root_cause"]        # Pydantic object
    decision = context["decision"]            # Pydantic object
    sensor_analysis = context["sensor_analysis"]

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    prompt = PromptTemplate(
        input_variables=[
            "timestamp",
            "state",
            "severity",
            "dominant_sensors",
            "probable_cause",
            "explanation",
            "recommended_action",
            "confidence"
        ],
        template="""
You are an industrial monitoring assistant.

Generate a concise report using ONLY the information below.

Timestamp: {timestamp}
State: {state}
Severity: {severity}
Dominant sensors: {dominant_sensors}
Probable cause: {probable_cause}
Explanation: {explanation}
Recommended action: {recommended_action}
Detection confidence: {confidence}

Respond clearly and professionally.
"""
    )

    response = model.invoke(
        prompt.format(
            timestamp=context["timestamp"],
            state=context["state"],
            severity=sensor_analysis["severity"],
            dominant_sensors=", ".join(sensor_analysis["dominant_sensors"]),
            probable_cause=root_cause.probable_cause,        
            explanation=root_cause.explanation,              
            recommended_action=decision.recommended_action, 
            confidence=sensor_analysis["confidence"]
        )
    )

    return response.content
