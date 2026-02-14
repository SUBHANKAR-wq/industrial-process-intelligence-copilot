from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from src.schema.root_cause_schema import RootCauseOutput

load_dotenv()

def root_cause_agent(sensor_analysis, state):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    ).with_structured_output(RootCauseOutput)

    prompt = PromptTemplate(
        input_variables=[
            "dominant_sensors",
            "severity",
            "confidence",
            "state"
        ],
        template="""
You are an industrial diagnostics expert.

Given:
- Dominant sensors: {dominant_sensors}
- System state: {state}
- Severity: {severity}
- Detection confidence: {confidence}

Infer the physical root cause.
"""
    )

    return llm.invoke(
        prompt.format(
            dominant_sensors=", ".join(sensor_analysis["dominant_sensors"]),
            severity=sensor_analysis["severity"],
            confidence=sensor_analysis["confidence"],
            state=state
        )
    )
