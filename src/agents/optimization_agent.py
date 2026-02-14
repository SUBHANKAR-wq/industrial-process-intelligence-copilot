from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from src.schema.optimization_schema import OptimizationOutput
import os
# load_dotenv() when u will run locally add it

def optimization_agent(sensor_analysis, root_cause):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GEMINI_API_KEY") #remove when u will run locally add it in .env file and load_dotenv()
    ).with_structured_output(OptimizationOutput)

    prompt = PromptTemplate(
        input_variables=[
            "severity",
            "probable_cause",
            "confidence"
        ],
        template="""
You are an industrial operations planner.

Given:
- Severity: {severity}
- Probable cause: {probable_cause}
- Detection confidence: {confidence}

Decide the best operational action.
"""
    )

    return llm.invoke(
        prompt.format(
            severity=sensor_analysis["severity"],
            probable_cause=root_cause.probable_cause,
            confidence=sensor_analysis["confidence"]
        )
    )
