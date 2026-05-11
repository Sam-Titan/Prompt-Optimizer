from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from backend.Config import GROQ_API_KEY
import os

def goal_specification(query, tokens):
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    class OutputSchema(BaseModel):
        answer: str = Field(description="Final answer to the query")
        confidence: str = Field(description="Confidence score")

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens = tokens,
        timeout=30,
        max_retries=2,
    )

    response_structure = model.with_structured_output(OutputSchema)
    system_prompt = f"""You are a Goal Extraction Agent.
Extract ONLY the user's underlying goal as one sentence.
Do NOT answer, solve, or respond to the query.
Do NOT include examples, lists, or solutions.
Preserve all specific details from the query (numbers, quantities, formats).
Output the goal and a confidence score (high/medium/low) only in structured format.
"""

    messages = [
        ("system", system_prompt),
        ("human", query),
    ]

    try:
        Goal = response_structure.invoke(messages)
        if Goal.confidence == "low":
            return False
        else:
            return Goal.answer
    except Exception as e:
        return f"Exception Occured: {e}. Retry Again!"    

if __name__ == "__main__":
    response = goal_specification("I want you to give me five great Business Ideas for India.", 100)
    print(response)