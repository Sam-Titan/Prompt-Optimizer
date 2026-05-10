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
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens = tokens,
        timeout=30,
        max_retries=2,
    )

    response_structure = model.with_structured_output(OutputSchema)
    system_prompt = f"""
You are a Goal Extraction Agent.

Your task:
- Extract the user's TRUE underlying goal from their query.
- If the query is vague or uncertain, infer the MOST LIKELY goal.
- Do NOT repeat the query.

Output format (STRICT):
Goal: <one clear sentence>
Confidence: <high/medium/low>

Rules:
- Keep response under {tokens} tokens.
- Be specific.
- If ambiguity exists, reflect it in confidence instead of guessing wildly.
"""

    messages = [
        ("system", system_prompt),
        ("human", query),
    ]

    try:
        Goal = response_structure.invoke(messages)
    except TimeoutError:
        return "The LLM model timed out. Retry Again!"
    
    if Goal.confidence == "low":
        return False
    else:
        return Goal.answer
    

if __name__ == "__main__":
    response = goal_specification("I am not sure.", 100)
    print(response)