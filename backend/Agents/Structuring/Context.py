from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain.tools import tool
from backend.Agents.Input.Goal_Specification import goal_specification

def context(query, Estimation_Tokens):
    class Output_Schema(BaseModel):
        Context:str = Field(description="Context for the query")

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens = Estimation_Tokens,
        timeout=30,
        max_retries=2,
    )

    response_structure = model.with_structured_output(Output_Schema)

    system_prompt = f"""
    You are a Context Generating Agent.

    Your task:
    - Build Context from their query.
    - If the query is vague or uncertain, infer the MOST LIKELY context.
    - Do NOT repeat the context.

    Output format (STRICT):
    Context: <bullet points>

    Rules:
    - Keep response under {Estimation_Tokens} tokens.
    - Three lines only.
    - If ambiguity exists, reflect it in confidence.
    """

    messages = [
        ("system", system_prompt),
        ("human", query),
    ]

    try:
        response = response_structure.invoke(messages)
    except Exception:
        print("Exception occured. Please retry again.")

    return response.Context

if __name__ == "__main__":
    response = goal_specification("I want to find the mountain or something similar. I am not sure.", 100)
    if response == False:
        print("The Goal is not specified")
    else:
        Context = context(response, 150)
        print(Context)