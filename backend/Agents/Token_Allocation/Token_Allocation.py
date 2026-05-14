from langchain_groq import ChatGroq
from backend.Agents.Input.Goal_Specification import goal_specification
from pydantic import BaseModel, Field

def Token_Estimation(query, Estimation_Tokens):

    class Output_Schema(BaseModel):
        Complexity:int = Field(description="Contains the complexity of the prompt")
        Tokens:int = Field(description="Contains the Token needed for the prompt")

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens = Estimation_Tokens,
        timeout=30,
        max_retries=2,
    )
    response_structure = model.with_structured_output(Output_Schema)

    token_prompt = """Estimate the complexity of the following task on a scale of 1–10,
    and estimate how many reasoning tokens are needed (1 = ~50 tokens, 10 = ~2000 tokens).

    Respond only in this format:
    complexity: N, tokens: N

    Task: f{query} """

    try:
        response = response_structure.invoke(token_prompt)
    except Exception:
        print("Exception occured. Please retry again.")
    
    Tokens_Required = response.Tokens
    Determined_Complexity = response.Complexity

    Tokens_Required = max(Tokens_Required, 100)   # minimum floor
    Tokens_Required = min(Tokens_Required, 1500)  # max cap

    return Determined_Complexity, Tokens_Required

if __name__ == "__main__":
    response = goal_specification("I want to find the mountain or something similar. I am not sure.", 100)
    if response == False:
        print("The Goal is not specified")
    else:
        Complexity, Tokens_Required = Token_Estimation(response, 100)
        print(Complexity, Tokens_Required)