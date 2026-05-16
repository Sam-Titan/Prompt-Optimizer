from langchain_groq import ChatGroq
from pydantic import BaseModel
from backend.Agents.Input.Goal_Specification import goal_specification
from typing import Optional

def context(query, Estimation_Tokens):
    class Output_Schema(BaseModel):
        Domain: Optional[str] = None
        Constraints: Optional[str] = None
        Audience: Optional[str] = None
        Tone: Optional[str] = None

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens = Estimation_Tokens,
        timeout=30,
        max_retries=2,
    )

    response_structure = model.with_structured_output(Output_Schema)

    system_prompt = f"""
    You are a Context Agent operating inside a prompt engineering pipeline.

    Your job:
    Given a user goal, extract the context a prompt engineer needs to build an effective prompt.

    Context means:
    - Domain (what field or subject does this goal belong to?)
    - Constraints (what limitations or boundaries are implied?)
    - Audience (who is the end user or target of this output?)
    - Tone (formal, technical, casual — what does the goal imply?)

    Rules:
    - If a dimension is clearly inferable, state it directly.
    - If a dimension cannot be inferred, make the most reasonable default assumption and prefix it with "Assumed:".
    - Never omit a dimension entirely. Always populate all four fields.
    - Be concise. Stay under {Estimation_Tokens} tokens.
    - No preamble. No explanation. Structured output only.
    """

    messages = [
        ("system", system_prompt),
        ("human", query),
    ]

    try:
        response = response_structure.invoke(messages)
        combined_response = f"Domain: {response.Domain}\nConstraints: {response.Constraints}\nAudience: {response.Audience}\nTone: {response.Tone}"
        if len(combined_response) < 20:
            print("Length of the response is too low")
            return False
        else:
            return combined_response
    except Exception as e:
        print(f"Exception occured {e}. Please retry again.")
        return False
    

if __name__ == "__main__":
    response = goal_specification("I want to climb a mountain.", 100)
    if response == False:
        print("The Goal is not specified")
    else:
        Context = context(response, 300)
        print(Context)