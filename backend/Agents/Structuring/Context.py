from langchain_groq import ChatGroq
from pydantic import BaseModel
from backend.Agents.Input.Goal_Specification import goal_specification
from typing import Optional

def context(query, Estimation_Tokens):
    class Output_Schema(BaseModel):
        Domain: Optional[str] = None
        Scope: Optional[str] = None
        Audience: Optional[str] = None
        Temporal: Optional[str] = None

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens = Estimation_Tokens,
        timeout=30,
        max_retries=2,
    )

    response_structure = model.with_structured_output(Output_Schema)

    system_prompt = fsystem_prompt = f"""
You are a Context Agent operating inside a prompt engineering pipeline.

Your job:
Given a user goal, extract the context a prompt engineer needs to build an effective prompt.

Context means:
- Domain: The specific field or subject this goal belongs to. Be precise not generic.
- Scope: The explicit or implied boundaries of the task. Include quantity, geography, constraints, or limitations directly stated or strongly implied by the goal.
- Audience: The intended consumer of the final output. State who will use or benefit from the result.

- Temporal: The timeframe relevant to this goal. 
  If the goal implies recency or a specific period, state it explicitly.
  If not inferable, default to the current year and prefix with Assumed:.

Rules:
- If a dimension is clearly inferable, state it with specificity. Do not use generic labels.
- If a dimension cannot be inferred, make the most reasonable specific assumption and prefix it with Assumed:.
- Never omit a dimension. Always populate all three fields.
- Be specific. Vague outputs like real-world applicability or general audience are not acceptable.
- No preamble. No explanation. Structured output only.
- Scope must not restate the goal. Scope means boundaries: industry sector, 
  geographic market, capital range, time horizon, or domain limitations 
  directly stated or strongly implied by the goal.
- If no boundaries are inferable, assume the most reasonable specific defaults 
  and prefix with Assumed:.
"""

    messages = [
        ("system", system_prompt),
        ("human", query),
    ]

    try:
        response = response_structure.invoke(messages)
        combined_response = f"Domain: {response.Domain}\nScope: {response.Scope}\nAudience: {response.Audience}\nTemporal: {response.Temporal}"
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