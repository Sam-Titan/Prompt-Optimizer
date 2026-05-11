from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from backend.Agents.Input.Goal_Specification import goal_specification

def Instruction(query, Estimation_Tokens):
    class Output_Schema(BaseModel):
        instruction: str = Field(description="The primary directive for the LLM")
        action_verb: str = Field(description="The leading action verb (Write, Analyze, Generate etc.)")

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens = Estimation_Tokens,
        timeout=30,
        max_retries=2,
    )

    response_structure = model.with_structured_output(Output_Schema)

    system_prompt = f"""
    You are an Instruction Agent operating inside a prompt engineering pipeline.

    Your job:
    Given a user goal, generate the core instruction a prompt engineer needs
    to build an effective prompt.

    The instruction means:
    - What exact task should the LLM perform?
    - What is the clearest actionable directive?
    - What is the primary action verb?

    Rules:
    - Start the instruction with a strong action verb.
    - Preserve the user's original intent.
    - Be specific and direct.
    - Define only one primary task.
    - Avoid vague verbs like:
    "help", "discuss", "talk about", "tell".
    - Prefer verbs like:
    "Write", "Analyze", "Generate",
    "Summarize", "Extract", "Classify",
    "Design", "Explain", "Compare".
    - If the task is vague, infer the most reasonable actionable instruction.
    - Do not add constraints, formatting, examples, or explanations.
    - Be concise. Stay under {Estimation_Tokens} tokens.
    - No preamble. No explanation. Structured output only.
    """

    messages = [
        ("system", system_prompt),
        ("human", query),
    ]

    try:
        response = response_structure.invoke(messages)
        combined_response = f"Instruction: {response.instruction}\nAction Verb: {response.action_verb}"
        if len(combined_response) < 20:
            print("Length of the response is too low")
            return False
        else:
            return combined_response
    except Exception as e:
        print(f"Exception occured {e}. Please retry again.")
        return False
    
if __name__ == "__main__":
    response = goal_specification("I want 5 great business Ideas for the real world", 100)
    if response == False:
        print("The Goal is not specified")
    else:
        instruction = Instruction(response, 300)
        print(instruction)