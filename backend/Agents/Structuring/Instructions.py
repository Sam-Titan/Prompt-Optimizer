from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from backend.Agents.Input.Goal_Specification import goal_specification

def Instruction(query, Estimation_Tokens):
    class Output_Schema(BaseModel):
        role: str = Field(description="The persona or expert role the LLM should adopt")
        instruction: str = Field(description="The primary directive for the LLM")
        action_verb: str = Field(description="The leading action verb")
        output_indicator: str = Field(description="The exact format and structure the output must follow")

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens = Estimation_Tokens,
        timeout=30,
        max_retries=2,
    )

    response_structure = model.with_structured_output(Output_Schema)

    system_prompt = """
You are an Instruction Agent operating inside a prompt engineering pipeline.

Your job:
Given a user goal, generate four things:

1. Role: The expert persona the LLM should adopt to answer this goal most effectively.
   - Format: "Act as a [specific expert title] with expertise in [specific domain]."
   - Be specific. Not "an expert" but "a senior business strategist specialising in emerging market entry."

2. Instruction: The precise directive telling the LLM exactly what to produce.
   - Start with a strong action verb: Generate, Analyze, Write, Extract, Compare, Design, Evaluate.
   - Include quantity, scope, and output type from the goal.
   - Preserve all specific details including numbers and formats.

3. Action Verb: The single leading verb from the instruction.

4. Output Indicator: The exact structure the response must follow.
   - Define every field the output must contain.
   - Be specific: "For each item provide: Name, Description, Market Opportunity, Implementation Steps, Estimated Capital Required."
   - If the task is a list, specify how each list item must be structured.

Rules:
- One role, one instruction, one output indicator only.
- Do not add constraints, examples, or context.
- Do not use vague verbs: help, discuss, talk about, tell, explore.
- No preamble. No explanation. Structured output only.
"""

    messages = [
        ("system", system_prompt),
        ("human", query),
    ]

    try:
        response = response_structure.invoke(messages)
        combined_response = f"Role: {response.role}\nInstruction: {response.instruction}\nAction Verb: {response.action_verb}\nOutput Indicator: {response.output_indicator}"
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