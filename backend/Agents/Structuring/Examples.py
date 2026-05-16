from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from backend.Agents.Input.Goal_Specification import goal_specification
from typing import Optional

def Examples(query, Estimation_Tokens):
    class Output_Schema(BaseModel):
        examples_needed: bool = Field(description="Whether examples are needed")
        examples: Optional[list[str]] = Field(description="List of input-output example pairs")

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens = Estimation_Tokens,
        timeout=30,
        max_retries=2,
    )

    response_structure = model.with_structured_output(Output_Schema)

    system_prompt = f"""
    You are an Examples Agent operating inside a prompt engineering pipeline.

    Your job:
    Given a user goal, determine whether few-shot examples
    are needed to improve prompt reliability and output quality.

    Examples are mainly needed when:
    - The output format is non-obvious.
    - The task involves classification.
    - The task requires structured outputs.
    - The task involves style imitation or pattern learning.
    - The task could benefit from demonstration-based guidance.

    Rules:
    - Set examples_needed to true only if examples would significantly
    improve consistency or clarity.
    - If examples are not needed, return examples_needed as false
    and examples as null.
    - Generate concise, high-quality examples only when necessary.
    - Examples must directly reflect the intended task pattern.
    - Prefer diverse examples over repetitive ones.
    - Keep examples short and easy to generalize from.
    - Do not generate unnecessary examples for simple direct tasks.
    - Do not add explanations or commentary.
    - Be concise. Stay under {Estimation_Tokens} tokens.
    - No preamble. No explanation. Structured output only.
    """
    messages = [
        ("system", system_prompt),
        ("human", query),
    ]

    try:
        response = response_structure.invoke(messages)
        combined_response = f"Examples Needed: {response.examples_needed}\nExamples: {response.examples}"
        if len(combined_response) < 20:
            print("Length of the response is too low")
            return False
        else:
            return combined_response
    except Exception as e:
        print(f"Exception occured {e}. Please retry again.")
        return False
    
if __name__ == "__main__":
    response = goal_specification("Extract the name, email, and phone number from resumes.", 100)
    if response == False:
        print("The Goal is not specified")
    else:
        print(response)
        example = Examples(response, 300)
        print(example)