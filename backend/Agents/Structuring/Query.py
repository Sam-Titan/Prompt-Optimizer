from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Optional
from backend.Agents.Input.Goal_Specification import goal_specification

def Query(query, Estimated_Tokens):
    class Output_Schema(BaseModel):
        query: str = Field(
            description="The cleaned, unambiguous version of the user input as it will appear in the final prompt"
        )
        query_type: str = Field(
            description="Classification of the query: factual, analytical, creative, advisory, comparative, multi-step, data-extraction"
        )
        placeholder_label: str = Field(
            description="The label wrapping the query in the final prompt structure. Example: 'User Query', 'Input', 'Task'"
        )
        is_complete: bool = Field(
            description="True if the query contains enough information to act on. False if critical information is missing."
        )
        missing_information: Optional[str] = Field(
            default=None,
            description="If is_complete is False, state exactly what information is missing. Otherwise null."
    )
    model = ChatGroq(
        model= "llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=Estimated_Tokens,
        timeout=30,
        max_retries=2,
    )

    response_structured = model.with_structured_output(Output_Schema)

    system_prompt = """
    You are a Query Agent operating inside a prompt engineering pipeline.

    Your input is a structured goal specification.
    Your job is to convert it into a clean, unambiguous query
    that will sit inside the final assembled prompt as the Input Data element.

    Rules:
    - Preserve the user's intent exactly. Do not add or remove meaning.
    - Write the query as a direct statement or question an LLM can act on immediately.
    - Do not write an instruction. Do not add context or constraints.
    - Classify the query type: factual, analytical, creative, advisory,
    comparative, multi-step, or data-extraction.
    - Choose the correct placeholder label:
    "User Query" for most tasks.
    "Input" when structured data or text is being processed.
    "Task" when the query is a direct action command.
    - Mark is_complete as False only when critical information
    is absent and cannot be reasonably inferred.
    State exactly what is missing — not a general observation.
    - No preamble. No explanation. Structured output only.
    """

    messages = [
        ("system", system_prompt),
        ("human", query),
    ]

    try:
        response = response_structured.invoke(messages)
        combined_response = f"Query: {response.query}\nQuery Type: {response.query_type}\nPlaceholder Label: {response.placeholder_label}\nIs_complete: {response.is_complete}\nMissing Information: {response.missing_information}"
        if len(combined_response) < 20:
            print("The length of the response is too low.")
            return False
        else:
            return combined_response
    except Exception as e:
        print(f"Exception occured {e}. Please retry again.")
        return False
    
if __name__ == "__main__":
    response = goal_specification("Extract the name, email, and phone number from resumes.", 100)
    if response == "False":
        print("The Query is too vague. Retry Again.")
    else:
        query = Query(response, 100)
        print(query)