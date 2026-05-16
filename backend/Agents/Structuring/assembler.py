from backend.Agents.Input.Goal_Specification import goal_specification
from backend.Agents.TokenAllocation.Token_Allocation import Token_Estimation
from backend.Agents.Structuring.Context import context
from backend.Agents.Structuring.Instructions import Instruction
from backend.Agents.Structuring.Examples import Examples
from backend.Agents.Structuring.Query import Query
import json

def Deterministic_Constraints(complexity, tokens_required, query_output, examples_output):
    with open("backend/PromptConstraints/Constraint.json", mode="r") as f:
        data = json.load(f)

    universal = data["prompt_constraints"]["universal"]
    conditional_data = data["prompt_constraints"]["conditional"]
    verbosity = data["prompt_constraints"]["verbosity"]["constraint"].replace("[TOKENS]", str(tokens_required))

    query_type = None
    is_complete = True
    examples_needed = False

    if query_output:
        for line in query_output.split("\n"):
            if line.startswith("Query Type:"):
                query_type = line.split(":", 1)[1].strip().lower()
            if line.startswith("Is_complete:"):
                is_complete = line.split(":", 1)[1].strip().lower() == "true"

    if examples_output:
        for line in examples_output.split("\n"):
            if line.startswith("Examples Needed:"):
                examples_needed = line.split(":", 1)[1].strip().lower() == "true"

    active_constraints = []
    active_constraints.extend(universal)
    active_constraints.append(verbosity)

    trigger_map = {
        "complexity_high": complexity >= 7,
        "complexity_mid": complexity >= 5 and complexity < 7,
        "complexity_low": complexity <= 3,
        "query_type_factual": query_type == "factual",
        "query_type_creative": query_type == "creative",
        "query_type_analytical": query_type == "analytical",
        "query_type_comparative": query_type == "comparative",
        "query_type_data_extraction": query_type == "data-extraction",
        "query_type_multi_step": query_type == "multi-step",
        "examples_present": examples_needed == True,
        "incomplete_query": is_complete == False,
    }

    for key, condition in trigger_map.items():
        if condition and key in conditional_data:
            active_constraints.append(conditional_data[key]["constraint"])

    return active_constraints


def assembler(instruction, context_output, examples_output, query_output, constraints):
    instruction_clean = None
    query_clean = None
    examples_needed = False
    examples_clean = None

    if instruction:
        for line in instruction.split("\n"):
            if line.startswith("Instruction:"):
                instruction_clean = line.split(":", 1)[1].strip()

    if query_output:
        for line in query_output.split("\n"):
            if line.startswith("Query:"):
                query_clean = line.split(":", 1)[1].strip()
            if line.startswith("Examples Needed:"):
                examples_needed = line.split(":", 1)[1].strip().lower() == "true"

    if examples_output:
        for line in examples_output.split("\n"):
            if line.startswith("Examples Needed:"):
                examples_needed = line.split(":", 1)[1].strip().lower() == "true"
            if line.startswith("Examples:"):
                examples_clean = line.split(":", 1)[1].strip()

    constraints_text = "\n".join([f"- {c}" for c in constraints])

    role_clean = None
    output_indicator_clean = None

    if instruction:
        for line in instruction.split("\n"):
            if line.startswith("Role:"):
                role_clean = line.split(":", 1)[1].strip()
            if line.startswith("Instruction:"):
                instruction_clean = line.split(":", 1)[1].strip()
            if line.startswith("Output Indicator:"):
                output_indicator_clean = line.split(":", 1)[1].strip()

    sections = []
    if role_clean:
        sections.append(f"{role_clean}")
    sections.append(f"Instruction:\n{instruction_clean}")
    sections.append(f"Context:\n{context_output}")
    if examples_needed and examples_clean:
        sections.append(f"Examples:\n{examples_clean}")
    if output_indicator_clean:
        sections.append(f"Output Format:\n{output_indicator_clean}")
    sections.append(f"Constraints:\n{constraints_text}")
    sections.append(f"Query:\n{query_clean}")

    optimized_prompt = "\n\n".join(sections)
    return optimized_prompt


if __name__ == "__main__":
    response = goal_specification("I want 5 great business Ideas for the real world", 100)
    if response == False:
        print("The Goal is not specified")
    else:
        complexity, tokens_required = Token_Estimation(response, 100)
        query_output = Query(response, tokens_required)
        examples_output = Examples(response, tokens_required)
        constraints = Deterministic_Constraints(complexity, tokens_required, query_output, examples_output)
        instruction = Instruction(response, tokens_required)
        context_output = context(response, tokens_required)
        prompt = assembler(instruction, context_output, examples_output, query_output, constraints)
        print(prompt)