from langchain_groq import ChatGroq
from backend.Agents.Input.Goal_Specification import goal_specification
from backend.Agents.TokenAllocation.Token_Allocation import Token_Estimation
from backend.Agents.Structuring.Context import context
from backend.Agents.Structuring.Instructions import Instruction
from backend.Agents.Structuring.Examples import Examples
from backend.Agents.Structuring.Query import Query
import json

def Deterministic_Constraints(query, tokens, query_output, examples_output):
    complexity, tokens_required = Token_Estimation(query, tokens)

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

    print(f"Debug — complexity: {complexity}, query_type: {query_type}, is_complete: {is_complete}, examples_needed: {examples_needed}")
    return active_constraints

if __name__ == "__main__":
    response = goal_specification("I want 5 great business Ideas for the real world", 100)
    if response == False:
        print("The Goal is not specified")
    else:
        query_output = Query(response, 100)
        examples_output = Examples(response, 200)
        Constraints = Deterministic_Constraints(response, 30, query_output, examples_output)
        print(Constraints)