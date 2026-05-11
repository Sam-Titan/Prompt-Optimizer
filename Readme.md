# Building a Prompt Optimization Engine

## Input
### Goal Specification Agent

I have built this Agent to determine the End Goal of the user if it is undetermined or vague. 
If the Goal is too vague the agent is designed to ask the User again for the prompt

## Token Allocation
### Token Estimation Agent

Through TALE-EP (Token-Aware Learning + Estimation Prompting), the system will determine the tokens needed for the end prompt.

This Agent will utilize the Step 1 of the method

## Structuring
### Context

I have built this Agent to determine the Context regarding the user Query. 
It's Schema consists of Domain, Audience, Constraints, and Tone.
