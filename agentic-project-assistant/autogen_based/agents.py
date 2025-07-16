"""AutoGen-based agent implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import autogen

# Basic AutoGen config helper

def _default_llm_config(api_key: str, base_url: str, model: str = "gpt-3.5-turbo") -> dict:
    """Return a basic llm_config dictionary for AutoGen agents."""
    return {
        "config_list": [
            {"model": model, "api_key": api_key, "base_url": base_url}
        ],
        "cache_seed": 42,
    }


@dataclass
class ActionPlanningAgent:
    """Agent that extracts ordered action steps from a prompt."""

    llm_config: dict
    knowledge: str

    def __post_init__(self) -> None:
        system_message = (
            "You are an action planning agent tasked with creating a clear, ordered "
            "sequence of instructions for completing a product development task.\n"
            "Your output must:\n"
            "- Be a numbered list (e.g., 1., 2., 3.)\n"
            "- Follow this strict order: 1. Product Manager -> 2. Program Manager -> 3. Development Engineer\n"
            "- Each numbered step must be a single sentence beginning with 'You as the [role] should...'\n"
            "Only use these roles:\n"
            "• Product Manager: Defines user stories from the product spec.\n"
            "• Program Manager: Extracts features from user stories.\n"
            "• Development Engineer: Translates features into development tasks.\n"
            "Do not create role headers. Do not split instructions across lines.\n"
            "Each item in the list must be a full, standalone directive.\n"
            f"Use this knowledge: {self.knowledge}"
        )
        self.assistant = autogen.AssistantAgent(
            name="action_planner",
            llm_config=self.llm_config,
            system_message=system_message,
        )
        self.user = autogen.UserProxyAgent(
            name="action_planner_user",
            human_input_mode="NEVER",
        )

    def extract_steps_from_prompt(self, prompt: str) -> List[str]:
        self.user.initiate_chat(self.assistant, message=prompt, summary_method="last_msg")
        response = self.user.last_message()["content"].strip()
        steps = [s.strip() for s in response.split("\n") if s.strip() and not s.startswith("Step")]
        return steps

    def respond(self, prompt: str) -> Any:
        return self.extract_steps_from_prompt(prompt)


@dataclass
class KnowledgeAgent:
    """Knowledge augmented agent implemented with AutoGen."""

    llm_config: dict
    persona: str
    knowledge: str

    def __post_init__(self) -> None:
        system_message = (
            f"Forget all previous context. You are {self.persona}.\n"
            "Use only the following knowledge to answer, do not use your own knowledge:\n"
            f"KNOWLEDGE: {self.knowledge} KNOWLEDGE END\n"
            "Answer the prompt based on this knowledge, not your own."
        )
        self.assistant = autogen.AssistantAgent(
            name=self.persona.replace(" ", "_").lower(),
            llm_config=self.llm_config,
            system_message=system_message,
        )
        self.user = autogen.UserProxyAgent(
            name=f"{self.assistant.name}_user",
            human_input_mode="NEVER",
        )

    def respond(self, prompt: str) -> str:
        self.user.initiate_chat(self.assistant, message=prompt, summary_method="last_msg")
        return self.user.last_message()["content"].strip()


@dataclass
class EvaluationAgent:
    """Evaluation agent that checks responses from another agent."""

    llm_config: dict
    persona: str
    evaluation_criteria: str
    worker: KnowledgeAgent
    max_interactions: int = 10

    def __post_init__(self) -> None:
        eval_system_message = (
            f"You are {self.persona}, an impartial evaluation agent.\n"
            f"You must decide if the following answer meets the evaluation criterion:\n"
            f"CRITERION: {self.evaluation_criteria}\n"
            "Your response must be strictly formatted:\n"
            "First line: 'Yes' or 'No'\n"
            "Second line: A one-sentence explanation why.\n"
            "Do not suggest improvements. Do not change or reinterpret the criterion."
        )
        self.evaluator = autogen.AssistantAgent(
            name="evaluator",
            llm_config=self.llm_config,
            system_message=eval_system_message,
        )
        instruction_system_message = (
            f"You are {self.persona}, a helpful assistant that generates concise correction instructions."
        )
        self.instructor = autogen.AssistantAgent(
            name="instructor",
            llm_config=self.llm_config,
            system_message=instruction_system_message,
        )
        self.user = autogen.UserProxyAgent(
            name="evaluation_user",
            human_input_mode="NEVER",
        )

    def evaluate_once(self, answer: str) -> str:
        self.user.initiate_chat(
            self.evaluator,
            message=f"Answer: {answer}",
            summary_method="last_msg",
        )
        return self.user.last_message()["content"].strip()

    def get_instructions(self, evaluation: str) -> str:
        self.user.initiate_chat(
            self.instructor,
            message=f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}",
            summary_method="last_msg",
        )
        return self.user.last_message()["content"].strip()

    def run(self, initial_prompt: str) -> dict[str, Any]:
        prompt_to_evaluate = initial_prompt
        worker_response = ""
        evaluation = ""
        for i in range(self.max_interactions):
            worker_response = self.worker.respond(prompt_to_evaluate)
            evaluation = self.evaluate_once(worker_response)
            if evaluation.lower().startswith("yes"):
                break
            instructions = self.get_instructions(evaluation)
            prompt_to_evaluate = (
                f"The original prompt was: {initial_prompt}\n"
                f"The response to that prompt was: {worker_response}\n"
                f"It has been evaluated as incorrect.\n"
                f"Make only these corrections, do not alter content validity: {instructions}"
            )
        return {
            "final_response": worker_response,
            "evaluation": evaluation,
            "iterations": i + 1,
        }

    def respond(self, prompt: str) -> Any:
        return self.run(prompt)

