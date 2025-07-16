from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import OpenAIEmbeddings

import numpy as np


@dataclass
class ActionPlanningChain:
    """LangChain implementation of ``ActionPlanningAgent``."""

    llm: ChatOpenAI
    knowledge: str
    chain: LLMChain = field(init=False)

    def __post_init__(self) -> None:
        system_msg = (
            "You are an action planning agent tasked with creating a clear, ordered sequence of instructions for completing a product development task.\n"
            "Your output must:\n"
            "\u2013 Be a numbered list (e.g., 1., 2., 3.)\n"
            "\u2013 Follow this strict order:\n   1. Product Manager \u2192 2. Program Manager \u2192 3. Development Engineer\n"
            "\u2013 Each numbered step must be a single sentence, beginning with: 'You as the [role] should...'\n"
            "Only use these roles:\n"
            "\u2022 Product Manager: Defines user stories from the product spec.\n"
            "\u2022 Program Manager: Extracts features from user stories.\n"
            "\u2022 Development Engineer: Translates features into development tasks.\n"
            "Do not create role headers. Do not split instructions across lines.\n"
            "Each item in the list must be a full, standalone directive.\n"
            "Use this knowledge: {knowledge}"
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_msg),
            HumanMessagePromptTemplate.from_template("{input}"),
        ])
        self.chain = LLMChain(llm=self.llm, prompt=prompt.partial(knowledge=self.knowledge))

    def run(self, prompt: str) -> List[str]:
        response = self.chain.run({"input": prompt})
        steps = [s.strip() for s in response.split("\n") if s.strip() and not s.startswith("Step")]
        return steps


@dataclass
class KnowledgeAgent:
    """Knowledge augmented agent implemented with LangChain."""

    llm: ChatOpenAI
    persona: str
    knowledge: str
    chain: LLMChain = field(init=False)

    def __post_init__(self) -> None:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Forget all previous context. You are {persona}, a knowledge-based assistant.\n"
                "Use only the following knowledge to answer, do not use your own knowledge:\n"
                "KNOWLEDGE: {knowledge} KNOWLEDGE END\n"
                "Answer the prompt based on this knowledge, not your own."
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
        ])
        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt.partial(persona=self.persona, knowledge=self.knowledge),
        )

    def run(self, prompt: str) -> str:
        return self.chain.run({"input": prompt})


@dataclass
class EvaluationAgent:
    """Evaluation agent that checks responses from another agent."""

    llm: ChatOpenAI
    persona: str
    evaluation_criteria: str
    worker: KnowledgeAgent
    max_interactions: int = 10
    eval_chain: LLMChain = field(init=False)
    instruction_chain: LLMChain = field(init=False)

    def __post_init__(self) -> None:
        eval_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are {persona}, an impartial evaluation agent.\n"
                "You must decide if the following answer meets the evaluation criterion:\n"
                "CRITERION: {criteria}\n"
                "Your response must be strictly formatted:\n"
                "First line: 'Yes' or 'No'\n"
                "Second line: A one-sentence explanation why.\n"
                "Do not suggest improvements. Do not change or reinterpret the criterion."
            ),
            HumanMessagePromptTemplate.from_template("Answer: {answer}"),
        ])
        self.eval_chain = LLMChain(
            llm=self.llm,
            prompt=eval_prompt.partial(persona=self.persona, criteria=self.evaluation_criteria),
        )

        instruction_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are {persona}, a helpful assistant that generates concise correction instructions."
            ),
            HumanMessagePromptTemplate.from_template(
                "Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
            ),
        ])
        self.instruction_chain = LLMChain(
            llm=self.llm,
            prompt=instruction_prompt.partial(persona=self.persona),
        )

    def run(self, initial_prompt: str) -> dict[str, Any]:
        prompt_to_evaluate = initial_prompt
        worker_response = ""
        evaluation = ""
        for i in range(self.max_interactions):
            worker_response = self.worker.run(prompt_to_evaluate)
            evaluation = self.eval_chain.run({"answer": worker_response}).strip()
            if evaluation.lower().startswith("yes"):
                break
            instructions = self.instruction_chain.run({"evaluation": evaluation}).strip()
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


@dataclass
class Route:
    name: str
    description: str
    func: Callable[[str], Any]
    embedding: List[float] | None = None


class EmbeddingRouter:
    """Simple router that selects the best route based on embedding similarity."""

    def __init__(self, routes: List[Route], embeddings: OpenAIEmbeddings) -> None:
        self.routes = routes
        self.embeddings = embeddings
        for route in self.routes:
            route.embedding = self.embeddings.embed_query(route.description)

    def route(self, prompt: str) -> Any:
        input_emb = self.embeddings.embed_query(prompt)
        best = None
        best_score = -1.0
        for route in self.routes:
            emb = route.embedding
            score = float(np.dot(input_emb, emb) / (np.linalg.norm(input_emb) * np.linalg.norm(emb)))
            if score > best_score:
                best_score = score
                best = route
        if best is None:
            return "Sorry, no suitable agent could be selected."
        return best.func(prompt)
