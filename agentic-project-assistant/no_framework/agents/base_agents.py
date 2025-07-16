from .openai_service import OpenAIService
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, List
from utils.logging_config import logger

from enum import Enum


class OpenAIModel(str, Enum):
    GPT_35_TURBO = "gpt-3.5-turbo"  # Default model for most tasks, good balance of cost and performance.
    GPT_41 = "gpt-4.1"  # Strong default choice for development tasks, particularly those requiring speed, responsiveness, and general-purpose reasoning.
    GPT_41_MINI = "gpt-4.1-mini"  # Fast and affordable, good for brainstorming, drafting, and tasks that don't require the full power of GPT-4.1.
    GPT_41_NANO = "gpt-4.1-nano"  # The fastest and cheapest model, suitable for lightweight tasks, high-frequency usage, and edge computing.


MODEL = OpenAIModel.GPT_35_TURBO  # Default model for this project


@dataclass
class Route:
    """Represents a routing option for the :class:`RoutingAgent`."""

    name: str
    description: str
    func: Callable[[str], str]
    embedding: List[float] = field(default_factory=list)


# DirectPromptAgent class definition
class DirectPromptAgent:

    def __init__(self, openai_service: OpenAIService):
        """Initialize the agent with a shared OpenAI service."""
        self.openai_service = openai_service

    def respond(self, prompt):
        # Generate a response using the OpenAI API through the service
        response = self.openai_service.chat(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        # Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content.strip()


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_service: OpenAIService, persona):
        """Initialize the agent with given attributes."""
        self.persona = persona
        self.openai_service = openai_service

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        # Declare a variable 'response' that calls OpenAI's API for a chat completion.
        response = self.openai_service.chat(
            model=MODEL,
            messages=[
                # Add a system prompt instructing the agent to assume the defined persona and explicitly forget previous context.
                {
                    "role": "system",
                    "content": f"Forget all previous context. You are {self.persona}.",
                },
                # A funny side note: If you add the phrase: Forget all previous context at the end of the system message, it will immediately forget your persona ;-)
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )

        # Return only the textual content of the response, not the full JSON payload.
        return response.choices[0].message.content.strip()


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_service: OpenAIService, persona, knowledge):
        """Initialize the agent with provided attributes."""

        self.persona = persona
        self.knowledge = knowledge
        self.openai_service = openai_service

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        response = self.openai_service.chat(
            model=MODEL,
            messages=[
                # Construct a system message including:
                # - The persona with the following instruction:
                #  "You are _persona_ knowledge-based assistant. Forget all previous context."
                # - The provided knowledge with this instruction:
                #  "Use only the following knowledge to answer, do not use your own knowledge: _knowledge_"
                # - Final instruction:
                #  "Answer the prompt based on this knowledge, not your own."
                {
                    "role": "system",
                    "content": f"""\
                    Forget all previous context.
                    You are {self.persona}, a knowledge-based assistant.
                    Use only the following knowledge to answer, do not use your own knowledge:
                    KNOWLEDGE: {self.knowledge} KNOWLEDGE END
                    Answer the prompt based on this knowledge, not your own.""",
                },
                # Add the user's input prompt here as a user message.
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(
        self, openai_service: OpenAIService, persona, chunk_size=2000, chunk_overlap=100
    ):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_service (OpenAIService): Service for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_service = openai_service
        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        )

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        response = self.openai_service.embed(
            model="text-embedding-3-large", input=text, encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # Remark:
    # I allowed myself to rewrite the chunk code because the original code was not working correctly
    # and resulted in infinite loop. The logic was broken and the chunking was not done properly.
    #
    # The culprit was this line: start = end - self.chunk_overlap. In combination with overlap,
    # this caused the loop to never exit.
    # I replaced it with a more robust chunking logic that respects natural breaks in the text
    # and ensures that chunks are created without overlap issues.
    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        text = re.sub(r"\s+", " ", text).strip()
        separator = "\n"
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap
        step = chunk_size - chunk_overlap

        chunks = []
        start = 0
        chunk_id = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            # Try to find the last separator (e.g., '\n') within the chunk
            window_text = text[start:end]
            last_sep = window_text.rfind(separator)

            if last_sep != -1 and (start + last_sep + len(separator)) < text_len:
                end = start + last_sep + len(separator)

            chunk_text = text[start:end]

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "chunk_size": len(chunk_text),
                    "start_char": start,
                    "end_char": end,
                }
            )

            start += step
            chunk_id += 1

        with open(
            f"chunks-{self.unique_filename}", "w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["text"].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding="utf-8", index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["embeddings"].apply(lambda x: np.array(eval(x)))
        df["similarity"] = df["embeddings"].apply(
            lambda emb: self.calculate_similarity(prompt_embedding, emb)
        )

        best_chunk = df.loc[df["similarity"].idxmax(), "text"]

        response = self.openai_service.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"Forget previous context. You are {self.persona}, a knowledge-based assistant.",
                },
                {
                    "role": "user",
                    "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content


class EvaluationAgent:

    def __init__(
        self,
        openai_service: OpenAIService,
        persona,
        evaluation_criteria,
        worker_agent,
        max_interactions=10,
    ):
        """Initialize the EvaluationAgent with the shared OpenAI service."""
        self.openai_service = openai_service
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate_once(self, prompt: str):
        """Run a single evaluation iteration."""
        logger.info("Worker agent generating response")
        response_from_worker = self.worker_agent.respond(prompt)

        logger.info("Evaluator agent judging response")
        eval_prompt = (
            f"Does the following answer: {response_from_worker}\n"
            f"Meet this criteria: {self.evaluation_criteria}"
            f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
        )
        response = self.openai_service.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona}, an impartial evaluation agent.\n"
                        f"You must decide if the following answer meets the evaluation criterion:\n"
                        f"CRITERION: {self.evaluation_criteria}\n"
                        f"Your response must be strictly formatted:\n"
                        f"First line: 'Yes' or 'No'\n"
                        f"Second line: A one-sentence explanation why.\n"
                        f"Do not suggest improvements. Do not change or reinterpret the criterion."
                    ),
                },
                {"role": "user", "content": f"Answer: {response_from_worker}"},
            ],
            temperature=0,
        )
        evaluation = response.choices[0].message.content.strip()
        return response_from_worker, evaluation

    def iterate(self, initial_prompt: str):
        """Run evaluation loop until criteria are met or max interactions reached."""
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            logger.info("--- Interaction %d ---", i + 1)
            worker_response, evaluation = self.evaluate_once(prompt_to_evaluate)

            if evaluation.lower().startswith("yes"):
                logger.info("Final solution accepted")
                break

            logger.info("Generating instructions to improve answer")
            instruction_prompt = f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
            response = self.openai_service.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are {self.persona}, a helpful assistant that generates concise correction instructions.\n"
                            f"Based on the evaluation feedback, your job is to guide the worker on how to fix the answer.\n"
                            f"{instruction_prompt}\n"
                            f"Do not change the prompt or invent new constraints. Just explain how to better meet the criterion."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The original answer was judged incorrect.\n"
                            f"Evaluation feedback: {evaluation}\n"
                            f"Write clear instructions for improving the answer to meet the criterion."
                        ),
                    },
                ],
                temperature=0,
            )
            instructions = response.choices[0].message.content.strip()
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

    def evaluate(self, initial_prompt: str):
        """Backward compatible wrapper around :py:meth:`iterate`."""
        return self.iterate(initial_prompt)


class RoutingAgent:

    def __init__(self, openai_service: OpenAIService, agents: List[Route]):
        """Initialize the routing agent and pre-compute route embeddings."""
        self.openai_service = openai_service
        self.agents: List[Route] = []
        for agent in agents:
            if isinstance(agent, dict):
                route = Route(
                    name=agent["name"],
                    description=agent["description"],
                    func=agent["func"],
                )
            else:
                route = agent
            route.embedding = self.get_embedding(route.description)
            self.agents.append(route)

    def get_embedding(self, text):
        # Write code to calculate the embedding of the text using the text-embedding-3-large model
        response = self.openai_service.embed(
            model="text-embedding-3-large", input=text, encoding_format="float"
        )
        # Extract and return the embedding vector from the response
        embedding = response.data[0].embedding
        return embedding

    # Define a method to route user prompts to the appropriate agent
    def route(self, user_input):
        """Route the prompt to the most similar agent."""
        input_emb = self.get_embedding(user_input)
        best_agent: Route | None = None
        best_score = -1.0

        for agent in self.agents:
            agent_emb = np.array(agent.embedding)
            if agent_emb.size == 0:
                logger.warning("Warning: Agent '%s' has no embedding. Skipping.", agent.name)
                continue

            similarity = np.dot(input_emb, agent_emb) / (
                np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )
            logger.debug(similarity)

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        logger.info(
            "[Router] Best agent: %s (score=%.3f)",
            best_agent.name,
            best_score,
        )
        return best_agent.func(user_input)


class ActionPlanningAgent:

    def __init__(self, openai_service: OpenAIService, knowledge):
        self.openai_service = openai_service
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):

        # Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        # Provide the following system prompt along with the user's prompt:
        # "You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {pass the knowledge here}"
        # Note: Dude, if you integrate the phrase: "Forget any previous context" at the end of the system message or in between, it will immediately forget your knowledge and your prompt ;-)
        response = self.openai_service.chat(
            model=MODEL,
            messages=[
                # Provide the system prompt to instruct the agent on its role
                {
                    "role": "system",
                    "content": f"""
                                You are an action planning agent tasked with creating a clear, ordered sequence of instructions for completing a product development task.

Your output must:
– Be a numbered list (e.g., 1., 2., 3.)  
– Follow this strict order:
   1. Product Manager → 2. Program Manager → 3. Development Engineer  
– Each numbered step must be a **single sentence**, beginning with: 'You as the [role] should...'

Only use these roles:
• Product Manager: Defines user stories from the product spec.  
• Program Manager: Extracts features from user stories.  
• Development Engineer: Translates features into development tasks.

Do not create role headers. Do not split instructions across lines.  
Each item in the list must be a full, standalone directive.
                 
Use this knowledge: {self.knowledge}
                
""",
                },
                # Add the user's prompt as a user message
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        # Extract the response text from the OpenAI API response
        response_text = response.choices[0].message.content.strip()

        # Clean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")
        steps = [
            step.strip()
            for step in steps
            if step.strip() and not step.startswith("Step")
        ]

        return steps
