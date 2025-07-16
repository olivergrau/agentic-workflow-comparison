import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent
from agents.openai_service import OpenAIService
from config import load_openai_api_key, load_openai_base_url
import logging

# Load OpenAI credentials using the shared config helper
openai_api_key = load_openai_api_key()
openai_base_url = load_openai_base_url()
openai_service = OpenAIService(api_key=openai_api_key, base_url=openai_base_url)
prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"

# Instantiate the KnowledgeAugmentedPromptAgent here
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_service=openai_service, persona=persona, knowledge=knowledge
)

# Parameters for the Evaluation Agent
persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."

# Instantiate the EvaluationAgent with a maximum of 10 interactions here
evaluation_agent = EvaluationAgent(
    openai_service=openai_service,
    persona=persona,
    evaluation_criteria=evaluation_criteria,
    worker_agent=knowledge_agent,
    max_interactions=10,
)

logging.basicConfig(level=logging.INFO)

# Evaluate the prompt and print the response from the EvaluationAgent
response = evaluation_agent.iterate(prompt)
print(f"Final Response: {response['final_response']}")
print(f"Evaluation: {response['evaluation']}")
print(f"Number of Iterations: {response['iterations']}")
print("✅ Evaluation completed successfully.")

