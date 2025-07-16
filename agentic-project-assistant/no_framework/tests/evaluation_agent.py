import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"

# Instantiate the KnowledgeAugmentedPromptAgent here
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key, persona=persona, knowledge=knowledge)

# Parameters for the Evaluation Agent
persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."

# Instantiate the EvaluationAgent with a maximum of 10 interactions here
evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    evaluation_criteria=evaluation_criteria,
    worker_agent=knowledge_agent,
    max_interactions=10
)

# Evaluate the prompt and print the response from the EvaluationAgent
response = evaluation_agent.evaluate(prompt)
print(f"Final Response: {response['final_response']}")
print(f"Evaluation: {response['evaluation']}")
print(f"Number of Iterations: {response['iterations']}")
print("✅ Evaluation completed successfully.")
