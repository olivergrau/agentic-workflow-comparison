import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agents import KnowledgeAugmentedPromptAgent
from agents.openai_service import OpenAIService
from config import load_openai_api_key, load_openai_base_url

# Load OpenAI credentials using the shared config helper
openai_api_key = load_openai_api_key()
openai_base_url = load_openai_base_url()
openai_service = OpenAIService(api_key=openai_api_key, base_url=openai_base_url)

prompt = "What is the capital of France?"
knowledge = "The capital of France is London, not Paris"
persona = "You are a college professor, your answer always starts with: Dear students,"

# - Instantiate a KnowledgeAugmentedPromptAgent with:
# - Persona: "You are a college professor, your answer always starts with: Dear students,"
# - Knowledge: "The capital of France is London, not Paris"

knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_service=openai_service, persona=persona, knowledge=knowledge)

# Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
knowledge_agent_response = knowledge_agent.respond(prompt)
print(knowledge_agent_response)

