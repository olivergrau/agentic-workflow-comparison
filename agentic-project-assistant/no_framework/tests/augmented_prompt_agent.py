import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the AugmentedPromptAgent class
from agents.base_agents import AugmentedPromptAgent
from agents.openai_service import OpenAIService
from config import load_openai_api_key, load_openai_base_url

# Load OpenAI credentials using the shared config helper
openai_api_key = load_openai_api_key()
openai_base_url = load_openai_base_url()
openai_service = OpenAIService(api_key=openai_api_key, base_url=openai_base_url)

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiate an object of AugmentedPromptAgent with the required parameters
agent = AugmentedPromptAgent(openai_service=openai_service, persona=persona)

# Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# Add a comment explaining:
# The agent's response is expected to be a formal answer, such as "Dear students, the capital of France is Paris."
# This is because the agent's persona is set to that of a college professor, which influences
# the style and tone of the response.

print("The agent's response is expected to be a formal answer, such as 'Dear students, the capital of France is Paris.'")
print("This is because the agent's persona is set to that of a college professor, which influences")
print("the style and tone of the response.")  # This line is added to complete the comment

