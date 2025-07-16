import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agents import ActionPlanningAgent
from agents.openai_service import OpenAIService
from dotenv import load_dotenv

# Load environment variables and define the openai_api_key variable with your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_service = OpenAIService(api_key=openai_api_key)

knowledge = """
# Fried Egg
1. Heat pan with oil or butter
2. Crack egg into pan
3. Cook until white is set (2-3 minutes)
4. Season with salt and pepper
5. Serve

# Scrambled Eggs
1. Crack eggs into a bowl
2. Beat eggs with a fork until mixed
3. Heat pan with butter or oil over medium heat
4. Pour egg mixture into pan
5. Stir gently as eggs cook
6. Remove from heat when eggs are just set but still moist
7. Season with salt and pepper
8. Serve immediately

# Boiled Eggs
1. Place eggs in a pot
2. Cover with cold water (about 1 inch above eggs)
3. Bring water to a boil
4. Remove from heat and cover pot
5. Let sit: 4-6 minutes for soft-boiled or 10-12 minutes for hard-boiled
6. Transfer eggs to ice water to stop cooking
7. Peel and serve
"""

# Instantiate the ActionPlanningAgent, passing the openai_api_key and the knowledge variable
action_planning_agent = ActionPlanningAgent(
    openai_service=openai_service,
    knowledge=knowledge
)

# Print the agent's response to the following prompt: "One morning I wanted to have scrambled eggs"
prompt = "One morning I wanted to have scrambled eggs"
response = action_planning_agent.extract_steps_from_prompt(prompt)
print(f"Action Planning Response: {response}")
