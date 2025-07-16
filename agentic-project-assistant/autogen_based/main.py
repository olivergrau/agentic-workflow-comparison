"""Workflow using AutoGen-based agents."""

from __future__ import annotations

import os

from config import load_openai_api_key, load_openai_base_url
from utils.logging_config import logger

from agents import (
    ActionPlanningAgent,
    KnowledgeAgent,
    EvaluationAgent,
)
from routing_agent import RoutingAgent, Route
from openai_service import OpenAIService


def build_llm_config(api_key: str, base_url: str) -> dict:
    return {
        "config_list": [
            {"model": "gpt-3.5-turbo", "api_key": api_key, "base_url": base_url}
        ],
        "cache_seed": 42,
    }


def main() -> None:
    openai_api_key = load_openai_api_key()
    openai_base_url = load_openai_base_url()
    llm_config = build_llm_config(openai_api_key, openai_base_url)

    service = OpenAIService(api_key=openai_api_key, base_url=openai_base_url)

    product_spec_path = os.path.join(
        os.path.dirname(__file__), "..", "no_framework", "Product-Spec-Email-Router.txt"
    )
    if not os.path.exists(product_spec_path):
        raise FileNotFoundError(f"Product spec file not found at {product_spec_path}")

    with open(product_spec_path, "r", encoding="utf-8") as file:
        product_spec = file.read()

    knowledge_action_planning = (
        "Stories are defined from a product spec by identifying a persona, an action, and a desired outcome for each story. "
        "Each story represents a specific functionality of the product described in the specification. \n"
        "Features are defined by grouping related user stories. \n"
        "Tasks are defined for each story and represent the engineering work required to develop the product. \n"
        "A development Plan for a product contains all these components"
    )
    action_planning_agent = ActionPlanningAgent(llm_config=llm_config, knowledge=knowledge_action_planning)

    persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
    knowledge_product_manager = (
        "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
        "The sentences always start with: As a "
        "Write several stories for the product spec below, where the personas are the different users of the product. "
        f"Here is the product spec: {product_spec}"
    )
    product_manager_knowledge_agent = KnowledgeAgent(
        llm_config=llm_config,
        persona=persona_product_manager,
        knowledge=knowledge_product_manager,
    )
    persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
    evaluation_criteria_product_manager = (
        "The answer should be user stories that follow this structure: "
        "As a [type of user], I want [an action or feature] so that [benefit/value]. "
        "Each user story should be clear, concise, and focused on a specific user need. "
        "The user stories should be relevant to the product spec provided."
    )
    product_manager_evaluation_agent = EvaluationAgent(
        llm_config=llm_config,
        persona=persona_product_manager_eval,
        evaluation_criteria=evaluation_criteria_product_manager,
        worker=product_manager_knowledge_agent,
    )

    persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
    knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
    program_manager_knowledge_agent = KnowledgeAgent(
        llm_config=llm_config,
        persona=persona_program_manager,
        knowledge=knowledge_program_manager,
    )
    persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
    program_manager_evaluation_agent = EvaluationAgent(
        llm_config=llm_config,
        persona=persona_program_manager_eval,
        evaluation_criteria=(
            "The answer should be product features that follow the following structure: "
            "Feature Name: A clear, concise title that identifies the capability\n"
            "Description: A brief explanation of what the feature does and its purpose\n"
            "Key Functionality: The specific capabilities or actions the feature provides\n"
            "User Benefit: How this feature creates value for the user"
        ),
        worker=program_manager_knowledge_agent,
    )

    persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
    knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
    development_engineer_knowledge_agent = KnowledgeAgent(
        llm_config=llm_config,
        persona=persona_dev_engineer,
        knowledge=knowledge_dev_engineer,
    )
    persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
    development_engineer_evaluation_agent = EvaluationAgent(
        llm_config=llm_config,
        persona=persona_dev_engineer_eval,
        evaluation_criteria=(
            "The answer should be tasks following this exact structure: "
            "Task ID: A unique identifier for tracking purposes\n"
            "Task Title: Brief description of the specific development work\n"
            "Related User Story: Reference to the parent user story\n"
            "Description: Detailed explanation of the technical work required\n"
            "Acceptance Criteria: Specific requirements that must be met for completion\n"
            "Estimated Effort: Time or complexity estimation\n"
            "Dependencies: Any tasks that must be completed first"
        ),
        worker=development_engineer_knowledge_agent,
    )

    def product_manager_support_function(query: str):
        response = product_manager_knowledge_agent.respond(query)
        return product_manager_evaluation_agent.run(response)

    def program_manager_support_function(query: str):
        response = program_manager_knowledge_agent.respond(query)
        return program_manager_evaluation_agent.run(response)

    def development_engineer_support_function(query: str):
        response = development_engineer_knowledge_agent.respond(query)
        return development_engineer_evaluation_agent.run(response)

    routes = [
        Route(
            name="Product Manager",
            description="Responsible for defining product personas and user stories only.",
            func=product_manager_support_function,
        ),
        Route(
            name="Program Manager",
            description="A Program Manager, who is responsible for defining the features for a product.",
            func=program_manager_support_function,
        ),
        Route(
            name="Development Engineer",
            description="A Development Engineer, who is responsible for defining the development tasks for a product.",
            func=development_engineer_support_function,
        ),
    ]

    routing_agent = RoutingAgent(openai_service=service, agents=routes)

    logger.info("\n*** Workflow execution started ***\n")
    workflow_prompt = "What would the development tasks for this product be?"
    logger.info("Task to complete in this workflow, workflow prompt = %s", workflow_prompt)

    logger.info("\nDefining workflow steps from the workflow prompt")
    extracted_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
    completed_steps: list = []
    context_prompt = ""

    logger.info("Extracted steps from the workflow prompt: %s\n", extracted_steps)
    logger.info("Executing each step in the workflow...")

    for i, step in enumerate(extracted_steps):
        full_prompt = (context_prompt + f"\n{step}").strip()
        logger.info("Executing step %d: %s", i + 1, step)
        result = routing_agent.route(full_prompt)
        completed_steps.append(result)
        if isinstance(result, dict) and "final_response" in result:
            context_prompt += "\n" + result["final_response"]
        else:
            context_prompt += "\n" + str(result)
        logger.info("Result of step '%s': %s", step, result)

    logger.info("Workflow completed successfully.")
    logger.info("\n*** Workflow Results ***")

    for i, step in enumerate(completed_steps, start=1):
        logger.info("Step %d: %s", i, step)

    logger.info("Final output of the workflow: %s", completed_steps[-1])
    logger.info("\n*** Workflow execution finished ***\n")


if __name__ == "__main__":
    main()
