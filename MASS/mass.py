# mass.py

import os
from agent import create_agent, optimize_prompt
from topology import optimize_topology
from workflow import optimize_workflow_prompts, get_final_optimized_system

# Set your OpenAI API Key (make sure to replace with your actual API key)
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Step 1: Initialize agents
agents = [create_agent("reflect"), create_agent("debate"), create_agent("aggregate")]

# Step 2: Block-level Prompt Optimization
optimized_agents = [optimize_prompt(agent) for agent in agents]

# Step 3: Workflow Topology Optimization
optimized_topology, topology_score = optimize_topology(optimized_agents)

# Step 4: Workflow-Level Prompt Optimization
workflow_optimized_prompts = optimize_workflow_prompts(optimized_topology)

# Step 5: Get Final Optimized System
final_optimized_system = get_final_optimized_system(optimized_agents, optimized_topology, workflow_optimized_prompts)

# Output the result
print("Final Optimized Multi-Agent System:")
for agent in final_optimized_system["agents"]:
    print(f"Agent: {agent.name}, Prompt: {agent.prompt}")
for prompt in final_optimized_system["workflow_prompts"]:
    print(prompt)
