# Function to perform workflow-level prompt optimization
def optimize_workflow_prompts(topology):
    # Fine-tune prompts for the entire system considering its topology
    optimized_prompts = []
    for agent in topology:
        optimized_prompts.append(f"Workflow-Optimized prompt for {agent.name}: {agent.prompt}")
    return optimized_prompts

# Function to return the final optimized multi-agent system
def get_final_optimized_system(optimized_agents, optimized_topology, workflow_optimized_prompts):
    final_system = {
        "agents": optimized_agents,
        "topology": optimized_topology,
        "workflow_prompts": workflow_optimized_prompts
    }
    return final_system
