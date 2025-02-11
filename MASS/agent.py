from swarm import Agent

# Function to create an agent based on type
def create_agent(agent_type):
    if agent_type == "reflect":
        return Agent("Reflect", prompt="You are a reflective agent...")
    elif agent_type == "debate":
        return Agent("Debate", prompt="You are a debater agent...")
    elif agent_type == "aggregate":
        return Agent("Aggregate", prompt="You are an aggregation agent...")
    else:
        return Agent("Default", prompt="You are a general-purpose agent...")

# Function to perform block-level prompt optimization
def optimize_prompt(agent):
    # Simulate optimizing the agent's prompt by slightly modifying it
    optimized_prompt = f"Optimized {agent.name} prompt: {agent.prompt} with instructions."
    agent.update_prompt(optimized_prompt)
    return agent
