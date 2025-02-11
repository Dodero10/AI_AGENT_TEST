# Function to perform workflow topology optimization
def optimize_topology(agents):
    # Simulate optimizing the interaction structure (topology) between agents
    # Simple chain structure: Reflect -> Debate -> Aggregate
    topology = [agents[0], agents[1], agents[2]]  # Example topology
    
    # Dummy evaluation logic: sum of prompt lengths for simplicity
    evaluation_score = sum([len(agent.prompt) for agent in topology])
    return topology, evaluation_score
