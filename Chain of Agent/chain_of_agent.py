from typing import List

from manager_agent import ManagerAgent
from worker_agent import WorkerAgent


class ChainOfAgents:
    def __init__(self, long_input: str, context_window_size: int, model: str, api_key: str, query: str = ''):
        self.long_input = long_input
        self.context_window_size = context_window_size
        self.model = model
        self.api_key = api_key
        self.query = query
    
    def process(self) -> str:
        # Split the long input into smaller chunks
        chunks = self.split_input_into_chunks(self.long_input, self.context_window_size)
        
        # Initialize the first worker's communication unit (empty string)
        communication_unit = ""
        
        # Stage 1: Process each chunk sequentially by worker agents
        for i, chunk in enumerate(chunks):
            worker = WorkerAgent(agent_name=f"Worker-{i+1}", model=self.model, api_key=self.api_key)
            communication_unit = worker.process_chunk(chunk, communication_unit, self.query)
        
        # Stage 2: The manager agent synthesizes the final output
        manager = ManagerAgent(model=self.model, api_key=self.api_key)
        final_output = manager.synthesize_final_output(communication_unit, self.query)
        
        return final_output
    
    def split_input_into_chunks(self, input_text: str, window_size: int) -> List[str]:
        # Split input into smaller chunks based on character length
        chunks = []
        current_chunk = ""
        words = input_text.split()
        
        for word in words:
            if len(current_chunk) + len(word) + 1 <= window_size:
                current_chunk += (" " + word if current_chunk else word)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks