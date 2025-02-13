from openai import OpenAI


class WorkerAgent:
    def __init__(self, agent_name, model, api_key):
        self.agent_name = agent_name
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    def process_chunk(self, chunk, communication_unit, query):
        prompt = f"""
        {chunk}
        
        Here is the summary of the previous source text: {communication_unit}

        Question: {query}

        You need to read the current source text and the summary of the previous source text (if any) and generate a summary that includes them both. Later, this summary will be used by other agents to answer the Query, if any.

        So please write a summary that contains essential information while ensuring it includes the evidence needed to answer the Query.
        """
        
        response = self._query_gpt(prompt)
        return response.choices[0].message.content.strip()
    
    def _query_gpt(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response
