from openai import OpenAI


class ManagerAgent:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def synthesize_final_output(self, final_communication_unit: str, query: str = '') -> str:
        prompt = f"""
        Task: Answer Generation

        The following are given passages. However, the source text is too long and has been summarized. You need to answer based on the summary provided.

        Summarized Source Text:
        {final_communication_unit}

        Question:
        {query}

        Instructions:
        - Carefully read the provided summarized text.
        - Generate a well-structured and concise response based on the available information.
        - Ensure your response is factual and directly addresses the query.
        - If the information is insufficient to answer the query, state that explicitly instead of guessing.

        Final Answer:
        """
        
        response = self._query_gpt(prompt)
        return response.choices[0].message.content.strip()
    
    def _query_gpt(self, input_text: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a highly intelligent assistant capable of summarization and reasoning."},
                {"role": "user", "content": input_text}
            ]
        )
        return response
