import openai

class ManagerAgent:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def synthesize_final_output(self, final_communication_unit: str, query: str = '') -> str:
        # Generate the final output using the final communication unit from workers
        input_text = f"Final Communication Unit: {final_communication_unit}\nQuery: {query}"
        response = self._query_gpt(input_text)
        return response['choices'][0]['message']['content'].strip()
    
    def _query_gpt(self, input_text: str) -> dict:
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
        )
        return response