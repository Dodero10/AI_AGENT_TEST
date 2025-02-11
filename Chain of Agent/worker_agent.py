import openai

class WorkerAgent:
    """
    A class to represent a worker agent that processes chunks of text
    using a specified model and communicates with previous interactions.

    Attributes:
        agent_name (str): The name of the agent.
        model (str): The model used for processing.
        api_key (str): The API key for authentication.
    """

    def __init__(self, agent_name: str, model: str, api_key: str):
        self.agent_name = agent_name
        self.model = model
        self.api_key = api_key
    
    def process_chunk(self, chunk: str, previous_communication: str, query: str = '') -> str:
        """
        Process a text chunk and return the response from the model.

        Args:
            chunk (str): The text chunk to process.
            previous_communication (str): The previous communication context.
            query (str, optional): An optional query to include. Defaults to ''.

        Returns:
            str: The processed response from the model.
        """
        # Process each chunk with GPT-4o-mini and communicate with the previous worker
        input_text = f"Previous Communication: {previous_communication}\nChunk: {chunk}\nQuery: {query}"
        response = self._query_gpt(input_text)
        return response['choices'][0]['message']['content'].strip()
    
    def _query_gpt(self, input_text: str) -> dict:
        """
        Query the GPT model with the provided input text.

        Args:
            input_text (str): The input text to send to the model.

        Returns:
            dict: The response from the GPT model.
        """
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
        )
        return response