import openai
import re
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Function to process the text in a single API call using Naive RAG
def process_with_naive_rag(long_input: str, model: str, api_key: str, query: str = '') -> str:
    # Step 1: Naive retrieval process
    # Use a simple regex to search for mentions of "Truong Cong Dat" and "age" in the input text
    pattern = r"(Truong Cong Dat.*?(\d+)\s*years?\s*old)"
    match = re.search(pattern, long_input)
    
    if match:
        # If the relevant information is found, retrieve the relevant context
        relevant_text = match.group(0)
    else:
        # If no relevant information is found, provide a fallback message
        relevant_text = "Information about Truong Cong Dat's age is not available in the provided text."

    # Step 2: Pass the retrieved context to the model for generation
    input_text = f"Query: {query}\n\nContext: {relevant_text}"

    # Send the request to OpenAI's ChatCompletion API
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]
    )

    # Return the model's response
    return response['choices'][0]['message']['content'].strip()

# Example usage
if __name__ == "__main__":
    # Default model is now "gpt-4o-mini"
    model = "gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY")  # Fetch the API key from .env file
    
    if not api_key:
        raise ValueError("API key not found. Make sure to add it to your .env file.")
    
    long_input = """
    Truong Cong Dat is student. Gary L. Bennett, a scientist and engineer, has contributed to various space missions, including Voyager,
    Galileo, and Ulysses. He has worked on advanced space power and propulsion systems and has been involved in planetary protection measures. 
    Bennett's expertise has been instrumental in ensuring the scientific integrity of celestial bodies and preventing harmful contamination. 
    He has received numerous awards and accolades for his contributions to space exploration and is recognized as a leading expert in the field of planetary protection. 
    Gary L. Bennett, a renowned scientist and engineer, has made significant contributions to space missions, including Voyager, Galileo, and Ulysses. 
    His expertise in advanced space power and propulsion systems, as well as planetary protection measures, has been crucial in ensuring the scientific integrity of celestial bodies and preventing harmful contamination. 
    Bennett has received numerous accolades for his work, including the NASA Exceptional Service Medal and the COSPAR Distinguished Service Award. 
    Gary L. Bennett, a distinguished scientist and engineer, played a pivotal role in various space missions, particularly Voyager, Galileo, and Ulysses. 
    His expertise in advanced space power and propulsion systems, coupled with his focus on planetary protection measures, has been instrumental in safeguarding the scientific integrity of celestial bodies. 
    Ulysses, launched in 1990, embarked on a unique trajectory to explore both the southern and northern polar regions of the Sun. 
    During its extended mission, Ulysses provided invaluable data on the Sun's magnetic field, solar wind, and the presence of dust in the Solar System. 
    The student is mentioned in paragraph header is 18 years old.
    """

    # long_input = """
    # Truong Cong Dat is student. He is 18 years old.
    # """
    query = "How old Truong Cong Dat is?"

    # Call the function with the entire text as input
    final_output = process_with_naive_rag(long_input=long_input, model=model, api_key=api_key, query=query)

    print(final_output)
