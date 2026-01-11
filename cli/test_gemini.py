import os
from dotenv import load_dotenv

from google import genai
from google.genai import types


if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    message = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
    response: types.GenerateContentResponse = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=message,
    )
    assert response.usage_metadata is not None
    print(response.text)
    print("Prompt Tokens:", response.usage_metadata.prompt_token_count)
    print("Response Tokens:", response.usage_metadata.candidates_token_count)
