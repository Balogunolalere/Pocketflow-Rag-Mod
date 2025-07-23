

import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv


def call_llm(prompt):
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        response_mime_type="text/plain",
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return response.text


def get_embedding(text):
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-embedding-001"
    result = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(),
    )
    # result.embeddings is a list of ContentEmbedding objects; get the vector from the first one
    if hasattr(result, 'embeddings') and result.embeddings:
        embedding_obj = result.embeddings[0]
        if hasattr(embedding_obj, 'values'):
            return np.array(embedding_obj.values, dtype=np.float32)
        elif isinstance(embedding_obj, list):
            return np.array(embedding_obj, dtype=np.float32)
    raise ValueError("No embedding values found in Gemini response.")

def fixed_size_chunk(text, chunk_size=2000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks

if __name__ == "__main__":
    print("=== Testing call_llm ===")
    prompt = "In a few words, what is the meaning of life?"
    print(f"Prompt: {prompt}")
    response = call_llm(prompt)
    print(f"Response: {response}")

    print("=== Testing embedding function ===")
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "Python is a popular programming language for data science."
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    print(f"Gemini Embedding 1 shape: {emb1.shape}")
    similarity = np.dot(emb1, emb2)
    print(f"Gemini similarity between texts: {similarity:.4f}")