# src/generator.py
"""
Text generation module using Hugging Face transformers (GPT-2).
This uses the pipeline API to generate text guided by a short prompt.
"""

from transformers import pipeline, set_seed

# Instantiate generator (downloads model the first time)
generator = pipeline("text-generation", model="gpt2")
set_seed(42)

def generate_text(prompt: str,
                  sentiment: str = "neutral",
                  max_length: int = 150,
                  temperature: float = 0.7,
                  top_p: float = 0.9) -> str:
    """
    Generate a paragraph aligned with the sentiment.
    - prompt: user's topic prompt
    - sentiment: 'positive'|'neutral'|'negative' (used to nudge GPT-2)
    - max_length: max tokens/length of generated text (approx)
    """
    # Guiding prefix to nudge sentiment
    guide = f"Write a {sentiment} paragraph about: {prompt}"
    out = generator(guide,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1)
    text = out[0]['generated_text']
    # If model echoes the guide, strip it
    if text.startswith(guide):
        text = text[len(guide):].strip()
    return text
