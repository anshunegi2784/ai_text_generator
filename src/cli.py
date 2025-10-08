print(">>> cli.py started <<<")

# src/cli.py
"""
Simple CLI to test the generator quickly from terminal.
Run: python src/cli.py
"""
from .app_core import ai_text_generator
#from src.app_core import ai_text_generator

def main():
    print("AI Sentiment Text Generator — CLI")
    prompt = input("Enter a prompt/topic: ").strip()
    if not prompt:
        print("No prompt given. Exiting.")
        return

    # Ask if user wants to override sentiment
    use_manual = input("Do you want to set sentiment manually? (y/N): ").strip().lower()
    manual = None
    if use_manual == 'y':
        ms = input("Enter sentiment (positive / neutral / negative): ").strip().lower()
        if ms in ("positive", "neutral", "negative"):
            manual = ms

    sentiment, text = ai_text_generator(prompt, manual_sentiment=manual)
    print(f"\nUsed sentiment: {sentiment}\n")
    print("Generated text:\n")
    print(text)

if __name__ == "__main__":
    main()
