# Methodology — Sentiment-based AI Text Generator

## 1. Model Choice

- **Text Generation:** GPT-2 (Hugging Face Transformers)
  - Pre-trained transformer-based language model
  - Capable of generating coherent paragraphs
  - Small model chosen (`gpt2`) for faster download and inference
  - Seed set for reproducibility

- **Sentiment Analysis:** DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)
  - Pre-trained model fine-tuned on SST-2 dataset
  - Classifies text as positive, neutral, or negative
  - Enables sentiment-aligned text generation

---

## 2. Sentiment Detection Approach

- Input prompt first passed to sentiment analysis pipeline
- Model predicts sentiment:
  - **Positive**, **Neutral**, **Negative**
- Optional: User can manually override detected sentiment
- Sentiment used as **guiding prefix** for text generation:


---

## 3. Dataset

- **No additional dataset needed**
- Pre-trained models from Hugging Face used
- Fine-tuning optional (future work):
- Could fine-tune GPT-2 on sentiment-specific paragraphs for better alignment

---

## 4. Challenges & Improvements

### Challenges

1. **Repetitive text generation**
 - GPT-2 sometimes repeats phrases
2. **First-run model download**
 - Takes time (~100-300MB per model)
3. **Windows + OneDrive path issues**
 - Symlink warnings, permissions warnings

### Improvements Implemented

- Added `repetition_penalty=1.2` in generator
- Explicit `truncation=True` to avoid token-length warnings
- Streamlit caching (`@st.cache_data`) for faster subsequent runs
- Interactive sliders for max_length and temperature
- Manual sentiment override option
- Improved UI with word count and copy-to-clipboard
