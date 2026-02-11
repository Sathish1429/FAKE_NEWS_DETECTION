import torch
from transformers import BertTokenizer, BertForSequenceClassification
from newspaper import Article
from urllib.parse import urlparse
import pandas as pd
import re
import shap
import numpy as np


# === CONFIG ===
MODEL_PATH = 'model_save/'
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load tokenizer and model ===
tokenizer = BertTokenizer.from_pretrained("model_save/")
model = BertForSequenceClassification.from_pretrained("model_save/")
model.eval()


# === Source credibility scores ===
SOURCE_CREDIBILITY = {
    "bbc": 0.95,
    "cnn": 0.90,
    "reuters": 0.93,
    "nytimes": 0.92,
    "foxnews": 0.75,
    "weirdnewsdaily": 0.2,
    "unknown": 0.5
}

def clean_text(text):
    return re.sub(r'\s+', ' ', str(text).lower()).strip()

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to extract article: {e}")
        return None

def get_domain_name(url):
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        return domain.split(".")[0]
    except:
        return "unknown"

def get_credibility_from_url(url):
    source = get_domain_name(url)
    return SOURCE_CREDIBILITY.get(source, 0.5), source

def predict_text(text, credibility_score=None):
    inputs = tokenizer.encode_plus(
        clean_text(text),
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)


    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    if credibility_score is not None:
        probs[0] *= (0.5 + credibility_score / 2)  # Real
        probs[1] *= (0.5 + (1 - credibility_score) / 2)  # Fake

    total = probs[0] + probs[1]
    probs /= total

    prob_real = round(probs[0] * 100, 2)
    prob_fake = round(probs[1] * 100, 2)

    if prob_real > prob_fake:
        label = "The News is Real"
    else:
        label = "The News is Fake"

    return label, prob_real, prob_fake  # ✅ return 3 values





def predict_article(text):
    label, real_pct, fake_pct = predict_text(text)
    prob_real = probs[0] * 100
    prob_fake = probs[1] * 100

    if prob_real > prob_fake:
        label = "The News is Real"
    else:
        label = "The News is Fake"

    return label, round(prob_real, 2), round(prob_fake, 2)

def predict_article(text):
    label, real_pct, fake_pct = predict_text(text)
    if "Real" in label:
        return f"✅ {label} ({real_pct:.2f}%)"
    else:
        return f"❌ {label} ({fake_pct:.2f}%)"

def predict_url(url):
    article_text = extract_text_from_url(url)
    if not article_text:
        return "❌ Could not extract article content."
    credibility, source = get_credibility_from_url(url)
    label, real_pct, fake_pct = predict_text(article_text, credibility)
    if "Real" in label:
        return f"✅ {label} ({real_pct:.2f}%)"
    else:
        return f"❌ {label} ({fake_pct:.2f}%)"


def predict_csv(csv_path):
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        text = row['text']
        source = row.get('source', 'unknown')
        credibility = SOURCE_CREDIBILITY.get(str(source).strip().lower(), 0.5)
        label, real_pct, fake_pct = predict_text(text, credibility)
        if "Real" in label:
            result = f"✅ {label} ({real_pct:.2f}%) - Source: {source}"
        else:
            result = f"❌ {label} ({fake_pct:.2f}%) - Source: {source}"
        results.append(result)
    return results

