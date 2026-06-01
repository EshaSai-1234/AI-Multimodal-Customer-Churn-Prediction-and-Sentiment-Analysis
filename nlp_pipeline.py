# nlp_pipeline.py
import numpy as np
import pandas as pd
import torch
import scipy.special
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class MultimodalTextExtractor:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Initialize tokenizer and sequence classifier onto available hardware."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def process_batch_logs(self, texts: list) -> pd.DataFrame:
        """Runs batch inference over customer chat logs to generate sentiment features."""
        neg_scores, neu_scores, pos_scores = [], [], []
        
        for text in texts:
            if not isinstance(text, str) or str(text).strip() == "":
                neg_scores.append(0.0)
                neu_scores.append(1.0)
                pos_scores.append(0.0)
                continue
                
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Transfer array weights back to CPU for extraction mapping
            logits = outputs.logits[0].cpu().numpy()
            probabilities = scipy.special.softmax(logits)
            
            # Mapping schema layout: 0 -> Negative, 1 -> Neutral, 2 -> Positive
            neg_scores.append(round(float(probabilities[0]), 4))
            neu_scores.append(round(float(probabilities[1]), 4))
            pos_scores.append(round(float(probabilities[2]), 4))
            
        return pd.DataFrame({
            'text_negative_score': neg_scores,
            'text_neutral_score': neu_scores,
            'text_positive_score': pos_scores
        })

if __name__ == "__main__":
    # Quick standalone unit test validation
    sample_complaints = [
        "Your service has constant downtime. My team can't access our dashboard!",
        "How can I download my annual tax invoice?"
    ]
    extractor = MultimodalTextExtractor()
    sentiment_df = extractor.process_batch_logs(sample_complaints)
    print("--- NLP Feature Engineering Outputs ---")
    print(sentiment_df)