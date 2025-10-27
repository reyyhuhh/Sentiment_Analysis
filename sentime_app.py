from transformers import pipeline
import gradio as gr

# Load pretrained sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return f"{result['label']} ({round(result['score'] * 100, 2)}%)"

# Launch Gradio app
gr.Interface(fn=analyze_sentiment, inputs="text", outputs="text",
             title="Movie Reviewer Analysis").launch()

#the main code
