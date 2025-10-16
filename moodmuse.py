from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gradio as gr

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
emotion_labels = {
    "sadness": "Feeling low, hopeless, or emotionally heavy.",
    "joy": "Feeling uplifted, happy, or grateful.",
    "anger": "Feeling frustrated, irritated, or resentful.",
    "fear": "Feeling anxious, worried, or uncertain.",
    "surprise": "Feeling shocked, amazed, or caught off guard.",
    "disgust": "Feeling repulsed, uncomfortable, or resistant."
}
emotion_texts = list(emotion_labels.values())
emotion_keys = list(emotion_labels.keys())
emotion_embeddings = embedder.encode(emotion_texts, convert_to_tensor=False)
emotion_index = faiss.IndexFlatL2(len(emotion_embeddings[0]))
emotion_index.add(np.array(emotion_embeddings))
def semantic_emotion(text):
    query_vec = embedder.encode([text])[0]
    D, I = emotion_index.search(np.array([query_vec]), k=1)
    return emotion_keys[I[0][0]]
def detect_emotion(text):
    model_emotion = emotion_classifier(text)[0]['label']
    semantic_fallback = semantic_emotion(text)
    return semantic_fallback  # You can fuse or compare if needed

documents = [
    # Sadness
    "Sadness flies away on the wings of time.",
    "Tears are words the heart can't express.",
    # Joy
    "Joy is not in things; it is in us.",
    "Happiness is a warm puppy.",
    # Anger
    "Speak when you are angry and you will make the best speech you will ever regret.",
    "Anger is one letter short of danger.",
    # Fear
    "Do one thing every day that scares you.",
    "Fear is only as deep as the mind allows.",
    # Surprise
    "Life is full of surprises, but the biggest one is discovering yourself.",
    # Disgust
    "Disgust is the body's way of saying 'no thank you.'",
    # Hope
    "Hope is the thing with feathers that perches in the soul.",
    "Even the darkest night will end and the sun will rise.",
    # Creativity
    "Creativity is allowing yourself to make mistakes. Art is knowing which ones to keep.",
    "Frustration is the compost from which the flowers of creativity grow."
]
doc_embeddings = embedder.encode(documents, convert_to_tensor=False)
quote_index = faiss.IndexFlatL2(len(doc_embeddings[0]))
quote_index.add(np.array(doc_embeddings))
def retrieve_quote(query):
    query_vec = embedder.encode([query])[0]
    D, I = quote_index.search(np.array([query_vec]), k=1)
    return documents[I[0][0]]

generator = pipeline("text-generation", model="gpt2")
def generate_poem(emotion, quote):
    prompt = f"Emotion: {emotion}\nQuote: {quote}\nWrite a short reflective poem that offers hope."
    output = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.9)[0]['generated_text']
    return output

def moodmuse_pipeline(user_input):
    emotion = detect_emotion(user_input)
    quote = retrieve_quote(f"{emotion} quotes")
    poem = generate_poem(emotion, quote)
    return f"### Detected Emotion: {emotion}\n\n**Retrieved Quote:**\n> {quote}\n\n**Generated Poem:**\n{poem}"

custom_css = """
.gradio-container {
    background: linear-gradient(to bottom right, #2c3e50, #3498db);
    color: white;
}
"""
if __name__ == "__main__":
    gr.Interface(
        fn=moodmuse_pipeline,
        inputs=gr.Textbox(
            lines=3,
            placeholder="💬 Share your current feeling...",
            label="🧠 What's on your mind?"
        ),
        outputs=gr.Textbox(
            lines=20,
            label="🎭 MoodMuse Response",
            show_copy_button=True
        ),
        title="🌈 MoodMuse: Emotion-Aware Creativity",
        description="""
        <div style="font-size: 16px; color: #fff;">
            Enter any emotional phrase — MoodMuse will detect your mood, retrieve a quote, and generate a reflective poem.<br>
            <b> Built with RAG, empathy, and poetic fusion.</b>
        </div>
        """,
        css=custom_css,
        allow_flagging="never"
    ).launch()