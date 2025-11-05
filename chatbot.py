from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Extended responses
responses = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! What’s up?",
    "how are you": "I’m just a bot, but I’m doing great! How about you?",
    "what is your name": "I’m ChatBot created by Dharshika!",
    "who created you": "I was created by Dharshika using Python and Machine Learning!",
    "what can you do": "I can chat with you, answer simple questions, and help you learn Python basics!",
    "what is python": "Python is a programming language used for AI, web apps, data science, and automation.",
    "tell me a joke": "Why did the computer show up at work late? Because it had a hard drive! 😄",
    "what is machine learning": "Machine Learning lets computers learn from data without being explicitly programmed.",
    "bye": "Goodbye! It was nice chatting with you!"
}

def chatbot_response(user_input):
    user_input = user_input.lower()
    all_inputs = list(responses.keys()) + [user_input]
    vectorizer = CountVectorizer().fit_transform(all_inputs)
    similarity = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    best_match = np.argmax(similarity)
    score = similarity[0][best_match]
    
    # If no close match, send default reply
    if score < 0.3:
        return "I’m not sure I understand. Could you please rephrase?"
    else:
        return list(responses.values())[best_match]

print("🤖 ChatBot: Hi! I’m your chatbot. Type 'bye' to end.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("🤖 ChatBot:", responses["bye"])
        break
    else:
        reply = chatbot_response(user_input)
        print("🤖 ChatBot:", reply)
