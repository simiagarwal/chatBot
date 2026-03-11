from textblob import TextBlob
import nltk

# Ensure the necessary NLTK data is downloaded for TextBlob to function
try:
    nltk.data.find('corpora/movie_reviews')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('movie_reviews')

class ChatBot:
    def __init__(self):
        self.intents = {
            "hours": {
                "keywords": ["hour", "open", "close", "time"],
                "response": "We are open from 9 AM to 5 PM, Monday to Friday."
            },
            "return": {
                "keywords": ["refund", "money back", "return", "back"],
                "response": "I'd be happy to help you process your return. Let me transfer you to a live agent."
            }
        }

    def get_response(self, message):
        clean_message = message.lower()
        
        # 1. Check for specific Keyword Intents first
        for intent_data in self.intents.values():
            if any(word in clean_message for word in intent_data["keywords"]):
                return intent_data["response"]
            
        # 2. If no keyword is found, perform Sentiment Analysis
        blob = TextBlob(clean_message)
        sentiment = blob.sentiment.polarity
        
        # Using a small threshold (0.1) for better "Neutral" detection
        if sentiment > 0.1:
            return "That's great to hear! Tell me more."
        elif sentiment < -0.1:
            return "I'm so sorry to hear that. How can I help?"
        else:
            return "I see. Can you tell me more about that?"

    def chat(self):
        print("ChatBot: Hi! I'm your NLP assistant. (Type 'quit' to stop)")
        while True:
            user_message = input("You: ").strip()
            if user_message.lower() in ['exit', 'quit', 'bye']:
                break
                
            response = self.get_response(user_message)
            print(f"ChatBot: {response}\n")
            
        print("ChatBot: Thank you for chatting. Have a great day!")

if __name__ == "__main__":
    ChatBot().chat()