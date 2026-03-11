#Every time the chatbot starts, it will look for a memory.json file. If it finds one, it loads the data; if not, it starts fresh. When the chat ends, it "dumps" the current user_data back into that file, ensuring that the next time the chatbot is launched, it can greet the user by name and recall previous interactions. This way, the chatbot can maintain a persistent memory across sessions, making the user experience more personalized and engaging.
import spacy
import json
import os
from textblob import TextBlob

# Load the NLP model
nlp = spacy.load("en_core_web_md")

class PersistentChatBot:
    def __init__(self, memory_file="memory.json"):
        self.memory_file = memory_file
        self.user_data = self.load_memory()
        
        self.intents = {
            "hours": {
                "ref_phrase": nlp("What are your opening hours and schedule?"),
                "response": "We are open from 9 AM to 5 PM, Monday to Friday."
            },
            "return": {
                "ref_phrase": nlp("I want to return an item or get a refund."),
                "response": "I'd be happy to help you process your return."
            }
        }

    def load_memory(self):
        """Loads user data from a JSON file if it exists."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return {"name": None, "last_interaction": None}

    def save_memory(self):
        """Saves current session data to the JSON file."""
        with open(self.memory_file, 'w') as f:
            json.dump(self.user_data, f)

    def extract_name(self, doc):
        """Identifies and saves the user's name."""
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                self.user_data["name"] = ent.text
                self.save_memory() # Save immediately when a name is found

    def get_response(self, message):
        user_doc = nlp(message)
        self.extract_name(user_doc)
        
        # Intent Similarity Logic
        best_intent = None
        highest_similarity = 0
        for intent, data in self.intents.items():
            similarity = user_doc.similarity(data["ref_phrase"])
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_intent = intent

        name_tag = f"{self.user_data['name']}, " if self.user_data["name"] else ""
        
        if highest_similarity > 0.7:
            return f"Well {name_tag}{self.intents[best_intent]['response']}"
        
        # Sentiment Fallback
        sentiment = TextBlob(message).sentiment.polarity
        if sentiment < -0.3:
            return f"I'm sorry you're upset, {name_tag if name_tag else 'friend'}. How can I fix this?"
            
        return f"I'm not sure about that, {name_tag if name_tag else 'but'} I'm learning every day!"

    def chat(self):
        # Greeting based on whether we recognize them
        if self.user_data["name"]:
            print(f"ChatBot: Welcome back, {self.user_data['name']}! How can I help you today?")
        else:
            print("ChatBot: Hello! I'm an NLP bot. What's your name?")

        while True:
            msg = input("You: ").strip()
            if msg.lower() in ['exit', 'quit', 'bye']:
                self.save_memory() # Final save before exiting
                print(f"ChatBot: Goodbye{', ' + self.user_data['name'] if self.user_data['name'] else ''}!")
                break
            print(f"ChatBot: {self.get_response(msg)}\n")

if __name__ == "__main__":
    bot = PersistentChatBot()
    bot.chat()