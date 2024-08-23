import tkinter as tk
from tkinter import scrolledtext
import random
import string
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
import sympy as sp  # For evaluating mathematical expressions

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK packages
nltk.download('punkt')  # Download 'punkt' tokenizer
nltk.download('wordnet')  # Download 'wordnet' for lemmatization

# Sample corpus (enhanced)
corpus = [
    "Hello, how can I assist you today?",
    "I am a chatbot created to help you with your queries.",
    "You can ask me anything about the topics I'm programmed to understand.",
    "Goodbye! Have a great day!",
    "I'm sorry, I didn't catch that. Could you please rephrase?",
    "I can help with basic programming, general queries, and more.",
    "Tell me more about what you're interested in.",
    "I can provide information on various subjects, or just have a chat!"
]

# Tokenization
sent_tokens = corpus  # Using the predefined corpus as sentences

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["Hi!", "Hey!", "Hello!", "Greetings!", "Hi there!"]

# Farewell Responses
FAREWELL_INPUTS = ("bye", "see you", "goodbye", "exit")
FAREWELL_RESPONSES = ["Bye! Take care.", "Goodbye!", "See you soon!", "It was nice talking to you!"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    if any(greeting in sentence.lower() for greeting in GREETING_INPUTS):
        return random.choice(GREETING_RESPONSES)

def farewell(sentence):
    """If user's input indicates farewell, return a farewell response"""
    if any(farewell in sentence.lower() for farewell in FAREWELL_INPUTS):
        return random.choice(FAREWELL_RESPONSES)

def evaluate_math_expression(expression):
    """Evaluate mathematical expressions"""
    try:
        result = sp.sympify(expression)
        return f"The result is: {result}"
    except:
        return "I couldn't understand the math problem."

def response(user_response):
    """Generate a response based on the user input"""
    user_response = user_response.lower()
    if any(term in user_response for term in ["solve", "calculate", "what is"]):
        # Check if the message is a math problem
        return evaluate_math_expression(user_response)
    
    # Process the response based on predefined corpus
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]
    if req_tfidf == 0:
        chatbot_response = "I'm sorry! I don't understand you."
    else:
        chatbot_response = sent_tokens[idx]
    sent_tokens.remove(user_response)  # Remove the user input to avoid repeated responses
    return chatbot_response

# GUI using tkinter
class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot")

        # Create chat display area
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20, width=80, font=("Arial", 12))
        self.chat_area.pack(padx=10, pady=10)
        self.chat_area.config(state=tk.DISABLED)

        # Create input field
        self.user_input = tk.Entry(root, width=80, font=("Arial", 12))
        self.user_input.pack(padx=10, pady=10)
        self.user_input.bind("<Return>", self.get_user_input)

        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.get_user_input, font=("Arial", 12))
        self.send_button.pack(padx=10, pady=10)

        # Initialize with a greeting message
        self.display_message("Hello! How can I help you today? (You can ask about basic programming, math problems, etc.)", "bot")

    def get_user_input(self, event=None):
        user_message = self.user_input.get().strip()
        if user_message:
            self.display_message(user_message, "user")
            self.user_input.delete(0, tk.END)
            bot_response = self.generate_bot_response(user_message)
            self.display_message(bot_response, "bot")

    def display_message(self, message, sender):
        self.chat_area.config(state=tk.NORMAL)
        if sender == "user":
            self.chat_area.insert(tk.END, f"You: {message}\n")
            self.chat_area.tag_add("user", "end-2l", "end-1l")
            self.chat_area.tag_configure("user", justify='right', background='#e0f7fa', foreground='black')
        else:
            self.chat_area.insert(tk.END, f"Bot: {message}\n")
            self.chat_area.tag_add("bot", "end-2l", "end-1l")
            self.chat_area.tag_configure("bot", justify='left', background='#f1f8e9', foreground='black')
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.yview(tk.END)

    def generate_bot_response(self, user_message):
        if farewell(user_message):
            return farewell(user_message)
        elif greeting(user_message):
            return greeting(user_message)
        else:
            return response(user_message)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
