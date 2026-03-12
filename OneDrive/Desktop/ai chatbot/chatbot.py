import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read knowledge base
with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    data = f.read()

sentences = data.split("\n")

# Greeting responses
greetings_input = ("hello", "hi", "hey")
greetings_response = ["Hello!", "Hi there!", "Hey! How can I help you?"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greetings_input:
            return random.choice(greetings_response)

def response(user_input):

    questions = []
    answers = []

    for line in sentences:
        if "-" in line:
            q, a = line.split("-", 1)
            questions.append(q.strip())
            answers.append(a.strip())

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions + [user_input])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    index = similarity.argmax()

    return answers[index]

# Chat loop
print("AI Chatbot: Ask me anything (type 'exit' to stop)")

while True:
    user_input = input("You: ").lower()

    if user_input == "exit":
        print("Chatbot: Goodbye!")
        break

    elif greet(user_input) != None:
        print("Chatbot:", greet(user_input))

    else:
        print("Chatbot:", response(user_input))
