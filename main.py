import json
from difflib import get_close_matches

# load the json File
def loadKnowledgeBase(filePath):
    """
    Read the knowledge base from a JSON file.

    :param file_path: The path to the JSON file containing the knowledge base.
    :return: A dictionary with the knowledge base data.
    """
    with open(filePath, 'r') as file:
        knowledgeBase = json.load(file)
    return knowledgeBase

# save data into the json File
def saveKnowledgeBase(filePath, data):
    """
    Write the updated knowledge base to a JSON file.

    :param file_path: The path to the JSON file to save the knowledge base.
    :param data: A dictionary with the knowledge base data.
    """
    with open(filePath, 'w') as file:
        json.dump(data, file, indent=2)

def findBestMatch(userQuestion, questions):
    # return best match and cutoff of 60%
    """
    Find the closest matching question in the knowledge base.

    :param user_question: The user's input question.
    :param questions: A list of questions from the knowledge base.
    :return: The closest matching question or None if no match is found.
    """
    matches = get_close_matches(userQuestion, questions, n=1, cutoff=0.6)
    return matches[0] if matches else None

def getAnswer(question, knowledgeBase):
    """
    Retrieve the answer for a given question from the knowledge base.

    :param question: The matched question from the knowledge base.
    :param knowledge_base: A dictionary containing the knowledge base data.
    :return: The answer to the question or None if the question is not found.
    """
    for q in knowledgeBase["questions"]:
        if q["question"] == question:
            return q["answer"]
        
def chatBot():
    """
    Run the chatbot to interact with the user, answer questions, and learn new information.

    The chatbot does the following:
    1. Load the knowledge base from a JSON file.
    2. Continuously prompt the user for questions.
    3. Find the closest matching question in the knowledge base.
    4. If a match is found, return the answer. Otherwise, ask the user to teach the chatbot.
    5. If the user provides a new answer, add it to the knowledge base and save the updated knowledge base to the JSON file.
    6. Exit the chatbot when the user types 'quit'.
    """
    knowledgeBase = loadKnowledgeBase('knowledgeBase.json')
    
    while True:
        userInput = input("You: ")

        if userInput.lower() == "quit":
            break

        bestMatch = findBestMatch(userInput, [q["question"] for  q in knowledgeBase["questions"]])

        if bestMatch:
            answer = getAnswer(bestMatch, knowledgeBase)
            print(f"Bot: {answer}")
        else:
            print("Bot: I don't know the answer, can you teach me?")
            newAnswer = input("Type your answer or 'skip':")

            if newAnswer.lower() != "skip":
                knowledgeBase["questions"].append({"question": userInput, "answer": newAnswer})
                saveKnowledgeBase("knowledgeBase.json", knowledgeBase)
                print("Bot: Thank you. I learned a new response")

if __name__ == "__main__":
    chatBot()

