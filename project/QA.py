from transformers import pipeline

# Load the pre-trained Question Answering model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define the context (document)
context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair.
"""

if __name__ == "__main__":
    while True:
        question = input("Ask a question about the document (exit to quit) :")
        if question.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        result = qa_pipeline(question=question, context=context)
        print("\nAnswer:", result["answer"])
        print("Confidence Score:", result["score"])
        print(result)