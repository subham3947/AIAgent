from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
model = OllamaLLM(model="llama3.2")
template = """
    You know everything about a pizza restaurant and you should be able to answer all questions regarding it
    
    Here are some reviews : {reviews}
    
    Here is the question to answer : {question}
    """
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
while True:
    print("\n\n")
    question = input("Ask your question (q to quit): ")
    if question == "q":
        break
    reviews=retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
