from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from src.prompt import chat_template
from dotenv import load_dotenv

load_dotenv()

chatModel = ChatOpenAI(model="gpt-4o")

def generate_response(question, chat_history, retriever):

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    retrieval_context = [doc.page_content for doc in docs]

    prompt = chat_template.invoke({
        "document": context,
        "topic": question,
        "chat_history": chat_history
    })

    response = chatModel.invoke(prompt)

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))

    # keep last 6 messages
    chat_history = chat_history[-6:]

    return response.content, chat_history, retrieval_context