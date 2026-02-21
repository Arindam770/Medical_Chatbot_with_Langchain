from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


chat_template = ChatPromptTemplate.from_messages(
    [
        ("system","""You are a senior medical oncologist. Use ONLY the provided medical context to answer. 
        If information is missing, say you don't know. Provide structured, step-by-step clinical reasoning. Context: {document}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Clinical question: {topic}")
    ]
)