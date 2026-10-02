#doing all the imports here 
import os
import hashlib
import time
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone,ServerlessSpec,PineconeApiException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from database import save_message, get_or_create_session

load_dotenv()
#loading the environment variables from the .env file



#initialising the pinecone parameters
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD"), region=os.getenv("PINECONE_REGION"))

#initialising the embedding model 
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#pinecone index name and if it is not present it will be handled by exception handling
index_name="medical-chatbot"
try:
    result=pc.Index(index_name).describe_index_stats()
except PineconeApiException as e:
    pc.create_index(
        name=index_name,
        dimension=embeddings.dimension,
        metric="cosine",
        spec=spec
    )  

##pinecone vector store initialisation for retrieval chain so that this can be chained with the chat model and the number of items it 
#can retrieve can be set altering the top k value
vectorstore=PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embeddings)
base_retriever=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)    

##initialising the llm for the chat model
llm=ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",temperature=0.1,
    max_tokens=1024,
    api_key=os.getenv("GOOGLE_API_KEY"),
    streaming=True 
)

##for the main RAG part this is exclusively for the system part 

##This is a crucial component in building a conversational RAG (Retrieval Augmented Generation) system, as it allows the chatbot to understand follow-up questions 
# that refer to earlier parts of the conversation.

#1)# Prompt to rephrase query based on chat history

#Purpose: This string is a system message that defines the task for 
#  Large Language Model (LLM). It instructs the LLM on how to behave when given a chat history and a new user question.

contextualize_question_system_prompt=(
    "Given a chat history and the latest user question"
    "which might reference context in the chat history,"
    "formulate a standalone question which can be understood"
    "without the chat history. Do NOT answer the question,"
    "just reformulate it if needed and otherwise return it as is."
)

#2)# Prompt to answer the question based on the retrieved context

# Purpose: This creates a structured prompt template that will be fed to the LLM. LangChain's ChatPromptTemplate is used to assemble a sequence of messages.
contexualised_q_prompt=ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_question_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever=create_history_aware_retriever(
    llm=llm,retriever=base_retriever,
    prompt=contexualised_q_prompt,
    )

#This chain generates answers from retrieved documents and 
# incorporates chat history for conversational context.

system_prompt=(
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. And DON'T answer questions outside mental health domain,"
    "If you don't know the answer, say that you "
    "don't know.\n\n{context}"
)

qa_prompt=ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)

#final touch and chat history 
rag_chain=create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain,
)

chat_history = []

def ask_question_streamed(user_question, current_chat_history, session_id=None):
    print("AI: ", end="")
    
    full_ai_response = ""
    start_time = time.perf_counter()

    for chunk in rag_chain.stream({
        "input": user_question,
        "chat_history": current_chat_history,
    }):
        # The 'answer' key should contain the streamed tokens from the LLM
        if "answer" in chunk and chunk["answer"] is not None:
            print(chunk["answer"], end="", flush=True)
            full_ai_response += chunk["answer"]
            
    print() # Newline after the full streamed response

    latency = round(time.perf_counter() - start_time, 4)

    # Persist to DB — wrapped in try/except so a DB failure never crashes chat
    if session_id:
        try:
            get_or_create_session(session_id)
            save_message(session_id, "user", user_question)
            save_message(
                session_id,
                "assistant",
                full_ai_response,
                latency_seconds=latency,
            )
        except Exception:
            pass  # DB is best-effort; chat continues regardless

    # Update in-memory chat history
    current_chat_history.append(HumanMessage(content=user_question))
    current_chat_history.append(AIMessage(content=full_ai_response))
    return full_ai_response



def get_document_id(document):
    """
    Use an existing document ID when available.
    Otherwise generate a stable ID from document text.
    """

    metadata = document.metadata or {}

    existing_id = (
        metadata.get("doc_id")
        or metadata.get("document_id")
        or metadata.get("id")
        or metadata.get("source")
    )

    if existing_id:
        return str(existing_id)

    text = document.page_content or ""

    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:12]


def ask_question_for_evaluation(
    user_question,
    current_chat_history=None,
    session_id=None,
):
    """
    Non-streaming chatbot call for evaluation.

    Returns:
        answer
        retrieved documents
        retrieved document IDs
    """

    if current_chat_history is None:
        current_chat_history = []

    start_time = time.perf_counter()

    result = rag_chain.invoke({
        "input": user_question,
        "chat_history": current_chat_history,
    })

    latency = round(time.perf_counter() - start_time, 4)

    answer = result.get("answer", "")
    documents = result.get("context", [])

    retrieved_documents = []

    for document in documents:
        retrieved_documents.append({
            "doc_id": get_document_id(document),
            "text": document.page_content,
            "metadata": document.metadata or {},
        })

    retrieved_doc_ids = [doc["doc_id"] for doc in retrieved_documents]

    # Persist to DB — wrapped in try/except so a DB failure never crashes eval
    if session_id:
        try:
            get_or_create_session(session_id)
            save_message(session_id, "user", user_question)
            save_message(
                session_id,
                "assistant",
                answer,
                retrieved_doc_ids=retrieved_doc_ids,
                latency_seconds=latency,
            )
        except Exception:
            pass  # DB is best-effort

    return {
        "answer": answer,
        "documents": retrieved_documents,
        "retrieved_doc_ids": retrieved_doc_ids,
        "latency_seconds": latency,
    }

def run_cli():
    chat_history = []

    while True:
        user_input = input(
            "Ask Any Question related to Mental Health: "
        )

        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Exiting the chat. Goodbye!")
            break

        ask_question_streamed(
            user_input,
            chat_history,
        )


if __name__ == "__main__":
    run_cli()