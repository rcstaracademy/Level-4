import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
#------------------------------------
# Connect to Environment
#----------------------------
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# 1. Connect to SQL Server
# -----------------------------
#connection_string = "mssql+pyodbc://sa:YourPassword@localhost/SalesDB?driver=ODBC+Driver+17+for+SQL+Server"

connection_string = (
    "mssql+pyodbc://@DESKTOP-DQHOSOL/TEMP ?"
    "driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

db = SQLDatabase.from_uri(connection_string)

# -----------------------------
# 2. Load Data from Tables
# -----------------------------
tables = db.get_usable_table_names()

documents = []

for table in tables:
    query = f"SELECT * FROM {table}"
    rows = db.run(query)

    for row in rows:
        documents.append(Document(page_content=str(row), metadata={"table": table}))

# -----------------------------
# 3. Chunk Data
# -----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# -----------------------------
# 4. Create Embeddings (RAG)
# -----------------------------
#embeddings = OpenAIEmbeddings()

embeddings = OpenAIEmbeddings(api_key=api_key)
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

# -----------------------------
# 5. LLM
# -----------------------------
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    api_key=api_key
)

# -----------------------------
# 6. SQL Agent
# -----------------------------
sql_agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True
)

# -----------------------------
# 7. Hybrid RAG + SQL Function
# -----------------------------
def ask_question(question):
    question = question.lower()

    if "table" in question:
        return "Available tables are:\n\n" + "\n".join(tables)

    docs = retriever.invoke(question)

    context = "\n".join(doc.page_content for doc in docs)

    prompt = f"""
    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return response.content


# def ask_question(question):
#     # Step 1: Try RAG
#     #rag_docs = retriever.get_relevant_documents(question)
#     rag_docs = retriever.invoke(question)

#     if rag_docs:
#         context = "\n".join([doc.page_content for doc in rag_docs[:3]])

#         prompt = f"""
#         Answer the question using the context below:
#         {context}

#         Question: {question}
#         """

#         return llm.invoke(prompt)

#     # Step 2: fallback to SQL
#     return sql_agent.run(question)

# -----------------------------
# 8. Run App
# -----------------------------
while True:
    query = input("\nAsk your question: ")
    if query.lower() == "exit":
        break

    answer = ask_question(query)
    print("\nAnswer:", answer)