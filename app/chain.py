# from langchain_pinecone import PineconeVectorStore
# from langchain_cohere import CohereEmbeddings

# index_name = "rag"
# embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# retriever = vectorstore.as_retriever()

# from langchain.prompts import ChatPromptTemplate

# # Multi Query: Different Perspectives
# template = """You are an AI language model assistant. Your task is to generate five 
# different versions of the given user question to retrieve relevant documents from a vector 
# database. By generating multiple perspectives on the user question, your goal is to help
# the user overcome some of the limitations of the distance-based similarity search. 
# Provide these alternative questions separated by newlines. Original question: {question}"""
# prompt_perspectives = ChatPromptTemplate.from_template(template)

# from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI

# google = GoogleGenerativeAI(model="gemini-pro")

# generate_queries = (
#     prompt_perspectives 
#     | google
#     | StrOutputParser() 
#     | (lambda x: x.split("\n")))

# from langchain.load import dumps, loads

# def get_unique_union(documents: list[list]):
#     """ Unique union of retrieved docs """
#     # Flatten list of lists, and convert each Document to string
#     flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
#     # Get unique documents
#     unique_docs = list(set(flattened_docs))
#     # Return
#     return [loads(doc) for doc in unique_docs]

# # Retrieve
# retrieval_chain = generate_queries | retriever.map() | get_unique_union

# from operator import itemgetter
# from langchain_openai import ChatOpenAI
# from langchain_core.runnables import RunnablePassthrough

# # RAG
# template = """Answer the following question based on this context:

# {context}

# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(template)

# from langchain_core.runnables import RunnablePassthrough


# chain = (
#     {"context": retrieval_chain, 
#      "question": RunnablePassthrough()} 
#     | prompt
#     | google
#     | StrOutputParser())





from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a comma-separated list."""


    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
chain = chat_prompt | GoogleGenerativeAI(model="gemini-pro") | CommaSeparatedListOutputParser()