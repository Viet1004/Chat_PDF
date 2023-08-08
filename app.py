__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st 
from langchain.document_loaders import PyPDFLoader
#from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from htlmTemplates import css, bot_template, user_template
from langchain.chains.combine_documents.refine import RefineDocumentsChain
import pdfplumber
import openai
from typing import Dict, Any
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

def get_documents(pdfs):
    # text = ""
    # for pdf in pdfs:
    #     pdf_loader = PyPDFLoader(pdf)
    #     pages = pdf_loader.load()
    #     for page in pages:
    #         text += page.page_content
    #     text += "/n"
    # return text 
    doc_lists = []
    for index, pdf in enumerate(pdfs):
        pdf_reader = PdfReader(pdf)
        doc_list = []
        meta = pdf_reader.metadata
        document_number = index+1
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            doc_list.append(Document(page_content = text, metadata={"source": f"Document {document_number}", "page": i+1}))
        doc_lists += doc_list
    return doc_list

def get_text_chunks(text_spliter, docs):
    texts = text_spliter.split_documents(docs)
    return texts

def get_vectorstore(embedding, documents):

#    vectorstore = FAISS.from_documents(texts=text_chunks, embedding=embeddings)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding)
    return vectorstore

def create_refine_chain():
    # This controls how each document will be formatted. Specifically,
    # it will be passed to `format_document` - see that function for more
    # details.
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )
    document_variable_name = "context"
    llm = OpenAI()
    # The prompt here should take as an input variable the
    # `document_variable_name`
    prompt = PromptTemplate.from_template(
        "Summarize this content: {context}"
    )
    initial_llm_chain = LLMChain(llm=llm, prompt=prompt)
    initial_response_name = "prev_response"
    # The prompt here should take as an input variable the
    # `document_variable_name` as well as `initial_response_name`
    prompt_refine = PromptTemplate.from_template(
        "Here's your first summary: {prev_response}. "
        "Now add to it based on the following context: {context}"
    )
    refine_llm_chain = LLMChain(llm=llm, prompt=prompt_refine)
    chain = RefineDocumentsChain(
        initial_llm_chain=initial_llm_chain,
        refine_llm_chain=refine_llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
    )
    return chain

# def st_display_pdf(pdf_file):
#     with open(pdf_file, "rb") as file:
#         base64_pdf = base64.b64decode(file.read()).decode('utf-8')
#     pdf_display = f'<embed src=”data:application/pdf;base64,{base64_pdf}” width=”700″ height=”1000″ type=”application/pdf”>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})


def create_conversation_chain(vectorstore, return_source = True):
    llm = ChatOpenAI(
        temperature = 0,
        model_name = "gpt-3.5-turbo",
        streaming = True,
        callbacks = [FinalStreamingStdOutCallbackHandler()]
    )
    memory = AnswerConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # if return_source:
    #     refine_doc_chain = create_refine_chain()
    #     template = (
    #         "Combine the chat history and follow up question into "
    #         "a standalone question. Chat History: {chat_history}"
    #         "Follow up question: {question}"
    #     )
    #     prompt = PromptTemplate.from_template(template)
    #     llm = OpenAI()
    #     question_generator_chain = LLMChain(llm=llm, prompt=prompt)
    #     conversation_chain = ConversationalRetrievalChain(
    #         combine_docs_chain=refine_doc_chain,
    #         retriever=vectorstore.as_retriever(),
    #         memory=memory,
    #         question_generator=question_generator_chain,
    #         return_source_documents=True
    #     )
    # else:
    #     conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(),
    #     memory = memory,
    # )

    llm = OpenAI(temperature=0)
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The index of documents, always has the pattern 'Document index', possibly followed by description of the Document ",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the documents",
            type="integer",
        ),
    ]

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        st.session_state.document_content_description,
        metadata_field_info,
#        verbose=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
#        retriever=vectorstore.as_retriever(search_type="mmr"),
        retriever=retriever,
        memory = memory,
        return_source_documents = True
    )
    return conversation_chain

def remove_identical_objects(input_list):
    i = 0
    while i < len(input_list):
        j = i + 1
        while j < len(input_list):
            if input_list[i] == input_list[j]:
                input_list.pop(j)
            else:
                j += 1
        i += 1


def handle_userinput(user_question):
#    st.write(st.session_state.conversation_chain.input_keys)
    response = st.session_state.conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']
#    st.write(type(response["source_documents"][0]))
    remove_identical_objects(response['source_documents'])
    st.session_state.source_documents.append(response['source_documents'])
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        if i % 2 == 1:
#            source = " ".join(st.session_state.source_documents[i//2].metadata["source"].split(" ")[:5])
#            page = str(st.session_state.source_documents[i//2].metadata["page"])
            source_documents__ = st.session_state.source_documents[i//2]
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            with st.expander(label="Source:"):
                st.write("This answer comes from:")
                for source_doc in source_documents__:
                    source = " ".join(source_doc.metadata["source"].split(" ")[:2])
                    page = source_doc.metadata["page"]
                    st.write(source_doc.page_content[:30]+ f"... from {source}, page {page}")

def main():
#    load_dotenv()
    st.set_page_config(page_title="Study with me", page_icon=":books:", layout="wide")
    if "source_documents" not in st.session_state:
        st.session_state.source_documents = []
    # Sidebar
    with st.sidebar:
        st.header(":squid: :squid: - :squid: :squid: :squid:")
        st.subheader("Your documents")
        api_container = st.empty()
        openai.api_key = api_container.text_input("Enter an OpenAI API key:")
        if openai.api_key != "":
            openai_api_key = openai.api_key
            api_container.empty()
            st.write("API is registered, upload your documents below.")

        pdf_docs = None
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'",type='pdf', accept_multiple_files=True)  

        st.session_state.document_content_description = st.text_input("Short description of your documents (Optional)")
        if not st.session_state.document_content_description:
            st.session_state.document_content_description = ""
        

        if st.button("Process PDFs"):
            with st.spinner("Processing"):
                # Get the texts
                docs = get_documents(pdf_docs)

                # Divide them into text chunks            
                # text_spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                #     chunk_size = 200, chunk_overlap = 20
                # )
                text_spliter = RecursiveCharacterTextSplitter(
                    chunk_size = 1500, chunk_overlap = 200
                )
                documents = get_text_chunks(text_spliter=text_spliter, docs=docs)

                 # Create vector store   
                embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vectorstore = get_vectorstore(embedding=embedding, documents=documents) 

                # Create conversation chain
                st.session_state.conversation_chain = create_conversation_chain(vectorstore)
            
        st.write("Tips: Give a short description of your docs might help. Otherwise, leave it blank. Specify document and page number for more precise results")
    # Columns
    col1, col2 = st.columns([5,5], gap="small")
    with col1:
        st.subheader("Display Documents (In progress)")
        try:
            if pdf_docs is not None:

                # with pdfplumber.open(pdf_docs[0]) as file:
                #     pages = file.pages
                #     for page in pages:
                #         st.write(pages.extract_text(page))
                for i, pdf in enumerate(pdf_docs):

                    pdf_reader = PdfReader(pdf)
                #    expander = st.expander(label=pdf_reader.metadata.title, expanded=True)
                #    with expander:
                    document_number = i+1
                    metadata = pdf_reader.metadata
                    if metadata.title is not None:
                        st.write(f"Document {document_number}: {metadata.title}")
                    else:
                        st.write(f"Document {document_number}:")
                    for j, page in enumerate(pdf_reader.pages):
                        page_number = j+1
                        with st.expander(label=f"Page {page_number}"):

#                            st.write(f"Page {page_number}:")
                            st.write(page.extract_text())                            
        except:
            pass
    with col2:
        st.subheader("Chat with your data!")

        st.write(css, unsafe_allow_html=True)


        if "conversation_chain" not in st.session_state:
                st.session_state.conversation_chain = None

        if "chat_history" not in st.session_state:
                st.session_state.chat_history = None

        instruction = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer."""
            
        #    instruction = st.text_input("Enter your instruction")
            
            # Build prompt
        template = instruction + """ 
            {context}
            Question: {question}
            Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        user_question = st.text_input("Ask me about your docs!")
        if user_question:
                handle_userinput(user_question)

    

           # videos = st.file_uploader("Upload your videos here", accept_multiple_files=True)
        # if st.button("Process videos"):
        #     with st.spinner("Processing"):
        #         # Get the texts
        #         # Divide them into text chunks
        #         # Create vector store   



if __name__ == '__main__':
    main()