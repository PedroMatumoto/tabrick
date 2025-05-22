import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

class RAGSystem:
    def __init__(self, openai_api_key, persist_directory="chroma_db"):
        self.openai_api_key = openai_api_key
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        self.loaded_files = {}
        
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
        if os.path.exists(persist_directory):
            self.load_vectorstore()
    
    def load_pdf(self, pdf_path):
        """Carrega um documento PDF e cria embeddings"""

        file_name = os.path.basename(pdf_path)
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        for page in pages:
            page.metadata["source"] = file_name
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(chunks)
        
        self.loaded_files[file_name] = {
            "path": pdf_path,
            "chunks": len(chunks)
        }
        
        self.setup_qa_chain()
        
        return {
            "file_name": file_name,
            "chunks_count": len(chunks)
        }
    
    def get_loaded_files(self):
        """Retorna a lista de arquivos carregados"""
        return self.loaded_files
    
    def load_vectorstore(self):
        """Carrega um vectorstore existente"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self.setup_qa_chain()
    
    def setup_qa_chain(self):
        """Configura a chain de QA"""
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20}
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini", api_key=self.openai_api_key),
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )
    
    def query(self, question, filter_files=None):
        """
        Executa uma consulta ao RAG
        
        Args:
            question (str): A pergunta a ser respondida
            filter_files (list): Lista de nomes de arquivos para filtrar (opcional)
        """
        if not self.qa_chain:
            raise ValueError("É necessário carregar documentos primeiro")
        

        search_kwargs = {}
        if filter_files:
            search_kwargs["filter"] = {"source": {"$in": filter_files}}
            
    
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 20, **search_kwargs}
            )
            
    
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4o-mini", api_key=self.openai_api_key),
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True
            )
            
            result = qa_chain.invoke(question)
        else:
            result = self.qa_chain.invoke(question)
        

        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", "desconhecida"),
                "source": doc.metadata.get("source", "desconhecido")
            })
        
        return {
            "answer": result["result"],
            "sources": sources
        }