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
        self.loaded_files = {}  # Dicionário para acompanhar os arquivos carregados
        
        # Inicializar embeddings
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
        # Verificar se já existem dados persistidos
        if os.path.exists(persist_directory):
            self.load_vectorstore()
    
    def load_pdf(self, pdf_path):
        """Carrega um documento PDF e cria embeddings"""
        # Extrair nome do arquivo do caminho
        file_name = os.path.basename(pdf_path)
        
        # Carregar o PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Adicionar metadados do arquivo
        for page in pages:
            page.metadata["source"] = file_name
        
        # Dividir o texto em chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        
        # Verificar se já existe uma vectorstore
        if self.vectorstore is None:
            # Criar vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # Adicionar à vectorstore existente
            self.vectorstore.add_documents(chunks)
        
        # Registrar arquivo carregado
        self.loaded_files[file_name] = {
            "path": pdf_path,
            "chunks": len(chunks)
        }
        
        # Criar retriever
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
        
        # Aplicar filtro de arquivos, se especificado
        search_kwargs = {}
        if filter_files:
            search_kwargs["filter"] = {"source": {"$in": filter_files}}
            
            # Atualizar o retriever com o filtro
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 20, **search_kwargs}
            )
            
            # Recriar a chain com o novo retriever
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4o-mini", api_key=self.openai_api_key),
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True
            )
            
            result = qa_chain.invoke(question)
        else:
            result = self.qa_chain.invoke(question)
        
        # Formatar a resposta
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
        
    def delete_document(self, file_name):
        """
        Remove um documento da base de conhecimento
        
        Args:
            file_name (str): Nome do arquivo a ser removido
        """
        if not self.vectorstore:
            raise ValueError("Não há vectorstore inicializada")
            
        # Remover documentos com o metadata.source igual ao file_name
        # Obtém os IDs dos documentos com source=file_name
        docs_to_delete = self.vectorstore._collection.get(
            where={"source": file_name}
        )
        
        if docs_to_delete and len(docs_to_delete['ids']) > 0:
            # Remover os documentos pelo ID
            self.vectorstore._collection.delete(ids=docs_to_delete['ids'])
            
            # Persistir alterações
            self.vectorstore.persist()
            
            # Remover do registro local
            if file_name in self.loaded_files:
                del self.loaded_files[file_name]
                
            return True
        
        return False