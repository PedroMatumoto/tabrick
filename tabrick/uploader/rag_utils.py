import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import pypdf
from langchain_community.document_loaders import TextLoader
from google.cloud import documentai_v1 as documentai
import tempfile

class RAGSystem:
    def __init__(self, openai_api_key, persist_directory="chroma_db", 
                 document_ai_project_id=None, document_ai_location=None, 
                 document_ai_processor_id=None):
        self.openai_api_key = openai_api_key
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        self.loaded_files = {}  # Dicionário para acompanhar os arquivos carregados
        
        # Document AI configuration
        self.document_ai_project_id = document_ai_project_id
        self.document_ai_location = document_ai_location
        self.document_ai_processor_id = document_ai_processor_id
        
        # Inicializar embeddings
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
        # Verificar se já existem dados persistidos
        if os.path.exists(persist_directory):
            self.load_vectorstore()
    def load_pdf(self, pdf_path):
        """Carrega um documento PDF e cria embeddings"""
        # Extrair nome do arquivo do caminho
        file_name = os.path.basename(pdf_path)
        used_document_ai = False
        
        # Carregar o PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Se o documento não contém texto legível, usar Document AI
        if not any(page.page_content.strip() for page in pages):
            print(f"O PDF {file_name} não tem texto legível. Tentando usar Document AI.")
            
            # Verificar se Document AI está configurado
            if all([self.document_ai_project_id, self.document_ai_location, self.document_ai_processor_id]):
                extracted_text = self.process_document_with_document_ai(pdf_path)
                
                if extracted_text and extracted_text.strip():
                    print(f"Document AI extraiu texto com sucesso de {file_name}")
                    used_document_ai = True
                    
                    # Criar documentos a partir do texto extraído
                    pages = []
                    
                    # Criar um documento para cada página (aproximadamente 3000 caracteres por página)
                    page_size = 3000
                    for i in range(0, len(extracted_text), page_size):
                        page_text = extracted_text[i:i+page_size]
                        if page_text.strip():
                            pages.append(
                                {
                                    "page_content": page_text,
                                    "metadata": {"source": file_name, "page": (i // page_size) + 1}
                                }
                            )
                    
                    if not pages:
                        raise ValueError(f"Document AI não conseguiu extrair texto útil de {file_name}")
                else:
                    raise ValueError(f"Document AI não conseguiu extrair texto de {file_name}")
            else:
                raise ValueError(f"O arquivo {file_name} não contém texto legível e Document AI não está configurado.")
        
          # Adicionar metadados do arquivo ou processar os dados do Document AI
        if not used_document_ai:
            # Caso normal: páginas carregadas pelo PyPDFLoader
            for page in pages:
                page.metadata["source"] = file_name
            
            # Dividir o texto em chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(pages)
        else:
            # Caso com Document AI: converter os dicionários em objetos Document
            from langchain.schema import Document
            documents = [Document(page_content=page["page_content"], metadata=page["metadata"]) for page in pages]
            
            # Dividir o texto em chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)
        
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
            "chunks": len(chunks),
            "used_document_ai": used_document_ai
        }
        
        # Criar retriever
        self.setup_qa_chain()
        
        return {
            "file_name": file_name,
            "chunks_count": len(chunks),
            "used_document_ai": used_document_ai
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
            
        try:
            # Remover documentos com o metadata.source igual ao file_name
            # Obtém os IDs dos documentos com source=file_name
            docs_to_delete = self.vectorstore._collection.get(
                where={"source": file_name}
            )
            
            if docs_to_delete and len(docs_to_delete['ids']) > 0:
                # Solução para erro "too many SQL variables": deletar em lotes
                batch_size = 100  # Reduzindo para um valor bem seguro
                ids_to_delete = docs_to_delete['ids']
                
                # Deletar em lotes
                for i in range(0, len(ids_to_delete), batch_size):
                    batch_ids = ids_to_delete[i:i + batch_size]
                    self.vectorstore._collection.delete(ids=batch_ids)
                
                # Persistir alterações
                self.vectorstore.persist()
                
                # Remover do registro local
                if file_name in self.loaded_files:
                    del self.loaded_files[file_name]
                    
                return True
            
            return False
        except Exception as e:
            print(f"Erro ao deletar documento: {e}")
            # Tentar uma abordagem alternativa se o primeiro método falhar
            try:
                # Tenta obter e deletar documentos em lotes ainda menores
                where_filter = {"source": file_name}
                while True:
                    # Pega apenas um pequeno lote por vez
                    docs_to_delete = self.vectorstore._collection.get(
                        where=where_filter,
                        limit=50  # Limite muito pequeno para evitar problemas
                    )
                    
                    if not docs_to_delete or len(docs_to_delete['ids']) == 0:
                        break
                        
                    self.vectorstore._collection.delete(ids=docs_to_delete['ids'])
                
                self.vectorstore.persist()
                return True
            except Exception as e2:
                print(f"Falha no método alternativo de deleção: {e2}")
                raise
    
    def is_pdf_editable(self, pdf_path):
        """
        Check if a PDF contains searchable text.
        Returns True if the PDF contains searchable text, False otherwise.
        """
        try:
            # Open the PDF
            pdf_reader = pypdf.PdfReader(pdf_path)
            
            # Check at least first 3 pages for text (or all pages if less than 3)
            pages_to_check = min(3, len(pdf_reader.pages))
            
            for i in range(pages_to_check):
                text = pdf_reader.pages[i].extract_text()
                # If we found text on any page, the PDF is likely editable
                if text.strip():
                    return True
            
            # If we checked all pages and found no text, the PDF is likely non-editable
            return False
        except Exception as e:
            print(f"Error checking PDF editability: {e}")
            # In case of error, assume it's not editable to use Document AI
            return False
    
    def process_document_with_document_ai(self, file_path):
        """
        Process a document using Google Document AI and return the extracted text.
        """
        if not all([self.document_ai_project_id, self.document_ai_location, self.document_ai_processor_id]):
            raise ValueError("Document AI configuration is not complete. Please provide project_id, location, and processor_id.")
            
        opts = {"api_endpoint": f"{self.document_ai_location}-documentai.googleapis.com"}
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        
        name = client.processor_path(self.document_ai_project_id, self.document_ai_location, self.document_ai_processor_id)
        
        with open(file_path, "rb") as image:
            image_content = image.read()
            
        raw_document = documentai.RawDocument(content=image_content, mime_type="application/pdf")
        request = documentai.ProcessRequest(name=name, raw_document=raw_document)
        
        try:
            result = client.process_document(request=request)
            document = result.document
            return document.text
        except Exception as e:
            print(f"Error processing document with Document AI: {e}")
            return None