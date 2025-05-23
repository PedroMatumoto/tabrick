import os
import csv
import json
import pandas as pd
import markdown
import re
import time
from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils.safestring import mark_safe
from django.conf import settings
from .forms import UploadFileForm, QueryForm
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from .rag_utils import RAGSystem

# Configuração do diretório de arquivos
UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configurações da API
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicializa o sistema RAG
rag_system = RAGSystem(openai_api_key=OPENAI_API_KEY, 
                       persist_directory=os.path.join(settings.BASE_DIR, 'chroma_db'))

def convert_markdown_table_to_html(markdown_table):
    """
    Converte uma tabela markdown para uma tabela HTML
    """
    try:
        
        lines = markdown_table.split('\n')
        header = lines[0]
        
        html_table = '<table class="table table-striped">\n<thead>\n<tr>'
        header_cells = header.split('|')[1:-1]  
        for cell in header_cells:
            html_table += f'<th>{cell.strip()}</th>'
        html_table += '</tr>\n</thead>\n<tbody>'
        
        
        for line in lines[2:]:
            if line.strip() and '---' not in line:
                cells = line.split('|')[1:-1]  
                html_table += '<tr>'
                for cell in cells:
                    html_table += f'<td>{cell.strip()}</td>'
                html_table += '</tr>'
        
        html_table += '</tbody>\n</table>'
        return html_table
    except Exception as e:
        print(f"Erro ao converter tabela: {e}")
        return None

def get_file_choices(request):
    loaded_files = request.session.get("loaded_files", {})
    choices = []
    
    for file_name, info in loaded_files.items():
        file_type = info.get("type", "desconhecido")
        if file_type == "pdf":
            label = f"{file_name} (PDF - {info.get('chunks', '?')} chunks)"
        elif file_type == "csv":
            label = f"{file_name} (CSV - {info.get('rows', '?')} linhas)"
        else:
            label = file_name
            
        choices.append((file_name, label))
    
    return choices

def upload_file(request):
    data = {}
    agent_response = None
    table_response = None
    rag_response = None
    rag_sources = None
    header = []
    csv_files = {}    # Inicializar variáveis que precisam existir em todos os caminhos
    form = UploadFileForm()
    
    # Inicializar o query_form com as escolhas de arquivos atuais
    if request.method == "POST":
        query_form = QueryForm(request.POST, file_choices=get_file_choices(request))
    else:
        query_form = QueryForm(file_choices=get_file_choices(request))
    
    # Inicializa a lista de arquivos na sessão se não existir
    if "loaded_files" not in request.session:
        request.session["loaded_files"] = {}
    
    # Inicializa o histórico de conversa se não existir
    if "conversation_history" not in request.session:
        request.session["conversation_history"] = []
    
    if request.method == "POST":
        action = request.POST.get("action", "")
        
        if action == "upload_file":
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                file = request.FILES["file"]
                file_extension = os.path.splitext(file.name)[1].lower()
                
                # Salvar arquivo para processamento
                file_path = os.path.join(UPLOAD_DIR, file.name)
                with open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                
                if file_extension == '.pdf':
                    # Processar PDF com RAG
                    try:
                        result = rag_system.load_pdf(file_path)
                        file_name = result["file_name"]
                        chunks_count = result["chunks_count"]
                        
                        # Atualizar lista de arquivos na sessão
                        loaded_files = request.session.get("loaded_files", {})                        
                        loaded_files[file_name] = {
                            "type": "pdf",
                            "chunks": chunks_count,
                            "path": file_path
                        }
                        request.session["loaded_files"] = loaded_files
                        
                        # Atualizar o formulário de consulta com a nova lista de arquivos
                        query_form = QueryForm(file_choices=get_file_choices(request))
                        
                        messages.success(request, f"Arquivo PDF '{file_name}' carregado com sucesso! {chunks_count} fragmentos processados.")
                    except Exception as e:
                        messages.error(request, f"Erro ao processar o PDF: {str(e)}")
                else:
                    # Processar CSV
                    try:
                        with open(file_path, 'r', encoding='latin-1') as csv_file:
                            decoded_file = csv_file.read().splitlines()
                        
                        reader = csv.reader(decoded_file, delimiter=";")
                        header = next(reader)
                        data_rows = list(reader)
                        
                        df = pd.DataFrame(data_rows, columns=header)
                        
                        # Armazenar CSV na sessão
                        csv_files = request.session.get("csv_files", {})
                        file_name = os.path.basename(file_path)
                        csv_files[file_name] = {
                            "data": df.to_json(),
                            "header": header
                        }
                        request.session["csv_files"] = csv_files
                          # Atualizar lista de arquivos na sessão
                        loaded_files = request.session.get("loaded_files", {})
                        loaded_files[file_name] = {
                            "type": "csv",
                            "rows": len(data_rows),
                            "path": file_path
                        }
                        request.session["loaded_files"] = loaded_files
                        
                        # Atualizar o formulário de consulta com a nova lista de arquivos
                        query_form = QueryForm(file_choices=get_file_choices(request))
                        
                        messages.success(request, f"Arquivo CSV '{file_name}' enviado com sucesso!")
                    except Exception as e:
                        messages.error(request, f"Erro ao processar o CSV: {str(e)}")
        
        elif action == "ask_question":
            query_form = QueryForm(request.POST, file_choices=get_file_choices(request))
            if query_form.is_valid():
                user_query = query_form.cleaned_data["question"]
                user_context = query_form.cleaned_data["context"]
                selected_files = query_form.cleaned_data["selected_files"]
                
                if user_query:
                    # Recuperar histórico de conversa
                    conversation_history = request.session.get("conversation_history", [])
                    context_with_history = "\n".join(
                        [f"Pergunta: {item['question']}\nResposta: {item['response']}" for item in conversation_history]
                    )
                    full_context = f"{user_context}\n{context_with_history}"
                    
                    # Verifica se há arquivos carregados
                    loaded_files = request.session.get("loaded_files", {})
                    
                    if not loaded_files:
                        messages.error(request, "Nenhum arquivo foi carregado para análise.")
                    else:
                        # Verifica os tipos de arquivos selecionados
                        pdf_files = []
                        csv_files_selected = []
                        
                        if selected_files:
                            for file in selected_files:
                                file_info = loaded_files.get(file, {})
                                if file_info.get("type") == "pdf":
                                    pdf_files.append(file)
                                elif file_info.get("type") == "csv":
                                    csv_files_selected.append(file)
                        else:
                            # Se nenhum arquivo for selecionado, considere todos
                            for file, info in loaded_files.items():
                                if info.get("type") == "pdf":
                                    pdf_files.append(file)
                                elif info.get("type") == "csv":
                                    csv_files_selected.append(file)
                        
                        # Processar PDFs com RAG
                        pdf_response = None
                        pdf_table = None
                        if pdf_files:
                            try:
                                # Usar o valor do campo pdf_query do formulário
                                pdf_extraction_query = query_form.cleaned_data.get("pdf_query")

                                # Usar uma consulta padrão se pdf_query estiver vazio ou não for fornecido
                                if not pdf_extraction_query:
                                    pdf_extraction_query = """
                                    Extraia TODAS as informações sobre:
                                    1. Redução de alíquotas da CBS ou IBS
                                    2. Regimes especiais com tratamento tributário diferenciado
                                    3. Categorias de serviços ou produtos com benefícios fiscais
                                    
                                    Liste TODAS as categorias/tipos de serviços que têm qualquer tipo de redução.
                                    Seja DETALHADO e ESPECÍFICO, mencionando os percentuais de redução quando disponíveis.
                                    """
                                
                                result = rag_system.query(pdf_extraction_query, filter_files=pdf_files if selected_files else None)
                                pdf_response = result["answer"]
                                rag_sources = result["sources"]
                                
                                # Verificar se há tabela na resposta do PDF
                                table_match = re.search(r'\|.*\|\n\|[-:| ]+\|\n((?:\|.*\|\n)+)', pdf_response, re.MULTILINE)
                                if table_match:
                                    markdown_table = table_match.group(0)
                                    pdf_table = convert_markdown_table_to_html(markdown_table)
                                    # Remover a tabela do texto principal
                                    pdf_response = re.sub(r'\|.*\|\n\|[-:| ]+\|\n((?:\|.*\|\n)+)', '', pdf_response, count=1)
                                
                            except Exception as e:
                                messages.error(request, f"Erro ao consultar PDFs: {str(e)}")
                        
                        # Processar CSVs
                        csv_response = None
                        csv_table = None
                        if csv_files_selected:
                            try:
                                # Combine all selected CSV files
                                all_csv_data = request.session.get("csv_files", {})
                                combined_data = []
                                combined_headers = []
                                
                                for file_name in csv_files_selected:
                                    if file_name in all_csv_data:
                                        file_data = all_csv_data[file_name]
                                        df = pd.read_json(file_data["data"])
                                        # Add file name as prefix to column names to avoid conflicts
                                        df = df.add_prefix(f"{file_name}_")
                                        combined_data.append(df)
                                        combined_headers.extend([f"{file_name}_{h}" for h in file_data["header"]])
                                
                                if combined_data:
                                    combined_df = pd.concat(combined_data, axis=1)
                                    
                                    # Cria uma string com informações sobre o CSV para contextualizar a consulta
                                    csv_context = "Arquivo(s) CSV contém os seguintes dados:\n"
                                    for file_name in csv_files_selected:
                                        if file_name in all_csv_data:
                                            file_data = all_csv_data[file_name]
                                            csv_context += f"- Arquivo {file_name} com colunas: {', '.join(file_data['header'])}\n"
                                    
                                    # Amostrar algumas linhas para dar contexto, mas enfatizar que o agente tem acesso a todos os dados
                                    sample_rows = min(5, len(combined_df))
                                    csv_context += f"\nExemplo de dados (mostrando {sample_rows} de {len(combined_df)} linhas totais):\n"
                                    csv_context += combined_df.head(sample_rows).to_markdown()
                                    
                                    # Se temos resposta do PDF, incluímos o contexto sobre o PDF
                                    full_query = ""
                                    if pdf_response:
                                        full_query = f"""
                                        Combine estas duas fontes de informação para responder:
                                        
                                        1. INFORMAÇÃO DO PDF: 
                                        {pdf_response}
                                        
                                        2. DADOS DO CSV (use o dataframe para análise):
                                        {csv_context}
                                        
                                        Com base nessas informações, responda à pergunta: {user_query}
                                        
                                        Se a pergunta solicitar análise específica sobre itens no CSV, mostre exemplos concretos do dataframe.
                                        """
                                    else:
                                        full_query = f"""
                                        Analise o dataframe e responda:
                                        
                                        {user_query}
                                        
                                        Contexto sobre os dados:
                                        {csv_context}
                                        
                                        Se a pergunta solicitar exemplos ou casos específicos, mostre-os diretamente do dataframe.
                                        """
                                    
                                    # Modificação para as linhas 266-286 (parte da consulta)
                                    if pdf_response:
                                        full_query = f"""
                                        TAREFA: Análise completa combinando dados do PDF e CSV.
                                        
                                        1. INFORMAÇÃO DO PDF: 
                                        {pdf_response}
                                        
                                        2. DADOS DO CSV:
                                        {csv_context}
                                        
                                        INSTRUÇÕES IMPORTANTES:
                                        - Você tem acesso a TODAS as {len(combined_df)} linhas do dataframe, não apenas as amostras acima
                                        - Analise o dataframe COMPLETO ao responder
                                        - Utilize df.shape, df.describe(), ou outras funções para examinar todo o conjunto de dados
                                        - Se relevante, faça análises estatísticas sobre todos os dados
                                        
                                        PERGUNTA DO USUÁRIO: {user_query}
                                        
                                        Responda de forma completa, utilizando todos os dados disponíveis no dataframe e as informações do PDF.
                                        Mostre exemplos específicos do dataframe quando relevante.
                                        """
                                    else:
                                        full_query = f"""
                                        TAREFA: Análise completa do dataframe.
                                        
                                        Contexto sobre os dados:
                                        {csv_context}
                                        
                                        INSTRUÇÕES IMPORTANTES:
                                        - Você tem acesso a TODAS as {len(combined_df)} linhas do dataframe, não apenas as amostras acima
                                        - Analise o dataframe COMPLETO ao responder
                                        - Utilize df.shape, df.describe(), ou outras funções para examinar todo o conjunto de dados
                                        - Se relevante, faça análises estatísticas sobre todos os dados
                                        
                                        PERGUNTA DO USUÁRIO: {user_query}
                                        
                                        Responda de forma completa, utilizando todos os dados disponíveis no dataframe.
                                        Mostre exemplos específicos quando relevante.
                                        """

                                    system_prompt = """Você é um assistente especializado em análise de dados que fala português.
                                    Sua tarefa é analisar dataframes pandas e responder perguntas sobre eles em português.
                                    Ao analisar dados:
                                    1. Examine cuidadosamente as colunas do dataframe para entender a estrutura dos dados
                                    2. Execute análises estatísticas quando necessário (média, mediana, contagens, etc)
                                    3. Crie visualizações descritivas em forma de tabelas markdown quando apropriado
                                    4. Se encontrar problemas nos dados, explique-os claramente
                                    5. Forneça exemplos concretos dos dados para apoiar suas conclusões
                                    
                                    Importante: O usuário está consultando em português, então responda em português também.
                                    """
                                    
                                    # Usar o agente para analisar o dataframe com o contexto
                                    agent = create_pandas_dataframe_agent(
                                        llm=ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY, model="gpt-4"),
                                        df=combined_df,
                                        agent_type=AgentType.OPENAI_FUNCTIONS,
                                        verbose=True,
                                        allow_dangerous_code=True,
                                        agent_kwargs={"system_prompt": system_prompt},
                                    )
                                    
                                    csv_result = agent.invoke(full_query)
                                    csv_response = csv_result["output"]
                                    
                                    # Verificar se há tabela na resposta do CSV
                                    table_match = re.search(r'\|.*\|\n\|[-:| ]+\|\n((?:\|.*\|\n)+)', csv_response, re.MULTILINE)
                                    if table_match:
                                        markdown_table = table_match.group(0)
                                        csv_table = convert_markdown_table_to_html(markdown_table)
                                        # Remover a tabela do texto principal
                                        csv_response = re.sub(r'\|.*\|\n\|[-:| ]+\|\n((?:\|.*\|\n)+)', '', csv_response, count=1)
                                        csv_response = mark_safe(markdown.markdown(csv_response))
                                
                                # Adicionar à conversa
                                conversation_history.append({
                                    "question": user_query,
                                    "response": csv_response,
                                    "files": csv_files_selected
                                })
                                request.session["conversation_history"] = conversation_history
                                
                            except Exception as e:
                                messages.error(request, f"Erro ao processar CSVs: {str(e)}")
                        
                        # Combinar as respostas do PDF e CSV
                        combined_response = ""
                        
                        if pdf_response:
                            combined_response += pdf_response
                        
                        if csv_response:
                            if combined_response:
                                combined_response += "\n\n"
                            combined_response += csv_response
                        
                        # Preparar a resposta final
                        if combined_response:
                            agent_response = mark_safe(markdown.markdown(combined_response))
                        
                        # Combinar as tabelas, se houver
                        if pdf_table or csv_table:
                            if pdf_table:
                                table_response = mark_safe(pdf_table)
                            if csv_table:
                                table_html = mark_safe(csv_table)
                                if table_response:
                                    table_response += "<hr>" + table_html
                                else:
                                    table_response = table_html
                                    
                        # Adicionar à conversa apenas uma vez
                        if combined_response:
                            files_used = pdf_files + csv_files_selected
                            conversation_history.append({
                                "question": user_query,
                                "response": combined_response,
                                "files": files_used
                            })
                            request.session["conversation_history"] = conversation_history
    
        elif action == "delete_file":
            file_name = request.POST.get("file_name", "")
            if file_name:
                return delete_file(request, file_name) # Ensure the redirect response is returned
        
        elif action == "clear_conversation":
            if "conversation_history" in request.session:
                del request.session["conversation_history"]
                request.session.modified = True
                messages.success(request, "Histórico da conversa limpo com sucesso!")
            else:
                messages.info(request, "O histórico da conversa já está vazio.")
            return redirect("upload_file")

        elif action == "reset_chroma":
            # Botão de emergência para redefinir a base ChromaDB
            reset_chroma_db_if_needed(request)
            return redirect("upload_file")

    else:
        # Carregar dados de CSV se existir na sessão
        data = []
        header = []
        all_csv_data = request.session.get("csv_files", {})
        
        # Exibir o primeiro CSV se houver algum
        loaded_files = request.session.get("loaded_files", {})
        for file_name, info in loaded_files.items():
            if info.get("type") == "csv" and file_name in all_csv_data:
                file_data = all_csv_data[file_name]
                df = pd.read_json(file_data["data"])
                data = df.values.tolist()
                header = file_data["header"]
                break

    return render(
        request,
        "upload.html",
        {
            "form": form,
            "query_form": query_form,
            "data": data,
            "header": header,
            "agent_response": agent_response,
            "table_response": table_response,
            "rag_sources": rag_sources,
            "conversation_history": request.session.get("conversation_history", []),
            "loaded_files": request.session.get("loaded_files", {}),
        },
    )

def clear_chroma(request):
    try:
        # Resetar o sistema RAG
        import shutil
        chroma_dir = os.path.join(settings.BASE_DIR, 'chroma_db')
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
            os.makedirs(chroma_dir, exist_ok=True)
        
        # Reinicializar o sistema RAG
        global rag_system
        rag_system = RAGSystem(openai_api_key=OPENAI_API_KEY, 
                               persist_directory=os.path.join(settings.BASE_DIR, 'chroma_db'))
        
        # Limpar arquivos PDF da sessão
        loaded_files = request.session.get("loaded_files", {})
        loaded_files = {k: v for k, v in loaded_files.items() if v.get("type") != "pdf"}
        request.session["loaded_files"] = loaded_files
        
        messages.success(request, "Base de conhecimento limpa com sucesso!")
    except Exception as e:
        messages.error(request, f"Erro ao limpar a base de conhecimento: {str(e)}")

def reset_chroma_db_if_needed(request):
    """Função alternativa para redefinir a base Chroma em caso de problemas críticos"""
    try:
        # Usar shutil para eliminar a base de dados e recriá-la vazia
        import shutil
        chroma_dir = os.path.join(settings.BASE_DIR, 'chroma_db')
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
            os.makedirs(chroma_dir, exist_ok=True)
        
        # Reinicializar o RAG system
        global rag_system
        rag_system = RAGSystem(openai_api_key=OPENAI_API_KEY, 
                              persist_directory=os.path.join(settings.BASE_DIR, 'chroma_db'))
        
        # Limpar apenas os PDFs da sessão
        loaded_files = request.session.get("loaded_files", {})
        loaded_files = {k: v for k, v in loaded_files.items() if v.get("type") != "pdf"}
        request.session["loaded_files"] = loaded_files
        request.session.modified = True
        
        messages.success(request, "Base de conhecimento reinicializada com sucesso devido a problemas técnicos.")
        return True
    except Exception as e:
        messages.error(request, f"Erro ao tentar reinicializar a base de conhecimento: {str(e)}")
        return False

def delete_file(request, file_name):
    try:
        # Remover arquivo da sessão
        loaded_files = request.session.get("loaded_files", {})
        if file_name in loaded_files:
            file_info = loaded_files[file_name]
            file_path = file_info.get("path")
            
            # Remover do sistema de arquivos se o caminho existir
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                
            # Se for um PDF, remover da base Chroma também
            if file_info.get("type") == "pdf":
                try:
                    # Tentar excluir do ChromaDB - se falhar, continue mesmo assim
                    rag_system.delete_document(file_name)
                except Exception as chroma_error:
                    print(f"Erro ao remover da base ChromaDB: {chroma_error}")
                    # Não falha completamente, continua removendo da sessão
                    messages.warning(request, f"O arquivo foi removido da sessão, mas pode haver resíduos na base de vetores. Erro: {str(chroma_error)[:100]}...")
            
            # Se for um CSV, remover da sessão de CSVs
            if file_info.get("type") == "csv":
                csv_files = request.session.get("csv_files", {}) 
                if file_name in csv_files:
                    del csv_files[file_name]
                    request.session["csv_files"] = csv_files
            
            # Remover da lista de arquivos carregados
            del loaded_files[file_name]
            request.session["loaded_files"] = loaded_files
            request.session.modified = True  # Forçar a atualização da sessão
            
            messages.success(request, f"Arquivo '{file_name}' removido com sucesso!")
        else:
            messages.error(request, f"Arquivo '{file_name}' não encontrado para remoção.")
            
    except Exception as e:
        messages.error(request, f"Erro ao remover o arquivo '{file_name}': {str(e)}")
    
    # Adicionar um parâmetro à URL para forçar a atualização da página
    response = redirect("upload_file")
    response['Location'] += f'?updated={int(time.time())}'  # Adiciona um timestamp para evitar cache
    return response
