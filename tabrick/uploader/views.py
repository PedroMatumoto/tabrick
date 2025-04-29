import csv
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import UploadFileForm
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def upload_file(request):
    data = {}
    agent_response = None

    if request.method == "POST":
        action = request.POST.get("action", "")
        form = UploadFileForm(request.POST, request.FILES)

        if action == "upload_file" and form.is_valid():
            # Handle file upload
            file = request.FILES['file']
            decoded_file = file.read().decode('utf-8').splitlines()
            reader = csv.reader(decoded_file)
            header = next(reader)
            data = list(reader)

            # Save the uploaded file to a temporary dataframe
            df = pd.DataFrame(data, columns=header)
            request.session['uploaded_data'] = df.to_json()  # Save DataFrame to session
            messages.success(request, 'Arquivo enviado com sucesso!')

        elif action == "ask_question":
            # Handle question submission
            user_query = request.POST.get("question", "")
            user_context = request.POST.get("context", "")
            if user_query:
                # Load DataFrame from session
                if 'uploaded_data' in request.session:
                    df = pd.read_json(request.session['uploaded_data'])
                    print(df)
                    # Initialize the agent
                    agent = create_pandas_dataframe_agent(
                        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                        df=df,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        verbose=True,
                        allow_dangerous_code=True,
                    )

                    # Process the query
                    agent_response = agent.invoke(f"Você é um analisador de documentos. Com base neste contexto: {user_context}, responda a pergunta: {user_query}")
                else:
                    messages.error(request, 'Nenhum arquivo foi enviado para análise.')

    else:
        form = UploadFileForm()

    return render(request, 'upload.html', {
        'form': form,
        'data': data,
        'agent_response': agent_response,
    })