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
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            file_name = file.name.lower()

            # Check if the uploaded file is an Excel file
            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                # Read all sheets into a dictionary of DataFrames
                excel_data = pd.read_excel(file, sheet_name=None)
                data = {sheet: df.head().to_html() for sheet, df in excel_data.items()}  # Preview first rows of each sheet

                # Combine all sheets into one DataFrame (optional)
                combined_df = pd.concat(excel_data.values(), ignore_index=True)

                # Initialize the agent with the combined DataFrame
                agent = create_pandas_dataframe_agent(
                    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                    df=combined_df,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    verbose=True,
                    allow_dangerous_code=True,
                )

                # Process user query if provided
                user_query = request.POST.get("query", "")
                if user_query:
                    agent_response = agent.invoke(user_query)

            else:
                messages.error(request, 'Por favor, envie um arquivo Excel válido (.xlsx ou .xls).')

            messages.success(request, 'Arquivo enviado com sucesso!')
    else:
        form = UploadFileForm()

    return render(request, 'upload.html', {
        'form': form,
        'data': data,
        'agent_response': agent_response,
    })