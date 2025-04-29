from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

df = pd.read_csv("df_rent.csv")

agent = create_pandas_dataframe_agent(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    df=df,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    allow_dangerous_code=True,
)

agent.invoke(
    "What is the average rent in the dataset? What is the average rent for 2 bedroom apartments? What is the average rent for 3 bedroom apartments?"
)
