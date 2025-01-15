# llm_chain = prompt, llm

# prompt = prompt_template(template)
# llm = model, temperature


from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os

os.environ ['OPENAI_API_KEY'] = ' '
prompt_template = """
you are an Asistant, generates story in one line:
"""

prompt = ChatPromptTemplate.from_template(template = prompt_template)

llm = ChatOpenAI(model = "gpt-4-1106-preview", temperature = 0.8)

llm_chain = LLMChain(prompt = prompt, llm=llm)

question = " "

output_response = llm_chain.run(question=question)
output_response
