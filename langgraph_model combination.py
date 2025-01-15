from langgraph import Node, Graph
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

# Qwen Node
def qwen_response(input_text):
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
    response, _ = model.chat(tokenizer, input_text, history=None)
    return response

qwen_node = Node(name="Qwen Model", func=qwen_response)

# Llama2 Node
def llama_response(input_text):
    llm = CTransformers(
        model="llama-2-model-path",
        model_type="llama",
        config={
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True
        },
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    template = f"Question: {{question}}\nAnswer:"
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run({"question": input_text})

llama_node = Node(name="Llama2 Model", func=llama_response)

# Combine Responses Node
def combine_responses(qwen_output, llama_output):
    return f"Combined response:\n- Qwen: {qwen_output}\n- Llama2: {llama_output}"

combine_node = Node(name="Combine Responses", func=combine_responses)

# Refine Response Node
def refine_response(combined_output):
    refined_prompt = f"Refine the following response:\n\n{combined_output}\n\nFinal Answer:"
    llm = CTransformers(
        model="llama-2-refine-model-path",
        model_type="llama",
        config={
            "max_new_tokens": 256,
            "temperature": 0.5,
            "top_p": 0.85,
            "stream": False
        },
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    prompt = PromptTemplate(template="{refined_prompt}", input_variables=["refined_prompt"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run({"refined_prompt": refined_prompt})

refine_node = Node(name="Refine Response", func=refine_response)

# Define the LangGraph Workflow
graph = Graph(
    nodes=[qwen_node, llama_node, combine_node, refine_node],
    edges=[
        (qwen_node, combine_node),
        (llama_node, combine_node),
        (combine_node, refine_node),
    ],
)

# Input and Run the Workflow
user_input = input("Enter your query: ")
final_response = graph.run(user_input)
print("Final Response:", final_response)
