from langgraph import Node
from flask import Flask, request, jsonify
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

openai.api_key = ""

# --- Qwen Model Node ---
def qwen_response(input_text):
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
    response, _ = model.chat(tokenizer, input_text, history=None)
    return response

# --- Llama2 Model Node ---
def llama_response(input_text):
    llm = CTransformers(
        model="llama-2-model-path",  # Replace with your model path
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

# --- Refine Response with GPT-4 ---
def refine_response_with_gpt4(combined_output):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Refine the following response:\n\n{combined_output}"}
        ]
    )
    return response['choices'][0]['message']['content']

# --- Define LangGraph Nodes ---
qwen_node = Node(name="Qwen Model", func=qwen_response)
llama_node = Node(name="Llama2 Model", func=llama_response)
gpt4_refine_node = Node(name="GPT-4 Refinement", func=refine_response_with_gpt4)

# --- Combine Nodes in a Workflow ---
def handle_query(input_text):
    qwen_output = qwen_node.run(input_text)  # Call the Qwen model node
    llama_output = llama_node.run(input_text)  # Call the Llama2 model node
    combined_output = f"Qwen Response: {qwen_output}\n\nLlama2 Response: {llama_output}"
    refined_response = gpt4_refine_node.run(combined_output)  # Refine the combined output with GPT-4
    return refined_response

# --- Flask API Setup ---
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_input = data.get("query", "")
    if not user_input:
        return jsonify({"error": "Query not provided"}), 400
    
    # Handle query and get refined response
    refined_response = handle_query(user_input)
    
    # Return the final refined response
    return jsonify({"response": refined_response})

if __name__ == "__main__":
    app.run(debug=True)
