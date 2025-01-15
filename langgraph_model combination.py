from flask import Flask, request, jsonify
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

=openai.api_key = ""

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

# --- Combine Responses ---
def combine_responses(qwen_output, llama_output):
    combined_output = f"Qwen Response: {qwen_output}\n\nLlama2 Response: {llama_output}"
    return combined_output

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

# --- Flask API Setup ---
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_input = data.get("query", "")
    if not user_input:
        return jsonify({"error": "Query not provided"}), 400
    
    # Run Qwen model
    qwen_output = qwen_response(user_input)
    
    # Run Llama2 model
    llama_output = llama_response(user_input)
    
    combined_output = combine_responses(qwen_output, llama_output)
    
    # Refine combined output with GPT-4
    refined_response = refine_response_with_gpt4(combined_output)
    
    # Return the final refined response
    return jsonify({"response": refined_response})

if __name__ == "__main__":
    app.run(debug=True)
