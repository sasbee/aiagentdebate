import logging
import os
from flask import Flask, request, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set environment variables for debugging CUDA issues
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # Resize model's token embeddings to accommodate new tokens

# Set pad_token_id for the model
model.config.pad_token_id = tokenizer.pad_token_id

# Enhanced agent function using GPT-2 from Hugging Face
def get_agent_response(persona, context, previous_turns):
    # Ensure context length is within limits
    max_context_length = 1024 - 150  # 1024 max tokens - 150 tokens for response
    truncated_context = previous_turns[-max_context_length:]

    input_prompt = f"The topic of debate is: {context}\n\n{persona} argument:\n{truncated_context}\n{persona} argument:"
    inputs = tokenizer.encode_plus(input_prompt, return_tensors='pt', padding=True, truncation=True)
    max_new_tokens = 100  # Define how many new tokens to generate
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=max_new_tokens,  # Ensure the total length does not exceed the model's limit
        num_return_sequences=1, 
        temperature=0.7,  # Balanced diversity
        top_p=0.9,       # Balanced diversity
        top_k=50,        # Balanced diversity
        repetition_penalty=1.1  # Reduce repetitions
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to simulate a debate between two agents on any topic
def simulate_debate(topic):
    debate_history = []
    unique_responses = set()
    initial_context = topic
    persona_1 = "Pro"
    persona_2 = "Con"
    
    # Generating initial responses
    response_1 = get_agent_response(persona_1, topic, "")
    response_2 = get_agent_response(persona_2, topic, f"Agent 1: {response_1}")

    if response_1 not in unique_responses and response_2 not in unique_responses:
        debate_history.append((f"Agent 1: {response_1}", f"Agent 2: {response_2}"))
        unique_responses.add(response_1)
        unique_responses.add(response_2)
    
    previous_turns = f"Agent 1: {response_1}\nAgent 2: {response_2}"
    
    for _ in range(4):  # Continue the debate for 4 more exchanges each, making 5 points per agent
        response_1 = get_agent_response(persona_1, topic, previous_turns)
        response_2 = get_agent_response(persona_2, topic, previous_turns + f"\nAgent 1: {response_1}")
        
        if response_1 not in unique_responses and response_2 not in unique_responses:
            previous_turns += f"\nAgent 1: {response_1}\nAgent 2: {response_2}"
            debate_history.append((f"Agent 1: {response_1}", f"Agent 2: {response_2}"))
            unique_responses.add(response_1)
            unique_responses.add(response_2)
    
    conclusion = generate_conclusion(debate_history, topic)
    debate_history.append(("Conclusion", conclusion))
    
    return debate_history

# Function to generate a conclusion based on the debate
def generate_conclusion(debate_history, topic):
    points_pro = [turn[0].split("Agent 1: ")[1] for turn in debate_history[:-1]]
    points_con = [turn[1].split("Agent 2: ")[1] for turn in debate_history[:-1]]

    conclusion = f"""
    The debate on '{topic}' presented strong arguments on both sides.
    Pro points included: {', '.join(points_pro)}.
    Con points included: {', '.join(points_con)}.
    In conclusion, the discussion highlighted the multifaceted nature of the topic, showing both the benefits and potential drawbacks.
    """
    return conclusion

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/debate', methods=['POST'])
def debate():
    try:
        topic = request.form['topic']
        debate_history = simulate_debate(topic)
        formatted_history = '\n'.join([f"{turn[0]}\n{turn[1]}" if turn[0] != "Conclusion" else f"{turn[0]}\n{turn[1]}" for turn in debate_history])
        return render_template('index.html', debate_history=formatted_history)
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return render_template('index.html', error='An internal server error occurred'), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
