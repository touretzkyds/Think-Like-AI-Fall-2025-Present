import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
import uuid

app = Flask(__name__)

print("Loading Transformer models...")
models = {
    'SmolLM-360M': {
        'tokenizer': AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-360M-Instruct'),
        'model': AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM-360M-Instruct',
                                                    torch_dtype=torch.float16) # Use float16 for speed
    }
}
print("Transformer models loaded.")


@app.route("/auto_login")
def auto_login():
    """
    Creates a temporary browser-based session parameter and redirects.
    """
    user_id = str(uuid.uuid4())
    return redirect(url_for("home", user_id=user_id))

@app.route("/", methods=["GET"])
@app.route("/login", methods=["GET"])
def study_login():
    """
    Shows the splash screen. When they click 'Let's Play', 
    the HTML button sends them to /auto_login.
    """
    return render_template("login.html")

@app.route("/exit")
def exit_app():
    """
    Terminal screen shown after a user exits from the chat page.
    Browsers block window.close() on tabs the user opened themselves, so this
    page is the fallback: the session is already cleared and there is no link
    back into the activity.
    """
    return render_template("exit.html")

@app.route("/about")
def about():
    """
    Static About page: project description, authors, publications, GitHub link.
    Accessible with or without a session; user_id is passed through so the
    'Back' button can return the user to where they came from.
    """
    user_id = request.args.get("user_id")
    return render_template("about.html", user_id=user_id)

@app.route("/home")
def home():
    user_id = request.args.get("user_id")
    if not user_id: return redirect(url_for("study_login"))
    return render_template("index.html", user_id=user_id)

def clean_token(token):
    special_tokens = ['</s>', '<pad>', '<|endoftext|>', '<unk>', '<|imend|>', '<|im_end|>', '<|im_start|>', 'Ċ', 'ĉ', '.*', '."']
    
    if token in special_tokens:
        return None

    cleaned_token = ""
    
    if token.startswith('Ġ'):
        cleaned_token = " " + token[1:]
    elif token.startswith(' '): 
        cleaned_token = " " + token[1:] 
    elif token.startswith('_'):
        cleaned_token = " " + token[1:] 
    else:
        cleaned_token = token 
    
    if not cleaned_token:
        return None
        
    return cleaned_token

def get_next_word_predictions(model_name, text, top_k=20, temperature=1.0, p_value=0.0):
    if model_name not in models:
        print(f"Error: Model {model_name} not found!")
        return [], [], []

    tokenizer = models[model_name]['tokenizer']
    model = models[model_name]['model']
    
    input_ids = tokenizer.encode(text, return_tensors='pt')
    current_temp = max(float(temperature), 0.01)

    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]

    top_logits, top_indices = torch.topk(logits, top_k)
    scaled_top_logits = top_logits / current_temp
    top_probs = torch.softmax(scaled_top_logits, dim=-1)
    top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())

    cleaned_tokens, cleaned_probs, cleaned_ids = [], [], []
    for token, prob, token_id in zip(top_tokens, top_probs.tolist(), top_indices.tolist()):
        cleaned = clean_token(token)
        if cleaned:
            cleaned_tokens.append(cleaned)
            display_prob = round(prob * 100, 1) 
            cleaned_probs.append(display_prob)
            cleaned_ids.append(token_id)
        
    if not cleaned_tokens and top_tokens:
        raw_top_token = top_tokens[0].replace(' ', ' ').strip()
        cleaned_tokens.append(raw_top_token if raw_top_token else top_tokens[0])
        cleaned_probs.append(top_probs[0].item())
        cleaned_ids.append(top_indices[0].item())
    
    return cleaned_tokens, cleaned_probs, cleaned_ids

@app.route("/chat")
def chat():
    user_id = request.args.get("user_id")
    if not user_id: return redirect(url_for("study_login"))
    return render_template('chat.html', user_id=user_id)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    model_name = data['model']
    text = data['text'].strip()
    temperature = float(data.get('temperature', 1.0))
    p_value = float(data.get('p_value', 0.0))

    predicted_tokens, probabilities, predicted_token_ids = get_next_word_predictions(
        model_name, text, 20, 1.0, p_value 
    )

    return jsonify({
        'predicted_tokens': predicted_tokens,
        'probabilities': probabilities,
        'predicted_token_ids': predicted_token_ids
    })

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        model_name = data.get("model", 'SmolLM-360M')
        query = data.get("query", "")
        
        raw_temp = float(data.get("temperature", 1.0))
        should_sample = raw_temp > 0.0  
        safe_temperature = max(raw_temp, 0.01) 

        top_p_value = max(0.01, min(float(data.get("top_p", 1.0)), 1.0)) 

        tokenizer = models[model_name]['tokenizer']
        model = models[model_name]['model']

        messages = [
            {"role": "system", "content": "You are a friendly assistant for middle school students. Keep answers brief and engaging."},
            {"role": "user", "content": query}
        ]
        
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            prompt = f"User: {query}\nAssistant:"

        inputs = tokenizer(prompt, return_tensors="pt")
        
        outputs = model.generate(
            inputs['input_ids'], 
            max_new_tokens=50, 
            do_sample=should_sample,       
            temperature=safe_temperature,  
            top_p=top_p_value, 
            repetition_penalty=1.2, 
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )
        
        output_ids = outputs[0][inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)