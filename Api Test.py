from flask import Flask, jsonify, request
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('Models')
model = GPT2LMHeadModel.from_pretrained('Models')

def generate(code, max_length=50):
    tokenized = tokenizer.encode(code, return_tensors='pt')
    resp = model.generate(
        tokenized,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )

    return (tokenizer.decode(resp[0]))

app = Flask(__name__)

@app.route('/api/data')
def get_data():
    data = {'message': 'Hello from Python API!'}
    return jsonify(data)

@app.route('/api/data', methods=['POST'])
def receive_data():
    received_data = request.json
    data = generate(received_data.get('code'), received_data.get('length'))
    print(data)
    return jsonify({'Generate': data})

if __name__ == '__main__':
    app.run(debug=True)
