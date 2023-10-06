from flask import Flask, request, jsonify
from transformers import pipeline


app = Flask(__name__)


"""classifier = pipeline(model="facebook/bart-large-mnli")

@app.route("/zero_shot_text_clf", methods=["POST"])
def classify_text():
    text = request.json['text']

    candidate_labels = request.json['candidate_labels']

    multi_label = bool(request.json['multi_label'])

    output = classifier(text, candidate_labels=candidate_labels, 
                        multi_label=multi_label)

    return jsonify(output)"""


classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route("/classify_text", methods=["POST"])
def classify_text():
    output = classifier(request.json['text'])

    return jsonify(output)


if __name__ == "__main__":
    app.run(host='0.0.0.0')