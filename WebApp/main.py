import os
import tensorflow as tf
from PIL import Image

import flask
from flask import Flask, render_template, session, request, jsonify
import uuid

import ImageTextRetrieval

print(tf.__version__)

app = Flask(__name__)
app.secret_key = "AsseBattipagliaRoma"

#model deafult paths
models_path = "models/"

model = None

custom_objects = {
    'ImageEncoder': ImageTextRetrieval.ImageEncoder,
    'TextEncoder': ImageTextRetrieval.TextEncoder,
    'ProjectionHead': ImageTextRetrieval.ProjectionHead
}


@app.route("/")
def index():
    flask.session['session_id'] = uuid.uuid4()
    return render_template("index.html")


@app.route('/returnmodelslist', methods=['GET'])
def returnModelsList():
    modelslist = os.walk(models_path)

    for _, dirs, n in modelslist:
        modelslist = dirs
        break
    return jsonify({'result': modelslist})


@app.route('/loadmodel', methods=['POST'])
def loadModel():
    global model

    model_path = request.args.get('model')

    model = tf.keras.models.load_model(os.path.join(models_path, model_path), custom_objects)
    ImageTextRetrieval.load_embeddings(os.path.join(models_path, model_path))

    return jsonify({'status': 'success', 'message': f'Model "{model_path}" loaded successfully'})

@app.route('/textprediction', methods=['POST'])
def textPrediction():
    global model
    
    image = request.files['image']
    image.save('static/input.jpeg')
    n = request.form.get('n')

    matches = ImageTextRetrieval.find_text_matches(model, 'static/input.jpeg', n=n)
    print(matches)
    return jsonify({'result': matches})


@app.route('/imageprediction', methods=['POST'])
def imagePrediction():
    global model

    text = request.form.get('text')
    n = request.form.get('n')
    print(text)

    #Generate the image embedding if it doesn't exists
    matches = ImageTextRetrieval.find_image_matches(model, text, n=n)
    print(matches)

    result=[]
    for i, image_path in enumerate(matches):
        image = Image.open(image_path)

        # Save the image to the destination path
        image.save('static/'+image_path)
        result.append('static/'+image_path)
    #The image MUST be saved into the 'static' folder

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(port=8000)