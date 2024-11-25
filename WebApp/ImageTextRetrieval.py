import tensorflow as tf
import os
import pandas as pd
import numpy as np
from transformers import TFBertModel, BertTokenizer, TFAutoModel
from tensorflow.keras import layers
class CFG:
    text_url = "google/bert_uncased_L-4_H-512_A-8"
    batch_size = 64
    size = 224
    pretrained = True
    trainable = False
    num_projection_layers = 2
    projection_dim = 512
    dropout = 0.1
    max_length = 200

class ImageEncoder(tf.keras.Model):
    def __init__(self, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super(ImageEncoder, self).__init__()

        # Upload ViT-small from Hugging Face
        self.model = TFAutoModel.from_pretrained('WinKawaks/vit-small-patch16-224',
                                                 trainable=trainable)
        self.model.trainable = trainable

        if not pretrained:
            self.model.init_weights()


    def call(self, x):
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        outputs = self.model(x)
        return outputs.last_hidden_state[:, 0]

class TextEncoder(tf.keras.Model):
    def __init__(self, model_name=CFG.text_url,
                 trainable=CFG.trainable):
        super(TextEncoder, self).__init__()
        self.bert_layer = TFBertModel.from_pretrained(model_name,
                                                      from_pt=True,
                                                      trainable=trainable)
        self.bert_layer.trainable = trainable


    def call(self, input_ids, attention_mask):
        token_type_ids = tf.zeros_like(input_ids)
        outputs = self.bert_layer(input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        return outputs.pooler_output

class ProjectionHead(tf.keras.Model):
    def __init__(self, num_projection_layers=CFG.num_projection_layers,
                 projection_dims=CFG.projection_dim, dropout_rate=CFG.dropout):
        super(ProjectionHead, self).__init__()
        self.projection_layers = tf.keras.Sequential()
        self.projection_layers.add(layers.Dense(units=projection_dims,
                                                activation="relu"))

        for _ in range(num_projection_layers - 1):
            self.projection_layers.add(layers.LayerNormalization(epsilon=1e-5))
            self.projection_layers.add(layers.Dense(projection_dims,
                                                    activation="relu"))
            self.projection_layers.add(layers.Dropout(dropout_rate))


    def call(self, inputs, training=False):
        return self.projection_layers(inputs, training=training)

data_path = os.path.join('data/')
images_path = os.path.join(data_path, 'images/')
csv_file = pd.read_csv(os.path.join(data_path, 'dataset.csv'), names=['ID', 'Caption'])

image_embeddings = None
text_embeddings = None

def load_embeddings(model_path):
    global image_embeddings
    global text_embeddings

    image_embeddings = np.loadtxt(os.path.join(model_path, 'image_embeddings.csv'), delimiter=',', dtype=np.float32)
    text_embeddings = np.loadtxt(os.path.join(model_path, 'text_embeddings.csv'), delimiter=',', dtype=np.float32)

captions = []
image_paths = []
for file in csv_file.values:
    path = images_path + file[0] + '.jpg'
    if os.path.isfile(path):
        image_paths.append(path)
        captions.append(file[1])

def read_image(image_path):
    image = tf.image.resize(
        tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3),
        size=(128, 128)
    )
    image = image / 255.0
    return image

def save_embeddings_csv(embeddings, file_path):
    np.savetxt(file_path, embeddings, delimiter=',')
    print(f"Image embeddings saved to {file_path}.")

def get_image_embeddings(model):
    global image_embeddings
    image_embeddings = []

    for i in range(0, len(image_paths), CFG.batch_size):
        print("Generating embedding for image n° {}".format(i))
        batch_paths = image_paths[i:i + CFG.batch_size]
        batch_images = []

        for image_path in batch_paths:
            image_file = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image_file, channels=3)
            image = tf.image.resize(image, [224, 224])
            image = tf.cast(image, tf.float32)
            image = (image - 127.5) / 127.5
            batch_images.append(image)

        batch_images = tf.stack(batch_images, axis=0)
        image_features = model.image_encoder(batch_images, training=False)
        batch_embeddings = model.image_projection(image_features)
        image_embeddings.extend(batch_embeddings)

    final_embeddings = tf.stack(image_embeddings, axis=0)
    save_embeddings_csv(final_embeddings, os.path.join(data_path, 'image_embeddings.csv'))
    print(f"Image embeddings shape: {final_embeddings.shape}.")

def get_text_embeddings(model):
    global text_embeddings
    text_embeddings = []

    tokenizer = BertTokenizer.from_pretrained(CFG.text_url)

    for i in range(0, len(captions), CFG.batch_size):
        print("Generating embedding for text n° {}".format(i))
        batch_captions = captions[i:i + CFG.batch_size]

        # Tokenization of batches of texts
        encoded_batch = tokenizer.batch_encode_plus(
            batch_captions,
            max_length=CFG.max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="tf"
        )

        # Extract input_ids and attention masks
        input_ids = encoded_batch['input_ids']
        attention_mask = encoded_batch['attention_mask']

        # Process texts
        text_features = model.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, training=False
        )
        batch_embeddings = model.text_projection(text_features)
        text_embeddings.extend(batch_embeddings)

    final_embeddings = tf.stack(text_embeddings, axis=0)
    print(final_embeddings)
    save_embeddings_csv(final_embeddings, os.path.join(data_path, 'text_embeddings.csv'))
    print(f"Text embeddings shape: {final_embeddings.shape}.")

def tokenize_map_function(caption):
    tokenizer = BertTokenizer.from_pretrained(CFG.text_url)
    encoded = tokenizer.encode_plus(
        caption.numpy().decode('utf-8'),
        add_special_tokens=True,
        max_length=CFG.max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )
    return encoded['input_ids'], encoded['attention_mask']

def find_image_matches(model, query, n=9):
    global image_embeddings
    if image_embeddings is None:
        get_image_embeddings(model)

    # Tokenize the query
    input_ids, attention_mask = tf.py_function(
        tokenize_map_function,
        [query],
        [tf.int32, tf.int32]
    )
    input_ids.set_shape([CFG.max_length])
    attention_mask.set_shape([CFG.max_length])

    # Add an other dimension to match what the model expects
    input_ids = tf.expand_dims(input_ids, 0)
    attention_mask = tf.expand_dims(attention_mask, 0)

    # Retrieve text features
    text_features = model.text_encoder(
        input_ids=input_ids, attention_mask=attention_mask, training=False
    )

    # Retrieve text embeddings
    text_embedding = model.text_projection(text_features)

    # Normalize embeddings
    image_embeddings_n = tf.math.l2_normalize(image_embeddings, axis=1)
    text_embeddings_n = tf.math.l2_normalize(text_embedding, axis=1)

    dot_similarity = tf.matmul(text_embeddings_n, image_embeddings_n, transpose_b=True)

    values, indices = tf.math.top_k(tf.squeeze(dot_similarity), k=int(n) * 5)

    # Find matches
    return [image_paths[idx] for idx in indices.numpy()[::5]]

def find_text_matches(model, image_path, n=9):
    global text_embeddings
    if text_embeddings is None:
        get_text_embeddings(model)
    print(text_embeddings)

    # Image pre-processing
    image_file = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_file, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5

    # Add a batch dimension
    image = tf.expand_dims(image, 0)

    # Compute the embeddings for the query image
    image_features = model.image_encoder(image, training=False)
    image_embedding = model.image_projection(image_features)

    # Normalize embeddings
    image_embedding_n = tf.math.l2_normalize(image_embedding, axis=1)
    text_embeddings_n = tf.math.l2_normalize(text_embeddings, axis=1)
    print(text_embeddings_n)

    # Compute similarity
    dot_similarity = tf.matmul(image_embedding_n, text_embeddings_n,
                               transpose_b=True)

    # Find the most n relevant captions
    values, indices = tf.math.top_k(tf.squeeze(dot_similarity), k=int(n))

    # Extract matches
    return [captions[idx] for idx in indices.numpy()]