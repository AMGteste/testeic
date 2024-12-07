from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import pickle
# Criação da aplicação FastAPI
app = FastAPI()

# Carregar o modelo salvo
model = load_model("flower_modelteste.h5")

model.compile(optimizer=Adam(learning_rate=0.00122195324402186004), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

with open("label_encoder.pkl", "rb") as f:
    nomes_codif = pickle.load(f)

# Classes de flores
flower_classes = list(nomes_codif.inverse_transform(range(len(nomes_codif.classes_))))
IMG_SIZE = 224

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Recebe uma imagem e retorna a classe predita.
    """
    try:
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Pré-processamento
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Predição
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        predicted_class = flower_classes[class_index]
        confidence = float(predictions[0][class_index])

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
