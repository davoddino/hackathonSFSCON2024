import cv2
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import re

# Carica il modello e il tokenizer di moondream
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    torch_dtype=torch.float16
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Funzione per analizzare un frame e ottenere una descrizione testuale
def analyze_frame(image, description_prompt=""):
    try:
        with torch.no_grad():  # Assicura che non si calcolino gradienti
            enc_image = model.encode_image(image)

            # Prompt per la descrizione
            question = f"{description_prompt}. Descrivi l'età approssimativa, il sesso, l'origine e l'emozione di ciascuna persona rilevata nella scena."

            # Ottiene la descrizione
            description = model.answer_question(enc_image, question, tokenizer)
            return description
    except Exception as e:
        print(f"Errore nell'analizzare il frame: {e}")
        return "Analisi non riuscita."

# Inizializza la webcam
cap = cv2.VideoCapture(0)  # Usa la webcam predefinita
if not cap.isOpened():
    print("Errore: impossibile aprire la webcam.")
    exit()

# Loop per acquisire frame ogni 5 secondi e analizzarli
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Errore nel leggere il frame.")
            break

        # Converti il frame in un'immagine PIL per l'analisi
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Descrivi cosa si vede
        description = analyze_frame(pil_image, "Analizza la scena")
        print(f"Descrizione trovata: {description}")

        # Mostra il frame con un overlay della descrizione nel terminale
        cv2.imshow('Webcam - Analisi in tempo reale', frame)

        # Aspetta 5 secondi prima di analizzare il prossimo frame
        if cv2.waitKey(5000) & 0xFF == ord('q'):
            break

finally:
    # Rilascia le risorse
    cap.release()
    cv2.destroyAllWindows()
