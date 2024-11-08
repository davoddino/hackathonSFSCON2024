from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import re
import torch
import os

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

moondream_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    revision=revision,
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

moondream_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

blip_model_id = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(blip_model_id)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id).to("cuda" if torch.cuda.is_available() else "cpu")

child_image_path = "../data/bambino.jpg"
adult_image_path = "../data/adulto.jpg"

def get_image_description(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = blip_processor(image, return_tensors="pt").to(moondream_model.device)
        with torch.no_grad():
            outputs = blip_model.generate(**inputs, max_length=100)
        return blip_processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Errore nella descrizione dell'immagine: {e}")
        return ""

def extract_structured_info(image_path, description):
    try:
        image = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            image_embeds = moondream_model.encode_image(image).to(moondream_model.device)
        question = (
            "Tell me if the person in the picture is a child or an adult. Respond only with 'Child' or 'Adult'. For example:\n"
            "Child"
        )
        response = moondream_model.answer_question(image_embeds, question, moondream_tokenizer)
        return parse_age_group(response)
    except Exception as e:
        print(f"Errore nell'estrazione delle informazioni strutturate: {e}")
        return {}

def parse_age_group(response):
    try:
        match = re.search(r'\b(Child|Adult)\b', response, re.IGNORECASE)
        if match:
            return {"age_group": match.group(1).capitalize()}
        else:
            print("Nessuna informazione valida trovata nell'output del modello.")
            return {"age_group": "Not specified"}
    except Exception as e:
        print(f"Errore nel parsing della classificazione dell'età: {e}")
        return {"age_group": "Not specified"}

def display_classification_image(age_group):
    try:
        if age_group == "Child" and os.path.exists(child_image_path):
            Image.open(child_image_path).show(title="Classification Result: Child")
        elif age_group == "Adult" and os.path.exists(adult_image_path):
            Image.open(adult_image_path).show(title="Classification Result: Adult")
        else:
            print("Classificazione dell'età non riconosciuta. Nessuna immagine verrà mostrata.")
    except Exception as e:
        print(f"Errore nella visualizzazione dell'immagine di classificazione: {e}")

def analyze_scene(image_path):
    description = get_image_description(image_path)
    if not description:
        print("Descrizione dell'immagine non disponibile.")
        return
    
    print("\nDescrizione dell'immagine:")
    print(description)
    
    structured_info = extract_structured_info(image_path, description)
    print("\nDescrizione Strutturata:")
    print(structured_info)
    
    age_group = structured_info.get("age_group", "Not specified")
    display_classification_image(age_group)

if __name__ == "__main__":
    image_path = "../data/dario.jpg"
    if not os.path.exists(image_path):
        print(f"L'immagine specificata non esiste: {image_path}")
    else:
        analyze_scene(image_path)
