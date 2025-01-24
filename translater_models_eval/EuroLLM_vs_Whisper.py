import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from llama_cpp import Llama
from langdetect import detect
from gtts import gTTS
import io
from pydub import AudioSegment
import numpy as np
import pandas as pd
import time

# Setup device and data types
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper model and processor
model_id = "openai/whisper-medium"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Create a pipeline for automatic speech recognition with translation
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1,
)

# Function to translate audio using Whisper
def translate_audio(audio_file):
    if isinstance(audio_file, io.BytesIO):
        audio_file.seek(0)
        audio = AudioSegment.from_file(audio_file, format="mp3")
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio_array = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        audio_file = {"array": audio_array, "sampling_rate": 16000}
    elif isinstance(audio_file, str):
        pass
    else:
        return "Invalid audio input format. Please provide a valid file or buffer."
    
    result = pipe(audio_file, generate_kwargs={"task": "translate"})
    return result["text"]

# Function to process text using Whisper
def process_text_whisper(text):
    try:
        # Detect language
        language = detect(text)

        # Convert text to speech and save as an in-memory buffer
        tts = gTTS(text=text, lang=language)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        # Translate the audio
        return translate_audio(audio_buffer)
    except Exception as e:
        return f"Error: {e}"

# Load EuroLLM models
model_path_1b = r"model\eurollm-1.7b-instruct-q4_k_m.gguf"
model_path_9b = r"model\EuroLLM-9B-Instruct-Q4_K_M.gguf"

llm_1b = Llama(model_path=model_path_1b, n_ctx=1024)
llm_9b = Llama(model_path=model_path_9b, n_ctx=1024)

# Function to process text using EuroLLM
def process_text_EuroLLM_1B(text, llm):
    input_prompt = f"""
    <|im_start|>system
    <|im_end|>
    <|im_start|>user
    Translate the following text to English:
    Text: {text}
    English: 
    <|im_end|>
    <|im_start|>assistant
    """
    output = llm(input_prompt, max_tokens=1024, temperature=0)
    print('\n\nOutput: ',output,'\n\n')
    full_response = output.get("choices", [{}])[0].get("text", "").strip()
    print('\n\nResponse: ',full_response,'\n\n')
    marker = "English:"
    if marker in full_response:
        translated_text = full_response.split(marker)[-1].strip()
        return translated_text if translated_text else "No valid translation provided."
    return "No valid translation provided."

# Function to process text using EuroLLM
def process_text_EuroLLM_9B(text, llm):
    input_prompt = f"""
    <|im_start|>system
    <|im_end|>
    <|im_start|>user
    Translate the following text to English:
    Text: {text}
    English: 
    <|im_end|>
    <|im_start|>assistant
    """
    output = llm(input_prompt, max_tokens=1024, temperature=0)
    print('\n\nOutput: ',output,'\n\n')
    full_response = output.get("choices", [{}])[0].get("text", "").strip()
    print('\n\nResponse: ',full_response,'\n\n')
    return full_response


# Function to evaluate models and store results
def evaluate_models(language, input_text):
    results = []

    # Whisper Model
    start_time = time.time()
    translated_text = process_text_whisper(input_text)
    end_time = time.time()
    results.append({
        "Model": "Whisper",
        "language":language,
        "Input": input_text,
        "Output": translated_text,
        "Time (s)": end_time - start_time,
    })

    start_time = time.time()
    translated_text = process_text_EuroLLM_1B(input_text, llm_1b)
    end_time = time.time()
    results.append({
            "Model": "EuroLLM-1.7B",
            "language":language,
            "Input": input_text,
            "Output": translated_text,
            "Time (s)": end_time - start_time,
        })
    
    start_time = time.time()
    translated_text = process_text_EuroLLM_9B(input_text, llm_9b)
    end_time = time.time()
    results.append({
            "Model": "EuroLLM-9B",
            "language":language,
            "Input": input_text,
            "Output": translated_text,
            "Time (s)": end_time - start_time,
        })



    return results

if __name__ == "__main__":
    # Define input text
    input_text = [
        ["English", "Spread out and search for heat signatures."],
        ["French", "Répartissez-vous et recherchez des signatures thermiques."],
        ["German", "Verteilen Sie sich und suchen Sie nach Wärmesignaturen."],
        ["Spanish", "Distribúyanse y busquen firmas de calor."],
        ["Italian", "Sparpagliatevi e cercate firme di calore."],
        ["Dutch", "Spreid uit en zoek naar warmtehandtekeningen."],
        ["Polish", "Rozproszyć się i szukać sygnatur cieplnych."],
        ["Portuguese", "Espalhem-se e procurem por assinaturas de calor."],
        ["Swedish", "Sprid ut er och sök efter värmesignaturer."],

        ["English", "Mark the location of survivors and form a path to guide rescue workers."],
        ["French", "Marquez l'emplacement des survivants et formez un chemin pour guider les secours."],
        ["German", "Markieren Sie den Standort der Überlebenden und bilden Sie einen Weg, um Rettungskräfte zu leiten."],
        ["Spanish", "Marquen la ubicación de los supervivientes y formen un camino para guiar a los rescatistas."],
        ["Italian", "Segnate la positione dei sopravvissuti e formate un percorso per guidare i soccorritori."],
        ["Dutch", "Markeer de locatie van overlevenden en vorm een pad om reddingswerkers te begeleiden."],
        ["Polish", "Oznacz miejsce przebywania ocalałych i wytycz ścieżkę, aby poprowadzić ratowników."],
        ["Portuguese", "Marque a localização dos sobreviventes e forme um caminho para guiar os socorristas."],
        ["Swedish", "Markera platsen för överlevande och skapa en väg för att guida räddningsarbetare."],

        ["English", "Scan the field for dry or unhealthy plants."],
        ["French", "Scannez le champ à la recherche de plantes sèches ou malades."],
        ["German", "Scannen Sie das Feld nach trockenen oder kranken Pflanzen."],
        ["Spanish", "Escaneen el campo en busca de plantas secas o enfermas."],
        ["Italian", "Scansionate il campo alla ricerca di piante secche o malsane."],
        ["Dutch", "Scan het veld op droge of ongezonde planten."],
        ["Polish", "Zeskanuj pole w poszukiwaniu suchych lub chorych roślin."],
        ["Portuguese", "Escaneie o campo em busca de plantas secas ou doentes."],
        ["Swedish", "Skanna fältet efter torra eller ohälsosamma växter."],

        ["English", "Cluster around unhealthy plants and emit a signal for manual inspection."],
        ["French", "Regroupez-vous autour des plantes malades et émettez un signal pour une inspection manuelle."],
        ["German", "Gruppieren Sie sich um kranke Pflanzen und senden Sie ein Signal für eine manuelle Inspektion."],
        ["Spanish", "Agrúpense alrededor de las plantas enfermas y emitan una señal para una inspección manual."],
        ["Italian", "Raggruppatevi intorno alle piante malsane ed emettete un segnale per un'ispezione manuale."],
        ["Dutch", "Groep rond ongezonde planten en zend een signaal voor handmatige inspectie."],
        ["Polish", "Skup się wokół chorych roślin i emituj sygnał do ręcznej inspekcji."],
        ["Portuguese", "Agrupe-se ao redor das plantas doentes e emita um sinal para inspeção manual."],
        ["Swedish", "Samlas runt ohälsosamma växter och sänd en signal för manuell inspektion."],

        ["English", "Search for plastic waste and collect it in one location."],
        ["French", "Recherchez les déchets plastiques et rassemblez-les en un seul endroit."],
        ["German", "Suchen Sie nach Plastikmüll und sammeln Sie ihn an einem Ort."],
        ["Spanish", "Busquen residuos plásticos y júntenlos en un solo lugar."],
        ["Italian", "Cercate i rifiuti di plastica e raccoglieteli in un unico posto."],
        ["Dutch", "Zoek naar plastic afval en verzamel het op één locatie."],
        ["Polish", "Szukaj plastikowych odpadów i zbierz je w jednym miejscu."],
        ["Portuguese", "Procure por resíduos plásticos e reúna-os em um único local."],
        ["Swedish", "Sök efter plastavfall och samla det på en plats."],

        ["English", "After cleaning, form a perimeter around the park."],
        ["French", "Après le nettoyage, formez un périmètre autour du parc."],
        ["German", "Bildet nach der Reinigung einen Perimeter um den Park."],
        ["Spanish", "Después de limpiar, formen un perímetro alrededor del parque."],
        ["Italian", "Dopo la pulizia, formate un perimetro attorno al parco."],
        ["Dutch", "Vorm na het schoonmaken een perimeter rond het park."],
        ["Polish", "Po sprzątaniu utwórz obwód wokół parku."],
        ["Portuguese", "Após a limpeza, forme um perímetro ao redor do parque."],
        ["Swedish", "Efter rengöring, skapa en perimeter runt parken."],

        ["English", "Form a foundation grid at these coordinates."],
        ["French", "Formez une grille de fondation à ces coordonnées."],
        ["German", "Bildet ein Fundamentraster an diesen Koordinaten."],
        ["Spanish", "Formen una cuadrícula de base en estas coordenadas."],
        ["Italian", "Formate una griglia di base a queste coordinate."],
        ["Dutch", "Vorm een funderingsrooster op deze coördinaten."],
        ["Polish", "Utwórz siatkę fundamentów w tych współrzędnych."],
        ["Portuguese", "Forme uma grade de fundação nestas coordenadas."],
        ["Swedish", "Skapa ett grundläggningsrutnät vid dessa koordinater."],

        ["English", "Illuminate the site from above."],
        ["French", "Illuminez le site depuis les airs."],
        ["German", "Erleuchten Sie die Stelle von oben."],
        ["Spanish", "Iluminen el sitio desde arriba."],
        ["Italian", "Illumina il sito dall'alto."],
        ["Dutch", "Verlicht de locatie van bovenaf."],
        ["Polish", "Oświetl miejsce z góry."],
        ["Portuguese", "Ilumine o local de cima."],
        ["Swedish", "Belys platsen uppifrån."],

        ["English", "Form a large circle and flash lights in sync with the music."],
        ["French", "Formez un grand cercle et clignotez les lumières en rythme avec la musique."],
        ["German", "Bilden Sie einen großen Kreis und blinken Sie die Lichter im Takt der Musik."],
        ["Spanish", "Formen un gran círculo y parpadeen las luces al ritmo de la música."],
        ["Italian", "Formate un grande cerchio e lampeggiate le luci a ritmo di musica."],
        ["Dutch", "Vorm een grote cirkel en knipper met de lichten in sync met de muziek."],
        ["Polish", "Utwórz duży okrąg i migaj światłami w rytm muzyki."],
        ["Portuguese", "Forme um grande círculo e pisque as luzes em sincronia com a música."],
        ["Swedish", "Skapa en stor cirkel och blinka med lamporna i takt med musiken."],

        ["English", "Morph into a star shape and rotate clockwise."],
        ["French", "Formez une étoile et tournez dans le sens des aiguilles d'une montre."],
        ["German", "Bilden Sie einen Stern und drehen Sie sich im Uhrzeigersinn."],
        ["Spanish", "Formen una estrella y giren en el sentido de las agujas del reloj."],
        ["Italian", "Formate una stella e ruotate in senso orario."],
        ["Dutch", "Vorm een ster en draai met de klok mee."],
        ["Polish", "Utwórz gwiazdę i obróć zgodnie z ruchem wskazówek zegara."],
        ["Portuguese", "Forme uma estrela e gire no sentido horário."],
        ["Swedish", "Bild en stjärna och rotera medsols."]
    ]



    # Initialize an empty list to collect results
    all_results = []

    # Evaluate models for each language-text pair
    for language, text in input_text:
        results = evaluate_models(language, text)
        print(results)
        all_results.extend(results)  # Append results to the main list

    # Create a DataFrame and display results
    df = pd.DataFrame(all_results)
    print(df)

    # Save to a CSV file
    df.to_csv("model_comparison_10-Examples.csv", index=False)