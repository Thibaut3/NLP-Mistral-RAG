import fitz
from doctr.models import ocr_predictor
import numpy as np
import io
from PIL import Image
import os
import pandas as pd
import json

def doctr_text(image,predictor):

    # Perform OCR on the image
    image = image.convert("RGB")
    result = predictor([np.asarray(image)])

    text_img = ""
    if len(result.pages[0].blocks) > 0:
        for i in range(len(result.pages[0].blocks[0].lines)):
            for j in range(len(result.pages[0].blocks[0].lines[i].words)):
                if result.pages[0].blocks[0].lines[i].words[j].confidence > 0.7:
                    text_img += result.pages[0].blocks[0].lines[i].words[j].value + " "
            text_img += "\n"

    return text_img

def extract_complet(pdf_path,predictor):
    doc = fitz.open(pdf_path)
    text_dict = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        text = text.strip()

        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
    
        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Charger l'image avec PIL
            image = Image.open(io.BytesIO(image_bytes))

            # Vérifier si l'image a un masque de transparence
            if "smask" in base_image:
                if base_image["smask"] != 0:
                    mask_bytes = doc.extract_image(base_image["smask"])["image"]
                    mask = Image.open(io.BytesIO(mask_bytes))

                    # Convertir en mode RGBA pour appliquer le masque
                    image = image.convert("RGBA")
                    mask = mask.convert("L")  # Masque en niveaux de gris

                    # Appliquer le masque (ajout d'un canal alpha)
                    image.putalpha(mask)

            if image.mode == "RGBA":
                # Remplacer la transparence par un fond blanc (ou une autre couleur)
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))  # Blanc
                background.paste(image, (0, 0), image)
                image = background.convert("RGB")  

            text_doctr = doctr_text(image,predictor)
            text += '\n' + text_doctr

        text_dict.append(text)
            
    return text_dict

def extract_text_xls(filepath, nb_page=1):
    # Déterminer l'extension du fichier
    file_ext = os.path.splitext(filepath)[1].lower()
    
    # Choisir le moteur approprié
    if file_ext == '.xls':
        engine = 'xlrd'
    else:
        engine = 'openpyxl'
    
    # Lire le fichier Excel
    try:
        df = pd.read_excel(filepath, sheet_name=None, engine=engine)
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier Excel : {e}")
    
    sheets = list(df.keys())
    
    if len(sheets) == 1:
        text = np.array(df[sheets[0]].values)
        text = text[~pd.isna(text)]

        taille = len(text) // nb_page
        tailles = [i * taille for i in range(nb_page + 1)]
        tailles[-1] = len(text)  # Ajuster la dernière position

        all_text = []
        for i in range(nb_page):
            text_part = text[tailles[i]:tailles[i+1]]
            text_x = "\n".join(str(s) for s in text_part)
            all_text.append(text_x)
    else:
        all_text = []
        for sheet in sheets:
            text = np.array(df[sheet].values)
            text = text[~pd.isna(text)]
            text_x = "\n".join(str(s) for s in text)
            all_text.append(text_x)
    
    return all_text

def get_all_file_names(directory_path):
    file_names = []
    
    # Parcourir le répertoire
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Ajouter le chemin complet du fichier
            file_path = os.path.join(root, file)
            file_names.append(file_path)
    
    return file_names

if __name__ == '__main__':
    directory = 'files/'  
    files = get_all_file_names(directory)

    files.sort()

    predictor = ocr_predictor('db_resnet50','crnn_mobilenet_v3_small',pretrained=True)

    for i,file in enumerate(files):
        print(f'File {i}')
        if file.split('.')[-1] == 'pdf':
            data = extract_complet(file,predictor)
        else:
            data = extract_text_xls(file)

        if data == []:
            data = 'Erreur MuPDF -- RIEN'

        file_name = file.split('/')[-1].split('.')[0]
        with open(f'extracted_text/{file_name}.txt','w') as f:
            json.dump(data,f)
    
    print("Fin de l'extraction")