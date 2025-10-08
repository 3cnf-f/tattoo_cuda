import cv2
import re
import os
import easyocr
import json
import numpy as np
from google.api_core.client_options import ClientOptions
from google.cloud import vision

# this program gets a list of fils in a folder and iterates over them, sending each to google cloud vision for ocr


folder_base = "145147"
folder = f"../{folder_base}_4fps_sound/"
json_file = f"{folder}{folder_base}_4fps_ocr_whole.json"

folder_contents= os.listdir(folder)
folder_contents.sort()


def get_coords_from_google(word):
    """
    Converts a Google Cloud Vision word bounding box to NumPy/OpenCV slices and min/max coords.
    
    Args:
        word: A Word object from Google Cloud Vision's text annotation.
    
    Returns:
        dict: {
            'slice_y': slice(y_min, y_max),
            'slice_x': slice(x_min, x_max),
            'x_min': int, 'x_max': int,
            'y_min': int, 'y_max': int,
            'bottom_left': (x_min, y_max)  # For cv2.putText org parameter
        }
    """
    vertices = word.bounding_box.vertices
    if not vertices:
        raise ValueError("Bounding box has no vertices.")
    
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Convert to integers, rounding if necessary (handles float coordinates)
    x_min = int(round(x_min))
    x_max = int(round(x_max))
    y_min = int(round(y_min))
    y_max = int(round(y_max))
    
    # Ensure valid bounds (x_min < x_max, y_min < y_max)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Invalid bounding box dimensions.")
    
    return {
        'slice_y': slice(y_min, y_max),
        'slice_x': slice(x_min, x_max),
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'bottom_left': (x_min, y_max)
    }

def g_cv_doc_text_detect(file_contents, api_key):
    """Detects text in the image file and returns words with confidence scores."""

    client_options = ClientOptions(api_key=api_key)
    client = vision.ImageAnnotatorClient(client_options=client_options)
    
    content = file_contents

    image = vision.Image(content=content)

    # Use document_text_detection for more detailed results including confidence
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

    # The first text annotation is the full text of the image.
    # Subsequent annotations are for individual words.
    # To get confidence for each word, we need to iterate through the fullTextAnnotation

    all_word_data = []

    # The response from document_text_detection contains a fullTextAnnotation
    # which is structured into pages, blocks, paragraphs, words, and symbols.
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    confidence = word.confidence
                    cv_2_bounding_box = get_coords_from_google(word)
                    
                    word_data = {
                        "word": word_text,
                        "confidence": confidence,
                        "cv2_bounding_box": cv_2_bounding_box,
                        "original_bounding_box": word.bounding_box.vertices
                        
                    }
                    all_word_data.append(word_data)
                                    


    return all_word_data

def g_wordlist_draw_annot(word_text,x_min,y_min,x_max,y_max,cv2_img):
    print(word_text,x_min,y_min,x_max,y_max,cv2_img)

    # Draw the text "hello" above the polygon
    # Using HERSHEY_SIMPLEX font, scale 0.5, blue color (255,0,0), thickness 1
    cv2.rectangle(cv2_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.putText(cv2_img,word_text, (int((x_min+x_max)/2), y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    return cv2_img

def is_it_a_temp(word_text,cv2_bounding_box):
    resultatata = re.search("([\d]{2})([\.]{0,1})([\d]{1})", word_text)
    
    if resultatata and (resultatata.group(1) != "40" ):
        
        return True, resultatata.group(1)+"."+resultatata.group(3)

    return False,None


api_key = os.environ.get('gooog')

allframesdict ={}
for filename in folder_contents[:3]:
    filepath=folder+filename
    framenumber = re.search("[\d_\w]+([\d]{6})\.png", filename).group(1)
    allframesdict[framenumber]={}
    allframesdict[framenumber]['filename'] = filename
    allframesdict[framenumber]['framenumber'] = framenumber
    




    with open(filepath, 'rb') as f:
        file_contents = f.read()
    


    img = cv2.imread(filepath)
    google_doc_word_list = g_cv_doc_text_detect(file_contents, api_key)
    result_dict = {}
    for this_result in google_doc_word_list:
        word = this_result['word'].strip().replace("\n","").replace(" ","")
        result_dict[word]={}
        result_dict[word]['cv2_bounding_box']=this_result['cv2_bounding_box']
        result_dict[word]['confidence'] = this_result['confidence']
        result_dict[word]['original_bounding_box'] = this_result['original_bounding_box']
        print(f"word:{word} with confidence:{int(this_result['confidence']*100)}" )
        #add to image:
        is_a_temp,parsedword = is_it_a_temp(this_result['word'],this_result['cv2_bounding_box'])
        if is_a_temp:
            img=g_wordlist_draw_annot(parsedword,this_result['cv2_bounding_box']['x_min'],this_result['cv2_bounding_box']['y_min'],this_result['cv2_bounding_box']['x_max'],this_result['cv2_bounding_box']['y_max'],img)
            result_dict[word]['is_atemp'] = True
            result_dict[word]['parsedword'] = parsedword
        else:
            result_dict[word]['is_atemp'] = False

    allframesdict[framenumber]['result_dict'] = result_dict


        
        

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(result_dict)
    input("Press enter to continue")
    input("Press enter again")
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(output_list, f, ensure_ascii=False, indent=4)
