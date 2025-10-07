import cv2
import os
import easyocr
import json
import numpy as np
from google.api_core.client_options import ClientOptions
from google.cloud import vision

# reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if no GPU/CUDA

folder_base = "145147"
folder = f"../{folder_base}_4fps_sound/"
json_file = f"{folder}{folder_base}_4fps_ocr_whole.json"

folder_contents= os.listdir(folder)
folder_contents.sort()
file = folder_contents[0]

def easyocr_frame(in_frame):
    result = reader.readtext(in_frame, allowlist='0123456789:.')  # Focus on numbers/timestamps
    
    
    
    return result

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
                    vertices = [(vertex.x, vertex.y)
                                for vertex in word.bounding_box.vertices]
                    
                    word_data = {
                        "word": word_text,
                        "confidence": confidence,
                        "bounding_box": vertices
                    }
                    all_word_data.append(word_data)
                                    


    return all_word_data

def g_wordlist_draw_boxes(result,cv2_img):
    vertices = result["bounding_box"]
    word_text = result["word"]
    confidence = result["confidence"]

    # Convert vertices to numpy array of integers for OpenCV
    # Each point is [x, y], reshaped to (-1, 1, 2) for polylines
    pts = np.array([[vertex[0], vertex[1]] for vertex in vertices], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # Draw the polygon outline on the image
    # True for closed shape, (0,255,0) for green color, 1 for thickness
    cv2.polylines(cv2_img, [pts], True, (0, 255, 0), 1)
    # poly_gone_away=poly_gone_away.reshape((-1,1,2))
    # Calculate position for text "hello" above the top of the bounding box
    min_x = min(vertex[0] for vertex in vertices)
    min_y = min(vertex[1] for vertex in vertices)
    text_pos = (int(max_x-min_x), int(min_y - 15))  # 5 pixels above the top edge
    # Draw the text "hello" above the polygon
    # Using HERSHEY_SIMPLEX font, scale 0.5, blue color (255,0,0), thickness 1
    cv2.putText(cv2_img,word_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 1)

    return cv2_img




api_key = os.environ.get('gooog')


output_list = []
for filename in folder_contents:
    filepath=folder+filename


    google_doc_word_list = g_cv_doc_text_detect(filepath, api_key)

    thisfileresults = []


    img = cv2.imread(filepath)

    for this_result in google_doc_word_list:
        # this_easyocr_dict = {'filename':filename,'bounding_box': str(this_result[0]),'text': str(this_result[1]), 'confidence': str(this_result[2])}
        thisfileresults.append(this_result)
        tmpimg = wordlist_draw_boxes(this_result, img)



        cv2.imshow("test", tmpimg)
        print(this_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    easyocr_dict = {'filename':filename,'result_list': easy_ocr_result}
# 760 100 / 1150 1060
# with open(json_file, "w", encoding="utf-8") as f:
#     json.dump(output_list, f, ensure_ascii=False, indent=4)
