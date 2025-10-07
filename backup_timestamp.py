import cv2
import os
import easyocr
import json

reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if no GPU/CUDA

folder_base = "145147"
folder = f"../{folder_base}_4fps_sound/"
json_file = f"{folder}{folder_base}_4fps_timestamps.json"

folder_contents= os.listdir(folder)
folder_contents.sort()
file = folder_contents[0]

def easyocr_frame(in_frame):
    result = reader.readtext(in_frame, allowlist='0123456789:.')  # Focus on numbers/timestamps
    
    
    
    return result





def return_timestamp_cropped_img(img):

# coordinates of the timestamp
    x=780
    w=1156-x
    y=1014
    h=1070-y

    img_out = img[y:y+h,x:x+w]
    return img_out
output_list = []
for filename in folder_contents:
    filepath=folder+filename


    img = cv2.imread(filepath)
    cropped_img = return_timestamp_cropped_img(img)
    # cv2.imshow("cropped", cropped_img)
    easy_ocr_result = (easyocr_frame(cropped_img))
    easyocr_dict = {'filename':filename,'bounding_box': str(easy_ocr_result[0][0]),'text': str(easy_ocr_result[0][1]), 'confidence': str(easy_ocr_result[0][2])}
    # print(easyocr_dict)
    output_list.append(easyocr_dict)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # input("Press enter to continue")
# 760 100 / 1150 1060
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(output_list, f, ensure_ascii=False, indent=4)
