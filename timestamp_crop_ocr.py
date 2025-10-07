import cv2
import os
import easyocr
import json

reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if no GPU/CUDA


def EasyOCR_frame(inframe):
    # Read text from image, restricted to digits and timestamp chars for optimization
    result = reader.readtext(in_frame, allowlist='0123456789:.')  # Focus on numbers/timestamps
    
    # Extract texts (will be mostly/only numbers if detected)
    texts = [r[1] for r in result]
    
    # Calculate approximate video timestamp
    
    return texts,result

# Save to JSON




folder_base = "150136"
folder = f"../{folder_base}_4fps_sound/"
json_file = f"{folder_base}_4fps.json"

folder_contents= os.listdir(folder)
folder_contents.sort()
file = folder_contents[0]
def return_timestamp_cropped_img(img):

# coordinates of the timestamp
    x=780
    w=1156-x
    y=1014
    h=1070-y

    img_out = img[y:y+h,x:x+w]
    return img_out

for filename in folder_contents:
    filepath=folder+filename
    print(filepath)


    img = cv2.imread(filepath)
    cropped_img = return_timestamp_cropped_img(img)
    cv2.imshow("cropped", cropped_img)
    print(easyocr_frame(cropped_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 760 100 / 1150 1060
