import cv2
import os
import easyocr
import json

reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if no GPU/CUDA

folder_base = "145147"
folder = f"../{folder_base}_4fps_sound/"
json_file = f"{folder}{folder_base}_4fps_ocr_whole.json"

folder_contents= os.listdir(folder)
folder_contents.sort()
file = folder_contents[0]

def easyocr_frame(in_frame):
    result = reader.readtext(in_frame, allowlist='0123456789:.')  # Focus on numbers/timestamps
    
    
    
    return result









output_list = []
for filename in folder_contents:
    filepath=folder+filename




    thisfileresults = []


    img = cv2.imread(filepath)
    easy_ocr_result = (easyocr_frame(img))
    for this_result in easy_ocr_result:
        # this_easyocr_dict = {'filename':filename,'bounding_box': str(this_result[0]),'text': str(this_result[1]), 'confidence': str(this_result[2])}
        thisfileresults.append(this_result)
        tmpimg = img.copy()

        cv2.rectangle(tmpimg, (int(this_result[0][0][0]), int(this_result[0][0][1])),  (int(this_result[0][2][0]), int(this_result[0][2][1])), (0, 0, 255), 2)
        cv2.putText(tmpimg, f" {this_result[1]}", (int(this_result[0][0][0]), int(this_result[0][0][1]) - 20),

                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow("test", tmpimg)
        print(this_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    easyocr_dict = {'filename':filename,'result_list': easy_ocr_result}
# 760 100 / 1150 1060
# with open(json_file, "w", encoding="utf-8") as f:
#     json.dump(output_list, f, ensure_ascii=False, indent=4)
