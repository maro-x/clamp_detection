from ultralytics import YOLO
from PIL import Image

tuned_model_1 = YOLO (r"C:/Users/DELL/runs/detect/train7/weights/best.pt")

def predict_image(img_path, model= 'model_1'):
   
    if model == 'model_1':
        result = tuned_model_1.predict(img_path)


    result = result[0]

    result = result.plot()

    pil_image = Image.fromarray(result)
    
    return pil_image



if __name__ == '__main__':
    pre_image = predict_image(r"C:/Users/DELL/OneDrive/Desktop/calmp_images/vlcsnap-2026-05-11-11h00m31s945.png",'model_1')
    pre_image.show()