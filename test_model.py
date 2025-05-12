import os
import cv2
import numpy as np
from ultralytics import YOLO

def predict_and_plot_split(model_path, images_dir, output_dir, show=False):
    model = YOLO(model_path)

    obb_dir = os.path.join(output_dir, 'obb_pain')
    pain_dir = os.path.join(output_dir, 'pain')
    os.makedirs(obb_dir, exist_ok=True)
    os.makedirs(pain_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            continue

        results = model.predict(image_path)

        digits = []
        for box, cls in zip(results[0].obb.xyxyxyxy, results[0].obb.cls):
            x1 = box[0][0].item()
            digit = int(cls.item())
            digits.append((x1, digit, box))
        digits.sort(key=lambda x: x[0])

        number = ''.join(str(d[1]) for d in digits)
        print(f"Imagen: {image_file} - numero detectado: {number}")

        plotted_image = results[0].plot()
        height, width, _ = plotted_image.shape
        extension_width = 200
        extended_image_obb = np.ones((height, width + extension_width, 3), dtype=np.uint8) * 255
        extended_image_obb[:, :width, :] = plotted_image

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 0)
        thickness = 2
        text_position = (width + 10, height // 2)
        cv2.putText(extended_image_obb, number, text_position, font, font_scale, font_color, thickness)

        output_obb_path = os.path.join(obb_dir, f"pred_{image_file}")
        cv2.imwrite(output_obb_path, extended_image_obb)
        print(f"Guardado (con OBBs): {output_obb_path}")

        extended_image_pain = np.ones((height, width + extension_width, 3), dtype=np.uint8) * 255
        extended_image_pain[:, :width, :] = image
        cv2.putText(extended_image_pain, number, text_position, font, font_scale, font_color, thickness)

        output_pain_path = os.path.join(pain_dir, f"pred_{image_file}")
        cv2.imwrite(output_pain_path, extended_image_pain)
        print(f"Guardado (sin OBBs): {output_pain_path}")

        if show:
            cv2.imshow(f'Dígitos detectados en {image_file}', extended_image_obb)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = './model02.pt'

    images_dir = 'cropped/digital'

    output_dir = 'predict_digital'

    show = False

    predict_and_plot_split(model_path, images_dir, output_dir, show)