import cv2
import os
import numpy as np
from pathlib import Path

class MeterDigitAnnotator:
    def __init__(self, input_folder, output_dir, meter_type):
        self.input_folder = input_folder
        self.output_dir = output_dir
        self.meter_type = meter_type  # 'analogico' o 'digital'
        self.points = []
        self.current_image = None
        self.image_name = None
        self.annotations = []  # Almacena (class_id, [x1,y1, x2,y2, x3,y3, x4,y4]) por dígito
        self.class_names = {str(i): i for i in range(10)}  # Clases: '0' a '9' mapeadas a 0-9

        # Crear directorios de salida
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.current_image, (x, y), 5, (0, 255, 0), -1)
            if len(self.points) > 1:
                cv2.line(self.current_image, self.points[-2], self.points[-1], (0, 255, 0), 2)
            if len(self.points) == 4:
                cv2.line(self.current_image, self.points[-1], self.points[0], (0, 255, 0), 2)
                cv2.imshow('Image', self.current_image)
                # Solicitar valor del dígito
                digit = input("Ingrese el valor del dígito (0-9) para este ROI: ")
                if digit in self.class_names:
                    self.annotations.append((self.class_names[digit], self.points[:]))
                    print(f"Dígito '{digit}' anotado")
                else:
                    print("Dígito inválido, ROI descartado")
                self.points = []  # Reiniciar puntos para el siguiente ROI
            cv2.imshow('Image', self.current_image)

    def annotate_image(self, image_path):
        self.points = []
        self.annotations = []
        self.current_image = cv2.imread(image_path)
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]

        if self.current_image is None:
            print(f"No se pudo leer la imagen: {image_path}")
            return False

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.mouse_callback)

        print(f"Anotando {image_path} ({self.meter_type})")
        print("Haga clic en 4 puntos por dígito (superior-izquierdo, superior-derecho, inferior-derecho, inferior-izquierdo)")
        print("Ingrese el valor del dígito (0-9) después de dibujar cada ROI")
        print("Presione 'q' para salir, 's' para guardar anotaciones y pasar a la siguiente imagen")

        while True:
            cv2.imshow('Image', self.current_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
            elif key == ord('s'):
                if self.annotations:
                    self.save_annotations(image_path)
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("No se han proporcionado anotaciones, anote al menos un dígito")
                    continue

    def save_annotations(self, image_path):
        # Guardar imagen en images_dir
        dest_image_path = os.path.join(self.images_dir, f"{self.image_name}.png")
        cv2.imwrite(dest_image_path, self.current_image)

        # Guardar anotaciones en labels_dir en formato YOLO OBB
        label_path = os.path.join(self.labels_dir, f"{self.image_name}.txt")
        with open(label_path, 'w') as f:
            for class_id, points in self.annotations:
                x1, y1 = points[0]  # Superior-izquierdo
                x2, y2 = points[1]  # Superior-derecho
                x3, y3 = points[2]  # Inferior-derecho
                x4, y4 = points[3]  # Inferior-izquierdo
                f.write(f"{class_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")

    def process_images(self):
        results = []
        image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            if self.annotate_image(image_path):
                digit_values = [str(k) for k, v in self.class_names.items() if v in [a[0] for a in self.annotations]]
                results.append({
                    'image': image_file,
                    'type': self.meter_type,
                    'digits': digit_values
                })

        # Crear data.yaml para YOLOv11
        self.create_data_yaml()

        return results

    def create_data_yaml(self):
        yaml_content = f"""
train: {self.images_dir}
val: {self.images_dir}
nc: {len(self.class_names)}
names: {list(self.class_names.keys())}
"""
        with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as f:
            f.write(yaml_content)

if __name__ == "__main__":
    '''# Configuración para medidores analógicos
    analog_folder = 'cropped/analogico'
    analog_output_dir = 'analog_digit_dataset'
    analog_annotator = MeterDigitAnnotator(analog_folder, analog_output_dir, 'analogico')
    analog_results = analog_annotator.process_images()'''

    # Configuración para medidores digitales
    digital_folder = 'cropped/digital'
    digital_output_dir = 'digital_digit_dataset'
    digital_annotator = MeterDigitAnnotator(digital_folder, digital_output_dir, 'digital')
    digital_results = digital_annotator.process_images()

    '''# Imprimir resultados
    print("\nResultados para medidores analógicos:")
    for res in analog_results:
        print(f"Imagen: {res['image']} - Tipo: {res['type']} - Dígitos: {res['digits']}")'''

    print("\nResultados para medidores digitales:")
    for res in digital_results:
        print(f"Imagen: {res['image']} - Tipo: {res['type']} - Dígitos: {res['digits']}")