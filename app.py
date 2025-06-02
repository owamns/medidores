from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
import traceback
import logging
from process_images import MeterProcessor

# Configurar logging para producción
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuración para producción - Centralizada con variables de entorno
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
app.config['JSON_SORT_KEYS'] = False

processor = None


def initialize_processor():
    global processor
    try:
        # Usar variables de entorno para rutas de modelos
        crop_model_path = os.environ.get('CROP_MODEL_PATH', 'models/best-recorte.pt')
        digital_model_path = os.environ.get('DIGITAL_MODEL_PATH', 'models/best-digital.pt')
        electronic_model_path = os.environ.get('ELECTRONIC_MODEL_PATH', 'models/best-electronico.pt')

        missing_models = []
        for name, path in [
            ('crop', crop_model_path),
            ('digital', digital_model_path),
            ('electronic', electronic_model_path)
        ]:
            if not os.path.exists(path):
                missing_models.append(f"{name}: {path}")

        if missing_models:
            logger.error(f"Missing model files: {', '.join(missing_models)}")
            return False

        # Usar variables de entorno para configuración
        scale_factor = int(os.environ.get('SCALE_FACTOR', 4))
        conf_threshold = float(os.environ.get('CONF_THRESHOLD', 0))

        processor = MeterProcessor(
            crop_model_path=crop_model_path,
            digital_model_path=digital_model_path,
            electronic_model_path=electronic_model_path,
            scale_factor=scale_factor,
            conf_threshold=conf_threshold
        )
        logger.info(
            f"MeterProcessor initialized successfully with scale_factor={scale_factor}, conf_threshold={conf_threshold}")
        return True
    except Exception as e:
        logger.error(f"Error initializing MeterProcessor: {str(e)}")
        return False


def base64_to_cv2(base64_string):
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_data))
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return cv2_image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")


def validate_image(image):
    if image is None:
        return False, "Image is None"
    if len(image.shape) != 3:
        return False, "Image must be a color image (3 channels)"
    height, width = image.shape[:2]
    if height < 50 or width < 50:
        return False, "Image too small (minimum 50x50 pixels)"
    if height > 5000 or width > 5000:
        return False, "Image too large (maximum 5000x5000 pixels)"
    return True, "Valid image"


@app.route('/health', methods=['GET'])
def health_check():
    global processor
    return jsonify({
        'status': 'healthy',
        'processor_loaded': processor is not None,
        'message': 'Meter Processing API is running'
    })


@app.route('/process-meter', methods=['POST'])
def process_meter():
    global processor
    if processor is None:
        logger.error("Processor not initialized")
        return jsonify({
            'success': False,
            'error': 'Processor not initialized',
            'message': 'Models not loaded properly'
        }), 500

    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Invalid request format',
                'message': 'Request must be JSON with base64 encoded image'
            }), 400

        data = request.get_json()
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing image data',
                'message': 'Please provide base64 encoded image in "image" field'
            }), 400

        try:
            image = base64_to_cv2(data['image'])
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Invalid image format',
                'message': str(e)
            }), 400

        is_valid, validation_message = validate_image(image)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Invalid image',
                'message': validation_message
            }), 400

        result = processor.process_image(image)

        response = {
            'success': True,
            'data': {
                'detected_number': result.get('detected_number', 0),
                'meter_type': result.get('meter_type'),
                'meter_type_description': {
                    'e': 'Electronic meter',
                    'd': 'Digital meter'
                }.get(result.get('meter_type'), 'Unknown')
            }
        }

        if 'error' in result:
            response['data']['processing_error'] = result['error']
            logger.warning(f"Processing error: {result['error']}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing meter: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Processing error',
            'message': 'An error occurred while processing the image'
        }), 500


@app.route('/process-meter-file', methods=['POST'])
def process_meter_file():
    global processor
    if processor is None:
        logger.error("Processor not initialized")
        return jsonify({
            'success': False,
            'error': 'Processor not initialized',
            'message': 'Models not loaded properly'
        }), 500

    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided',
                'message': 'Please upload an image file'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'message': 'Please select a file'
            }), 400

        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': 'Invalid file type',
                'message': f'Allowed file types: {", ".join(allowed_extensions)}'
            }), 400

        try:
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Invalid image file',
                'message': 'Could not read image file'
            }), 400

        is_valid, validation_message = validate_image(image)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Invalid image',
                'message': validation_message
            }), 400

        result = processor.process_image(image)

        response = {
            'success': True,
            'data': {
                'filename': file.filename,
                'detected_number': result.get('detected_number', 0),
                'meter_type': result.get('meter_type'),
                'meter_type_description': {
                    'e': 'Electronic meter',
                    'd': 'Digital meter'
                }.get(result.get('meter_type'), 'Unknown')
            }
        }

        if 'error' in result:
            response['data']['processing_error'] = result['error']
            logger.warning(f"Processing error: {result['error']}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing meter file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Processing error',
            'message': 'An error occurred while processing the image'
        }), 500


@app.route('/models/info', methods=['GET'])
def models_info():
    global processor
    model_info = {
        'crop_model': 'models/best-recorte.pt',
        'digital_model': 'models/best-digital.pt',
        'electronic_model': 'models/best-electronico.pt'
    }

    models_status = {}
    for model_name, model_path in model_info.items():
        models_status[model_name] = {
            'path': model_path,
            'exists': os.path.exists(model_path),
            'size_mb': round(os.path.getsize(model_path) / (1024 * 1024), 2) if os.path.exists(model_path) else 0
        }

    return jsonify({
        'processor_initialized': processor is not None,
        'models': models_status,
        'configuration': {
            'scale_factor': processor.scale_factor if processor else None,
            'conf_threshold': processor.conf_threshold if processor else None
        }
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this endpoint'
    }), 405


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': 'The uploaded file is too large'
    }), 413


if __name__ == '__main__':
    # Solo para desarrollo/testing local
    # En producción se usa Gunicorn
    if initialize_processor():
        logger.info("Starting Flask API server in development mode...")
        logger.info("For production, use: gunicorn --config gunicorn.conf.py app:app")

        port = int(os.environ.get('PORT', 5000))
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=True
        )
    else:
        logger.error("Failed to initialize MeterProcessor. Please check your model files.")
        exit(1)

# Inicializar el processor al importar el módulo (para Gunicorn)
if not initialize_processor():
    logger.error("Failed to initialize MeterProcessor during module import")
    raise RuntimeError("Cannot initialize MeterProcessor")