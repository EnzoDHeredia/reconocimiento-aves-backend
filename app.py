from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
import os
import json
import logging
from werkzeug.utils import secure_filename

# Constantes
IMG_SIZE = (224, 224)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.65  # Umbral de confianza para la predicción
MODEL_PATH = os.environ.get('MODEL_PATH', 'model/modelo_102_B0.pth')
EBIRD_TAXONOMY_PATH = os.environ.get('EBIRD_TAXONOMY_PATH', 'ebird_taxonomy.json')
CLASSES_PATH = os.environ.get('CLASSES_PATH', 'classes.json')
NOMBRE_COMUN_PATH = os.environ.get('NOMBRE_COMUN_PATH', 'nombre_comun_a_cientifico.json')

# Configuración de logging
logging.basicConfig(level=logging.INFO)

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error cargando {path}: {e}")
        return None

# Cargar clases y taxonomía
CLASSES = load_json(CLASSES_PATH)
if CLASSES is None:
    raise RuntimeError(f"No se pudo cargar {CLASSES_PATH}")

NOMBRE_COMUN_A_CIENTIFICO = load_json(NOMBRE_COMUN_PATH)
if NOMBRE_COMUN_A_CIENTIFICO is None:
    raise RuntimeError(f"No se pudo cargar {NOMBRE_COMUN_PATH}")

# Diccionario inverso: nombre_cientifico -> nombre_comun
CIENTIFICO_A_NOMBRE_COMUN = {v: k for k, v in NOMBRE_COMUN_A_CIENTIFICO.items()}

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocesamiento
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

idx_to_class = {i: name for i, name in enumerate(CLASSES)}

# Cargar modelo EfficientNet_B0
model = efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(idx_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Cargar taxonomía de eBird local reducida (solo las especies de tu modelo)
EBIRD_TAXONOMY = load_json(EBIRD_TAXONOMY_PATH)
if EBIRD_TAXONOMY is None:
    raise RuntimeError(f"No se pudo cargar {EBIRD_TAXONOMY_PATH}")

# Crear el diccionario solo con las especies presentes en CLASSES
CLASSES_SET = set([c.lower() for c in CLASSES])
EBIRD_SCI_TO_INFO = {bird['sciName'].lower(): {
    'speciesCode': bird.get('speciesCode'),
    'comName': bird.get('comName'),
    'sciName': bird.get('sciName'),
    'category': bird.get('category'),
    'order': bird.get('order'),
    'familyComName': bird.get('familyComName'),
    'familySciName': bird.get('familySciName'),
} for bird in EBIRD_TAXONOMY if bird['sciName'].lower() in CLASSES_SET}

def get_ebird_info(class_name):
    return EBIRD_SCI_TO_INFO.get(class_name.lower(), {})

def preprocess_image(img):
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    return tensor

def predict_class(tensor):
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item()
    return predicted.item(), confidence

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception:
        return jsonify({'error': 'Invalid image file'}), 400

    tensor = preprocess_image(image)
    pred_idx, confidence = predict_class(tensor)

    if confidence < THRESHOLD:
        predicted_class = "no se identifica ave"
        pred_idx = -1
        ebird_info = None
    else:
        predicted_class = idx_to_class[pred_idx]
        ebird_info = get_ebird_info(predicted_class)

    return jsonify({
        'class': predicted_class,
        'index': pred_idx,
        'confidence': confidence,
        'ebird_info': ebird_info
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)