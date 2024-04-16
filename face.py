from flask import Flask, request, jsonify
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torchvision.transforms as transforms
import time

app = Flask(__name__)

# Initialize the MTCNN and InceptionResnetV1 models on the appropriate device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print(f"Using device: {device}")

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Resize the image to fit the model input
    transforms.ToTensor()
])

@app.route('/detect', methods=['POST'])
def detect_faces():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided.'}), 400

    # Start the timer
    start_time = time.time()

    results = []
    # Process each file
    for file in request.files.getlist('images'):
        image = Image.open(file.stream).convert('RGB')
        # Resize and detect faces
        image_resized = image.resize((160, 160))  # Resize image to a standard size
        boxes, _ = mtcnn.detect(image_resized)

        if boxes is not None:
            # Extract and encode faces
            face_encodings = optimized_face_processing(image_resized, boxes)
            results.append({'file': file.filename, 'faces': face_encodings})
        else:
            results.append({'file': file.filename, 'message': 'No faces detected'})

    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time

    return jsonify({
        'results': results,
        'processing_time_seconds': processing_time
    })

def optimized_face_processing(image_resized, detected_boxes):
    # Crop detected faces and prepare tensors
    detected_faces_tensors = [transform(image_resized.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))) for box in detected_boxes]

    # Stack tensors to create a batch
    faces_tensor_batch = torch.stack(detected_faces_tensors).to(device)

    # Perform batch face encoding
    face_encodings_batch = resnet(faces_tensor_batch).detach().cpu().numpy()

    # Convert batch encodings to list for storage or further processing
    return face_encodings_batch.tolist()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
