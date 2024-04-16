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

target_size = 800

# Define the transformation pipeline for face images
transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Resize the image to fit the model input
    transforms.ToTensor()
])

def process_in_batches(detected_boxes, image_resized):
    # Adjust the batch size based on your GPU capacity and model requirements
    batch_size = 32
    for i in range(0, len(detected_boxes), batch_size):
        batch_boxes = detected_boxes[i:i + batch_size]
        faces_tensors = [transform(image_resized.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))) for box in batch_boxes]
        faces_tensor_batch = torch.stack(faces_tensors).to(device)
        yield faces_tensor_batch

def optimized_face_processing(image_resized, detected_boxes):
    face_encodings = []
    for faces_tensor_batch in process_in_batches(detected_boxes, image_resized):
        # Perform batch face encoding
        batch_encodings = resnet(faces_tensor_batch).detach().cpu().numpy()
        face_encodings.extend(batch_encodings.tolist())
    return face_encodings

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
        aspect_ratio = image.width / image.height
        image_resized = image.resize((target_size, int(target_size / aspect_ratio)))
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
