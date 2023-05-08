from flask import Flask, flash, request, redirect, url_for, render_template
import os
from Kick_Punch_Classifier.image_classifier import ImageClassifier
import cv2


script_dir = os.path.dirname(os.path.abspath(__file__))

train_images_Kicks = os.path.join(script_dir, '..', 'Kick_Punch_Classifier', 'Train', 'Kicks')
train_images_Punches = os.path.join(script_dir, '..', 'Kick_Punch_Classifier', 'Train', 'Punches')

train_paths_Kicks = os.listdir(train_images_Kicks)
train_paths_Punches = os.listdir(train_images_Punches)

classifier = ImageClassifier()

for f in train_paths_Kicks:
    classifier.add_img(os.path.join(train_images_Kicks, f), 'kick')

for f in train_paths_Punches:
    classifier.add_img(os.path.join(train_images_Punches, f), 'punch')

test_images = os.path.join(script_dir, '..', 'Kick_Punch_Classifier', 'Test')

test_paths = os.listdir(test_images)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg'}
app.secret_key = 'super secret key'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('app.html', images=images)


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        result, distance = classifier.predict(cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
        flash(f'The image is classified as: {result}')
        flash(f'Distance: {distance}')
        return render_template('app.html', result=result)
    else:
        flash('Allowed image types are - jpg')
        return redirect(request.url)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
