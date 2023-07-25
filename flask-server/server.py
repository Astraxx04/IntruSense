import cv2
import csv
from flask import Flask, render_template, Response, request, redirect, url_for
from recognize_video import generate_frames,cleanup,video_feed_active
from checkdnn import dnn
from extract_embeddings import extactemb
from train_model import train

app = Flask(__name__,template_folder='templates', static_folder='static')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # Stream the video frames

@app.route('/process', methods=['POST'])
def process():
    name = request.form['name']
    # Run your script using the name here
    # Replace the print statement with your script logic
    print("Running script for name: "+name)
    dnn(name)
    extactemb()
    train()

    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    with open('identified_persons.csv', 'r') as file:
        reader = csv.reader(file)
        table_data = list(reader)
    return render_template('dashboard.html', table_data=table_data)
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['pass']
        
        if username == 'admin@go.com' and password == 'Password@123':
            return redirect(url_for('dashboard'))
        else:
            return redirect(url_for('login'))


    return render_template('index.html')

if __name__ == '__main__':
    app.run()
    


