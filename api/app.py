from flask import Flask,request,jsonify,send_file
import torch
import os
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.route("/lipsync",methods=['POST'])
def sync_lip():
    video_file=request.files['video']
    audio_file=request.files['audio']
    video_path = os.path.join('uploads',secure_filename(video_file.filename))
    audio_path = os.path.join('uploads',secure_filename(audio_file.filename))
    video_file.save(video_path)
    audio_file.save(audio_path)
    output_path = os.path.join("/Users/useradmin/Documents/fevealAI/api/Wav2Lip/results",'result_voice.mp4')
    command =f"cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face {os.path.abspath(video_path)} --audio {os.path.abspath(audio_path)}"
    subprocess.run(command,shell=True)
    return send_file(output_path,as_attachment=True)
