from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)


pipe = pipeline("automatic-speech-recognition", model="ginpham614/wav2vec2-large-xlsr-53-demo-colab")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:

        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        temp_audio_path = "temp_audio.wav"
        audio_file.save(temp_audio_path)
        result = pipe(temp_audio_path)
        import os
        os.remove(temp_audio_path)
        return jsonify({"transcription": result["text"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
