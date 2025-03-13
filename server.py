from flask import Flask, request, jsonify
from transformers import pipeline, AutoProcessor, AutoModelForPreTraining
import os
import ffmpeg

app = Flask(__name__)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


pipe = pipeline("automatic-speech-recognition", model="ginpham614/wav2vec2-large-xlsr-53-vietnamese-finetuned-demo-v2.0-colab")
pipe_en = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def convert_to_wav(input_path, output_path="processed_audio.wav"):
    try:
        ffmpeg.input(input_path).output(
            output_path,
            format="wav",
            acodec="pcm_s16le",
            ar=16000,
            ac=1,
            ).run( overwrite_output=True)
        return output_path
    except Exception as e:
        print(f"Error converting file {input_path}: {e}")
        return None


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:

        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        temp_audio_path = "temp_audio.wav"
        audio_file.save(temp_audio_path)
        
        processed_audio_path = convert_to_wav(temp_audio_path)
        if processed_audio_path is None:
            return jsonify({"error": "Error processing audio file"}), 500
        
        
        
        result = pipe(processed_audio_path)
        os.remove(temp_audio_path)
        os.remove(processed_audio_path)
        return jsonify({"transcription": result["text"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/transcribe_en', methods=['POST'])
def transcribe_en():
    try:

        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        temp_audio_path = "temp_audio.wav"
        audio_file.save(temp_audio_path)
        
        processed_audio_path = convert_to_wav(temp_audio_path)
        if processed_audio_path is None:
            return jsonify({"error": "Error processing audio file"}), 500
        
        with open(processed_audio_path, "rb") as f:
            audio_bytes = f.read()
        
        result = pipe_en(audio_bytes)
        os.remove(temp_audio_path)
        os.remove(processed_audio_path)
        return jsonify({"transcription": result["text"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
