from fastapi import FastAPI, File, UploadFile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import soundfile as sf
import tempfile
import shutil
from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffmpeg = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

app = FastAPI()

# Load model từ Hugging Face
MODEL_NAME = "ginpham614/wav2vec2-large-xlsr-53-demo-colab"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    # if file.filename.endswith(".mp3"):
    #     mp3_audio = AudioSegment.from_file(file.file, format="mp3")
    #     mp3_audio = mp3_audio.set_frame_rate(16000).set_channels(1)  
    #     mp3_audio.export(temp_path, format="wav")
    # else:
    #     shutil.copyfileobj(file.file, temp)
    
    # Đọc file âm thanh
    speech, sr = librosa.load(temp_path, sr=16000)
    input_values = processor(speech, return_tensors="pt",
                             sampling_rate=16000).input_values

    # Dự đoán
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return {"text": transcription}
