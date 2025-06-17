# 用 FastAPI 封装 similarity_by_segments 

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from wespeaker.cli.speaker import load_model

import oss2
import os
import uuid
import json
from dotenv import load_dotenv

app = FastAPI(title="WeSpeaker API")

load_dotenv()

BUCKET_NAME = os.getenv("OSS_BUCKET_NAME")
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT")

_auth   = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
bucket = oss2.Bucket(_auth, OSS_ENDPOINT, BUCKET_NAME)

VOICE_FILE = "./data/voiceprint.json"

model = load_model("chinese")

class Segment(BaseModel):
    start: float
    end: float

class Segments(BaseModel):
    segments: List[Segment]


def load_user_voices(filepath: str) -> dict[str, List[float]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        user_voices = json.load(f)
    # 文件是 []  或者空
    print(user_voices)
    for user, vals in user_voices:
        user_voices[user] = [float(v) for v in vals]
    return user_voices

def persist_user_voices(filepath: str, user_voices: dict):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(user_voices, f, ensure_ascii=False, indent=2)

user_voice_dict = load_user_voices(VOICE_FILE)

@app.post("/calculate_similarities")
def calculate_similarities(
    audio: UploadFile = File(...),
    segments: Segments = ...,
    user_id: str = ...
):
    if not audio or not segments or not user_id:
        return JSONResponse({"error": "input error"}, status_code=400)
    
    ref_embedding = []
    if user_id not in user_voice_dict:
        return JSONResponse({"error": "voiceprint not registered"}, status_code=400)
    else:
        try:
            tmp_filename = tmp_wav_filepath()
            audio_data = audio.file.read()
            with open(tmp_filename, "wb") as f:
                f.write(audio_data)
            ref_embedding = user_voice_dict[user_id]
            segments_list = [(seg.start, seg.end) for seg in segments.segments]
            similarities = model.compute_similarity_by_segments(tmp_filename, segments_list, ref_embedding)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        finally:
            os.remove(tmp_filename)
    return {"similarities": similarities}

@app.post("/register_voiceprint")
def register_voiceprint(
    user_id: str = ...,
    audio_osskey: str = ...
):
    if not audio_osskey or not user_id:
        return JSONResponse({"success": False, "message": "input error"}, status_code=400)
    try:
        response = bucket.get_object(audio_osskey)
        audio_data = response.read()
        filename = tmp_wav_filepath()
        with open(filename, "wb") as f:
            f.write(audio_data)
        print(f"extracting embedding from file: {filename}, size: {len(audio_data)} bytes")
        print(f"os.getcwd(): {os.getcwd()}")
        embedding = model.extract_embedding(filename)
        print(f"embedding: {embedding}")
    except Exception as e:
        print(f"error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)
    #finally:
        #os.remove(filename)
    user_voice_dict[user_id] = embedding
    persist_user_voices(VOICE_FILE, user_voice_dict)
    return {"success": True}

def tmp_wav_filepath():
    return f"./tmp/tmp_{uuid.uuid4()}.wav"

if __name__ == "__main__":
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=13700) 
