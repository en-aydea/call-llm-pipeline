# app/utils.py
import logging
import json
from typing import Optional

def setup_logging():
    """Temel loglama ayarlarını yapılandırır."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_json_file(file_path: str):
    """JSON formatındaki konu hiyerarşisini yükler."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"HATA: {file_path} dosyası okunamadı: {e}")
        return None

def load_text_file(file_path: str):
    """Metin formatındaki ürün listesini yükler."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"HATA: {file_path} dosyası okunamadı: {e}")
        return ""

def get_call_start(transcript: str, num_words: int = 40) -> str:
    """
    Transkriptin ilk 'num_words' kelimesini alır.
    'intent' tespiti için kullanılır.
    """
    words = transcript.split()
    start = " ".join(words[:num_words])
    return start