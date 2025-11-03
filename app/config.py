# app/config.py
from dotenv import load_dotenv
import os

# .env dosyasındaki değişkenleri yükle
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY ortam değişkeni bulunamadı. .env dosyasını kontrol edin.")

# Veritabanı URL'si (models.py'dan alıyoruz)
from app.models import DATABASE_URL
DB_URL = DATABASE_URL

# İşlem ayarları
BATCH_SIZE = 5          # Her döngüde kaç çağrı işlenecek
MAX_RETRIES = 3          # Hata durumunda kaç kez denenecek
LLM_MODEL = "gpt-5-nano"     # Önerilen model (veya gpt-4-turbo)

# Dosya yolları
TOPIC_HIERARCHY_PATH = "data/topic_hierarchy.json"
PRODUCT_LIST_PATH = "data/product_list.txt"