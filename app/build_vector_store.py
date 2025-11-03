# app/build_vector_store.py
import logging
from app.utils import load_json_file, setup_logging
from app.config import TOPIC_HIERARCHY_PATH, OPENAI_API_KEY
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import shutil 

log = setup_logging()
FAISS_INDEX_PATH = "data/faiss_index"

def build_vector_store():
    """
    topic_hierarchy.json dosyasını okur.
    "Alt Konu + Örnekler" metnini vektörize eder.
    Bu, RAG doğruluğu için en zengin bağlamı sağlar.
    """
    
    # Önceki hatalı (gürültüsüz ama aptal) index'i temizle
    if os.path.exists(FAISS_INDEX_PATH):
        log.warning(f"Mevcut hatalı index '{FAISS_INDEX_PATH}' bulundu ve siliniyor...")
        try:
            shutil.rmtree(FAISS_INDEX_PATH)
            log.info("Eski index başarıyla silindi.")
        except Exception as e:
            log.error(f"Eski index silinirken hata: {e}. Lütfen manuel olarak silin.")
            return

    log.info("Zenginleştirilmiş vektör veritabanı oluşturma işlemi başlıyor...")
    
    topic_data = load_json_file(TOPIC_HIERARCHY_PATH)
    if not topic_data:
        log.error("Konu hiyerarşisi yüklenemedi.")
        return

    docs = []
    for item in topic_data:
        ana_konu = item.get("ana_konu")
        for alt_konu_item in item.get("alt_konular", []):
            alt_konu = alt_konu_item.get("alt_konu")
            ornekler = alt_konu_item.get("ornekler")
            
            # --- GERÇEK DÜZELTME BURADA ---
            # Vektörize edilecek metin (page_content):
            # Alt konunun kendisi + Anlamsal bağlam için örnekler
            # "Ana Konu" metnini çıkararak gürültüyü azaltıyoruz.
            page_content = f"Alt Konu: {alt_konu}\nÖrnekler: {ornekler}"
            
            # Ana konuyu ve alt konuyu hala metadata'da tutuyoruz
            metadata = {
                "ana_konu": ana_konu,
                "alt_konu": alt_konu
            }
            
            docs.append(Document(page_content=page_content, metadata=metadata))

    log.info(f"Vektörize edilmek üzere {len(docs)} adet zenginleştirilmiş konu dokümanı oluşturuldu.")

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        log.info("Dokümanlar vektörize ediliyor ve FAISS index'i oluşturuluyor...")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        vector_store.save_local(FAISS_INDEX_PATH)
        log.info(f"Zenginleştirilmiş vektör veritabanı başarıyla '{FAISS_INDEX_PATH}' adresine kaydedildi.")
        
    except Exception as e:
        log.error(f"VVektör veritabanı oluşturulurken hata: {e}")

if __name__ == "__main__":
    # Bu script artık eski index'i otomatik silip yeniden kurar
    build_vector_store()