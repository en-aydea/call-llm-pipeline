# app/setup_db.py
import pandas as pd
from sqlalchemy.orm import sessionmaker
from app.models import engine, create_db_and_tables, CallInput, Base
import logging

# XLSX dosyanızın yolu
XLSX_PATH = "data/new_calls.xlsx"
# Transkriptlerin olduğu sütunun adı (Sizdeki adı buraya yazın)
TRANSCRIPT_COLUMN_NAME = "Transkript" 
# Benzersiz bir ID sütunu varsa (yoksa index'i kullanırız)
CALL_ID_COLUMN_NAME = "Çağrı ID" 

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_xlsx_to_db():
    log.info("Veritabanı ve tablolar oluşturuluyor...")
    create_db_and_tables()
    
    log.info(f"'{XLSX_PATH}' dosyasından veriler okunuyor...")
    try:
        df = pd.read_excel(XLSX_PATH)
    except FileNotFoundError:
        log.error(f"HATA: '{XLSX_PATH}' dosyası bulunamadı.")
        return
    except Exception as e:
        log.error(f"XLSX okuma hatası: {e}")
        return

    # Veritabanı oturumu başlatma
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    log.info("Transkriptler 'calls_input' tablosuna yükleniyor...")
    count = 0
    try:
        for index, row in df.iterrows():
            call_id = str(row.get(CALL_ID_COLUMN_NAME, f"call_{index}"))
            transcript = row.get(TRANSCRIPT_COLUMN_NAME)

            if not transcript or pd.isna(transcript):
                log.warning(f"Satır {index} (ID: {call_id}) atlanıyor: Transkript boş.")
                continue

            # Bu call_id daha önce eklendi mi diye kontrol et
            exists = session.query(CallInput).filter_by(call_id=call_id).first()
            if not exists:
                new_call = CallInput(
                    call_id=call_id,
                    transcript=str(transcript),
                    status="pending" # Başlangıç durumu
                )
                session.add(new_call)
                count += 1
            
        session.commit()
        log.info(f"Başarıyla {count} adet yeni çağrı transkripti veritabanına eklendi.")
    except Exception as e:
        session.rollback()
        log.error(f"Veritabanına yazma hatası: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    load_xlsx_to_db()