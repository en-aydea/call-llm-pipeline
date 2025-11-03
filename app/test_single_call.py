# app/test_single_call.py
import json
import time
import asyncio # Asenkron RAG görevleri için eklendi
from app.utils import setup_logging, load_text_file, get_call_start
from app.config import PRODUCT_LIST_PATH
from app.llm_chain import create_extraction_chain
from app.models import SessionLocal, CallInput
from app.main import load_retriever 

TEST_CALL_ID = 21
log = setup_logging()

async def run_single_test_async(): # Fonksiyonu asenkron hale getiriyoruz
    """
    Tekil çağrıyı 16 kriterli "İkili-Arama RAG" akışı ile test eder.
    (HATA DÜZELTMESİ: Her 'sub_topic_free' için AYRI RAG araması yapar)
    """
    log.info(f"Tekil Çağrı Testi Başlatılıyor (Çağrı ID: {TEST_CALL_ID})...")
    
    db_session = SessionLocal()
    
    try:
        test_call = db_session.query(CallInput).filter(CallInput.id == TEST_CALL_ID).first()
        if not test_call:
            log.error(f"HATA: ID'si {TEST_CALL_ID} olan çağrı veritabanında bulunamadı.")
            return
        log.info("Test transkripti veritabanından başarıyla alındı.")
        full_transcript = test_call.transcript
    except Exception as e:
        log.error(f"Veritabanından çağrı okunurken hata: {e}")
        return
    finally:
        db_session.close()

    vector_store = load_retriever()
    if not vector_store:
        log.error("Test durduruldu: Retriever yüklenemedi.")
        return

    product_list_str = load_text_file(PRODUCT_LIST_PATH)
    extraction_chain = create_extraction_chain(product_list_str)
    
    log.info("Çağrı başı ve tüm transkript hazırlanıyor...")
    transcript_start = get_call_start(full_transcript)
    
    log.info("RAG zinciri hazır. Adım 1: OpenAI'ye Çıkarım İsteği gönderiliyor...")
    start_time = time.time()

    try:
        # --- ADIM 1: LLM ÇIKARIM ---
        input_data = {
            "transcript_start": transcript_start,
            "full_transcript": full_transcript
        }
        # invoke() yerine async olan ainvoke() kullanıyoruz
        partial_result = await extraction_chain.ainvoke(input_data)
        
        log.info("Adım 1 tamamlandı. Serbest konular alınıyor...")
        print("\n--- ZİNCİR 1 ÇIKTISI ---")
        print(f"INTENT (Arama Niyeti): {partial_result.intent}")
        print(f"ANA KONU (Genel): {partial_result.main_topic_free}")
        print(f"ALT KONULAR (Genel): {partial_result.sub_topics_free}")
        print("--------------------------\n")

        # --- ADIM 2: İKİLİ RAG ARAMA (DÜZELTİLMİŞ MANTIK) ---
        final_result = partial_result 
        
        # --- ARAMA 1: ALT KONULAR (HER BİRİ İÇİN AYRI AYRI) ---
        if partial_result.sub_topics_free:
            log.info(f"RAG Araması (Alt Konular) {len(partial_result.sub_topics_free)} adet konu için ayrı ayrı yapılıyor...")
            
            sub_topic_tasks = []
            for sub_query in partial_result.sub_topics_free:
                # Her bir alt konu için k=1 (en iyi) araması
                sub_topic_tasks.append(vector_store.asimilarity_search_with_score(sub_query, k=1))
            
            # Tüm alt konu RAG aramalarını paralel çalıştır
            sub_topic_results_list = await asyncio.gather(*sub_topic_tasks)
            
            print("--- RAG ARAMA SONUÇLARI (ALT KONULAR) ---")
            found_alt_konular = []
            for i, result_list in enumerate(sub_topic_results_list):
                if result_list:
                    best_doc, score = result_list[0]
                    found_alt_konu = best_doc.metadata['alt_konu']
                    found_alt_konular.append(found_alt_konu)
                    print(f"  SORGULANAN: '{partial_result.sub_topics_free[i]}'")
                    print(f"  BULUNAN (SKOR: {score:.4f}): '{found_alt_konu}'")
            print("-------------------------------------------\n")
            
            final_result.sub_topics_guided = list(set(found_alt_konular))
        else:
            log.info("RAG Araması (Alt Konular) atlandı: Serbest alt konu bulunamadı.")

        # --- ARAMA 2: ANA KONU İÇİN (k=1) ---
        if partial_result.main_topic_free:
            main_topic_query = partial_result.main_topic_free
            log.info(f"RAG Araması (Ana Konu) yapılıyor...")
            
            # Ana konu için asenkron arama
            main_topic_results = await vector_store.asimilarity_search_with_score(main_topic_query, k=1)
            
            if main_topic_results:
                best_match_doc, best_score = main_topic_results[0]
                found_main_topic = best_match_doc.metadata.get('ana_konu')
                
                print("--- RAG ARAMA SONUCU (ANA KONU) ---")
                print(f"SKOR: {best_score:.4f} | Eşleşen Ana Konu: {found_main_topic}")
                print("-------------------------------------------\n")
                
                final_result.main_topic_guided = found_main_topic
            else:
                log.info("RAG Araması (Ana Konu) sonuç bulamadı.")
        else:
            log.info("RAG Araması (Ana Konu) atlandı: Serbest ana konu bulunamadı.")

        # --- ADIM 3: BİRLEŞTİRİLMİŞ SONUÇ ---
        end_time = time.time()
        log.info(f"Çağrı (LLM+RAG) {end_time - start_time:.2f} saniyede başarıyla işlendi.")
        
        log.info("---------- BİRLEŞTİRİLMİŞ TEST SONUCU (16 KRİTER) ----------")
        result_json = final_result.model_dump_json(indent=2, ensure_ascii=False)
        print(result_json)
        log.info("----------------------------------------------------------")

    except Exception as e:
        log.error(f"Zincir çalıştırılırken bir hata oluştu: {e}")

if __name__ == "__main__":
    # Asenkron fonksiyonu çalıştırmak için asyncio.run() kullanılır
    asyncio.run(run_single_test_async())