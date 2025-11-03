# app/main.py
import asyncio
import time
from sqlalchemy.orm import sessionmaker
from app.models import engine, CallInput, CallOutput, SessionLocal
from app.config import BATCH_SIZE, OPENAI_API_KEY
from app.utils import setup_logging, load_text_file, get_call_start
from app.config import PRODUCT_LIST_PATH
from app.llm_chain import create_extraction_chain
from app.models import CallAnalysisOutput

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

log = setup_logging()
FAISS_INDEX_PATH = "data/faiss_index"

def load_retriever():
    """Lokal FAISS veritabanını yükler ve bir vector_store nesnesi döner."""
    try:
        log.info(f"Vektör veritabanı '{FAISS_INDEX_PATH}' yükleniyor...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True) 
        return vector_store
    except Exception as e:
        log.error(f"HATA: FAISS index'i yüklenemedi: {e}")
        log.error(f"Lütfen önce 'python app/build_vector_store.py' komutunu çalıştırdığınızdan emin olun.")
        return None

async def process_batch(extraction_chain, vector_store, db_session, call_batch):
    """
    "İkili-Arama RAG" akışı (HATA DÜZELTMESİ: Her alt konu için ayrı RAG)
    """
    log.info(f"{len(call_batch)} adet çağrı işleniyor (Adım 1: LLM Çıkarım)...")
    
    inputs_for_chain = []
    for call in call_batch:
        full_transcript = call.transcript
        transcript_start = get_call_start(full_transcript)
        inputs_for_chain.append({
            "transcript_start": transcript_start,
            "full_transcript": full_transcript
        })
    
    start_time = time.time()
    
    try:
        # --- ADIM 1: LLM ÇIKARIM (BATCH) ---
        partial_results = await extraction_chain.abatch(inputs_for_chain)
        
        # --- ADIM 2: İKİLİ RAG ARAMA (DÜZELTİLMİŞ MANTIK) ---
        
        main_topic_tasks = []
        sub_topic_group_tasks = [] # Her çağrı için bir "grup" görevi
        empty_async_list = asyncio.Future()
        empty_async_list.set_result([])

        for result in partial_results:
            # Görev 1: Ana Konu (k=1)
            if result.main_topic_free:
                query = result.main_topic_free
                main_topic_tasks.append(vector_store.asimilarity_search_with_score(query, k=1))
            else:
                main_topic_tasks.append(empty_async_list)
                
            # Görev 2: Alt Konular (k=1, her biri için)
            if result.sub_topics_free:
                # Her bir alt konu için ayrı RAG görevleri oluştur
                tasks_for_this_call = [
                    vector_store.asimilarity_search_with_score(sub_query, k=1) 
                    for sub_query in result.sub_topics_free
                ]
                # Bu görev grubunu toplayan bir 'gather' görevi oluştur
                sub_topic_group_tasks.append(asyncio.gather(*tasks_for_this_call))
            else:
                sub_topic_group_tasks.append(empty_async_list)

        log.info(f"{len(call_batch)} çağrı için İkili-RAG araması yapılıyor...")
        
        # Tüm RAG aramalarını (hem ana hem de alt konu grupları) paralel çalıştır
        all_main_topic_results = await asyncio.gather(*main_topic_tasks)
        all_sub_topic_group_results = await asyncio.gather(*sub_topic_group_tasks)
        
        end_time = time.time()
        log.info(f"{len(call_batch)} çağrı {end_time - start_time:.2f} saniyede (LLM+RAG) işlendi.")
        
        # --- ADIM 3: BİRLEŞTİRME VE VERİTABANINA YAZMA ---
        for i in range(len(call_batch)):
            call_input = call_batch[i]
            final_output = partial_results[i] 
            
            # Alt konu sonuçlarını işle
            # all_sub_topic_group_results[i] -> [[(doc, score)], [(doc, score)], ...]
            sub_topic_results_list = all_sub_topic_group_results[i]
            found_alt_konular = []
            if sub_topic_results_list:
                for result_list in sub_topic_results_list:
                    if result_list: # Arama sonucu boş değilse
                        best_doc, score = result_list[0] # k=1
                        found_alt_konular.append(best_doc.metadata['alt_konu'])
            final_output.sub_topics_guided = list(set(found_alt_konular))
            
            # Ana konu sonuçlarını işle
            main_topic_results_with_scores = all_main_topic_results[i]
            if main_topic_results_with_scores:
                best_match_doc, best_score = main_topic_results_with_scores[0]
                final_output.main_topic_guided = best_match_doc.metadata.get('ana_konu')
            
            # Veritabanına yaz
            try:
                new_output = CallOutput(
                    input_call_id=call_input.id,
                    intent=final_output.intent,
                    summary=final_output.summary,
                    main_topic_free=final_output.main_topic_free,
                    main_topic_guided=final_output.main_topic_guided,
                    sub_topics_free=", ".join(final_output.sub_topics_free or []),
                    sub_topics_guided=", ".join(final_output.sub_topics_guided or []),
                    sentiment=final_output.sentiment.value, 
                    is_complaint=final_output.is_complaint,
                    complaint_reason=final_output.complaint_reason,
                    is_product_offer=final_output.is_product_offer,
                    is_escalation=final_output.is_escalation,
                    is_regulatory_mention=final_output.is_regulatory_mention,
                    is_other_bank_mention=final_output.is_other_bank_mention,
                    nps_score=final_output.nps_score,
                    nps_rationale=final_output.nps_rationale,
                    top_keywords=", ".join(final_output.top_keywords or [])
                )
                db_session.add(new_output)
                call_input.status = "processed"
            except Exception as e:
                log.error(f"Çağrı ID {call_input.id} veritabanına yazılırken hata: {e}")
                call_input.status = "failed"
                
    except Exception as e:
        log.error(f"Batch işleme hatası (LLM veya RAG): {e}")
        for call in call_batch:
            call.status = "failed"
            
    finally:
        db_session.commit()

def run_pipeline():
    """Ana pipeline fonksiyonu (İkili-Arama RAG / 16 Kriter)."""
    log.info("Çağrı Merkezi 16-Kriter Analiz Pipeline'ı Başlatılıyor...")
    
    vector_store = load_retriever()
    if not vector_store:
        return

    log.info("Ürün listesi yükleniyor...")
    product_list_str = load_text_file(PRODUCT_LIST_PATH)
    
    log.info("LangChain Çıkarım Zinciri (Zincir 1) oluşturuluyor...")
    extraction_chain = create_extraction_chain(product_list_str)

    db_session = SessionLocal()
    try:
        while True:
            log.info(f"'pending' statüsündeki {BATCH_SIZE} adet çağrı aranıyor...")
            
            call_batch = db_session.query(CallInput).filter(
                CallInput.status == "pending"
            ).limit(BATCH_SIZE).all()

            if not call_batch:
                log.info("İşlenecek yeni çağrı bulunamadı. Pipeline tamamlandı.")
                break
            
            asyncio.run(process_batch(extraction_chain, vector_store, db_session, call_batch))

    except Exception as e:
        log.error(f"Pipeline'da kritik hata: {e}")
        db_session.rollback()
    finally:
        db_session.close()
        log.info("Veritabanı bağlantısı kapatıldı.")

if __name__ == "__main__":
    run_pipeline()