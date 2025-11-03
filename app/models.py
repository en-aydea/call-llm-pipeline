# app/models.py
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from sqlalchemy.orm import sessionmaker

# Banka simülasyonu için lokal SQLite veritabanı yolu
DATABASE_URL = "sqlite:///./bank_calls.db"

Base = declarative_base()
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class CallInput(Base):
    """Gelen çağrı transkriptlerinin tutulduğu kaynak tablo."""
    __tablename__ = "calls_input"
    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(String, unique=True, index=True) # Çağrıya ait benzersiz bir ID (örn: dosya adı)
    transcript = Column(Text, nullable=False)
    status = Column(String, default="pending") # (pending, processed, failed)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CallOutput(Base):
    """LLM tarafından işlenen ve 16 kriterin yazıldığı hedef tablo."""
    __tablename__ = "calls_output"
    id = Column(Integer, primary_key=True, index=True)
    input_call_id = Column(Integer, index=True) # calls_input tablosundaki id'ye referans

    # --- YENİ ALAN (16. KRİTER) ---
    intent = Column(String, nullable=True) # Müşterinin ilk arama niyeti
    
    # Diğer 15 Kriter
    summary = Column(Text)
    main_topic_free = Column(String)
    main_topic_guided = Column(String)
    sub_topics_free = Column(String) 
    sub_topics_guided = Column(String) 
    sentiment = Column(String) 
    is_complaint = Column(Boolean)
    complaint_reason = Column(String, nullable=True)
    is_product_offer = Column(Boolean)
    is_escalation = Column(Boolean)
    is_regulatory_mention = Column(Boolean)
    is_other_bank_mention = Column(Boolean)
    nps_score = Column(Integer, nullable=True)
    nps_rationale = Column(Text, nullable=True)
    top_keywords = Column(String) 
    
    processed_at = Column(DateTime(timezone=True), server_default=func.now())

def create_db_and_tables():
    """Veritabanı ve tabloları oluşturur."""
    Base.metadata.create_all(bind=engine)


SentimentEnum = Literal["POZITIF", "NEGATIF", "NOTR"]

# --- GÜNCELLENMİŞ PYDANTIC MODELİ (16 KRİTER) ---
class CallAnalysisOutput(BaseModel):
    """
    LLM'den dönmesini beklediğimiz 16 kriterli yapı.
    """
    
    # --- YENİ ALAN (16. KRİTER) ---
    intent: str = Field(
        description="Müşterinin aramayı BAŞLATMA NİYETİ. Sadece çağrının ilk cümlelerine bakarak 1-2 kelime ile doldurulur."
    )
    
    summary: str = Field(description="Çağrının TAMAMININ 2-3 cümlelik kısa özeti.")
    
    main_topic_free: str = Field(
        description="Çağrının TAMAMINA bakarak belirlenen, konuşmanın genel ana konusu."
    )
    
    main_topic_guided: Optional[str] = Field(
        default=None,
        description="RAG tarafından doldurulacak güdümlü ana konu."
    )
    
    sub_topics_free: Optional[List[str]] = Field(
    description="Çağrının TAMAMINA bakarak müşterinin dile getirdiği TÜM FARKLI konu başlıkları (max 3). "
                "Bu liste, çağrının ana konusunu VE buna ek olarak sorulan tüm ikincil/farklı talepleri "
                "(örn: 'limit sorgulama', 'fatura ödeme', 'adres güncelleme' vb.) içermelidir."
)
    
    sub_topics_guided: Optional[List[str]] = Field(
        default=None,
        description="RAG tarafından doldurulacak güdümlü alt konular."
    )
    
    sentiment: SentimentEnum = Field(description="Müşterinin genel duygu durumu.")
    
    is_complaint: bool = Field(description="Müşteri net bir şikayet ('şikayetçiyim', 'rezalet', 'çok mağdur oldum') belirtiyorsa `true`, yoksa `false`.")
    
    complaint_reason: Optional[str] = Field(description="Şikayet varsa, kısaca nedeni.")
    
    is_product_offer: bool = Field(description="Temsilci ürün teklifinde bulundu mu?")
    
    is_escalation: bool = Field(description="Çağrı aktarıldı mı?")
    
    is_regulatory_mention: bool = Field(description="BDDK, CİMER gibi yasal veya  regülatif kurumlara atıf var mı?")
    
    is_other_bank_mention: bool = Field(description="Rakip bankalardan (Akbank dışında) birinin adı geçiyor mu? Örneğin Garanti, İş Bankası vb.")
    
    nps_score: Optional[int] = Field(description="Müşteri ifadelerinden bir memnuniyet skoru tahmin edebilirsen 0 ile 10 arasında not ver edemezsen `null` bırak.")
    
    nps_rationale: Optional[str] = Field(description="NPS skorunu çıkarsadığın ifadelerin aslına sadık kısa özeti.")
    
    top_keywords: List[str] = Field(description="MÜŞTERİ'nin (temsilcinin değil) kullandığı, çağrı içinde önem derecesi en yüksek ve en iyi anlatan 4 kelime veya kısa kelime öbeği seç.", max_items=4)