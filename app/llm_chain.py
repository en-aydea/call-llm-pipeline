# app/llm_chain.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.models import CallAnalysisOutput
from app.config import OPENAI_API_KEY, LLM_MODEL, MAX_RETRIES

def create_extraction_chain(product_list_str: str):
    """
    ZİNCİR 1: Zenginleştirilmiş İkili-Bağlamlı Çıkarım Zinciri.
    'intent' alanını çağrı başından, diğerlerini tamamından çıkarır.
    """
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    ).with_retry(stop_after_attempt=MAX_RETRIES)

    parser = PydanticOutputParser(pydantic_object=CallAnalysisOutput)

 
    prompt_template = """
    Sen bir banka çağrı merkezinde çalışan, deneyimli bir kalite ve analiz uzmanısın.
    Görevin, sana verilen müşteri hizmetleri çağrı transkriptini analiz ederek farklı kritere göre yapılandırılmış bir JSON formatında raporlamaktır.
    iki farklı metin parçası verilecek: Çağrının başı ve çağrının tamamı.
    Ayrıca, bankaya ait ürünleri tanıyabilmene yardımcı olması için ürün listesi ve tanımları iletilecek.
    
    ### Görev Talimatları:
    1. `intent` (Müşteri Niyeti) alanını doldurmak için SADECE 'ÇAĞRI BAŞLANGICI' metnini kullan. Bu, müşterinin asıl arama nedenidir.
    2. `main_topic_free` (Serbest Ana Konu) alanını doldurmak için 'TÜM TRANSKRİPT' metnine bakarak çağrının en baskın (dominant) ve en çok zaman alan konusunu belirle.
    3. `summary`, `sub_topics_free`, `sentiment` vb. DİĞER TÜM alanları doldurmak için 'TÜM TRANSKRİPT' metnini kullan.
    4   Ürün teklifini belirlerken, 'is_product_offer' kriteri için ürün listesine başvur.
    5.  `sub_topics_free` (Serbest Alt Konular) alanını doldururken, 'TÜM TRANSKRİPT' metnini dikkatle incele. Müşterinin dile getirdiği tüm farklı konuları, talepleri veya soruları listele.
        **ÖNEMLİ:** Müşteri, ana konudan (`main_topic_free`) tamamen bağımsız, "bir de şunu sorayım", "bu arada...", "aklıma gelmişken..." gibi ifadelerle ikincil bir talepte bulunursa (örneğin, önce EFT sorununu konuşup sonra "kart limitim ne kadardı?" diye sorarsa), bu ikincil talebi MUTLAKA ayrı bir alt konu olarak eklemelisin.
    6. **ÖNEMLİ**: `main_topic_guided` ve `sub_topics_guided` alanlarını DOLDURMA. Bu alanları 'null' veya boş bırak.

    ### Referans Bilgiler (Bağlam):

    Aşağıdaki ürün listesi, 'is_product_offer' kriterini değerlendirmenize yardımcı olacaktır (Tüm transkripte bakarak):
    ---[ÜRÜN LİSTESİ]----
    {product_list}
    -----------------------

    ### Analiz Edilecek Metinler:

    ---[ÇAĞRI BAŞLANGICI (Sadece 'intent' tespiti için)]----
    {transcript_start}
    -------------------------------------------------------

    ---[TÜM TRANSKRİPT ('main_topic_free' ve diğer tüm alanlar için)]----
    {full_transcript}
    ------------------------------------------------

    ### Çıktı Formatı:
    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "product_list": product_list_str
        }
    )

    chain = prompt | llm | parser
    
    return chain