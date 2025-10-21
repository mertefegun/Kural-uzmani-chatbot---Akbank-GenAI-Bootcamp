# Gerekli kütüphaneleri içe aktar
import os                           # İşletim sistemiyle ilgili işlemler için (örn: dosya yolları, ortam değişkenleri)
import sys                          # Sistemle ilgili parametreler ve fonksiyonlar için (örn: hata mesajları, programdan çıkış)
import uuid                         # Benzersiz kimlikler (UUID) oluşturmak için (session ID'leri için)
from datetime import datetime       # Tarih ve zaman işlemleri için (sohbet zaman damgaları)
import traceback                    # Hata ayıklama sırasında detaylı hata izi yazdırmak için

import google.generativeai as genai # Google Generative AI (Gemini) kütüphanesi
import markdown                     # Metni Markdown formatından HTML'e çevirmek için
from dotenv import load_dotenv      # .env dosyasındaki ortam değişkenlerini yüklemek için
from flask import Flask, jsonify, redirect, render_template, request, session, url_for # Web framework'ü Flask ve ilgili modüller
from langchain_google_genai import GoogleGenerativeAIEmbeddings # LangChain ile Google embedding modeli entegrasyonu
from langchain_community.vectorstores import Chroma             # LangChain ile Chroma veritabanı entegrasyonu
import shutil                       # Dosya ve klasör işlemleri için (örn: chroma_db silme)

# ----- Yapılandırma ve Kurulum -----

# Proje ana dizinindeki .env dosyasını bul ve içindeki değişkenleri yükle
load_dotenv()
# Yüklenen ortam değişkenlerinden GOOGLE_API_KEY değerini al
api_key = os.getenv("GOOGLE_API_KEY")

# API anahtarı yüklenememişse hata ver ve programı durdur
if not api_key:
    print("Hata: GOOGLE_API_KEY ortam değişkeni bulunamadı.", file=sys.stderr)
    print("Lütfen proje ana dizininde '.env' dosyasını oluşturup içine 'GOOGLE_API_KEY=...' satırını eklediğinizden emin olun.", file=sys.stderr)
    sys.exit(1)

# Google Generative AI kütüphanesini alınan API anahtarıyla yapılandır
try:
    genai.configure(api_key=api_key)
    print("Google API Anahtarı başarıyla yüklendi ve yapılandırıldı.")
except Exception as e:
    # Yapılandırma başarısız olursa hata ver ve programı durdur
    print(f"Hata: Google API yapılandırılamadı: {e}", file=sys.stderr)
    sys.exit(1)

# Kullanılacak Gemini modelini (LLM) yükle ve yapılandır
try:
    # Model adı: 'gemini-1.5-flash' gibi daha yeni modeller de denenebilir.
    # generation_config: Modelin cevap üretme davranışını ayarlar.
    # temperature: Cevapların ne kadar rastgele/yaratıcı olacağını belirler (0=deterministik, 1=yaratıcı). Kural açıklamaları için düşük tutulur.
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config=genai.types.GenerationConfig(temperature=0.4) # Monopoly için 0.4 ayarlandı
    )
    print("Google Gemini modeli ('gemini-2.0-flash', temperature=0.4) başarıyla yüklendi.")
except Exception as e:
    # Model yüklenemezse hata ver ve programı durdur
    print(f"Hata: Google Gemini modeli yüklenemedi: {e}", file=sys.stderr)
    sys.exit(1)

# ----- Veritabanı Fonksiyonları -----

def load_database():
    """
    Önceden 'create_database.py' ile oluşturulmuş olan Chroma vektör veritabanını yükler.
    Veritabanı bulunamazsa veya yüklenirken bir hata oluşursa programı durdurur.
    Returns:
        Chroma: Yüklenmiş LangChain Chroma vektör veritabanı nesnesi.
    """
    print("Monopoly veritabanı yükleniyor...")
    # Veritabanını yüklemek için de embedding fonksiyonuna ihtiyaç var
    try:
        # Google'ın text-embedding-004 modelini LangChain entegrasyonu ile kullan
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=api_key
        )
        print("Google Embedding modeli başarıyla ayarlandı.")
    except Exception as e:
        # Embedding modeli ayarlanamazsa hata ver ve durdur
        print(f"Hata: Google Embedding modeli ayarlanamadı: {e}", file=sys.stderr)
        sys.exit(1)

    # Veritabanının kaydedildiği klasör yolu
    db_path = "./chroma_db"
    # Veritabanı içindeki koleksiyon adı (create_database.py ile aynı olmalı!)
    collection_name = "gaih_monopoly_comprehensive"

    # Veritabanı klasörünün ve içindeki önemli bir dosyanın (örn: sqlite dosyası) varlığını kontrol et
    db_file_path = os.path.join(db_path, "chroma.sqlite3")
    if not os.path.exists(db_path) or not os.path.exists(db_file_path):
        # Veritabanı bulunamazsa kullanıcıyı bilgilendir ve programı durdur
        print(f"Hata: Veritabanı '{db_path}' klasöründe bulunamadı veya geçersiz.", file=sys.stderr)
        print("Lütfen önce 'python create_database.py' komutunu çalıştırarak veritabanını oluşturun.", file=sys.stderr)
        sys.exit(1) # Veritabanı olmadan uygulama başlayamaz

    # Veritabanını Chroma kütüphanesiyle yüklemeyi dene
    try:
        print(f"Mevcut veritabanı '{db_path}' klasöründen '{collection_name}' koleksiyonu yükleniyor...")
        # Chroma'ya kalıcı depolama yolunu, embedding fonksiyonunu ve koleksiyon adını vererek yükle
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
        print("Veritabanı başarıyla yüklendi.")
        # Yüklenen veritabanı nesnesini döndür
        return vectordb
    except Exception as e:
        # Yükleme sırasında bir hata oluşursa (örn: bozuk dosya, sürüm uyumsuzluğu)
        print(f"Hata: Vektör veritabanı yüklenemedi: {e}", file=sys.stderr)
        print("Veritabanı dosyaları bozulmuş olabilir veya 'create_database.py' ile uyumsuzluk olabilir.", file=sys.stderr)
        print(f"'{db_path}' klasörünü silip 'python create_database.py' komutunu tekrar çalıştırmayı deneyin.", file=sys.stderr)
        traceback.print_exc() # Hatanın tam kaynağını logla
        sys.exit(1) # Yükleme başarısızsa uygulama durmalı

# ----- RAG (Retrieval-Augmented Generation) ve AI Fonksiyonları -----

def get_answer(query, vectordb, top_k=5):
    """
    Kullanıcının sorusunu alır, vektör veritabanından ilgili bilgi parçalarını çeker (retrieve),
    bu parçaları ve soruyu bir prompt ile birleştirip Gemini modeline göndererek yanıt üretir (generate).
    Args:
        query (str): Kullanıcının sorduğu soru.
        vectordb (Chroma): Yüklenmiş Chroma vektör veritabanı nesnesi.
        top_k (int): Veritabanından çekilecek en ilgili belge (chunk) sayısı.
    Returns:
        str: Gemini modeli tarafından üretilen yanıt metni.
    """
    print(f"Alınan soru: '{query}'")
    print(f"Veritabanında en ilgili {top_k} Monopoly bilgisi aranıyor...")
    # Hata durumunda LLM'e gönderilecek varsayılan context
    context = "Monopoly veritabanı aranırken bir hata oluştu."

    # Adım 1: Retrieval (Bilgi Çekme)
    try:
        # Veritabanı nesnesini bir retriever olarak kullan
        # search_type="mmr": Max Marginal Relevance - Hem benzerliği hem de sonuçların çeşitliliğini dikkate alır.
        # search_kwargs: k -> döndürülecek sonuç sayısı, fetch_k -> MMR'ın çeşitliliği sağlamak için başlangıçta çekeceği sonuç sayısı (genellikle k'dan büyük)
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={'k': top_k, 'fetch_k': 15}
        )
        # Kullanıcının sorusunu kullanarak ilgili belgeleri retriever'dan al
        retrieved_docs = retriever.invoke(query)

        # Eğer hiç belge bulunamazsa
        if not retrieved_docs:
             print("Uyarı: Veritabanından bu soruyla ilgili bilgi bulunamadı.")
             context = "İlgili bilgi bulunamadı." # LLM'e bu durumu bildir
        else:
            # Bulunan belgelerin içeriklerini birleştirerek LLM için context oluştur
            # Belgeler arasına ayırıcı eklemek modelin belgeleri ayırt etmesine yardımcı olabilir
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            print(f"{len(retrieved_docs)} adet ilgili bilgi parçası bulundu (MMR ile).")

    except Exception as e:
        # Veritabanı araması sırasında hata olursa logla
        print(f"Hata: Veritabanı araması sırasında sorun oluştu: {e}", file=sys.stderr)
        traceback.print_exc()
        # context zaten hata mesajı olarak ayarlı

    # Adım 2: Prompt Oluşturma
    # LLM'e gönderilecek talimatları ve çekilen bilgileri içeren prompt metni
    prompt = f"""
Sen Monopoly Emlak Ticareti Oyunu için bir Yardımcı Asistansın. Görevin, SADECE sana aşağıda verilen MONOPOLY BİLGİ ALINTILARI'nı kullanarak oyuncunun sorduğu soruları yanıtlamaktır. Bu alıntılar hem resmi kuralları hem de oyunla ilgili ek bilgileri içerebilir.

Yanıtlarken UYMAN GEREKEN KURALLAR:
1.  Cevabın KESİNLİKLE verilen MONOPOLY BİLGİ ALINTILARI içinde yer almalıdır.
2.  Eğer cevap bu alıntılarda yoksa veya alıntılar yetersizse, "Üzgünüm, sağlanan bilgilerde bu soruya net bir cevap bulamadım. Belki kural kitapçığının kendisine bakmak isteyebilirsiniz." şeklinde yanıt ver. ASLA tahmin yürütme veya alıntılar dışında bilgi verme.
3.  Cevabını net, anlaşılır ve doğrudan sorulan konuyla ilgili ver. Mümkünse adım adım açıkla veya madde imleri kullan.
4.  Oyun stratejisi verme, sadece bilgi aktarımı yap. Örneğin "Ev kurmak iyi bir stratejidir" yerine "Ev kurmanın kuralları şunlardır..." gibi cevap ver.
5.  Eğer alıntılarda birden fazla ilgili bilgi varsa, bunları mantıklı bir sıra ile birleştirerek kapsamlı bir yanıt oluştur.
6.  "Monopoly kurallarına göre...", "Sağlanan bilgilere göre..." gibi ifadelerle başla. Yanıtının sonunda alıntıların dışına çıktığını belirten bir ifade KULLANMA.

MONOPOLY BİLGİ ALINTILARI:
---
{context}
---

OYUNCUNUN SORUSU:
{query}

MONOPOLY YARDIMCI ASİSTANI YANITI:"""

    # Adım 3: Generation (Yanıt Üretme)
    print("Prompt Gemini modeline gönderiliyor...")
    # Hata durumunda varsayılan yanıt
    answer = "Üzgünüm, sorunuzu yanıtlarken bir teknik sorunla karşılaştım. Lütfen tekrar deneyin."
    try:
        # Hazırlanan prompt'u Gemini modeline gönder
        response = model.generate_content(prompt)

        # Modelden gelen yanıtı kontrol et
        # response.parts: Modelin ürettiği metin parçalarını içerir. Başarılıysa dolu olur.
        if response.parts:
            answer = response.text # Üretilen metni al
            print("Gemini modelinden yanıt alındı.")
        # response.candidates: Alternatif yanıt adayları ve bitiş nedenini içerir.
        # finish_reason != 'STOP': Modelin normal şekilde bitmediğini gösterir (örn: güvenlik filtresi, uzunluk limiti).
        elif response.candidates and response.candidates[0].finish_reason != 'STOP':
             reason = response.candidates[0].finish_reason # Bitiş nedenini al
             print(f"Uyarı: Gemini yanıtı tamamlayamadı. Neden: {reason}")
             answer = f"Yanıt tam olarak üretilemedi (Neden: {reason}). Sorunuzu farklı şekilde sormayı deneyin."
        # Eğer response.parts boşsa ve bitiş nedeni de STOP değilse (örn: tamamen engellendi)
        else:
             print("Uyarı: Gemini modelinden boş yanıt alındı (Muhtemelen güvenlik filtresi).")
             answer = "Modelden geçerli bir yanıt alınamadı (güvenlik filtresine takılmış olabilir). Lütfen sorunuzu farklı şekilde ifade etmeyi deneyin."

    except Exception as e:
        # API çağrısı sırasında bir hata oluşursa logla
        print(f"Hata: Gemini modeli yanıt üretirken sorun oluştu: {e}", file=sys.stderr)
        traceback.print_exc() # Hatanın detayını logla
        # answer zaten varsayılan hata mesajı olarak ayarlı

    # Üretilen veya hata mesajı olan yanıtı döndür
    return answer

# ----- Yardımcı Fonksiyonlar -----

def render_markdown_html(text):
    """
    Verilen metni Markdown formatından HTML formatına dönüştürür.
    Bu fonksiyon Flask template'i içinde Jinja2 tarafından kullanılır.
    Args:
        text (str): Markdown formatındaki metin.
    Returns:
        str: HTML formatına dönüştürülmüş metin (veya hata durumunda düz metin).
    """
    try:
        # Markdown kütüphanesini kullanarak dönüşüm yap
        # extensions: Markdown'a ek özellikler kazandırır (kod blokları, tablolar, listeler vb.)
        html = markdown.markdown(
            text or "", # Metin None ise boş string kullan
            extensions=[
                "markdown.extensions.fenced_code", # ```python ... ``` gibi kod blokları
                "markdown.extensions.nl2br",      # Satır sonlarını <br> etiketine çevir
                "markdown.extensions.tables",     # Markdown tablolarını HTML tablosuna çevir
                "markdown.extensions.sane_lists", # İç içe ve düzgün listeler oluştur
            ],
            output_format="html5", # Modern HTML5 çıktısı üret
        )
        return html
    except Exception as e:
        # Markdown dönüşümü sırasında hata olursa logla ve güvenli bir HTML döndür
        print(f"Markdown render hatası: {e}", file=sys.stderr)
        import html as html_escaper
        # Metni HTML'den kaçış karakterleriyle güvenli hale getir ve <pre> içinde göster
        escaped_text = html_escaper.escape(text or "")
        return f"<p><i>(İçerik görüntülenirken hata oluştu)</i></p><pre>{escaped_text}</pre>"

# ----- Flask Uygulama Kurulumu -----

# Flask uygulamasını oluştur
app = Flask(__name__)
# Kullanıcı oturumlarını (session) şifrelemek ve güvende tutmak için rastgele bir gizli anahtar oluştur
# Bu anahtar, sunucu her yeniden başladığında değişir (kalıcı olması için sabit bir değer de atanabilir ama daha az güvenli)
app.secret_key = os.urandom(24)

# Uygulama başlarken vektör veritabanını yükle
print("Uygulama başlatılıyor, Monopoly veritabanı yükleniyor...")
try:
    vectordb = load_database() # Veritabanı yükleme fonksiyonunu çağır
    print("Monopoly veritabanı hazır. Flask uygulaması çalışmaya hazır.")
except SystemExit: # load_database hata verip çıkarsa uygulamayı başlatma
    print("Veritabanı yüklenemediği için Flask uygulaması başlatılamıyor.", file=sys.stderr)
    sys.exit(1) # Uygulamayı başlatmadan çık

# Sohbet geçmişlerini sunucu hafızasında (RAM) tutmak için bir sözlük yapısı
# Basit uygulamalar için yeterli, ancak sunucu yeniden başlarsa tüm sohbetler kaybolur.
# Kalıcı depolama için Redis, veritabanı vb. çözümler gerekir.
conversations = {}

# ----- Flask Rotaları (Web Sayfaları ve API Endpoints) -----

# Ana sayfa (örn: http://127.0.0.1:5000/) için endpoint
@app.route("/")
def index():
    """
    Ana HTML sayfasını render eder. Kullanıcının oturumunu yönetir,
    mevcut sohbeti ve geçmiş sohbet listesini template'e gönderir.
    """
    # Kullanıcının tarayıcısında kayıtlı session ID'sini al
    session_id = session.get("session_id")

    # Eğer session ID yoksa (ilk ziyaret) veya hafızadaki sohbetlerde bu ID yoksa (sunucu yeniden başlamış olabilir)
    if not session_id or session_id not in conversations:
        # Yeni bir benzersiz session ID oluştur
        session_id = str(uuid.uuid4())
        # Bu ID'yi kullanıcının tarayıcısına (cookie olarak) kaydet
        session["session_id"] = session_id
        # Sunucu hafızasında bu yeni ID için boş bir sohbet kaydı oluştur
        conversations[session_id] = {
            "id": session_id,
            "title": f"Yeni Monopoly Oyunu {datetime.now().strftime('%d.%m %H:%M')}", # Varsayılan başlık
            "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "messages": [], # Mesaj listesi başlangıçta boş
        }
        print(f"Yeni oturum başlatıldı ve ayarlandı: {session_id}")

    # Mevcut veya yeni oluşturulan sohbet verisini hafızadan al
    current_conversation = conversations.get(session_id)

    # Sol menüde gösterilecek tüm sohbetlerin listesini al ve tarihe göre (en yeni üste) sırala
    all_conversations = sorted(
        list(conversations.values()),
        key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"), # Sıralama için tarih formatını parse et
        reverse=True # En yeni en üstte olacak şekilde ters sırala
    )

    # HTML template'ine gönderilecek sayfa başlıkları ve diğer bilgiler
    page_config = {
        "page_title": "Monopoly Yardımcı Asistanı",
        "header_title": "Monopoly Yardımcı Asistanı",
        "header_subtitle": "Monopoly kuralları ve oyunu hakkında sorularınızı yanıtlar."
    }

    # Flask'ın render_template fonksiyonu ile index.html dosyasını oluştur ve tarayıcıya gönder
    # Template'e çeşitli değişkenler ve fonksiyonlar (renderMarkdown gibi) gönderilir
    return render_template(
        "index.html",
        conversation_history=current_conversation.get("messages", []), # Mevcut sohbetin mesajları
        conversations=all_conversations, # Sol menü için tüm sohbet başlıkları
        current_session_id=session_id, # Aktif sohbeti vurgulamak için
        renderMarkdown=render_markdown_html, # Template'in Markdown'ı HTML'e çevirmesi için
        **page_config # page_config sözlüğündeki tüm anahtar-değerleri template'e değişken olarak gönderir
    )

# Mesaj gönderme API endpoint'i (JavaScript tarafından POST isteği ile çağrılır)
@app.route("/send_message", methods=["POST"])
def send_message():
    """
    Kullanıcıdan AJAX (JavaScript) ile gelen mesajı alır, RAG ile yanıt üretir,
    sohbet geçmişini günceller ve yanıtı JSON formatında geri döndürür.
    """
    # Kullanıcının session ID'sini al
    session_id = session.get("session_id")
    # Geçerli bir session ID var mı kontrol et (güvenlik ve tutarlılık için)
    if not session_id or session_id not in conversations:
         print(f"Hata: /send_message - Geçersiz veya kayıp session ({session_id}).", file=sys.stderr)
         # İstemciye (JavaScript) hata mesajı döndür
         return jsonify({"response": "Oturum bulunamadı veya süresi doldu. Lütfen sayfayı yenileyin.", "conversations": []}), 400 # 400 Bad Request HTTP durum kodu

    # Gelen isteği ve yanıt üretimini try-except bloğu içine alarak hataları yakala
    try:
        # Gelen isteğin JSON içeriğini al
        data = request.json
        # JSON içinden 'message' anahtarının değerini al, baş/sondaki boşlukları sil
        user_message = data.get("message", "").strip()

        # Mesaj boşsa, kullanıcıyı uyar ve işlemi durdur
        if not user_message:
            all_conversations = sorted(list(conversations.values()), key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"), reverse=True)
            return jsonify({"response": "Lütfen Monopoly hakkında bir soru sorun.", "conversations": all_conversations})

        # Kullanıcının mesajını ilgili session'ın mesaj listesine ekle
        conversations[session_id]["messages"].append({"role": "user", "content": user_message})
        print(f"Oturum {session_id}: Soru: '{user_message}'")

        # Eğer bu, kullanıcının bu sohbetteki ilk mesajıysa, sohbet başlığını ayarla
        # (Listede 'user' rolüne sahip mesaj sayısını kontrol et)
        if len([msg for msg in conversations[session_id]["messages"] if msg['role'] == 'user']) == 1:
            # Başlığı mesajın ilk 35 karakteri yap (çok uzunsa kısalt)
            title = user_message[:35] + "..." if len(user_message) > 35 else user_message
            conversations[session_id]["title"] = title
            print(f"Oturum {session_id}: Başlık güncellendi: '{title}'")

        # RAG fonksiyonunu çağırarak bot yanıtını al
        bot_response_text = get_answer(user_message, vectordb, top_k=5) # Veritabanından en fazla 5 ilgili parça al

        # Bot yanıtını ilgili session'ın mesaj listesine ekle
        conversations[session_id]["messages"].append({"role": "bot", "content": bot_response_text})
        print(f"Oturum {session_id}: Yanıt eklendi.")

        # Güncel sohbet listesini (sol menü için) hazırla
        all_conversations = sorted(list(conversations.values()), key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"), reverse=True)

        # Yanıtı (düz metin olarak) ve güncel sohbet listesini JSON formatında döndür
        # JavaScript bu JSON'ı alıp arayüzü güncelleyecek
        return jsonify({"response": bot_response_text, "conversations": all_conversations})

    except Exception as e:
        # Beklenmedik bir hata oluşursa logla ve genel bir hata mesajı döndür
        print(f"Hata: /send_message sırasında beklenmedik hata: {e}", file=sys.stderr)
        traceback.print_exc() # Hatanın tam izini yazdır
        current_conversations = []
        try: # Hata olsa bile mevcut sohbet listesini göndermeyi dene
            current_conversations = sorted(list(conversations.values()), key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"), reverse=True)
        except Exception: pass # Liste alınamazsa boş gönder
        # Kullanıcıya teknik olmayan bir hata mesajı göster
        return jsonify({
                "response": "Üzgünüm, sorunuzu yanıtlarken beklenmedik bir sunucu hatası oluştu. Lütfen tekrar deneyin veya daha sonra tekrar gelin.",
                "conversations": current_conversations,
            }), 500 # 500 Internal Server Error HTTP durum kodu

# Yeni sohbet başlatma API endpoint'i (JavaScript tarafından POST isteği ile çağrılır)
@app.route("/new_chat", methods=["POST"])
def new_chat():
    """Yeni bir sohbet oturumu başlatır ve tarayıcıyı yönlendirmek için yeni ID'yi döndürür."""
    print("Yeni sohbet başlatılıyor...")
    # Yeni benzersiz ID oluştur
    session_id = str(uuid.uuid4())
    # Tarayıcının session bilgisini bu yeni ID ile güncelle
    session["session_id"] = session_id
    # Hafızada bu yeni ID için boş bir sohbet kaydı oluştur
    conversations[session_id] = {
        "id": session_id,
        "title": f"Yeni Monopoly Oyunu {datetime.now().strftime('%d.%m %H:%M')}",
        "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "messages": [],
    }
    print(f"Yeni oturum oluşturuldu: {session_id}")
    # JavaScript'in yönlendirme yapabilmesi için başarı durumu ve yeni ID'yi döndür
    return jsonify({"success": True, "new_session_id": session_id})

# Belirli bir sohbeti yüklemek için endpoint (sol menüdeki linkler tarafından kullanılır)
@app.route("/conversation/<session_id>")
def load_conversation(session_id):
    """Kullanıcı sol menüden eski bir sohbete tıkladığında o sohbeti aktif hale getirir."""
    # Gelen session_id hafızamızda (conversations sözlüğünde) var mı kontrol et
    if session_id in conversations:
        # Varsa, tarayıcının session bilgisini bu ID ile güncelle
        session["session_id"] = session_id
        print(f"Oturuma geçildi: {session_id}")
    else:
        # Yoksa (geçersiz link veya sunucu yeniden başlatılmış olabilir), uyarı ver ve tarayıcı session'ını temizle
        print(f"Uyarı: Geçersiz sohbet ID'si ({session_id}) yüklenmeye çalışıldı. Yeni oturum açılacak.")
        session.pop('session_id', None) # Tarayıcıdaki geçersiz ID'yi sil
    # Her durumda kullanıcıyı ana sayfaya yönlendir (index fonksiyonu durumu ele alacaktır)
    return redirect(url_for('index')) # 'index' -> index() fonksiyonunun adıdır

# ----- Uygulama Başlatma Noktası -----

# Bu script doğrudan çalıştırıldığında (örn: python app.py) aşağıdaki kod bloğu çalışır
if __name__ == "__main__":
    # Uygulamanın çalışacağı portu belirle (Ortam değişkeni PORT yoksa 5000 kullan)
    port = int(os.environ.get("PORT", 5000))
    # Debug modunu ortam değişkeninden al (FLASK_DEBUG=1 veya True ise aktif, yoksa pasif)
    # debug=True: Kod değişikliklerinde otomatik yeniden başlatma ve tarayıcıda detaylı hata gösterme sağlar. Deploy ederken False olmalı!
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() in ["true", "1", "yes"]
    print(f"Flask uygulaması http://0.0.0.0:{port} adresinde (Debug Modu: {debug_mode}) başlatılıyor...")
    # Flask uygulamasını geliştirme sunucusu ile çalıştır
    # host='0.0.0.0' -> Uygulamanın sadece yerel makineden değil, ağdaki diğer cihazlardan da erişilebilir olmasını sağlar (örn: aynı ağdaki telefon)
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
