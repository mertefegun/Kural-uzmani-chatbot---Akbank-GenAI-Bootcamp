import os
import sys
import uuid # Benzersiz session ID'leri oluşturmak için
from datetime import datetime # Zaman damgaları için
import traceback # Hata ayıklama için

import google.generativeai as genai # Google Gemini API
import markdown # Sunucu tarafında eski mesajları render etmek için
from dotenv import load_dotenv # .env dosyasını okumak için
from flask import Flask, jsonify, redirect, render_template, request, session, url_for # Web framework'ü
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Google embedding modeli
from langchain_community.vectorstores import Chroma # Chroma veritabanı entegrasyonu
# create_database fonksiyonunu doğrudan çağırmıyoruz, sadece varlığını kontrol ediyoruz.
import shutil # Veritabanı temizleme için (gerektiğinde)

# ----- Yapılandırma ve Kurulum -----

# .env dosyasındaki ortam değişkenlerini (GOOGLE_API_KEY) yükle
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# API Anahtarı var mı kontrol et
if not api_key:
    print("Hata: GOOGLE_API_KEY ortam değişkeni bulunamadı.", file=sys.stderr)
    print("Lütfen proje ana dizininde '.env' dosyasını oluşturup içine 'GOOGLE_API_KEY=...' satırını eklediğinizden emin olun.", file=sys.stderr)
    sys.exit(1) # Anahtar yoksa uygulama başlamaz

# Google Generative AI kütüphanesini API anahtarı ile yapılandır
try:
    genai.configure(api_key=api_key)
    print("Google API Anahtarı başarıyla yüklendi ve yapılandırıldı.")
except Exception as e:
    print(f"Hata: Google API yapılandırılamadı: {e}", file=sys.stderr)
    sys.exit(1)

# Kullanılacak Gemini modelini (LLM) ayarla
try:
    # 'gemini-2.0-flash' yerine 'gemini-1.5-flash' veya başka bir uygun model de seçilebilir.
    # temperature: Cevapların ne kadar rastgele/yaratıcı olacağını belirler (0: en net, 1: en yaratıcı). Kural açıklamaları için düşük tutmak iyi.
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config=genai.types.GenerationConfig(temperature=0.3)
        # Gerekirse güvenlik ayarları burada tanımlanabilir:
        # safety_settings=[...]
    )
    print("Google Gemini modeli ('gemini-2.0-flash', temperature=0.3) başarıyla yüklendi.")
except Exception as e:
    print(f"Hata: Google Gemini modeli yüklenemedi: {e}", file=sys.stderr)
    sys.exit(1)

# ----- Veritabanı Fonksiyonları -----

def load_database():
    """
    Önceden oluşturulmuş Chroma vektör veritabanını yükler.
    Eğer veritabanı bulunamazsa veya yüklenemezse programı durdurur.
    """
    print("Kural veritabanı yükleniyor...")
    # Embedding modelini yapılandır (veritabanı yüklenirken gerekli)
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=api_key
            # Sorgulama için task_type belirtmek genellikle gerekmez, LangChain halleder.
        )
        print("Google Embedding modeli ('text-embedding-004') başarıyla ayarlandı.")
    except Exception as e:
        print(f"Hata: Google Embedding modeli ayarlanamadı: {e}", file=sys.stderr)
        sys.exit(1)

    # Veritabanı klasörünün ve koleksiyon adının yolları
    db_path = "./chroma_db"
    # Bu ad, create_database.py'deki collection_name ile aynı olmalı!
    collection_name = "gaih_tmars_rules"

    # Veritabanı klasörü var mı ve içinde gerekli dosya var mı kontrol et
    db_file_path = os.path.join(db_path, "chroma.sqlite3")
    if not os.path.exists(db_path) or not os.path.exists(db_file_path):
        print(f"Hata: Veritabanı '{db_path}' klasöründe bulunamadı veya 'chroma.sqlite3' dosyası eksik.", file=sys.stderr)
        print("Lütfen önce 'python create_database.py' komutunu başarıyla çalıştırdığınızdan emin olun.", file=sys.stderr)
        sys.exit(1) # Veritabanı olmadan uygulama başlayamaz

    # Veritabanını yüklemeyi dene
    try:
        print(f"Mevcut veritabanı '{db_path}' klasöründen '{collection_name}' koleksiyonu yükleniyor...")
        vectordb = Chroma(
            persist_directory=db_path,           # Kalıcı depolama klasörü
            embedding_function=embedding_function, # Kullanılan embedding modeli
            collection_name=collection_name,     # Yüklenecek koleksiyonun adı
        )
        # Veritabanının boş olmadığını doğrulamak için küçük bir test (opsiyonel)
        # count = vectordb._collection.count()
        # if count == 0:
        #     print("Uyarı: Veritabanı yüklendi ancak içinde hiç belge bulunmuyor!", file=sys.stderr)
        # else:
        #     print(f"Veritabanında {count} belge bulundu.")

        print("Veritabanı başarıyla yüklendi.")
        return vectordb
    except Exception as e:
        # Yükleme sırasında hata oluşursa (örn: sürüm uyumsuzluğu, bozuk dosya)
        print(f"Hata: Vektör veritabanı yüklenemedi: {e}", file=sys.stderr)
        print("Veritabanı dosyaları bozulmuş olabilir veya 'create_database.py' ile uyumsuzluk olabilir.", file=sys.stderr)
        print(f"'{db_path}' klasörünü silip 'python create_database.py' komutunu tekrar çalıştırmayı deneyin.", file=sys.stderr)
        traceback.print_exc() # Hatanın tam kaynağını görmek için
        sys.exit(1) # Yükleme başarısızsa uygulama durmalı

# ----- RAG ve AI Fonksiyonları -----

def get_answer(query, vectordb, top_k=5):
    """
    Kullanıcının kural sorusunu alır, vektör veritabanından ilgili kural parçalarını çeker (retrieve),
    bu parçaları ve soruyu Gemini modeline göndererek bir yanıt üretir (generate).
    Args:
        query (str): Kullanıcının sorduğu kural sorusu.
        vectordb (Chroma): Yüklenmiş Chroma vektör veritabanı nesnesi.
        top_k (int): Veritabanından çekilecek en ilgili kural parçası sayısı.
    Returns:
        str: Gemini modeli tarafından üretilen yanıt metni.
    """
    print(f"Alınan soru: '{query}'")
    print(f"Veritabanında en ilgili {top_k} kural parçası aranıyor...")
    context = "Kural veritabanı aranırken bir hata oluştu." # Hata durumunda varsayılan context

    try:
        # Retriever kullanarak arama yapalım (MMR veya similarity)
        # Max Marginal Relevance (MMR) hem benzerliği hem de çeşitliliği dikkate alır.
        retriever = vectordb.as_retriever(
            search_type="mmr", # "similarity" de kullanılabilir
            search_kwargs={'k': top_k, 'fetch_k': 15} # Daha fazla sonuç alıp çeşitliliği artırır
        )
        # LangChain'in güncel invoke metodunu kullanalım
        retrieved_docs = retriever.invoke(query)

        # Eğer hiç sonuç bulunamazsa
        if not retrieved_docs:
             print("Uyarı: Veritabanından bu soruyla ilgili kural parçası bulunamadı.")
             context = "İlgili kural bulunamadı." # LLM'e bu bilgiyi verelim
        else:
            # Bulunan belgelerin içeriklerini birleştirerek 'context' oluştur
            # Belgeler arasına ayırıcı koymak LLM'in anlamasına yardımcı olabilir
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            print(f"{len(retrieved_docs)} adet ilgili kural parçası bulundu (MMR ile).")
            # İsteğe bağlı: Bulunan context'i loglamak için
            # print("--- Bulunan Context Başlangıcı ---")
            # print(context)
            # print("--- Bulunan Context Sonu ---")

    except Exception as e:
        print(f"Hata: Veritabanı araması sırasında sorun oluştu: {e}", file=sys.stderr)
        traceback.print_exc()
        # context zaten hata mesajı olarak ayarlı

    # Gemini modeline gönderilecek Prompt Şablonu
    # Modelin rolünü, görevini, kullanacağı kaynağı ve uyması gereken kuralları netleştirir.
    prompt = f"""
Sen Terraforming Mars oyunu için bir Kural Uzmanısın. Görevin, SADECE sana aşağıda verilen KURAL KİTAPÇIĞI ALINTILARI'nı kullanarak oyuncunun sorduğu kural sorusunu yanıtlamaktır.

Yanıtlarken UYMAN GEREKEN KURALLAR:
1.  Cevabın KESİNLİKLE verilen KURAL KİTAPÇIĞI ALINTILARI içinde yer almalıdır.
2.  Eğer cevap bu alıntılarda yoksa veya alıntılar yetersizse, "Üzgünüm, sağlanan kural alıntılarında bu soruya net bir cevap bulamadım. Kural kitapçığının ilgili bölümünü kontrol etmenizi öneririm." şeklinde yanıt ver. ASLA tahmin yürütme veya alıntılar dışında bilgi verme.
3.  Cevabını net, anlaşılır ve doğrudan sorulan kuralla ilgili ver. Mümkünse adım adım açıkla veya madde imleri kullan.
4.  Oyun stratejisi, kart yorumu, tavsiye veya kişisel görüş belirtme. Sadece kuralları olduğu gibi aktar.
5.  Eğer alıntılarda birden fazla ilgili kural varsa, bunları mantıklı bir sıra ile birleştirerek kapsamlı bir yanıt oluştur.
6.  "Kural kitapçığı alıntılarına göre...", "İlgili kurallar şöyle diyor:" gibi ifadelerle başla. Yanıtının sonunda alıntıların dışına çıktığını belirten bir ifade KULLANMA.

KURAL KİTAPÇIĞI ALINTILARI:
---
{context}
---

OYUNCUNUN SORUSU:
{query}

KURAL UZMANI YANITI:"""

    print("Prompt Gemini modeline gönderiliyor...")
    answer = "Üzgünüm, kural açıklamasını üretirken bir teknik sorunla karşılaştım. Lütfen tekrar deneyin." # Varsayılan hata yanıtı
    try:
        # Modeli çağır ve yanıtı al
        response = model.generate_content(prompt)

        # Cevabın içeriğini ve neden bittiğini kontrol et
        if response.parts:
            answer = response.text
            print("Gemini modelinden yanıt alındı.")
        # Eğer model yanıtı bitiremediyse (güvenlik, uzunluk vb. nedenlerle)
        elif response.candidates and response.candidates[0].finish_reason != 'STOP':
             reason = response.candidates[0].finish_reason
             print(f"Uyarı: Gemini modeli yanıtı tamamlayamadı. Neden: {reason}")
             answer = f"Yanıt tam olarak üretilemedi (Neden: {reason}). Lütfen sorunuzu daha basit veya farklı bir şekilde tekrar sormayı deneyin."
        # Eğer hiçbir içerik dönmediyse
        else:
             print("Uyarı: Gemini modelinden boş yanıt alındı.")
             answer = "Modelden geçerli bir yanıt alınamadı. Lütfen sorunuzu farklı şekilde sormayı deneyin veya API anahtarınızı kontrol edin."

    except Exception as e:
        # API çağrısı sırasında bir hata oluşursa
        print(f"Hata: Gemini modeli yanıt üretirken sorun oluştu: {e}", file=sys.stderr)
        traceback.print_exc() # Hatanın detayını logla
        # answer zaten varsayılan hata mesajı olarak ayarlı

    return answer

# ----- Yardımcı Fonksiyonlar -----

# Markdown metnini HTML'e çeviren fonksiyon (Jinja template içinde kullanılacak)
def render_markdown_html(text):
    """Markdown metnini HTML formatına dönüştürür."""
    try:
        # Temel ve güvenli markdown uzantıları
        html = markdown.markdown(
            text or "", # Boş metin gelirse hata vermesin
            extensions=[
                "markdown.extensions.fenced_code", # ```kod``` blokları için
                "markdown.extensions.nl2br",      # Yeni satırlar <br> olur
                "markdown.extensions.tables",     # Tablolar | Başlık | ... |
                "markdown.extensions.sane_lists", # Listeler - * 1.
            ],
            output_format="html5", # Modern HTML çıktısı
        )
        return html
    except Exception as e:
        # Render sırasında hata olursa, metni güvenli şekilde göster
        print(f"Markdown render hatası: {e}", file=sys.stderr)
        import html as html_escaper
        escaped_text = html_escaper.escape(text or "")
        # Hata mesajını da ekleyelim
        return f"<p><i>(Markdown formatı işlenirken hata oluştu)</i></p><pre>{escaped_text}</pre>"


# ----- Flask Uygulama Kurulumu -----
app = Flask(__name__)
# Session'ları (kullanıcı oturumlarını) güvende tutmak için rastgele bir anahtar
app.secret_key = os.urandom(24)

# Uygulama başlarken veritabanını yükle
print("Uygulama başlatılıyor, kural veritabanı yükleniyor...")
try:
    vectordb = load_database() # Yükleme fonksiyonunu çağır
    print("Kural veritabanı hazır. Flask uygulaması çalışmaya hazır.")
except SystemExit: # load_database hata verip çıkarsa uygulamayı başlatma
    print("Veritabanı yüklenemediği için Flask uygulaması başlatılamıyor.", file=sys.stderr)
    # Uygulama başlamadan çıkış yapacak
    sys.exit(1)


# Sohbet geçmişlerini sunucu hafızasında tutmak için bir sözlük
# Not: Sunucu yeniden başladığında bu hafıza silinir. Kalıcı depolama için veritabanı gerekir.
conversations = {}

# ----- Flask Rotaları (Web Sayfaları ve API Endpoints) -----

@app.route("/")
def index():
    """Ana sayfayı gösterir, mevcut sohbeti veya yeni sohbeti yükler."""
    session_id = session.get("session_id")

    # Eğer kullanıcı ilk defa giriyorsa veya session kaybolmuşsa yeni bir tane oluştur
    if not session_id or session_id not in conversations:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id # Tarayıcıya yeni ID'yi kaydet
        # Hafızada yeni sohbeti başlat
        conversations[session_id] = {
            "id": session_id,
            "title": f"Yeni Oyun {datetime.now().strftime('%d.%m %H:%M')}", # Varsayılan başlık
            "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "messages": [], # Boş mesaj listesi
        }
        print(f"Yeni oturum başlatıldı ve ayarlandı: {session_id}")

    # Mevcut veya yeni oluşturulan sohbeti al
    current_conversation = conversations.get(session_id)

    # Sol menü için tüm sohbetleri al ve tarihe göre (en yeni üste) sırala
    all_conversations = sorted(
        list(conversations.values()),
        key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"),
        reverse=True
    )

    # HTML template'ine gönderilecek sayfa başlıkları
    page_config = {
        "page_title": "Terraforming Mars Kural Uzmanı",
        "header_title": "Terraforming Mars Kural Uzmanı",
        "header_subtitle": "Oyun kuralları hakkında sorularınızı yanıtlar."
    }

    # index.html dosyasını render et ve gerekli verileri gönder
    return render_template(
        "index.html",
        conversation_history=current_conversation.get("messages", []), # Mevcut sohbetin mesajları
        conversations=all_conversations, # Sol menü için tüm sohbetler
        current_session_id=session_id, # Aktif session ID'si (CSS için)
        renderMarkdown=render_markdown_html, # Template içinde markdown render fonksiyonu
        **page_config # Sayfa başlıkları
    )

@app.route("/send_message", methods=["POST"])
def send_message():
    """
    Kullanıcıdan AJAX (JavaScript) ile gelen mesajı alır, RAG ile yanıt üretir,
    sohbet geçmişini günceller ve yanıtı JSON formatında geri döndürür.
    """
    session_id = session.get("session_id")
    # Geçerli bir session ID var mı kontrol et
    if not session_id or session_id not in conversations:
         print(f"Hata: /send_message - Geçersiz veya kayıp session ({session_id}).", file=sys.stderr)
         # Kullanıcıya hata mesajı döndür
         return jsonify({"response": "Oturum bulunamadı veya süresi doldu. Lütfen sayfayı yenileyin.", "conversations": []}), 400 # 400 Bad Request

    try:
        # Gelen JSON verisini al
        data = request.json
        user_message = data.get("message", "").strip() # Mesajı al ve boşlukları temizle

        # Boş mesaj gelirse hata döndür
        if not user_message:
            # Güncel sohbet listesini alıp dönelim
            all_conversations = sorted(list(conversations.values()), key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"), reverse=True)
            return jsonify({"response": "Lütfen bir kural sorun.", "conversations": all_conversations})

        # Kullanıcı mesajını hafızadaki sohbet geçmişine ekle
        conversations[session_id]["messages"].append({"role": "user", "content": user_message})
        print(f"Oturum {session_id}: Soru: '{user_message}'")

        # Eğer bu, kullanıcının ilk mesajıysa (listede sadece 1 kullanıcı mesajı varsa) sohbet başlığını ayarla
        if len([msg for msg in conversations[session_id]["messages"] if msg['role'] == 'user']) == 1:
            # Başlığı mesajın ilk 35 karakteri yap (çok uzunsa kısalt)
            title = user_message[:35] + "..." if len(user_message) > 35 else user_message
            conversations[session_id]["title"] = title
            print(f"Oturum {session_id}: Başlık güncellendi: '{title}'")

        # RAG fonksiyonunu çağırarak bot yanıtını al
        bot_response_text = get_answer(user_message, vectordb, top_k=5) # top_k=5 belge kullan

        # Bot yanıtını hafızadaki sohbet geçmişine ekle
        conversations[session_id]["messages"].append({"role": "bot", "content": bot_response_text})
        print(f"Oturum {session_id}: Yanıt eklendi.")

        # Güncel sohbet listesini (sol menü için) hazırla
        all_conversations = sorted(list(conversations.values()), key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"), reverse=True)

        # Yanıtı (düz metin olarak) ve güncel sohbet listesini JSON formatında döndür
        # JavaScript tarafı bu yanıtı alıp Markdown'ı HTML'e çevirecek (Marked.js ile)
        return jsonify({"response": bot_response_text, "conversations": all_conversations})

    except Exception as e:
        # Beklenmedik bir hata oluşursa logla ve genel bir hata mesajı döndür
        print(f"Hata: /send_message sırasında beklenmedik hata: {e}", file=sys.stderr)
        traceback.print_exc() # Hatanın tam izini yazdır
        current_conversations = []
        try: # Hata olsa bile mevcut sohbet listesini göndermeyi dene
            current_conversations = sorted(list(conversations.values()), key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"), reverse=True)
        except Exception: pass
        # Kullanıcıya teknik olmayan bir hata mesajı göster
        return jsonify({
                "response": "Üzgünüm, kural bulunurken beklenmedik bir sunucu hatası oluştu. Lütfen tekrar deneyin veya daha sonra tekrar gelin.",
                "conversations": current_conversations,
            }), 500 # 500 Internal Server Error

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
        "title": f"Yeni Oyun {datetime.now().strftime('%d.%m %H:%M')}",
        "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "messages": [],
    }
    print(f"Yeni oturum oluşturuldu ve aktif: {session_id}")
    # JavaScript'in yönlendirme yapabilmesi için başarı durumu ve yeni ID'yi döndür
    return jsonify({"success": True, "new_session_id": session_id})

@app.route("/conversation/<session_id>")
def load_conversation(session_id):
    """Kullanıcı sol menüden eski bir sohbete tıkladığında o sohbeti aktif hale getirir."""
    # Gelen session_id hafızamızda (conversations sözlüğünde) var mı kontrol et
    if session_id in conversations:
        # Varsa, tarayıcının session bilgisini bu ID ile güncelle
        session["session_id"] = session_id
        print(f"Mevcut oturuma geçildi: {session_id}")
    else:
        # Yoksa (geçersiz link veya sunucu yeniden başlatılmış olabilir), uyarı ver ve tarayıcı session'ını temizle
        print(f"Uyarı: Geçersiz sohbet ID'si ({session_id}) yüklenmeye çalışıldı. Yeni oturum açılacak.")
        session.pop('session_id', None) # Tarayıcıdaki geçersiz ID'yi sil
    # Her durumda kullanıcıyı ana sayfaya yönlendir (index fonksiyonu durumu ele alacaktır)
    return redirect(url_for('index'))

# ----- Uygulama Başlatma Noktası -----
if __name__ == "__main__":
    # Uygulamanın çalışacağı portu belirle (Ortam değişkeni yoksa 5000)
    port = int(os.environ.get("PORT", 5000))
    # Debug modunu ortam değişkeninden al (varsayılan True)
    # Deploy ederken FLASK_DEBUG=0 olarak ayarlamak önemlidir!
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() in ["true", "1", "yes"]
    print(f"Flask uygulaması http://0.0.0.0:{port} adresinde (Debug Modu: {debug_mode}) başlatılıyor...")
    # Uygulamayı çalıştır
    # host='0.0.0.0' -> Ağdaki diğer cihazlardan erişilebilir yapar
    # debug=True -> Kod değişikliklerinde otomatik yeniden başlatma ve tarayıcıda hata gösterme sağlar
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
