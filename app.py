import os
import sys
import uuid
from datetime import datetime
import traceback # Hata ayıklama için

import google.generativeai as genai
import markdown
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

# ----- Yapılandırma ve Kurulum -----
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Hata: GOOGLE_API_KEY bulunamadı.", file=sys.stderr)
    sys.exit(1)

try:
    genai.configure(api_key=api_key)
    print("Google API Anahtarı başarıyla yüklendi.")
except Exception as e:
    print(f"Hata: Google API yapılandırılamadı: {e}", file=sys.stderr)
    sys.exit(1)

try:
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config=genai.types.GenerationConfig(temperature=0.4) # Monopoly için biraz daha serbest olabilir
    )
    print("Google Gemini modeli ('gemini-2.0-flash', temperature=0.4) başarıyla yüklendi.")
except Exception as e:
    print(f"Hata: Google Gemini modeli yüklenemedi: {e}", file=sys.stderr)
    sys.exit(1)

# ----- Veritabanı Fonksiyonları -----
def load_database():
    """Oluşturulmuş Monopoly Chroma veritabanını yükler."""
    print("Monopoly veritabanı yükleniyor...")
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=api_key
        )
        print("Google Embedding modeli başarıyla ayarlandı.")
    except Exception as e:
        print(f"Hata: Google Embedding modeli ayarlanamadı: {e}", file=sys.stderr)
        sys.exit(1)

    db_path = "./chroma_db"
    # <<<--- GÜNCELLENDİ: Koleksiyon adı ---<<<
    collection_name = "gaih_monopoly_comprehensive"

    db_file_path = os.path.join(db_path, "chroma.sqlite3")
    if not os.path.exists(db_path) or not os.path.exists(db_file_path):
        print(f"Hata: Veritabanı '{db_path}' klasöründe bulunamadı veya geçersiz.", file=sys.stderr)
        print("Lütfen önce 'python create_database.py' komutunu çalıştırarak veritabanını oluşturun.", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Mevcut veritabanı '{db_path}' klasöründen '{collection_name}' koleksiyonu yükleniyor...")
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
        print("Veritabanı başarıyla yüklendi.")
        return vectordb
    except Exception as e:
        print(f"Hata: Vektör veritabanı yüklenemedi: {e}", file=sys.stderr)
        print(f"'{db_path}' klasörünü silip 'python create_database.py' komutunu tekrar çalıştırmayı deneyin.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

# ----- RAG ve AI Fonksiyonları -----
def get_answer(query, vectordb, top_k=5):
    """Kullanıcının Monopoly sorusuna RAG ile cevap üretir."""
    print(f"Alınan soru: '{query}'")
    print(f"Veritabanında en ilgili {top_k} Monopoly bilgisi aranıyor...")
    context = "Monopoly veritabanı aranırken bir hata oluştu." # Varsayılan hata

    try:
        # Retriever ile arama (MMR veya similarity)
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={'k': top_k, 'fetch_k': 15}
        )
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
             print("Uyarı: Veritabanından bu soruyla ilgili bilgi bulunamadı.")
             context = "İlgili bilgi bulunamadı."
        else:
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            print(f"{len(retrieved_docs)} adet ilgili bilgi parçası bulundu (MMR ile).")

    except Exception as e:
        print(f"Hata: Veritabanı araması sırasında sorun oluştu: {e}", file=sys.stderr)
        traceback.print_exc()

    # <<<--- GÜNCELLENDİ: Prompt ---<<<
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

    print("Prompt Gemini modeline gönderiliyor...")
    answer = "Üzgünüm, sorunuzu yanıtlarken bir teknik sorunla karşılaştım. Lütfen tekrar deneyin." # Varsayılan hata
    try:
        response = model.generate_content(prompt)
        if response.parts:
            answer = response.text
            print("Gemini modelinden yanıt alındı.")
        elif response.candidates and response.candidates[0].finish_reason != 'STOP':
             reason = response.candidates[0].finish_reason
             print(f"Uyarı: Gemini yanıtı tamamlayamadı. Neden: {reason}")
             answer = f"Yanıt tam olarak üretilemedi (Neden: {reason}). Sorunuzu farklı şekilde sormayı deneyin."
        else:
             print("Uyarı: Gemini modelinden boş yanıt alındı.")
             answer = "Modelden geçerli bir yanıt alınamadı. Lütfen sorunuzu farklı şekilde sormayı deneyin."

    except Exception as e:
        print(f"Hata: Gemini modeli yanıt üretirken sorun oluştu: {e}", file=sys.stderr)

    return answer

# ----- Yardımcı Fonksiyonlar -----
def render_markdown_html(text):
    """Markdown metnini HTML formatına dönüştürür."""
    try:
        html = markdown.markdown(
            text or "", extensions=[ "markdown.extensions.fenced_code", "markdown.extensions.nl2br", "markdown.extensions.tables", "markdown.extensions.sane_lists" ], output_format="html5",
        )
        return html
    except Exception as e:
        print(f"Markdown render hatası: {e}", file=sys.stderr)
        import html as html_escaper
        escaped_text = html_escaper.escape(text or "")
        return f"<p><i>(İçerik görüntülenirken hata oluştu)</i></p><pre>{escaped_text}</pre>"

# ----- Flask Uygulama Kurulumu -----
app = Flask(__name__)
app.secret_key = os.urandom(24)

print("Uygulama başlatılıyor, Monopoly veritabanı yükleniyor...")
try:
    vectordb = load_database()
    print("Monopoly veritabanı hazır. Flask uygulaması çalışmaya hazır.")
except SystemExit:
    print("Veritabanı yüklenemediği için uygulama başlatılamıyor.", file=sys.stderr)
    sys.exit(1)

conversations = {}

# ----- Flask Rotaları -----
@app.route("/")
def index():
    """Ana sayfayı ve sohbet geçmişini gösterir."""
    session_id = session.get("session_id")

    if not session_id or session_id not in conversations:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
        conversations[session_id] = {
            "id": session_id,
            "title": f"Yeni Monopoly Oyunu {datetime.now().strftime('%d.%m %H:%M')}", # <<<--- Güncellendi
            "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "messages": [],
        }
        print(f"Yeni oturum başlatıldı: {session_id}")

    current_conversation = conversations.get(session_id)
    all_conversations = sorted( list(conversations.values()), key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"), reverse=True )

    # <<<--- GÜNCELLENDİ: Sayfa Başlıkları ---<<<
    page_config = {
        "page_title": "Monopoly Yardımcı Asistanı",
        "header_title": "Monopoly Yardımcı Asistanı",
        "header_subtitle": "Monopoly kuralları ve oyunu hakkında sorularınızı yanıtlar."
    }

    return render_template(
        "index.html",
        conversation_history=current_conversation.get("messages", []),
        conversations=all_conversations,
        current_session_id=session_id,
        renderMarkdown=render_markdown_html,
        **page_config
    )

@app.route("/send_message", methods=["POST"])
def send_message():
    """AJAX isteği ile gelen mesajı işler ve yanıt döndürür."""
    session_id = session.get("session_id")
    if not session_id or session_id not in conversations:
         print(f"Hata: /send_message - Geçersiz session ({session_id}).", file=sys.stderr)
         return jsonify({"response": "Oturum bulunamadı, sayfayı yenileyin.", "conversations": []}), 400

    try:
        data = request.json
        user_message = data.get("message", "").strip()

        if not user_message:
            all_conversations = sorted(list(conversations.values()), key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"), reverse=True)
            # <<<--- Güncellendi: Hata Mesajı ---<<<
            return jsonify({"response": "Lütfen Monopoly hakkında bir soru sorun.", "conversations": all_conversations})

        conversations[session_id]["messages"].append({"role": "user", "content": user_message})
        print(f"Oturum {session_id}: Soru: '{user_message}'")

        if len([msg for msg in conversations[session_id]["messages"] if msg['role'] == 'user']) == 1:
            title = user_message[:35] + "..." if len(user_message) > 35 else user_message
            conversations[session_id]["title"] = title
            print(f"Oturum {session_id}: Başlık: '{title}'")

        bot_response_text = get_answer(user_message, vectordb, top_k=5)
        conversations[session_id]["messages"].append({"role": "bot", "content": bot_response_text})
        print(f"Oturum {session_id}: Yanıt eklendi.")

        all_conversations = sorted(list(conversations.values()), key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"), reverse=True)

        return jsonify({"response": bot_response_text, "conversations": all_conversations})

    except Exception as e:
        print(f"Hata: /send_message sırasında beklenmedik hata: {e}", file=sys.stderr)
        traceback.print_exc()
        current_conversations = []
        try:
            current_conversations = sorted(list(conversations.values()), key=lambda c: datetime.strptime(c["created_at"], "%d.%m.%Y %H:%M"), reverse=True)
        except Exception: pass
        return jsonify({
                "response": "Üzgünüm, sorunuzu yanıtlarken beklenmedik bir sunucu hatası oluştu.",
                "conversations": current_conversations,
            }), 500

@app.route("/new_chat", methods=["POST"])
def new_chat():
    """Yeni sohbet başlatır."""
    print("Yeni sohbet başlatılıyor...")
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    conversations[session_id] = {
        "id": session_id,
        # <<<--- Güncellendi: Başlık ---<<<
        "title": f"Yeni Monopoly Oyunu {datetime.now().strftime('%d.%m %H:%M')}",
        "created_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "messages": [],
    }
    print(f"Yeni oturum oluşturuldu: {session_id}")
    return jsonify({"success": True, "new_session_id": session_id})

@app.route("/conversation/<session_id>")
def load_conversation(session_id):
    """URL'den gelen session ID ile sohbeti yükler."""
    if session_id in conversations:
        session["session_id"] = session_id
        print(f"Oturuma geçildi: {session_id}")
    else:
        print(f"Uyarı: Geçersiz sohbet ID ({session_id}). Yeni oturum açılacak.")
        session.pop('session_id', None)
    return redirect(url_for('index'))

# ----- Uygulama Başlatma Noktası -----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() in ["true", "1", "yes"]
    print(f"Flask uygulaması http://0.0.0.0:{port} adresinde (Debug Modu: {debug_mode}) başlatılıyor...")
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
