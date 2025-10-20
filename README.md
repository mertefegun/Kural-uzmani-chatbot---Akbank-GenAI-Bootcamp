# Akbank-GenAI-Bootcamp-Yeni-Nesil-Proje-Kampı-
🎲 Monopoly Yardımcı Asistanı Chatbot
Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, RAG (Retrieval-Augmented Generation) tabanlı bir chatbot uygulamasıdır. Chatbot'un amacı, popüler masa oyunu Monopoly Emlak Ticareti Oyunu'nun kuralları ve oyunla ilgili bilgilere dayanarak oyuncuların sorularını yanıtlamaktır.

🎯 Projenin Amacı
Monopoly oynarken sıkça karşılaşılan kural sorularına ("İpotekli yerden kira alınır mı?", "Kodes'ten nasıl çıkılır?", "Evleri nasıl kurmalıyım?") hızlı ve doğru yanıtlar sunmak. Chatbot, yanıtlarını hem resmi kural kitapçığına hem de oyunla ilgili ek bilgilere (SSS vb.) dayandırarak oyunculara yardımcı olmayı hedefler.

📊 Veri Seti Hakkında Bilgi
Bu chatbot'un ana bilgi kaynağı, data klasörüne yerleştirilecek olan monopoly_kapsamli_veri.pdf dosyasıdır. Bu dosya, aşağıdakilerin birleşiminden oluşur:

1- Resmi Monopoly Emlak Ticareti Oyunu Kural Kitapçığı'ndan çıkarılan metinler.

2- Monopoly hakkında ek genel bilgiler.

3- Oyunla ilgili sıkça sorulan sorular (SSS) ve cevapları.

. Veri Genişletme Notu: Veri setindeki SSS bölümü, projenin geliştirilmesi sırasında Google Gemini gibi üretken yapay zeka modellerinden faydalanılarak, muhtemel kullanıcı soruları ve cevapları türetilerek genişletilmiştir. Bu yaklaşım, chatbot'un daha fazla soruya hazırlıklı olmasını sağlamak amacıyla kullanılmıştır.

create_database.py script'i, bu kapsamlı PDF dosyasındaki metin içeriğini otomatik olarak çıkarıp işleyerek RAG sistemi için hazırlar.

🛠️ Kullanılan Teknolojiler ve Yöntemler
. Web Framework: Flask

. RAG Pipeline Framework: LangChain

. Generation Model (LLM): Google Gemini (gemini-2.0-flash)

. Embedding Model: Google Embedding Model (models/text-embedding-004)

. Vektör Veritabanı: Chroma

. Veri Kaynağı: Kapsamlı Monopoly Bilgi PDF (monopoly_kapsamli_veri.pdf)

. PDF İşleme: pypdf kütüphanesi

⚙️ Çözüm Mimarisi
1- Veri Hazırlama: create_database.py, pypdf kullanarak data/monopoly_kapsamli_veri.pdf dosyasından metin çıkarır. Metin, RecursiveCharacterTextSplitter ile parçalara ayrılır.

2- Embedding & Depolama: Her parça, Google text-embedding-004 ile vektöre dönüştürülür ve Chroma veritabanına (chroma_db) kaydedilir.

3- Sorgu İşleme: Kullanıcının sorusu vektöre dönüştürülür.

4- Retrieval: Chroma DB'de soruya en yakın bilgi parçaları (context) bulunur (MMR veya similarity search kullanılarak).

5- Generation: Bulunan parçalar ve soru, özel bir prompt ile Google Gemini (gemini-2.0-flash) modeline gönderilir. Model, sağlanan bilgilere dayanarak yanıt üretir.

🚀 Kurulum ve Çalıştırma Kılavuzu 
Projeyi test etmek için aşağıdaki adımları takip edebilirsiniz:

1- Google API Anahtarı Alın:

Google AI Studio adresinden API anahtarınızı oluşturun ve kopyalayın.

2- Depoyu Klonlayın:

git clone https://github.com/mertefegun/Kural-uzmani-chatbot---Akbank-GenAI-Bootcamp.git
cd Kural-uzmani-chatbot---Akbank-GenAI-Bootcamp

3- Sanal Ortam Oluşturun (Önerilir):

python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows

4- Bağımlılıkları Yükleyin:

pip install -r requirements.txt

5- API Anahtarını Ayarlayın:

. Proje ana klasöründe .env adında yeni bir dosya oluşturun.

. İçine GOOGLE_API_KEY=BURAYA_API_ANAHTARINIZI_YAPIŞTIRIN satırını ekleyin.

. Dosyayı kaydedin.

6- Veritabanını Oluşturun (PDF'i İşleyecek):

. Not: data klasöründe monopoly_kapsamli_veri.pdf dosyası zaten bulunmaktadır.

. Aşağıdaki komutu çalıştırın:

python create_database.py

. "Veritabanı başarıyla oluşturuldu..." mesajını bekleyin.

7- Uygulamayı Başlatın:

python app.py

8- Test Edin:

. Web tarayıcınızı açıp terminalde belirtilen adrese (genellikle http://127.0.0.1:5000) gidin.

. Chatbot arayüzü açılacaktır. Aşağıdaki gibi sorular sorarak test edebilirsiniz:

  . "Bankada ev veya otel kalmazsa ne olur?"

  . "Kodes'ten çıkmak için ne kadar ödemem gerekir?"

  . "İpotekli mülk satılabilir mi?"

  . "Açık artırma nasıl yapılır?"

  . "Kamu kuruluşlarının kirası nasıl hesaplanır?"

  . "Evleri nasıl kurarım? Sırayla mı kurmak zorundayım?"

📁 Proje Yapısı
.
├── data/
│   └── monopoly_kapsamli_veri.pdf # Kapsamlı Monopoly Veri PDF'i
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── create_database.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

🌐 Web Arayüzü & Product Kılavuzu
Canlı Demo Linki: [ Projenizi deploy ettikten sonra canlı linki buraya ekleyin ]

Kullanım:

1- Uygulama linkine gidin veya yerelde çalıştırıp http://localhost:5000 adresine gidin.

2- Sol menüden "Yeni Sohbet Başlat" diyerek temiz bir sayfa açabilir veya varsa eski sohbetinize tıklayarak devam edebilirsiniz.

3- Alt kısımdaki metin kutusuna Monopoly ile ilgili sorunuzu yazın.

4- "Gönder" butonuna tıklayın veya Enter'a basın.

5- Chatbot, yüklenen PDF dosyasındaki bilgilere dayanarak sorunuza yanıt vermeye çalışacaktır.

(Buraya Ekran Görüntüleri veya Video Ekleyin)

. Chatbot arayüzünün genel görünümü.

. Örnek bir Monopoly sorusu ve chatbot'un verdiği yanıt.

✨ Sonuçlar ve Değerlendirme
Bu proje ile Monopoly oyunu için kapsamlı bir bilgi kaynağını temel alan, RAG tabanlı bir "Monopoly Yardımcı Asistanı" chatbot geliştirilmiştir. PDF verisinden metin çıkarma, LangChain ile RAG akışı, ChromaDB vektör veritabanı ve Google Gemini API entegrasyonu başarıyla kullanılmıştır. Veri seti, Gemini kullanılarak potansiyel sorular ve cevaplarla zenginleştirilmiştir. Flask ile sunulan web arayüzü, oyuncuların oyun sırasında sorularına hızlı yanıt almasını sağlar. Proje, Akbank GenAI Bootcamp proje gereksinimlerini karşılamaktadır.

📚 Yardımcı Kaynaklar
. [Monopoly Kural Kitapçığı ve Ek Bilgiler (monopoly_kapsamli_veri.pdf)]

. LangChain Documentation

. ChromaDB Documentation

. Gemini API Documentation

. Flask Documentation

. pypdf Documentation

Sonraki Adımlar:

1- Dosyaları Güncelle: Bilgisayarındaki create_database.py, app.py ve README.md dosyalarını yukarıdaki içeriklerle değiştir/üzerine yaz.

2- Veri Dosyasını Yerleştir: data klasörünün içine monopoly_kapsamli_veri.pdf adıyla oluşturduğun birleştirilmiş PDF dosyasını koyduğundan emin ol (varsa eski PDF'i sil).

3- Veritabanını Yeniden Oluştur: VS Code terminalinde ((venv) aktifken) şu komutu çalıştır:

python create_database.py

4- Uygulamayı Başlat: Terminalde ((venv) aktifken) şu komutu çalıştır:

python app.py

5- Test Et: Tarayıcıdan http://localhost:5000 adresine giderek Monopoly soruları sor.
