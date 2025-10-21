# Gerekli kütüphaneleri içe aktar
import os                           # Dosya/klasör yolları ve ortam değişkenleri için
import shutil                       # Klasör silme işlemi için
import sys                          # Sistem (hata mesajları, çıkış) işlemleri için
from dotenv import load_dotenv      # .env dosyasını okumak için
import google.generativeai as genai # Google AI (API yapılandırması)
from langchain.schema import Document # LangChain'in metin parçalarını temsil eden Document sınıfı
from langchain.text_splitter import RecursiveCharacterTextSplitter # Metni parçalara (chunk) ayırmak için
from langchain_community.vectorstores import Chroma             # Chroma veritabanı ile LangChain entegrasyonu
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Google embedding modeli için LangChain entegrasyonu
try:
    from pypdf import PdfReader     # PDF dosyalarını okumak için kütüphane
except ImportError:
    # Eğer pypdf kurulu değilse kullanıcıyı bilgilendir ve çık
    print("Hata: 'pypdf' kütüphanesi bulunamadı.", file=sys.stderr)
    print("Lütfen yüklemek için 'pip install pypdf' komutunu çalıştırın.", file=sys.stderr)
    sys.exit(1)
import traceback                    # Hata ayıklama için detaylı hata izi

# PDF dosyasından metin içeriğini çıkaran fonksiyon
def extract_text_from_pdf(pdf_path):
    """
    Verilen yoldaki PDF dosyasını okur, her sayfadaki metni çıkarır,
    basit temizlik yapar ve tüm metni tek bir string olarak birleştirir.
    Args:
        pdf_path (str): İşlenecek PDF dosyasının tam yolu.
    Returns:
        str: PDF'ten çıkarılmış ve birleştirilmiş metin içeriği.
             Hata durumunda veya dosya bulunamazsa None döner.
    """
    print(f"PDF dosyasından metin çıkarılıyor: {pdf_path}")
    # Fonksiyonun başında dosyanın var olup olmadığını kontrol et
    if not os.path.exists(pdf_path):
        print(f"Hata: Belirtilen yolda PDF dosyası bulunamadı: {pdf_path}", file=sys.stderr)
        return None
    try:
        # pypdf kütüphanesi ile PDF dosyasını aç
        reader = PdfReader(pdf_path)
        # Tüm metni depolamak için boş bir string başlat
        full_text = ""
        print(f"Toplam {len(reader.pages)} sayfa bulundu.")
        # PDF'in her sayfası için döngü başlat
        for i, page in enumerate(reader.pages):
            # Sayfadaki metni çıkarmayı dene
            page_text = page.extract_text()
            # Eğer sayfadan metin çıkarılabildiyse
            if page_text:
                # Metin temizleme adımları:
                # 1. Satır sonunda tire ile bölünen kelimeleri birleştir
                page_text = page_text.replace("-\n", "").replace("- \n", "")
                # 2. Birden fazla boşluğu veya satır başını tek boşluğa indir
                page_text = ' '.join(page_text.split())
                # Temizlenmiş sayfa metnini ana metne ekle, sayfalar arasına çift satır başı koy (paragraf ayrımı gibi)
                full_text += page_text + "\n\n"
        print("PDF metin çıkarma işlemi tamamlandı.")
        # Sonuç metninin başındaki/sonundaki olası boşlukları temizle ve döndür
        return full_text.strip()
    except Exception as e:
        # PDF okuma/işleme sırasında hata olursa logla ve None döndür
        print(f"Hata: PDF dosyası okunurken veya işlenirken sorun oluştu: {e}", file=sys.stderr)
        traceback.print_exc() # Hatanın tam detayını yazdır
        return None

# Vektör veritabanını oluşturan ana fonksiyon
def create_database():
    """
    PDF veri kaynağını okur, metni LangChain ile parçalara ayırır,
    Google embedding modeli ile vektörlere dönüştürür ve sonuçları
    kalıcı bir Chroma veritabanına kaydeder.
    """
    print("Veritabanı oluşturma işlemi başlıyor...")
    # .env dosyasındaki ortam değişkenlerini yükle
    load_dotenv()
    # GOOGLE_API_KEY değerini al
    api_key = os.getenv("GOOGLE_API_KEY")

    # API anahtarının varlığını kontrol et
    if not api_key:
        print("Hata: GOOGLE_API_KEY bulunamadı. Lütfen .env dosyanızı kontrol edin.", file=sys.stderr)
        sys.exit(1)

    # Google AI API'sini yapılandır
    try:
        genai.configure(api_key=api_key)
        print("Google API Anahtarı başarıyla yüklendi.")
    except Exception as e:
        print(f"Hata: Google API yapılandırılamadı: {e}", file=sys.stderr)
        sys.exit(1)

    # İşlenecek PDF dosyasının yolu
    data_path = "data/monopoly_kapsamli_veri.pdf"
    print(f"Veri dosyası olarak kullanılacak: {data_path}")

    # 'data' klasörünün varlığını kontrol et
    if not os.path.exists("data"):
        print("Hata: 'data' klasörü bulunamadı.", file=sys.stderr)
        print(f"Lütfen 'data' klasörünü oluşturun ve '{os.path.basename(data_path)}' dosyasını içine koyun.", file=sys.stderr)
        sys.exit(1)
    # PDF dosyasının varlığını kontrol et
    if not os.path.exists(data_path):
        print(f"Hata: Veri dosyası '{data_path}' bulunamadı.", file=sys.stderr)
        sys.exit(1)

    # PDF'ten metin içeriğini çıkar
    text_content = extract_text_from_pdf(data_path)
    # Metin çıkarma başarısız olduysa veya içerik boşsa işlemi durdur
    if text_content is None or not text_content.strip():
        print("Hata: PDF'ten metin çıkarılamadı veya metin boş. İşlem durduruldu.", file=sys.stderr)
        sys.exit(1)

    # Metni RAG için uygun parçalara (chunk) ayıracak splitter'ı ayarla
    text_splitter = RecursiveCharacterTextSplitter(
        # chunk_size: Her bir parçanın yaklaşık maksimum karakter sayısı. Deneyerek ayarlanabilir.
        chunk_size=1500,
        # chunk_overlap: Parçalar arasında anlam bütünlüğünü korumak için ortak karakter sayısı.
        chunk_overlap=200,
        # length_function: Parça boyutunu hesaplamak için kullanılacak fonksiyon (genellikle len yeterli).
        length_function=len,
        # separators: Metni bölmek için kullanılacak karakter dizileri (öncelik sırasına göre).
        # PDF'ten gelen metinlerde çift satır başı genellikle paragrafları ayırır.
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        # separators listesinin regex olup olmadığını belirtir (False daha hızlıdır).
        is_separator_regex=False,
    )
    print("Metin parçalayıcı (Text Splitter) oluşturuldu.")

    # PDF metnini splitter kullanarak parçalara ayır
    chunks = text_splitter.split_text(text_content)
    # Her bir metin parçasını LangChain'in Document formatına çevir ve boş parçaları filtrele
    documents = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]

    # Eğer hiç parça oluşturulamadıysa hata ver
    if not documents:
        print("Hata: Metin parçalara ayrılamadı. Splitter ayarlarını veya PDF içeriğini kontrol edin.", file=sys.stderr)
        sys.exit(1)

    print(f"Toplam {len(documents)} adet metin parçası (document chunk) oluşturuldu.")

    # Metin parçalarını vektörlere çevirecek embedding modelini ayarla
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=api_key
            # Veritabanı oluştururken task_type belirtmek genellikle performansı artırabilir:
            # task_type="RETRIEVAL_DOCUMENT"
        )
        print("Google Embedding modeli başarıyla ayarlandı.")
    except Exception as e:
        print(f"Hata: Google Embedding modeli ayarlanamadı: {e}", file=sys.stderr)
        sys.exit(1)

    # Vektör veritabanının kaydedileceği klasör
    db_path = "./chroma_db"
    # Veritabanı içindeki koleksiyonun (tablo gibi düşünülebilir) adı
    collection_name = "gaih_monopoly_comprehensive" # app.py'deki ile aynı olmalı
    print(f"Vektör veritabanı '{db_path}' klasörüne '{collection_name}' koleksiyonu ile kaydedilecek.")

    # Eğer daha önceden oluşturulmuş bir veritabanı klasörü varsa, onu sil (temiz kurulum için)
    if os.path.exists(db_path):
        print(f"Mevcut '{db_path}' klasörü siliniyor...")
        try:
            shutil.rmtree(db_path) # Klasörü ve içindekileri sil
            print(f"'{db_path}' klasörü başarıyla silindi.")
        except Exception as e:
            # Silme işlemi başarısız olursa sadece uyar, devam etmeyi dene
            print(f"Uyarı: '{db_path}' klasörü silinirken hata: {e}", file=sys.stderr)

    # Chroma veritabanını oluştur ve belgeleri ekle
    try:
        print("Chroma veritabanı oluşturuluyor ve belgeler işleniyor (Bu işlem biraz zaman alabilir)...")
        # Chroma.from_documents: Verilen belgeleri (Documents listesi) alır,
        # embedding_function ile vektörlere çevirir ve belirtilen klasöre (`persist_directory`)
        # kalıcı olarak kaydeder (`collection_name` ile).
        vectordb = Chroma.from_documents(
            documents=documents,             # Vektöre çevrilecek metin parçaları
            embedding=embedding_function,    # Vektöre çevirme işlemi için fonksiyon
            collection_name=collection_name, # Koleksiyon adı
            persist_directory=db_path,       # Kaydedileceği klasör
        )
        # Verilerin diske yazıldığından emin olmak için persist çağrılabilir (genellikle from_documents sonrası gerekmez)
        vectordb.persist()
        print(f"Veritabanı başarıyla oluşturuldu ve '{db_path}' klasörüne kaydedildi.")
        # Oluşturulan veritabanı nesnesini döndür (opsiyonel)
        return vectordb
    except Exception as e:
        # Veritabanı oluşturma sırasında hata olursa logla ve programı durdur
        print(f"Hata: Chroma veritabanı oluşturulamadı: {e}", file=sys.stderr)
        traceback.print_exc() # Hatanın detayını yazdır
        sys.exit(1)

# Bu script doğrudan çalıştırıldığında (python create_database.py) create_database() fonksiyonunu çağır
if __name__ == "__main__":
    create_database()
