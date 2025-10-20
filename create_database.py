import os
import shutil
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
try:
    from pypdf import PdfReader
except ImportError:
    print("Hata: 'pypdf' kütüphanesi bulunamadı.", file=sys.stderr)
    print("Lütfen yüklemek için 'pip install pypdf' komutunu çalıştırın.", file=sys.stderr)
    sys.exit(1)
import traceback # Hata ayıklama için

# PDF dosyasından metin çıkaran fonksiyon
def extract_text_from_pdf(pdf_path):
    """Verilen yoldaki PDF dosyasını okur ve metni çıkarır."""
    print(f"PDF dosyasından metin çıkarılıyor: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"Hata: Belirtilen yolda PDF dosyası bulunamadı: {pdf_path}", file=sys.stderr)
        return None
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        print(f"Toplam {len(reader.pages)} sayfa bulundu.")
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                page_text = page_text.replace("-\n", "").replace("- \n", "") # Tire ile bölünen kelimeler
                page_text = ' '.join(page_text.split()) # Çoklu boşlukları tek boşluğa indir
                full_text += page_text + "\n\n" # Sayfalar arasına çift satır boşluk
        print("PDF metin çıkarma işlemi tamamlandı.")
        return full_text.strip()
    except Exception as e:
        print(f"Hata: PDF dosyası okunurken veya işlenirken sorun oluştu: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

# Veritabanını oluşturan ana fonksiyon
def create_database():
    """PDF'i okur, işler ve Chroma veritabanını oluşturur."""
    print("Veritabanı oluşturma işlemi başlıyor...")
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Hata: GOOGLE_API_KEY bulunamadı. Lütfen .env dosyanızı kontrol edin.", file=sys.stderr)
        sys.exit(1)

    try:
        genai.configure(api_key=api_key)
        print("Google API Anahtarı başarıyla yüklendi.")
    except Exception as e:
        print(f"Hata: Google API yapılandırılamadı: {e}", file=sys.stderr)
        sys.exit(1)

    # <<<--- GÜNCELLENDİ: Okunacak dosya adı ---<<<
    data_path = "data/monopoly_kapsamli_veri.pdf"
    print(f"Veri dosyası olarak kullanılacak: {data_path}")

    if not os.path.exists("data"):
        print("Hata: 'data' klasörü bulunamadı.", file=sys.stderr)
        print(f"Lütfen 'data' klasörünü oluşturun ve '{os.path.basename(data_path)}' dosyasını içine koyun.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(data_path):
        print(f"Hata: Veri dosyası '{data_path}' bulunamadı.", file=sys.stderr)
        sys.exit(1)


    text_content = extract_text_from_pdf(data_path)
    if text_content is None or not text_content.strip():
        print("Hata: PDF'ten metin çıkarılamadı veya metin boş.", file=sys.stderr)
        sys.exit(1)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        is_separator_regex=False,
    )
    print("Metin parçalayıcı (Text Splitter) oluşturuldu.")

    chunks = text_splitter.split_text(text_content)
    documents = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]

    if not documents:
        print("Hata: Metin parçalara ayrılamadı.", file=sys.stderr)
        sys.exit(1)

    print(f"Toplam {len(documents)} adet metin parçası (document chunk) oluşturuldu.")

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
    print(f"Vektör veritabanı '{db_path}' klasörüne '{collection_name}' koleksiyonu ile kaydedilecek.")

    if os.path.exists(db_path):
        print(f"Mevcut '{db_path}' klasörü siliniyor...")
        try:
            shutil.rmtree(db_path)
            print(f"'{db_path}' klasörü başarıyla silindi.")
        except Exception as e:
            print(f"Uyarı: '{db_path}' klasörü silinirken hata: {e}", file=sys.stderr)

    try:
        print("Chroma veritabanı oluşturuluyor ve belgeler işleniyor...")
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory=db_path,
        )
        vectordb.persist()
        print(f"Veritabanı başarıyla oluşturuldu ve '{db_path}' klasörüne kaydedildi.")
        return vectordb
    except Exception as e:
        print(f"Hata: Chroma veritabanı oluşturulamadı: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    create_database()
