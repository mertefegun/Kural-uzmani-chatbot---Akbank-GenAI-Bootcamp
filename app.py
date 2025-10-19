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
# create_database fonksiyonunu import etmiyoruz, sadece varlığını kontrol edeceğiz.
import shutil

# ----- Yapılandırma ve Kurulum -----
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

try:
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config=genai.types.GenerationConfig(temperature=0.3)
    )
    print("Google Gemini modeli ('gemini-2.0-flash') başarıyla yüklendi.")
except Exception as e:
    print(f"Hata: Google Gemini modeli yüklenemedi: {e}", file=sys.stderr)
    sys.exit(1)

# ----- Veritabanı Fonksiyonları -----
def load_database():
    """Chroma vektör veritabanını yükler veya yoksa hata verir."""
    print("Kural veritabanı yükleniyor...")
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=api_key
        )
        print("Google Embedding modeli ('text-embedding-004') başarıyla ayarlandı.")
    except Exception as e:
        print(f"Hata: Google Embedding modeli ayarlanamadı: {e}", file=sys.stderr)
        sys.exit(1)

    db_path = "./chroma_db"
    collection_name = "gaih_tmars_rules" # create_database.py ile aynı

    # Veritabanı klasörü var mı ve içinde veri var mı kontrol et
    # Basit kontrol: sqlite3 dosyası var mı?
    db_file_exists = os.path.exists(os.path.join(db_path, "chroma.sqlite3"))

    if not os.path.exists(db_path) or not db_file_exists:
        print(f"Hata: Veritabanı '{db_path}' klasöründe bulunamadı veya geçerli değil.", file=sys.stderr)
        print("Lütfen önce 'python create_database.py' komutunu çalıştırarak veritabanını oluşturun.", file=sys.stderr)
        sys.exit(1) # Veritabanı olmadan uygulama başlayamaz

    try:
        print(f"Mevcut veritabanı '{db_path}' klasöründen '{collection_name}' koleksiyonu yükleniyor...")
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
        print("Veritabanı başarıyla yüklendi.")
        # Küçük bir test sorgusu ile doğrula (opsiyonel ama faydalı)
        # print("Veritabanı test ediliyor...")
        # _ = vectordb.similarity_search("test", k=1)
        # print("Veritabanı testi başarılı.")
        return vectordb
    except Exception as e:
        print(f"Hata: Vektör veritabanı yüklenemedi: {e}", file=sys.stderr)
        print("Veritabanı dosyaları bozulmuş olabilir veya koleksiyon adı yanlış olabilir.", file=sys.stderr)
        print(f"'{db_path}' klasörünü silip 'python create_database.py' komutunu tekrar çalıştırmayı deneyin.", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

# ----- RAG ve AI Fonksiyonları -----
def get_answer(query, vectordb, top_k=5):
    """Kullanıcının kural sorusuna RAG ile cevap üretir."""
    print(f"Alınan soru: '{query}'")
    print(f"Veritabanında benzerlik araması yapılıyor (top_k={top_k})...")
    context = "Kural veritabanı aranırken bir hata oluştu." # Varsayılan hata mesajı
    try:
        # Daha iyi sonuçlar için MMR (Max Marginal Relevance) retriever kullanmayı deneyebiliriz
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={'k': top_k, 'fetch_k': 15} # fetch_k > k olmalı
        )
        retrieved_docs = retriever.invoke(query) # invoke kullanımı güncel LangChain'de daha yaygın

        # retrieved_docs = vectordb.similarity_search(query, k=top_k) # Alternatif: Basit benzerlik

        if not retrieved_docs:
             print("Uyarı: Veritabanından ilgili kural bulunamadı.")
             context = "İlgili kural bulunamadı." # LLM'e bilgi ver
        else:
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            print(f"{len(retrieved_docs)} adet ilgili kural parçası bulundu (MMR ile).")

    except Exception as e:
        print(f"Hata: Veritabanı araması sırasında sorun oluştu: {e}", file=sys.stderr)
        traceback.print_exc()
        # Context zaten hata mesajı olarak ayarlı

    # Kural Uzmanı persona'sına uygun güncellenmiş prompt
    prompt = f"""
