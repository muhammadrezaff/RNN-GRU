import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Konfigurasi halaman menjadi full screen (wide)
st.set_page_config(page_title="Skripsi", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
    
    * {
        font-family: 'Playfair Display', sans-serif;
    }

    /* Heading (h1, h2, h3) warna merah */
    h1, h2, h3 {
        color: navy;
    }

    /* Teks biasa warna navy */
    p, div, span, li {
        color: #2F2F2F;
    }
    .stApp {
        background-color:#ffffff;  
    }
    </style>
    """, unsafe_allow_html=True)

# Custom CSS untuk menghilangkan padding/margin dan menyesuaikan konten agar full screen
st.markdown(
    """
    <style>
    /* Hilangkan margin dan padding default Streamlit */
    .main .block-container {
        padding: 0; /* Hilangkan padding */
        margin: 0; /* Hilangkan margin */
    }
    /* Penyesuaian header dengan logo */
    .header-container {
        display: flex;
        justify-content: flex-start; /* Logo di kiri atas */
        align-items: center;
        padding: 10px;
        background-color: #ffffff;
    }
    .logo {
        height: 50px; /* Tinggi logo */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header dengan logo di kiri dan tulisan "Seminar Proposal" di kanan
st.markdown(
    """
    <style>
    .header-container {
        display: flex;
        justify-content: space-between; /* Membuat ruang antara logo dan tulisan */
        align-items: center;
        padding: 5px;
        background-color:#ffffff; 
    }
    .seminar-text {
        font-size: 20px;
        font-weight: bold;
        color: #333;
    }
    .logo-container {
        display: flex;
        align-items: center;
    }
    .logo {
        height: 50px; /* Menyesuaikan ukuran logo */
    }
    .css-1d391kg {  /* Class untuk menu navigasi Streamlit */
        font-size: 10px;  /* Menyesuaikan ukuran font */
    }
    </style>
    <div class="header-container">
        <!-- Logo di sebelah kiri -->
        <div class="logo-container">
            <img src="LOGO-UNIMUS-UNGGUL-1.png" class="logo" alt="Logo">
        </div>
        <!-- Teks Seminar Hasil di sebelah kanan -->
        <span class="seminar-text">Sidang Skripsi</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigasi Utama
selected_main = option_menu(
    menu_title=None,  # Tidak ada judul menu utama
    options=["Cover", "Introduction", "Data Source", "Preprocessing", "Modeling", "Hasil", "Kesimpulan"],  # Menu utama
    icons=["book", "list-task", "database", "gear", "bar-chart-line", "clipboard-check", "check-circle"],  # Ikon untuk setiap menu
    menu_icon="cast",  # Ikon untuk menu utama
    default_index=0,  # Default ke "Cover"
    orientation="horizontal",  # Membuat menu horizontal
)

# Konten berdasarkan pilihan di menu utama
if selected_main == "Cover":
    # CSS untuk memusatkan judul dan mengatur barisan untuk informasi lainnya
    st.markdown(
        """
        <style>
        .cover-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            height: 80vh; /* Tinggi penuh halaman */
            background-color: #f9f9f9; /* Warna latar belakang lembut */
            padding: 10px;
        }
        .cover-title {
            font-size: 20px;
            font-weight: bold;
            margin-top: 5px;
            margin-bottom: 30px;
            margin-left: 5cm;
            margin-right: 5cm;
        }
        .cover-text {
            font-size: 12px;
            line-height: 1.15;
        }
        .info-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
            margin-bottom: 20px;
        }
        .info-top {
            display: flex;
            justify-content: center;
            width: 100%;
            margin-bottom: 1cm;
        }
        .info-left,
        .info-right {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-left: 3cm;
            margin-right: 3cm;
        }
        .info-item {
            margin-bottom: 10px; /* Jarak antar baris */
            align-items: left;
        }
        .info-bottom {
            display: flex;
            justify-content: space-between;
            width: 100%;
        }
        .logoo {
        height: 100px; 
        }
        .style {
            </style>
        <div class="cover-container">
        <!-- Logo di sebelah kiri -->
        <div class="logo-container">
            <img src= "Logo Unimus.png" class="logoo" alt="logo.bawah" style="margin-top:100px;">
        </div>
        </style>
        <div class="cover-container">
            <div class="cover-title">PREDIKSI MAGNITUDO GEMPA BUMI DI INDONESIA MENGGUNAKAN METODE <i>RECURRENT NEURAL NETWORK GATED RECURRENT UNIT</i> (RNN-GRU)</div>
            <div class="info-container">
                <!-- Nama dan NIM di tengah -->
                <div class="info-top">
                    <div class="info-left">
                        <div class="info-item"><strong>Nama:</strong> Muhammad Reza Fadillah</div>
                        <div class="info-item"><strong>NIM:</strong> B2A021055</div>
                    </div>
                </div>
                <!-- Dosen Pembimbing kiri dan Dosen Penguji kanan -->
                <div class="info-bottom">
                    <div class="info-left">
                        <div class="info-item"><strong>Pembimbing 1:</strong> Tiani Wahyu Utami, M.Si </div>
                        <div class="info-item"><strong>Pembimbing 2:</strong> Dannu Purwanto, S.T., M.Kom </div>
                    </div>
                    <div class="info-right">
                        <div class="info-item"><strong>Penguji 1:</strong> M. Al Haris, M.Si</div>
                        <div class="info-item"><strong>Penguji 2:</strong> Prizka Rismawati Arum, S.Si., M.Stat</div>
                    </div>
                </div>
            </div>
            <div class="cover-text">
                <p>PROGRAM STUDI STATISTIKA</p>
                <p>FAKULTAS SAINS DAN TEKNOLOGI PERTANIAN</p>
                <p>UNIVERSITAS MUHAMMADIYAH SEMARANG</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif selected_main == "Introduction":
    st.title("Introduction")

    # Sub-navigasi dalam Introduction
    sub_intro = st.radio(
        "Pilih BAB:",
        ["BAB 1 Pendahuluan", "BAB 2 Tinjauan Pustaka"]
    )

    if sub_intro == "BAB 1 Pendahuluan":
        st.header("BAB 1 Pendahuluan")

        # Sub-sub-slide untuk BAB 1
        sub_bab1 = st.selectbox(
            "Pilih Sub-Bab Pendahuluan:",
            ["Latar Belakang", "Penelitian Sebelumnya", "Rumusan Masalah", "Tujuan Penelitian", "Manfaat Penelitian", "Batasan Masalah"]
        )
        
        if sub_bab1 == "Latar Belakang":
            st.subheader("Latar Belakang")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>1. Indonesia sebagai negara yang terletak di jalur Ring of Fire memiliki tingkat kerentanan yang sangat tinggi terhadap gempa bumi akibat interaksi lempeng tektonik aktif. Frekuensi gempa yang tinggi dan potensi dampaknya terhadap keselamatan manusia dan infrastruktur menjadikan prediksi gempa bumi sebagai kebutuhan mendesak untuk mendukung upaya mitigasi bencana.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>2. Indonesia menempati peringkat tinggi dalam frekuensi gempa bumi secara global, bahkan tercatat sebagai negara dengan jumlah gempa terbanyak pada tahun 2023. Hal ini menunjukkan perlunya penelitian yang lebih mendalam untuk mengantisipasi dan memitigasi risiko gempa.</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("Frekuensi_Gempa_Bumi-transformed.png", caption="Diagram jumlah gempa bumi terbanyak di dunia tahun 2023", width=500)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>3. Penelitian menggunakan metode Recurrent Neural Network Gated Recurrent Unit (RNN-GRU) untuk menganalisis pola data deret waktu dan memprediksi magnitudo gempa bumi. Data magnitudo gempa, yang bersifat sekuensial dan memiliki pola temporal, sangat sesuai untuk diproses menggunakan metode RNN-GRU karena kemampuannya dalam menangkap pola data berurutan dengan akurasi tinggi.</p>", unsafe_allow_html=True)
        elif sub_bab1 == "Penelitian Sebelumnya":
            st.subheader("Penelitian Sebelumnya")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>1. <strong>Judul</strong>: Penerapan Metode Resilient Backpropagation (RPROP) untuk Prediksi Aktivitas Gempa Bumi Berdasarkan Skala Magnitudo<br><strong>Peneliti</strong>: Widyaningrum & Ramadhani, 2024<br> <strong>Hasil</strong>: Prediksi gempa bumi menggunakan metode Jaringan Saraf Tiruan Resilient Backpropagation dengan model arsitektur 6-10-1 dan learning rate 0,6 menunjukkan tingkat akurasi RMSE 0,04778 dan MAE 0,03509.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>2. <strong>Judul</strong>: Prediksi Kekuatan Gempa Bumi Indonesia Berdasarkan Nilai Magnitudo Menggunakan Neural Network<br><strong>Peneliti</strong>: Somantri, 2021<br> <strong>Hasil</strong>: Diperoleh nilai root mean square error (RMSE) sebesar 0,718 untuk model neural network yang diusulkan. Setelah dilakukan optimasi dengan algoritma genetika (GA), nilai RMSE berhasil ditingkatkan menjadi 0,708.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>3. <strong>Judul</strong>: Hybrid Deep Learning Model Using Recurrent Neural Network and Gated Recurrent Unit for Heart Disease Prediction<br><strong>Peneliti</strong>: Krishnan et al., 2021<br> <strong>Hasil</strong>: Model yang diusulkan ini menghasilkan akurasi luar biasa sebesar 98,6876%, yang merupakan yang tertinggi di antara model RNN yang ada.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>4. <strong>Judul</strong>: Predicting Machine Failure Using Recurrent Neural Network-Gated Recurrent Unit (RNN-GRU) Through Time Series Data<br><strong>Peneliti</strong>: Zainuddin et al., 2021<br> <strong>Hasil</strong>: penelitian ini menawarkan algoritma pembelajaran mendalam yang disebut Recurrent Neural Network-Gated Recurrent Unit (RNN-GRU) untuk meramalkan kondisi mesin yang menghasilkan data deret waktu di sektor minyak dan gas. RNN-GRU adalah struktur yang sederhana dengan akurasi 87% pada prediksi.</p>", unsafe_allow_html=True)
        elif sub_bab1 == "Rumusan Masalah":
            st.subheader("Rumusan Masalah")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>1. Bagaimana menentukan model prediksi magnitudo gempa bumi di Indonesia menggunakan metode Recurrent Neural Network Gated Recurrent Unit (RNN-GRU)?</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>2. Bagaimana akurasi model Recurrent Neural Network Gated Recurrent Unit (RNN-GRU) dalam memprediksi magnitudo gempa bumi di Indonesia?</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>3. Bagaimana hasil prediksi magnitudo gempa bumi di Indonesia menggunakan metode Recurrent Neural Network Gated Recurrent Unit (RNN-GRU)?</p>", unsafe_allow_html=True)
        elif sub_bab1 == "Tujuan Penelitian":
            st.subheader("Tujuan Penelitian")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>1. Mengetahui model prediksi magnitudo gempa bumi di Indonesia menggunakan metode Recurrent Neural Network Gated Recurrent Unit (RNN-GRU).</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>2. Mendapatkan akurasi model Recurrent Neural Network Gated Recurrent Unit (RNN-GRU) dalam memprediksi magnitudo gempa bumi di Indonesia.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>3. Mendapatkan hasil prediksi magnitudo gempa bumi di Indonesia menggunakan metode Recurrent Neural Network Gated Recurrent Unit (RNN-GRU).</p>", unsafe_allow_html=True)
        elif sub_bab1 == "Manfaat Penelitian":
            st.subheader("Manfaat Penelitian")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>1. Manfaat Teoritis</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Bagi Pembaca:</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>- Meningkatkan pemahaman pembaca mengenai magnitudo gempa bumi.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>- Berperan dalam pengembangan ilmu pengetahuan mengenai metode peramalan, sehingga dapat dijadikan referensi atau sumber informasi bagi peneliti selanjutnya terkait metode RNN-GRU maupun metode prediksi lainnya.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Bagi Universitas Muhammadiyah Semarang:</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>-	Menjadi referensi yang bermanfaat dalam mengembangkan pemahaman mengenai peramalan magnitudo gempa bumi dengan metode RNN-GRU.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>2. Manfaat Praktis</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Bagi Pihak Terkait:", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>-	Memberikan pengetahuan mengenai gempa bumi serta analisis prediksi magnitudo gempa yang dapat mendukung proses pengambilan keputusan. Bagi pemerintah, penelitian ini diharapkan dapat menjadi pertimbangan dalam mengambil langkah-langkah untuk mengurangi risiko akibat gempa bumi.</p>", unsafe_allow_html=True)
        elif sub_bab1 == "Batasan Masalah":
            st.subheader("Batasan Masalah")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>1. Data yang digunakan merupakan data harian magnitudo gempa bumi di Indonesia dari tanggal 1 Januari 2020 hingga 30 September 2024.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>2. Pembagian data training dan testing dilakukan dalam tiga skenario, yaitu 80:20, 75:25, dan 70:30. Dengan demikian, data training digunakan sebesar 80%, 75%, dan 70%, sementara data testing sebesar 20%, 25%, dan 30%.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>3. Prediksi magnitudo gempa bumi dilakukan menggunakan pendekatan Recurrent Neural Network Gated Recurrent Unit dengan optimasi Adaptive Moment Estimation (ADAM).</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>4. Analisis data menggunakan bahasa pemrograman python dengan framework streamlit.</p>", unsafe_allow_html=True)

    elif sub_intro == "BAB 2 Tinjauan Pustaka":
        st.header("BAB 2 Tinjauan Pustaka")

        # Sub-sub-slide untuk BAB 2
        sub_bab2 = st.selectbox(
            "Pilih Sub-Bab Tinjauan Pustaka:",
            ["Statistika Deskriptif", "Data Deret Waktu (Time Series)", "Peramalan", "Prediksi",
             "Artificial Neural Network (ANN)", "Recurrent Neural Network (RNN)", "Gated Recurrent Unit (GRU)",
             "Hyperparameter", "Normalisasi Data", "Evaluasi Model", "Denormalisasi Data", "Gempa Bumi"]
        )
        
        if sub_bab2 == "Statistika Deskriptif":
            st.subheader("Statistika Deskriptif")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Statistika deskriptif merupakan metode yang mencakup cara mengumpulkan data angka, menyusun angka-angka tersebut dalam bentuk tabel, menggambarkannya, mengolah dan menganalisis data tersebut, serta memberikan interpretasi melalui penafsiran. Dengan kata lain, metode ini menjelaskan langkah-langkah untuk mencatat data angka, menyajikannya dalam bentuk grafik, kemudian menganalisis dan menafsirkannya untuk menarik suatu kesimpulan. Pada statistik deskriptif, data dapat disajikan dalam bentuk tabel atau diagram. Biasanya, data dijelaskan menggunakan ukuran seperti mean, median, modus, dan standar deviasi.</p>", unsafe_allow_html=True)
        elif sub_bab2 == "Data Deret Waktu (Time Series)":
            st.subheader("Data Deret Waktu (Time Series)")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Data deret waktu atau yang bisa disebut dengan time series data merupakan kumpulan data yang dikumpulkan dari rentang waktu tertentu untuk melihat bagaimana perkembangan data tersebut. Faktor yang mempengaruhi data deret waktu adalah tipe atau pola dari data tersebut. Terdapat empat macam tipe pola deret waktu, yaitu:</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>a. Pola horizontal merupakan pola yang terjadi ketika data memiliki nilai rata – rata fluktuasi yang sama.</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("Pola Horizontal.jpg", caption="Pola Horizontal", width=300)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>b. Pola musiman merupakan pola yang terjadi ketika data memiliki nilai yang terjadi secara berulang pada suatu periodik waktu tertentu.</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("Pola Musiman.jpg", caption="Pola Musiman", width=300)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>c. Pola tren merupakan pola yang terjadi secara kontinu atau bergerak stabil dan terdapat arah perkembangan secara umum (naik atau turun).</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("Pola tren.jpg", caption="Pola Tren", width=300)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>d. Pola siklis merupakan pola yang terjadi ketika data memiliki nilai yang terjadi secara berulang setelah jangka waktu tertentu, namun tidak secara periodik.</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("Pola Siklis.jpg", caption="Pola Siklis", width=300)
        elif sub_bab2 == "Peramalan":
            st.subheader("Peramalan")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Peramalan (forecasting) merupakan teknik analisis yang menggunakan pendekatan kualitatif maupun kuantitatif untuk memprediksi kejadian di masa depan. Proses ini didasarkan pada data historis guna mengurangi dampaknya. Peramalan umumnya terbagi menjadi tiga kategori, yaitu jangka pendek, jangka menengah, dan jangka panjang. Peramalan jangka pendek digunakan untuk memprediksi kejadian dalam periode harian, mingguan, hingga beberapa bulan mendatang. Peramalan jangka menengah melibatkan analisis data untuk memproyeksikan kejadian selama satu hingga dua tahun ke depan. Sementara itu, peramalan jangka panjang digunakan untuk memperkirakan kejadian yang berlangsung lebih dari dua tahun ke depan.</p>", unsafe_allow_html=True)
        elif sub_bab2 == "Prediksi":
            st.subheader("Prediksi")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Secara umum, prediksi adalah metode untuk menggambarkan kejadian di masa mendatang. Menurut Kamus Besar Bahasa Indonesia (KBBI), prediksi didefinisikan sebagai hasil dari aktivitas meramal, meramalkan, atau memperkirakan nilai di masa depan berdasarkan data masa lalu. Kualitas sebuah prediksi tidak hanya dipengaruhi oleh metode yang digunakan, tetapi juga oleh keakuratan informasi yang menjadi dasar analisis. Jika informasi yang digunakan kurang meyakinkan, maka hasil peramalan yang dihasilkan juga cenderung sulit dipercaya tingkat akurasinya.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Keberhasilan sebuah prediksi bergantung pada beberapa faktor, di antaranya:</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>a. Pemahaman terhadap teknik pengumpulan data masa lalu, baik yang bersifat kuantitatif maupun kualitatif.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>b. Pemilihan teknik dan metode yang tepat serta sesuai dengan pola data yang diperoleh.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Gambaran tentang perkembangan di masa mendatang diperoleh melalui analisis data yang dikumpulkan dari hasil penelitian sebelumnya. Perkembangan di masa depan merupakan prediksi dari apa yang mungkin terjadi, sehingga ramalan selalu menjadi bagian penting dalam penelitian. Meskipun ramalan ramalan sangat diutamakan, perlu disadari bahwa setiap ramalan selalu mengandung unsur kesalahan. Oleh karena itu, upaya untuk meminimalkan kesalahan dalam peramalan menjadi hal yang penting.</p>", unsafe_allow_html=True)
        elif sub_bab2 == "Artificial Neural Network (ANN)":
            st.subheader("Artificial Neural Network (ANN)")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Artificial Neural Network (ANN) atau bisa disebut Neural Network (NN) adalah algoritma yang umumnya sangat efektif dalam mengenali pola. Algoritma ini bekerja dengan meniru cara kerja jaringan saraf manusia, yang mampu menyimpan berbagai informasi dan membentuk tujuan tertentu dalam sistem saraf tersebut. Artificial Neural Network mengadaptasi mekanisme kerja otak manusia dalam mempelajari sesuatu, yaitu dengan memodifikasi sinapsis berdasarkan stimulasi yang diterima dari lingkungan.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Kecerdasan buatan (Artificial Intelligence/AI) semakin populer dan mengalami perkembangan yang sangat pesat. Perkembangan ini dimulai dengan hadirnya metode deep learning yang diperkenalkan dari penemuan jaringan saraf tiruan oleh Warren McCulloch pada tahun 1943. Sementara itu, deep learning sendiri dikembangkan sekitar 40 tahun kemudian oleh Geoffrey Hinton. Sejak saat itu, dengan kemajuan penelitian di bidang komputasi tingkat tinggi (super computing) dan pemrosesan data yang semakin cepat, pengembangan deep learning terus mengalami peningkatan. Prinsip utama deep learning adalah memperbarui bobot menggunakan metode gradien selama proses backpropagation untuk menghasilkan model yang sesuai dengan konteks input dan output tertentu.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Keunggulan ANN dibandingkan metode prediksi lainnya terletak pada kemampuannya untuk menerapkan berbagai algoritma guna meminimalkan kesalahan dan menghasilkan nilai prediksi yang mendekati kondisi aktual. Secara sederhana, ANN merupakan alat pemodelan data statistik nonlinier yang dapat memodelkan hubungan kompleks antara input dan output sekaligus mengidentifikasi pola dalam data. Dalam bentuk paling dasar, ANN mampu menangani data dalam jumlah besar (big data) dan mengidentifikasi pola-pola tertentu dari data tersebut.</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("Arsitektur ANN.png", caption="Arsitektur Artificial Neural Network (ANN)", width=600)
        elif sub_bab2 == "Recurrent Neural Network (RNN)":
            st.subheader("Recurrent Neural Network (RNN)")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Recurrent Neural Network (RNN) merupakan sejenis jaringan syaraf tiruan yang dapat memproses bahasa alami, prediksi time series dan melihat korelasi tersembunyi pada aplikasi pengenalan suara pada suatu data. Pada umumnya neural network memiliki anggapan bahwa input dan output pada suatu data saling bebas. Untuk menyelesaikan masalah pemodelan urutan RNN baik digunakan, karena RNN memiliki koneksi yang berulang dan jejak informasi sebelumnya dan informasi input. RNN umumnya diterapkan untuk mengatasi permasalahan yang melibatkan data deret waktu, dirancang untuk memproses data yang berurutan atau bersambung dan merupakan variasi dari Artificial Neural Network.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Terdapat empat jenis Recurrent Neural Network (RNN) yang digunakan berdasarkan fungsinya, yaitu:</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>a. One to One: Jenis RNN ini umumnya digunakan untuk menyelesaikan masalah sederhana dalam pembelajaran mesin. Dikenal juga sebagai jaringan saraf vanilla, jenis ini hanya mampu menerima satu input dan menghasilkan satu output.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>b. One to Many: Tipe RNN ini mampu menghasilkan beberapa output dari satu input. Biasanya, tipe ini diaplikasikan pada tugas seperti pembuatan caption untuk gambar.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>c. Many to One: RNN jenis ini menerima banyak input dan menghasilkan satu output. Biasanya digunakan untuk analisis sentimen, seperti mengklasifikasikan teks berdasarkan emosi negatif, netral, atau positif.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>d. Many to Many: Jenis RNN ini dapat memproses banyak input dan menghasilkan beberapa output dalam urutan tertentu. Aplikasi umumnya adalah pada mesin penerjemah bahasa.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Recurrent Neural Network (RNN) merupakan algoritma yang digunakan untuk data sekuensial, RNN terus berkembang dan banyak modifikasi untuk menyempurnakannya. Cara kerja RNN adalah dengan memproses input yang diberikan dengan berbagai informasi yang pernah diperoleh sebelumnya. RNN tidak hanya menjalankan proses secara linear namun juga terjadi perulangan yang membentuk siklus pada arsitektur RNN pada setiap neuron dan layer.</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("Struktur RNN.jpg", caption="Struktur Reccurent Neural Network", width=500)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("Rumus RNN.jpg",  width=400)
            st.image("Ket Rumus RNN.jpg",  width=300)
        elif sub_bab2 == "Gated Recurrent Unit (GRU)":
            st.subheader("Gated Recurrent Unit (GRU)")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Gated Recurrent Unit (GRU) adalah metode yang diperkenalkan oleh Kyunghun Cho dan teman-temannya, yang merupakan varian dari Recurrent Neural Network (RNN). Metode ini memungkinkan setiap unit recurrent untuk secara adaptif menangkap dependensi pada berbagai skala waktu. GRU dikembangkan untuk mengatasi masalah vanishing gradient yang sering ditemui pada RNN. Dibandingkan dengan varian lain dari RNN, seperti Long Short-Term Memory (LSTM), GRU memiliki arsitektur yang lebih sederhana karena hanya memiliki dua pintu, yaitu update gate dan reset gate.</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("Struktur GRU.jpg", caption="Struktur Gated Recurrent Unit", width=500)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("Rumus GRU.jpg",  width=500)
            st.image("Keterangan Rumus GRU.png",  width=400)
        
        elif sub_bab2 == "Hyperparameter":
            st.subheader("Hyperparameter")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Menentukan hyperparameter yang tepat adalah kunci keberhasilan model RNN-GRU, karena setiap pengaturan dapat secara langsung mempengaruhi kinerja model dalam memahami pola data dan menghasilkan prediksi yang akurat.</p>", unsafe_allow_html=True)

            # Add sub-selection for Hyperparameters
            sub_hyperparameter = st.selectbox(
            "Pilih Sub-Bagian Hyperparameter:",
            ["Candidate Hidden State", "Final Hidden State", "Batch Size", "Epoch", "Optimizer", 
             "Drop Out", "Dense Layer", "Loss Function", "Activation Function"]
            )

            if sub_hyperparameter == "Candidate Hidden State":
                st.subheader("Candidate Hidden State")
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Candidate hidden state merupakan penggabungan informasi dari hidden state sebelumnya dengan input yang ada.</p>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image("Candidate Hidden State.jpg", caption="(a) struktur candidate hidden state pertama (b) Struktur candidate hidden state kedua", width=700)
            elif sub_hyperparameter == "Final Hidden State":
                st.subheader("Final Hidden State")
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Final hidden state merupakan output yang akan diteruskan pada timestep berikutnya.</p>", unsafe_allow_html=True)
                st.image("Final Hidden State.jpg", caption="(a) Struktur final hidden state pertama (b) Struktur final hidden state kedua (c) Struktur final hidden state ketiga", width=900)
            elif sub_hyperparameter == "Batch Size":
                st.subheader("Batch Size")
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Batch size adalah jumlah kelompok data sampel yang digunakan dalam pelatihan model. Sampel-sampel tersebut akan diproses oleh model agar dapat memahami, mengingat, dan mempelajari pola data melalui proses iterasi. Batch size mengacu pada jumlah sampel yang akan diproses dalam satu iterasi. Setiap kali model memproses data sebanyak ukuran batch yang ditentukan, parameter model akan diperbarui berdasarkan hasil yang diperoleh dari proses tersebut. Namun, memilih ukuran batch yang tepat memerlukan pertimbangan yang matang, karena ukuran batch yang terlalu kecil dapat memperlambat konvergensi model, sementara ukuran batch yang terlalu besar berisiko membuat model kurang adaptif terhadap variasi data. Oleh karena itu, pemilihan ukuran batch perlu disesuaikan dengan karakteristik data, kapasitas perangkat keras, dan tujuan pelatihan.</p>", unsafe_allow_html=True)
            elif sub_hyperparameter == "Epoch":
                st.subheader("Epoch")
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Epoch adalah nilai yang menunjukkan banyaknya proses pelatihan model untuk mempelajari dan menganalisis seluruh sampel data. Dalam penerapannya, satu epoch menggambarkan satu siklus penuh di mana seluruh data pelatihan digunakan untuk memperbarui parameter model internal. Selama proses ini, dataset dibagi menjadi beberapa batch, dan setiap batch diproses secara bergantian. Semakin banyak epoch yang digunakan, semakin sering model akan melihat seluruh dataset, memungkinkan pembelajaran algoritma untuk mengurangi kesalahan secara bertahap. Biasanya, jumlah epoch yang diterapkan cukup besar, sering kali mencapai ratusan hingga ribuan, guna meningkatkan akurasi dan kualitas model yang dihasilkan.</p>", unsafe_allow_html=True)
            elif sub_hyperparameter == "Optimizer":
                st.subheader("Optimizer")
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Optimizer merupakan algoritma atau metode dalam kecerdasan buatan yang berperan penting dalam menyesuaikan parameter seperti bobot dan bias, dengan tujuan mengurangi fungsi kerugian atau meningkatkan efisiensi produksi, sehingga memfasilitasi perubahan nilai bobot dan penyesuaian laju pembelajaran dalam jaringan saraf agar kerugian dapat diminimalkan. Oleh karena itu, pemilihan optimizer yang tepat dapat memberikan dampak signifikan terhadap akurasi dan efisiensi model. Pada penelitian ini menggunakan Adaptive Moment Estimation (ADAM) yaitu menghitung estimasi pertama (mean) dan kedua (uncentered variance) dari gradien, dan menggunakan kedua estimasi tersebut untuk mengatur learning rate untuk setiap parameter. Rumusnya meliputi:</p>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image("Rumus ADAM.jpg",  width=500)
                st.image("Ket Rumus ADAM.jpg",  width=700)
            elif sub_hyperparameter == "Drop Out":
                st.subheader("Drop Out")
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Drop out merupakan teknik regularisasi untuk menurunkan overfitting pada neural network yang menghalangi ko-adaptasi kompleks pada data latih. Dalam dropout layer dilakukan proses dropout yaitu menghilangkan koneksi secara acak pada unit neural network pada proses pelatihan.</p>", unsafe_allow_html=True)
            elif sub_hyperparameter == "Dense Layer":
                st.subheader("Dense Layer")
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Dense layer adalah komponen dalam jaringan saraf tradisional yang berfungsi untuk melakukan klasifikasi sesuai dengan kelas pada output. Jumlah lapisan padat mempengaruhi tingkat kompleksitas model yang dibangun, semakin banyak lapisan padat, semakin kompleks model tersebut, sehingga meningkatkan kemampuan pembelajaran model terhadap data yang lebih rumit. Namun, penggunaan lapisan padat yang terlalu sedikit dapat menyebabkan underfitting, di mana model tidak mampu memahami data latihan dengan baik. Sebaliknya, lapisan yang terlalu padat dapat mengakibatkan overfitting, karena model mempelajari data latihan secara berlebihan, termasuk noise yang seharusnya diabaikan. Selain itu, penambahan lapisan padat juga berdampak pada peningkatan waktu pelatihan dan penggunaan memori, karena model harus mempelajari lebih banyak parameter dalam jaringannya.</p>", unsafe_allow_html=True)
            elif sub_hyperparameter == "Loss Function":
                st.subheader("Loss Function")
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Loss function yang digunakan untuk mengukur seberapa baik model memprediksi target yang benar. Loss function, juga dikenal sebagai a cost function or objective function, adalah fungsi matematika yang mengukur perbedaan antara keluaran yang diprediksi dan target sebenarnya untuk masukan tertentu dalam tugas machine learning. Tujuannya adalah untuk meminimalkan loss function ini, karena fungsi ini sebagai ukuran seberapa baik performa model yang dilatih. Semakin besar nilai loss function maka model tidak mampu menangkap pola dengan baik.</p>", unsafe_allow_html=True)
            elif sub_hyperparameter == "Activation Function":
                st.subheader("Activation Function")
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Activation function secara khusus digunakan pada jaringan saraf tiruan agar dapat mempelajari pola-pola kompleks pada data. Keakuratan prediksi Jaringan Syaraf Tiruan bergantung pada jumlah lapisan yang digunakan dan yang lebih penting lagi pada jenis fungsi aktivasi yang digunakan. Fungsi aktivasi juga digunakan untuk mentransformasi data input menjadi dimensi yang lebih tinggi dan untuk menghitung jumlah bobot dan bias dari model RNN-GRU.</p>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: justify; font-size: 25px;'>a. Fungsi Aktivasi Sigmoid</p>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Fungsi aktivasi sigmoid merupakan fungsi matematika yang memiliki bentuk kurva S, berperan untuk mengubah nilai menjadi rentang antara 0 hingga 1. Fungsi ini juga dikenal sebagai kurva logistik atau fungsi logistik. Fungsi ini merupakan salah satu jenis fungsi aktivasi non-linear yang paling umum digunakan.</p>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image("Rumus Sigmoid.jpg",  width=300)
                st.image("Ket Rumus Sigmoid.jpg",  width=600)
                st.markdown("<p style='text-align: justify; font-size: 25px;'>b. Fungsi AKtivasi Tanh</p>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Fungsi aktivasi Tanh, atau Tangens Hiperbolik, merupakan salah satu fungsi aktivasi yang sering digunakan dan cukup populer dalam deep learning. Fungsi ini mengubah nilai masukan ke dalam rentang antara -1 hingga 1.</p>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image("Rumus Tanh.jpg",  width=300)
                st.image("Ket Rumus Tanh.jpg",  width=600)

        elif sub_bab2 == "Normalisasi Data":
            st.subheader("Normalisasi Data")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Normalisasi data adalah proses konversi skala atribut data sehingga data berada pada rentang tertentu. Pada penelitian ini, normalisasi dilakukan dengan metode Min Max Scaling. Data observasi dikurangi nilai terkecil dari data observasi yang kemudian hasil dari pengurangan tersebut dibagi dengan hasil pengurangan dari nilai terbesar terhadap nilai terkecil data observasi sehingga data berada pada interval 0 hingga 1. Tujuan dari dilakukannya normalisasi adalah untuk menyamakan fitur nilai setiap data observasi dan meminimalisir error.</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("Rumus Normalisasi.jpg",  width=400)
            st.image("Ket Rumus Normalisasi.jpg",  width=800)
        elif sub_bab2 == "Evaluasi Model":
            st.subheader("Evaluasi Model")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Model yang telah bentuk dievaluasi untuk mengukur performa model apakah model yang dihasilkan sudah cukup baik atau memiliki performa yang kurang baik. Performa model yang baik memiliki ukuran kesalahan yang paling kecil pada tahap training dan testing.</p>", unsafe_allow_html=True)
            # Add sub-selection for Evaluasi Model
            sub_evaluasimodel = st.selectbox(
            "Pilih Sub-Bagian Evaluasi Model:",
            ["Mean Absolute Percentage Error (MAPE)", "Root Mean Square Error (RMSE)"]
            )

            if sub_evaluasimodel == "Mean Absolute Percentage Error (MAPE)":
                st.subheader("Mean Absolute Percentage Error (MAPE)")
                st.markdown("<p style='text-align: justify; font-size: 25px;'>Mean Absolute Percentage Error (MAPE) adalah salah satu ukuran keakuratan yang paling populer dan sering dijadikan standar utama dalam berbagai kompetisi. Suatu model dianggap sangat baik jika nilai MAPE-nya berada di bawah 10%, dan dinilai baik jika nilai MAPE berada dalam rentang 10% hingga 20%.</p>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image("Rumus MAPE.jpg",  width=400)
                st.image("Ket Rumus MAPE.jpg",  width=600)
            elif sub_evaluasimodel == "Root Mean Square Error (RMSE)":
                st.subheader("Root Mean Square Error (RMSE)")
                st.markdown("<p style='text-align: justify; font-size: 25px;'>RMSE (Root Mean Square Error) adalah rata-rata dari akar kuadrat jumlah kesalahan yang dikuadratkan. Nilai ini sering digunakan untuk mengukur tingkat kesalahan yang dihasilkan oleh suatu model prediksi. Hasil prediksi dianggap baik jika nilai RMSE yang diperoleh semakin kecil.</p>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image("Rumus RMSE.jpg",  width=400)
                st.image("Ket Rumus RMSE.jpg",  width=600)

        elif sub_bab2 == "Denormalisasi Data":
            st.subheader("Denormalisasi Data")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Denormalisasi adalah proses mengembalikan nilai hasil prediksi ke rentang yang sebenarnya untuk memperoleh nilai yang sebenarnya dan memberikan kinerja model berdasarkan hasil tersebut. Data yang sebelumnya dinormalisasi akan dikonversi kembali ke bentuk aslinya sehingga nilai prediksi atau perkiraan saat ini dapat ditentukan.</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                 st.image("Rumus Denormalisasi.jpg",  width=900)
            st.image("Ket Rumus Denormalisai.jpg",  width=500)
        elif sub_bab2 == "Gempa Bumi":
            st.subheader("Gempa Bumi")
            st.markdown("<p style='text-align: justify; font-size: 25px;'>Gempa bumi adalah peristiwa getaran yang terjadi akibat retakan dan pergeseran lempeng bumi yang kuat, sehingga menyebabkan permukaan bumi berguncang hebat. Proses ini menghasilkan pelepasan energi dalam jumlah besar yang merambat ke permukaan dalam bentuk gelombang seismik, sehingga menyebabkan guncangan yang dapat dirasakan di wilayah sekitar episentrum. Besarnya kekuatan gempa bumi ini diukur menggunakan magnitudo, yaitu parameter yang menunjukkan seberapa besar energi yang dilepaskan dari sumber terjadinya gempa. Magnitudo menjadi salah satu indikator penting dalam menentukan potensi dampak dan tingkat kerusakan yang ditimbulkan oleh suatu gempa bumi. Berdasarkan besarnya magnitudo, gempa bumi dapat diklasifikasikan ke dalam beberapa kategori, seperti yang disajikan pada Tabel berikut:</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                 st.image("Klasifikasi Gempa Bumi.png",  width=900)

elif selected_main == "Data Source":
    st.title("Sumber Data")
    st.markdown("<p style='text-align: justify; font-size: 25px;'>Jenis data yang digunakan dalam penelitian ini adalah data sekunder. Data ini merupakan data historis magnitudo gempa bumi yang diperbarui datanya pada setiap harinya dan diperoleh dari website https://repogempa.bmkg.go.id. Dalam penelitian ini mangambil data harian magnitudo gempa bumi dalam periode waktu 1 Januari 2020 sampai dengan 30 September 2024 yang diakses pada tanggal 22 Desember 2024. Sehingga total data yang digunakan adalah sebanyak 47.683 ribu data penelitian.</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
         st.image("Logo BMKG.png", caption="Sumber Data Gempa Bumi", width=300)

    st.markdown("<p style='text-align: justify; font-size: 25px;'>Variabel yang digunakan dalam penelitian ini adalah data magnitudo gempa bumi di Indonesia dari 1 Januari 2020 hingga 30 September 2024 sebanyak 47.683 ribu data.</p>", unsafe_allow_html=True)

    # Input file upload
    file_path = "DATA MAGNITUDO.csv"
    st.title("Struktur Data")
    if file_path is not None:
        try:
            # Membaca file CSV
            data = pd.read_csv(file_path, skip_blank_lines=False, encoding='utf-8')

            # Menampilkan jumlah total baris dan kolom
            st.write(f"Jumlah baris dalam data: {data.shape[0]}")
            st.write(f"Jumlah kolom dalam data: {data.shape[1]}")

        # Menampilkan data dengan opsi tampilan
            if st.checkbox("Tampilkan seluruh data"):
                st.write(data)
            else:
                rows_per_page = st.number_input("Jumlah baris per halaman:", min_value=10, max_value=5000, value=1000, step=10)
                page = st.number_input("Halaman:", min_value=1, max_value=(len(data) // rows_per_page) + 1, step=1)

                start_idx = (page - 1) * rows_per_page
                end_idx = start_idx + rows_per_page
                st.dataframe(data.iloc[start_idx:end_idx], use_container_width=True)
       
            # Statistik deskriptif
            st.title("Statistik Deskriptif")

            # Hanya ambil kolom numerik untuk statistik deskriptif
            numeric_data = data.select_dtypes(include=[np.number])

            if not numeric_data.empty:
                desc_table = numeric_data.describe().to_html()
                
                st.markdown(f"""
                    <div style='display: flex; justify-content: center;'>
                        <div>
                            {desc_table}
                        <div>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.warning("Tidak ada kolom numerik dalam dataset. Pastikan file CSV memiliki data numerik.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

        st.markdown("<p style='text-align: justify; font-size: 25px;'>Analisis awal terhadap data tersebut menghasilkan sejumlah informasi statistik deskriptif yang merangkum karakteristik utama dari magnitudo gempa selama periode tersebut. Pada Tabel di atas dijelaskan bahwa data magnitudo gempa bumi yang merupakan data harian memiliki sebanyak 47.683 data dengan rata – rata magnitudo keseluruhan adalah 3,483849 SR dengan magnitudo terendah 0,638354 SR dan magnitudo tertinggi 7,894698 SR.</p>", unsafe_allow_html=True)


    else:
        st.write("Silakan unggah file CSV untuk melihat datanya.")

elif selected_main == "Preprocessing":
    st.title("Pembagian Dataset")

    file_path = "DATA MAGNITUDO.csv"
    data = pd.read_csv(file_path)

    # Pastikan data sudah di-load sebelumnya
    if 'data' in locals():
        try:
            data['DATE'] = pd.to_datetime(data['DATE'])

            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)
            
            # Mengurutkan data berdasarkan tanggal
            train_data = train_data.sort_values(by='DATE')
            test_data = test_data.sort_values(by='DATE')
            
            # Menampilkan jumlah data training dan testing
            st.subheader("Pembagian Dataset dengan Skenario 80:20")
            st.write(f"Jumlah Data Training: {len(train_data)}")
            st.write(f"Jumlah Data Testing: {len(test_data)}")

            # Plot time series
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_data['DATE'], train_data['MAGNITUDO (SR)'], label='Training Data', color='blue', alpha=0.7, linewidth=1)
            ax.plot(test_data['DATE'], test_data['MAGNITUDO (SR)'], label='Testing Data', color='orange', alpha=0.7, linewidth=1)
            ax.set_title('Plot Training 80% vs Testing 20%', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Magnitude', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            train_data2, test_data2 = train_test_split(data, test_size=0.25, random_state=42, shuffle=False)
            
            # Mengurutkan data berdasarkan tanggal
            train_data2 = train_data2.sort_values(by='DATE')
            test_data2 = test_data2.sort_values(by='DATE')
            
            # Menampilkan jumlah data training dan testing
            st.subheader("Pembagian Dataset dengan Skenario 75:25")
            st.write(f"Jumlah Data Training: {len(train_data2)}")
            st.write(f"Jumlah Data Testing: {len(test_data2)}")

            # Plot time series
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_data2['DATE'], train_data2['MAGNITUDO (SR)'], label='Training Data', color='blue', alpha=0.7, linewidth=1)
            ax.plot(test_data2['DATE'], test_data2['MAGNITUDO (SR)'], label='Testing Data', color='orange', alpha=0.7, linewidth=1)
            ax.set_title('Plot Training 75% vs Testing 25%', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Magnitude', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            train_data3, test_data3 = train_test_split(data, test_size=0.30, random_state=42, shuffle=False)
            
            # Mengurutkan data berdasarkan tanggal
            train_data3 = train_data3.sort_values(by='DATE')
            test_data3 = test_data3.sort_values(by='DATE')
            
            # Menampilkan jumlah data training dan testing
            st.subheader("Pembagian Dataset dengan Skenario 70:30")
            st.write(f"Jumlah Data Training: {len(train_data3)}")
            st.write(f"Jumlah Data Testing: {len(test_data3)}")

            # Plot time series
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_data3['DATE'], train_data3['MAGNITUDO (SR)'], label='Training Data', color='blue', alpha=0.7, linewidth=1)
            ax.plot(test_data3['DATE'], test_data3['MAGNITUDO (SR)'], label='Testing Data', color='orange', alpha=0.7, linewidth=1)
            ax.set_title('Plot Training 70% vs Testing 30%', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Magnitude', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.title("Normalisasi Data")
            scaler = MinMaxScaler()
            numeric_features = ['MAGNITUDO (SR)']

            # Menggabungkan train dan test untuk normalisasi secara keseluruhan
            full_data = pd.concat([train_data, test_data])
            full_scaled = scaler.fit_transform(full_data[numeric_features])
            
            # Konversi hasil normalisasi ke DataFrame
            full_scaled_df = pd.DataFrame(full_scaled, columns=['Scaled Magnitude'])
            
            # Menampilkan seluruh hasil normalisasi
            st.subheader("Seluruh data setelah MinMax Scaling:")
            st.write(full_scaled_df)
            
            train_scaled = scaler.fit_transform(train_data[numeric_features])
            test_scaled = scaler.transform(test_data[numeric_features])
            
            # Menampilkan seluruh hasil normalisasi
            train_scaled_df = pd.DataFrame(train_scaled, columns=['Scaled Magnitude'])
            train_scaled_df['DATE'] = train_data['DATE'].values
            
            test_scaled_df = pd.DataFrame(test_scaled, columns=['Scaled Magnitude'])
            test_scaled_df['DATE'] = test_data['DATE'].values
            
            # Plot data yang telah dinormalisasi
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_scaled_df['DATE'], train_scaled_df['Scaled Magnitude'], label='Training Data (Scaled)', color='blue', linewidth=1)
            ax.plot(test_scaled_df['DATE'], test_scaled_df['Scaled Magnitude'], label='Testing Data (Scaled)', color='orange', linewidth=1)
            ax.set_title('Plot Data After Scaling 80:20', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Scaled Magnitude', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            scaler = MinMaxScaler()
            numeric_features = ['MAGNITUDO (SR)']

            # Menggabungkan train dan test untuk normalisasi secara keseluruhan
            full_data = pd.concat([train_data2, test_data2])
            full_scaled = scaler.fit_transform(full_data[numeric_features])
            
            # Konversi hasil normalisasi ke DataFrame
            full_scaled_df = pd.DataFrame(full_scaled, columns=['Scaled Magnitude'])
            
            train_scaled = scaler.fit_transform(train_data2[numeric_features])
            test_scaled = scaler.transform(test_data2[numeric_features])
            
            # Menampilkan seluruh hasil normalisasi
            train_scaled_df = pd.DataFrame(train_scaled, columns=['Scaled Magnitude'])
            train_scaled_df['DATE'] = train_data2['DATE'].values
            
            test_scaled_df = pd.DataFrame(test_scaled, columns=['Scaled Magnitude'])
            test_scaled_df['DATE'] = test_data2['DATE'].values
            
            # Plot data yang telah dinormalisasi
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_scaled_df['DATE'], train_scaled_df['Scaled Magnitude'], label='Training Data (Scaled)', color='blue', linewidth=1)
            ax.plot(test_scaled_df['DATE'], test_scaled_df['Scaled Magnitude'], label='Testing Data (Scaled)', color='orange', linewidth=1)
            ax.set_title('Plot Data After Scaling 75:25', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Scaled Magnitude', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            scaler = MinMaxScaler()
            numeric_features = ['MAGNITUDO (SR)']

            # Menggabungkan train dan test untuk normalisasi secara keseluruhan
            full_data = pd.concat([train_data3, test_data3])
            full_scaled = scaler.fit_transform(full_data[numeric_features])
            
            # Konversi hasil normalisasi ke DataFrame
            full_scaled_df = pd.DataFrame(full_scaled, columns=['Scaled Magnitude'])
            
            train_scaled = scaler.fit_transform(train_data3[numeric_features])
            test_scaled = scaler.transform(test_data3[numeric_features])
            
            # Menampilkan seluruh hasil normalisasi
            train_scaled_df = pd.DataFrame(train_scaled, columns=['Scaled Magnitude'])
            train_scaled_df['DATE'] = train_data3['DATE'].values
            
            test_scaled_df = pd.DataFrame(test_scaled, columns=['Scaled Magnitude'])
            test_scaled_df['DATE'] = test_data3['DATE'].values
            
            # Plot data yang telah dinormalisasi
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_scaled_df['DATE'], train_scaled_df['Scaled Magnitude'], label='Training Data (Scaled)', color='blue', linewidth=1)
            ax.plot(test_scaled_df['DATE'], test_scaled_df['Scaled Magnitude'], label='Testing Data (Scaled)', color='orange', linewidth=1)
            ax.set_title('Plot Data After Scaling 70:30', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Scaled Magnitude', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.markdown("<p style='text-align: justify; font-size: 25px;'>Pada Gambar Gambar di Atas dapat terlihat bahwa hasil scaling tidak merubah bentuk dan fluktuatif data sehingga hasil prediksi akan tetap akurat dan mengikuti data aktual.</p>", unsafe_allow_html=True)
    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
    else:
     st.warning("Data belum dimuat. Pastikan Anda sudah mengunggah file data sebelumnya.")
    
elif selected_main == "Modeling":
    st.title("Pembentukan Arsitektur RNN-GRU")
    st.subheader("Pembentukan Arsitektur Pertama")

    file_path = "DATA MAGNITUDO.csv"
    data = pd.read_csv(file_path) 

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)

    # Scaling data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[['MAGNITUDO (SR)']])
    test_scaled = scaler.transform(test_data[['MAGNITUDO (SR)']])

    # Preparing training data
    time_step = 6
    X_train, y_train = [], []
    for i in range(time_step, len(train_scaled)):
        X_train.append(train_scaled[i - time_step:i, 0])
        y_train.append(train_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Preparing testing data
    X_test, y_test = [], []
    for i in range(time_step, len(test_scaled)):
        X_test.append(test_scaled[i - time_step:i, 0])
        y_test.append(test_scaled[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Building the model
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(128, activation='tanh', return_sequences=True, input_shape=(time_step, 1), kernel_initializer=tf.keras.initializers.GlorotUniform()),
        tf.keras.layers.GRU(64, activation='tanh', return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotUniform()),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Model summary
    st.subheader('Model Summary Arsitektur 1')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
          st.image("Arsitektur 1 RNN-GRU.png",  width=900)

    # Train model
    history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    st.success('Model training complete!')

    # Plot training history
    if history.history['loss'] and history.history['val_loss']:
        epochs = range(1, len(history.history['loss']) + 1)

        fig, ax = plt.subplots()
        ax.plot(epochs, history.history['loss'], 'r', label='Training loss')
        ax.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
        ax.set_title('Training and validation loss')
        ax.legend()

        st.pyplot(fig)

    # Evaluate model
    y_train_pred = model.predict(X_train_reshaped)
    y_test_pred = model.predict(X_test_reshaped)

    # Rescale predictions
    y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_train_pred_rescaled = scaler.inverse_transform(y_train_pred[:, -1, 0].reshape(-1, 1))
    y_test_pred_rescaled = scaler.inverse_transform(y_test_pred[:, -1, 0].reshape(-1, 1))

    # Calculate evaluation metrics
    mape_train = mean_absolute_percentage_error(y_train_rescaled, y_train_pred_rescaled)
    rmse_train = np.sqrt(mean_squared_error(y_train_rescaled, y_train_pred_rescaled))

    mape_test = mean_absolute_percentage_error(y_test_rescaled, y_test_pred_rescaled)
    rmse_test = np.sqrt(mean_squared_error(y_test_rescaled, y_test_pred_rescaled))

    # Display results
    st.header('Evaluasi Model')
    st.write('Training Metrics')
    st.write(f"MAPE (Train): {mape_train * 100:.2f}%")
    st.write(f"RMSE (Train): {rmse_train:.4f}")

    st.write('Testing Metrics')
    st.write(f"MAPE (Test): {mape_test * 100:.2f}%")
    st.write(f"RMSE (Test): {rmse_test:.4f}")

    
    st.subheader("Pembentukan Arsitektur Kedua")

    file_path = "DATA MAGNITUDO.csv"
    data = pd.read_csv(file_path) 

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)

    # Scaling data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[['MAGNITUDO (SR)']])
    test_scaled = scaler.transform(test_data[['MAGNITUDO (SR)']])

    # Preparing training data
    time_step = 6
    X_train, y_train = [], []
    for i in range(time_step, len(train_scaled)):
        X_train.append(train_scaled[i - time_step:i, 0])
        y_train.append(train_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Preparing testing data
    X_test, y_test = [], []
    for i in range(time_step, len(test_scaled)):
        X_test.append(test_scaled[i - time_step:i, 0])
        y_test.append(test_scaled[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Building the model
    model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(128, activation='tanh', return_sequences=True, input_shape=(time_step, 1), kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dropout(0.2),  # Dropout setelah RNN pertama
    tf.keras.layers.GRU(64, activation='tanh', return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dropout(0.2),  # Dropout setelah GRU
    tf.keras.layers.GRU(32, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotUniform()),  # GRU tambahan tanpa return_sequences
    tf.keras.layers.Dense(1)  # Output layer
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Model summary
    st.subheader('Model Summary Arsitektur 2')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
          st.image("Arsitektur 2 RNN-GRU.png",  width=900)


    # Train model
    history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    st.success('Model training complete!')

    # Plot training history
    if history.history['loss'] and history.history['val_loss']:
        epochs = range(1, len(history.history['loss']) + 1)

        fig, ax = plt.subplots()
        ax.plot(epochs, history.history['loss'], 'r', label='Training loss')
        ax.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
        ax.set_title('Training and validation loss')
        ax.legend()

        st.pyplot(fig)

    # Evaluate model
    y_train_pred = model.predict(X_train_reshaped)  # Prediksi untuk data training
    y_test_pred = model.predict(X_test_reshaped)  # Prediksi untuk data testing

    # Rescale predictions
    y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_train_pred_rescaled = scaler.inverse_transform(y_train_pred)
    y_test_pred_rescaled = scaler.inverse_transform(y_test_pred)

    # Calculate evaluation metrics
    mape_train = mean_absolute_percentage_error(y_train_rescaled, y_train_pred_rescaled)
    rmse_train = np.sqrt(mean_squared_error(y_train_rescaled, y_train_pred_rescaled))

    mape_test = mean_absolute_percentage_error(y_test_rescaled, y_test_pred_rescaled)
    rmse_test = np.sqrt(mean_squared_error(y_test_rescaled, y_test_pred_rescaled))

    # Display results
    st.header('Evaluasi Model 2')
    st.write('Training Metrics')
    st.write(f"MAPE (Train): {mape_train * 100:.2f}%")
    st.write(f"RMSE (Train): {rmse_train:.4f}")

    st.write('Testing Metrics')
    st.write(f"MAPE (Test): {mape_test * 100:.2f}%")
    st.write(f"RMSE (Test): {rmse_test:.4f}")

    st.title("Prediksi Data Uji dengan Model RNN-GRU")

    # Prediksi pada data uji
    y_pred_rescaled = model.predict(X_test_reshaped)

    # Konversi prediksi ke skala asli (jika menggunakan MinMaxScaler)
    y_test_pred_original = scaler.inverse_transform(y_pred_rescaled)

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi pada Data Uji")
    st.write(y_test_pred_original.flatten())

    df_prediksi = pd.DataFrame({'Prediksi': y_test_pred_original.flatten()})

   # Tombol untuk mengunduh hasil prediksi
    csv = df_prediksi.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Unduh Hasil Prediksi sebagai CSV",
        data=csv,
        file_name='hasil_prediksi.csv',
        mime='text/csv'
    )

elif selected_main == "Hasil":
    st.title("Hasil Prediksi Magnitudo Gempa Bumi")

    # Upload file data aktual dan prediksi
    actual_file = st.file_uploader("Upload file data aktual (CSV)", type='csv')
    predicted_file = st.file_uploader("Upload file data prediksi (CSV)", type='csv')

    if actual_file and predicted_file:
        # Load data
        actual_df = pd.read_csv(actual_file)
        predicted_df = pd.read_csv(predicted_file)

        # Pastikan kolom tanggal dan magnitudo sesuai
        actual_df['DATE'] = pd.to_datetime(actual_df['DATE']).dt.date
        actual_data = actual_df['MAGNITUDO (SR)']

        predicted_data = predicted_df['Prediksi']  # Pastikan kolom sesuai nama di file prediksi

        # Truncate data agar panjang sama
        min_len = min(len(actual_data), len(predicted_data))
        actual_data = actual_data[:min_len].to_numpy().flatten()
        predicted_data = predicted_data[:min_len]
        actual_dates = actual_df['DATE'][:min_len]

        st.title("Hasil Prediksi Magnitudo Gempa Bumi")

        # Hitung metrik evaluasi (menggunakan semua data)
        mse = mean_squared_error(actual_data, predicted_data)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actual_data, predicted_data)

        # Tampilkan hasil evaluasi
        st.write("### Evaluation Metrics")
        st.write(f"MSE: {mse:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAPE: {mape:.4f}")

        # Tampilkan tabel seluruh data prediksi dan data asli
        combined_df = pd.DataFrame({
            'Date': actual_dates,
            'Actual': actual_data,
            'Prediksi': predicted_data
        })
        st.write("### Tabel Data Aktual dan Prediksi")
        st.dataframe(combined_df)

        # Filter data hanya untuk magnitudo antara 3.0 dan 3.8
        mask = (actual_data >= 3.0) & (actual_data <= 3.8)
        filtered_actual_data = actual_data[mask]
        filtered_predicted_data = predicted_data[mask]

        # Cek apakah ada data setelah difilter
        if len(filtered_actual_data) > 0:
            # Visualisasi data aktual vs prediksi (hanya untuk magnitudo 3.0 - 3.8)
            comparison_df = pd.DataFrame({
                'Index': range(len(filtered_actual_data)),
                'Actual': filtered_actual_data,
                'Predicted': filtered_predicted_data
            })

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(comparison_df['Index'], comparison_df['Actual'], label='Actual')
            ax.plot(comparison_df['Index'], comparison_df['Predicted'], label='Predicted')
            ax.set_xlabel('Kejadian Ke-')
            ax.set_ylabel('Magnitude (SR)')
            ax.set_title('Perbandingan Data Aktual dan Data Prediksi')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.write("Tidak ada data dengan magnitudo antara 3.0 dan 3.8 SR.")

        st.markdown("<p style='text-align: justify; font-size: 25px;'>Berdasarkan Gambar di atas pola antara data aktual dan data prediksi menunjukkan kemiripan yang cukup baik, meskipun masih terdapat beberapa perbedaan pada titik-titik tertentu. Grafik ini menggambarkan bagaimana model mampu mengikuti fluktuasi nilai magnitudo gempa dengan cukup akurat, meskipun dalam beberapa kejadian terdapat selisih antara nilai aktual dan prediksi.</p>", unsafe_allow_html=True)

    # Plot hanya data prediksi
    st.write("### Visualisasi Data Prediksi")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(predicted_data)), predicted_data, label='Predicted Data', color='orange')
    ax.set_xlabel('Kejadian Ke-')
    ax.set_ylabel('Magnitude (SR)')
    ax.set_title('Plot Prediksi')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("<p style='text-align: justify; font-size: 25px;'>Pada Gambar terlihat bahwa model memprediksi magnitudo gempa dengan nilai yang relatif stabil, meskipun tetap mengalami fluktuasi dalam rentang 3,0 – 3,8 SR. Variasi prediksi magnitudo yang cukup signifikan mengindikasikan bahwa model menangkap perubahan magnitudo dari waktu ke waktu, namun tetap dalam kisaran yang relatif kecil. Berdasarkan Tabel 2.2, hasil prediksi menunjukkan bahwa gempa bumi yang diperkirakan termasuk dalam kategori Gempa Bumi Lemah dengan Status Waspada. Getaran yang ditimbulkan terasa lemah dan umumnya hanya dirasakan di wilayah tertentu. Selain itu, gempa dalam kategori ini jarang menyebabkan kerusakan yang signifikan, namun tetap perlu diwaspadai sebagai upaya mitigasi risiko.</p>", unsafe_allow_html=True)

elif selected_main == "Kesimpulan":
    st.title("Kesimpulan")
    st.markdown("<p style='text-align: justify; font-size: 25px;'>Berdasarkan analisis yang telah dilakukan dalam prediksi magnitudo di Indonesia menggunakan metode Neural Network Gated Reccurent Unit (RNN-GRU), maka dapat diambil kesimpulan sebagai berikut: </p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; font-size: 25px;'>1. Model prediksi magnitudo gempa bumi di Indonesia berhasil dibentuk dengan melakukan uji coba (trial and error) hingga mendapatkan hyperparameter yang sesuai untuk membentuk model yang terbaik. Model awal menggunakan 1 layer RNN sebanyak 128 unit, layer GRU sebanyak 64 unit dan 1 layer dropout sebesar 20% Selanjutnya dilakukan penambahan layer agar arsitektur lebih optimal menjadi 1 layer RNN sebanyak 128 unit, 2 layer GRU sebanyak 64 dan 32 unit, 2 layer dropout sebesar 20% dengan iterasi sebanyak 50 epochs dan dibagi menjadi 32 batch size.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 25px;'>Arsitektur 1</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
          st.image("RNN-GRU 1.png",  width=700)
    st.markdown("<p style='text-align: center; font-size: 25px;'>Arsitektur 2</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
          st.image("RNN-GRU 2.png",  width=700)

    st.markdown("<p style='text-align: justify; font-size: 25px;'>2. Model terbaik yang dihasilkan memiliki akurasi dengan nilai MAPE sebesar 18,41% dan RMSE sebesar 0,7346, menunjukkan kinerja yang baik dalam memprediksi magnitudo gempa bumi.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; font-size: 25px;'>3. Berdasarkan hasil prediksi, gempa bumi dalam rentang magnitudo 3,0 hingga 3,8 SR dikategorikan sebagai Gempa Bumi Lemah dengan Status Waspada. Prediksi tersebut menunjukkan tingkat kestabilan yang baik, dengan nilai MAPE sebesar 19,29% dan RMSE sebesar 0,7721. Getaran yang ditimbulkan terasa lemah dan jarang menyebabkan kerusakan yang signifikan, namun tetap perlu diwaspadai sebagai langkah antisipasi.</p>", unsafe_allow_html=True)

    st.title("Saran")
    st.markdown("<p style='text-align: justify; font-size: 25px;'>Berdasarkan analisis yang telah dilakukan didapatkan beberapa saran yang diberikan peneliti sebagai berikut: </p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; font-size: 25px;'>1. Pada penelitian ini, waktu pemrosesan model terbilang cukup lama, sehingga disarankan untuk mengoptimalkan model agar lebih efisien. Selain itu, penelitian selanjutnya dapat menggunakan data dengan cakupan waktu yang lebih panjang atau menambahkan variabel lain yang relevan, seperti lokasi atau kedalaman gempa, agar model dapat memberikan informasi yang lebih kaya dan akurat.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: justify; font-size: 25px;'>2. Metode RNN-GRU yang digunakan dalam penelitian ini sudah menunjukkan kinerja yang baik, namun disarankan untuk membandingkannya dengan metode lain seperti LSTM atau model hybrid. Perbandingan ini penting untuk memastikan bahwa metode yang digunakan benar-benar merupakan pendekatan yang paling efektif dan akurat dalam memprediksi magnitudo gempa.</p>", unsafe_allow_html=True)

