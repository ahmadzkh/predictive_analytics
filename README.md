# Laporan Proyek Machine Learning - Ahmad Zaky Humami

## Domain Proyek
Prestasi akademik siswa menjadi indikator penting bagi sekolah, orang tua, dan pembuat kebijakan untuk memahami efektivitas proses belajar–mengajar, mengalokasikan sumber daya, serta memberikan intervensi yang tepat waktu. Sejumlah penelitian menunjukkan bahwa faktor–faktor demografis (usia, jenis kelamin), kebiasaan belajar (lama belajar per minggu, kehadiran), serta dukungan sosial (bimbingan belajar, dukungan orang tua, kegiatan ekstrakurikuler) secara signifikan mempengaruhi hasil belajar siswa (Pei, 2023; Li & Wang, 2024). Di Indonesia, riset oleh Ambarita et al. (2024) dan Mentari & Nurhaeka (2024) juga mengonfirmasi peran faktor-faktor tersebut dalam memprediksi nilai akhir siswa SD dan SMA.

Mengapa Masalah Ini Harus Diselesaikan?
  - Deteksi Dini Siswa Berisiko
    > Mengidentifikasi siswa dengan potensi prestasi rendah memungkinkan sekolah melakukan pembinaan lebih awal.
  - Optimalisasi Intervensi
    > Data-driven insights membantu memilih jenis dukungan (bimbingan belajar, konseling orang tua, program ekstrakurikuler) yang paling efektif.
  - Pengambilan Keputusan Berbasis Bukti
    > Otomatisasi prediksi dengan machine learning mempercepat evaluasi kebijakan pendidikan dan alokasi anggaran.

## Business Understanding
### Problem Statements
1. Bagaimana memprediksi kategori klasifikasi prestasi (GradeClass: A, B, C, D, F) siswa berdasarkan atribut demografis dan perilaku akademik mereka?
2. Faktor manakah (misal GPA, jam belajar mingguan, absensi, dukungan orang tua, partisipasi ekstrakurikuler) yang paling berpengaruh terhadap prediksi GradeClass?
3. Seberapa andal model machine learning (misalnya XGBoost, Random Forest) dalam memprediksi GradeClass—diukur melalui metrik accuracy, F1-score, dan confusion matrix—pada data testing yang terpisah?

### Goals
1. Membangun model klasifikasi yang dapat mencapai ≥ 80 % accuracy pada data testing.
2. Melihat Faktor penting yang mempengaruhi Grade Class
3. Mengukur precision, recall, F1-score per kelas dan menghasilkan confusion matrix.

## Data Understanding
### Sumber Data
Pada proyek ini, kita menggunakan **Students Performance Dataset** yang diambil dari Kaggle. Dataset ini berisi data akademik dan demografis siswa sekolah menengah atas, dengan **2392 baris** dan **15 kolom** fitur, antara lain usia, jam belajar per minggu, jumlah absensi, nilai GPA, partisipasi dalam bimbingan belajar, dukungan orang tua, dan keterlibatan dalam berbagai kegiatan ekstrakurikuler. Anda dapat mengunduh dataset lengkap di:  
[Students Performance Dataset – Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset/data)

### Target: **GradeClass**
Variabel target **GradeClass** dikonstruksi dari nilai akhir (GPA) siswa dan dikelompokkan menjadi lima kategori, yaitu:

- **Grade A**: Siswa dengan GPA ≥ 3.7  
- **Grade B**: Siswa dengan GPA ≥ 3.0 dan < 3.7  
- **Grade C**: Siswa dengan GPA ≥ 2.3 dan < 3.0  
- **Grade D**: Siswa dengan GPA ≥ 1.7 dan < 2.3  
- **Grade F**: Siswa dengan GPA < 1.7  

Kelima kategori inilah yang menjadi **target prediksi** model klasifikasi di proyek ini.  

### Deskripsi Variabel
Variabel | Keterangan
----------|----------
StudentID | A unique identifier assigned to each student (1001 to 3392).
Age | The age of the students ranges from 15 to 18 years.
Gender | Gender of the students, where 0 represents Male and 1 represents Female.
Ethnicity | Ethnic background of the student. 0: Caucasian, 1: African American, 2: Asian, 3: Other.
ParentalEducation | Level of parental support for the student\'s education. 0: None 1: High School, 2: Some College, 3: Bachelor`s, 4: Higher
StudyTimeWeekly | Weekly study time in hours, ranging from 0 to 20.
Absences |  Number of absences during the school year, ranging from 0 to 30.
Tutoring | Tutoring status, where 0 indicates No and 1 indicates Yes.
ParentalSupport | The education level of the parents, coded as follows: 0: None 1: High School, 2: Some College, 3: Bachelor`s, 4: Higher
Extracurricular | Participation in extracurricular activities, where 0 indicates No and 1 indicates Yes.
Sports | Participation in sports, where 0 indicates No and 1 indicates Yes.
Music	| Participation in music activities, where 0 indicates No and 1 indicates Yes.
Volunteering	| Participation in volunteering, where 0 indicates No and 1 indicates Yes.
GPA	| Grade Point Average on a scale from 2.0 to 4.0, influenced by study habits, parental involvement, and extracurricular activities.
GradeClass | Classification of students grades based on GPA: 0: 'A' (GPA >= 3.5) 1: 'B' (3.0 <= GPA < 3.5) 2: 'C' (2.5 <= GPA < 3.0) 3: 'D' (2.0 <= GPA < 2.5) 4: 'F' (GPA < 2.0)

```
Dataset Info :

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2392 entries, 0 to 2391
Data columns (total 15 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   StudentID          2392 non-null   int64  
 1   Age                2392 non-null   int64  
 2   Gender             2392 non-null   int64  
 3   Ethnicity          2392 non-null   int64  
 4   ParentalEducation  2392 non-null   int64  
 5   StudyTimeWeekly    2392 non-null   float64
 6   Absences           2392 non-null   int64  
 7   Tutoring           2392 non-null   int64  
 8   ParentalSupport    2392 non-null   int64  
 9   Extracurricular    2392 non-null   int64  
 10  Sports             2392 non-null   int64  
 11  Music              2392 non-null   int64  
 12  Volunteering       2392 non-null   int64  
 13  GPA                2392 non-null   float64
 14  GradeClass         2392 non-null   float64
dtypes: float64(3), int64(12)
memory usage: 280.4 KB
```

#### Menangani Missing Value
Column | Missing Value
----------|----------
StudentID | 0
Age | 0
Gender | 0
Ethnicity | 0
ParentalEducation | 0
StudyTimeWeekly | 0
Absences |  0
Tutoring | 0
ParentalSupport | 0
Extracurricular | 0
Sports | 0
Music	| 0
Volunteering	| 0
GPA	| 0
GradeClass | 0

Untuk Mising Value
- Dari hasil yang ditampilkan, data tidak memiliki nilai kosong (null) pada setiap kolom dataset

#### Memeriksa Duplikasi Data
```
Jumlah duplikasi:  0
```
Untuk Duplikasi Data
- Hasil yang ditampilkan adalah 0, dengan demikian data tidak ada yang ganda (dupikat)

#### Konversi nilai Numeric pada Column Categorical ke Object
```
# Mengganti nilai number kategori ke String Keterangan
def convert_numerical_to_object ():
  student_df['Gender'] = student_df['Gender'].replace({1: 'Wanita', 0: 'Pria'})
  student_df['Extracurricular'] = student_df['Extracurricular'].replace({1:'Yes', 0:'No'})
  student_df['Tutoring'] = student_df['Tutoring'].replace({1: 'Yes', 0: 'No'})
  student_df['Music'] = student_df['Music'].replace({1: 'Yes', 0: 'No'})
  student_df['Sports'] = student_df['Sports'].replace({1: 'Yes', 0: 'No'})
  student_df['Volunteering'] = student_df['Volunteering'].replace({1: 'Yes', 0: 'No'})
  student_df['ParentalSupport'] = student_df['ParentalSupport'].replace({4: 'Very High', 3: 'High', 2: 'Moderate', 1: 'Low', 0: 'None'})
  student_df['GradeClass'] = student_df['GradeClass'].replace({4.0: 'Grade F', 3.0: 'Grade D', 2.0: 'Grade C', 1.0: 'Grade B', 0.0: 'Grade A'})

# Memanggil fungsi konversi numerik ke objek(string)
convert_numerical_to_object()
```
Fungsi convert_numerical_to_object() pada bagian “Konversi nilai Numeric pada Column Categorical ke Object” bertujuan untuk mengubah kolom–kolom yang secara semula berisi kode angka (integer atau float) menjadi tipe data object dengan label yang lebih deskriptif. Ini sangat berguna untuk:
- Mempermudah interpretasi saat eksplorasi data (bar-plot, pivot‐table, dsb.).
- Menyiapkan data untuk encoding selanjutnya (label/one-hot encoding), karena algoritma visualisasi maupun beberapa library analisis lebih nyaman bekerja dengan string kategori.

## Exploratory Data Analysis (EDA)
### Unvariative Analysis EDA
#### Distribusi Categorical Column menggunakan Bar Plot

<p align="center">
  <img src="https://github.com/user-attachments/assets/5ebe6f6b-1228-433b-94bd-984a1ce4af48" width="500"/>
</p>

📊 Distribution of Student Gender:

Kategori:
  - Wanita
  - Pria

Insight:
  - Jumlah siswa wanita lebih banyak daripada pria, meskipun tidak terlalu jauh perbedaannya.
  - Ini mengindikasikan distribusi gender cukup seimbang, namun ada sedikit dominan wanita.

<p align="center">
  <img src="https://github.com/user-attachments/assets/343d0596-ae59-4e60-82f8-a2555dc6c556" width="500"/>
</p>


📊 Distribution of Student Ethnicity:

Kategori:
  - Caucasian
  - Asian
  - African American
  - Other

Insight:
  - Etnis Caucasian mendominasi populasi siswa dalam dataset.
  - Etnis Asian dan African American jumlahnya hampir sama, tapi jauh di bawah Caucasian.
  - Kategori Other merupakan yang paling sedikit, menunjukkan keberagaman etnis yang relatif kecil.

<p align="center">
  <img src="https://github.com/user-attachments/assets/393f267f-6f67-4ba0-8949-3a4599d6f931" width="500"/>
</p>

📊 Distribution of Student Parental Education:

Kategori:
  - 0: None
  - 1: High School
  - 2: Some College
  - 3: Bachelors
  - 4: Higher

Insight:
  - Mayoritas orang tua siswa memiliki latar belakang Some College dan High School.
  - Hanya sebagian kecil yang mencapai tingkat pendidikan Bachelors dan lebih tinggi.
  - Sekitar 250 siswa berasal dari keluarga dengan orang tua tidak memiliki pendidikan formal.
  - Ini bisa berdampak pada pola dukungan dan pemahaman orang tua terhadap pendidikan anak.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f179e136-bac6-4dfa-bac9-e46512729aac" width="500"/>
</p>

📊 Distribution of Student Tutoring

Kategori:
  - Yes (mengikuti bimbingan belajar)
  - No (tidak mengikuti bimbingan belajar)

Insight:
  - Mayoritas siswa tidak mengikuti bimbingan belajar.
  - Hanya sekitar 30% yang mengikuti program tutoring, selaras dengan statistik deskriptif sebelumnya (mean ≈ 0.3).
  - Hal ini dapat berpengaruh pada variasi prestasi akademik antar siswa.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c2185450-ccbf-45cf-bb17-b42830c25f21" width="500"/>
</p>

📊 Distribution of Student Parental Support:

Kategori:
  - None
  - Low
  - Moderate
  - High
  - Very High

Insight:
  - Sebagian besar orang tua memberikan dukungan sedang (Moderate) dan tinggi (High) terhadap pendidikan anaknya.
  - Dukungan sangat tinggi (Very High) relatif sedikit.
  - Sekitar 200+ siswa tidak mendapatkan dukungan sama sekali dari orang tua, yang bisa menjadi indikator risiko terhadap prestasi akademik mereka.
  - Distribusi dukungan cukup beragam, menunjukkan adanya variasi signifikan dalam latar belakang keluarga siswa.

<p align="center">
  <img src="https://github.com/user-attachments/assets/19b09ac1-3801-4b06-9226-0f7c304823c0" width="500"/>
</p>

📊 Distribution of Student Extracurricular:

Kategori:
  - Yes (ikut)
  - No (tidak ikut)

Insight:
  - Lebih dari 60% siswa tidak ikut ekstrakurikuler, sisanya (~40%) ikut.
  - Ini bisa menunjukkan bahwa ekstrakurikuler belum sepenuhnya dimanfaatkan siswa sebagai sarana pengembangan diri.
  - Perlu digali apakah partisipasi dalam ekstrakurikuler berdampak positif pada GPA atau prestasi lainnya.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d01ff743-85a8-485d-84fe-8f88828834b3" width="500"/>
</p>

📊 Distribution of Student Sports:

Kategori:
  - Yes (ikut kegiatan olahraga)
  - No (tidak ikut)

Insight:
  - Sebagian besar siswa (~70%) tidak terlibat dalam kegiatan olahraga.
  - Hanya sekitar 30% siswa aktif dalam olahraga.
  - Ini bisa berdampak pada keseimbangan fisik-mental siswa, karena olahraga berkontribusi pada kesehatan dan performa akademik.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9e6e0b5f-3fc6-4ae4-ac6b-a5dcd16eae5a" width="500"/>
</p>

📊 Distribution of Student Musics:

Kategori:
  - Yes (ikut musik)
  - No (tidak ikut)

Insight:
  - Hanya sekitar 20% siswa yang ikut kegiatan musik, sedangkan mayoritas (~80%) tidak ikut.
  - Hal ini mencerminkan kurangnya minat atau akses terhadap program musik di sekolah atau dalam lingkungan siswa.
  - Aktivitas musik seringkali berkorelasi dengan keterampilan kognitif dan kreativitas, sehingga bisa menjadi area untuk ditingkatkan.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3fcfa29b-cb2c-4ea5-8016-c0eb873cbe8e" width="500"/>
</p>

📊 Distribution of Student Volunteering:

Kategori:
  - Yes (pernah menjadi relawan)
  - No (tidak pernah)

Insight:
  - Sebagian besar siswa (lebih dari 80%) tidak pernah terlibat dalam kegiatan sosial atau volunteering.
  - Ini menunjukkan bahwa keterlibatan siswa dalam kegiatan sosial masih rendah, meskipun kegiatan ini penting untuk membentuk karakter dan soft skill.
  - Program sekolah bisa lebih mendorong siswa untuk ikut serta dalam volunteering.

<p align="center">
  <img src="https://github.com/user-attachments/assets/931edb08-ca80-4252-ac7e-6536d9e4e946" width="500"/>
</p>

📊 Distribution of Student GradeClass:

Kategori:
  - Grade A
  - Grade B
  - Grade C
  - Grade D
  - Grade F

Insight:
  - Grade C adalah yang paling banyak dicapai oleh siswa, diikuti oleh Grade D dan Grade F, yang menunjukkan performa akademik cenderung menengah ke bawah.
  - Hanya sedikit siswa yang mendapatkan Grade A, menunjukkan bahwa pencapaian tertinggi masih jarang.
  - Ini mungkin menjadi indikator adanya masalah dalam efektivitas proses belajar atau faktor eksternal (dukungan orang tua, motivasi, kegiatan tambahan, dsb).

#### Distribusi Numerical Column menggunakan Box Plot

<p align="center">
  <img src="https://github.com/user-attachments/assets/bd3b2ae4-c166-4e41-a9f9-945db7a247dc" width="500"/>
</p>

📊 Box Plot of Student Age:

Insight:
  - Rentang usia siswa adalah dari 15 hingga 18 tahun.
  - Mayoritas siswa berusia antara 15,5 hingga 17 tahun, dengan median di sekitar 16 tahun.
  - Tidak terdapat outlier, menandakan distribusi usia cukup normal untuk siswa SMA.

<p align="center">
  <img src="https://github.com/user-attachments/assets/324743ee-0ad5-44d2-a42d-a5cb91a370e8" width="500"/>
</p>

📊 Box Plot of Weekly Study Time:

Insight:
  - Rentang waktu belajar mingguan berkisar antara 0 hingga 20 jam.
  - Median berada sekitar 10 jam/minggu.
  - Hampir semua nilai berada dalam rentang interkuartil, tidak terlihat outlier.
  - Ini mengindikasikan bahwa sebagian besar siswa belajar sekitar 1-2 jam per hari secara konsisten.

<p align="center">
  <img src="https://github.com/user-attachments/assets/22fe4f56-e901-4284-a860-97dbc4ab1a69" width="500"/>
</p>

📊 Box Plot of Student Absences:

Insight:
  - Ketidakhadiran siswa bervariasi antara 0 hingga hampir 30 kali.
  - Median berada di sekitar 15 kali absen, dengan distribusi cukup merata. -
  - Tidak ada outlier ekstrem, tapi ada siswa dengan ketidakhadiran yang cukup tinggi (>25 kali).
  - Ini bisa menjadi indikator penting: siswa dengan banyak absen kemungkinan memiliki GPA lebih rendah atau keterlibatan yang minim.

<p align="center">
  <img src="https://github.com/user-attachments/assets/dcf561fb-7d6d-4f72-a81e-82e26378dfb9" width="500"/>
</p>

📊 Box Plot of Student GPA:

Insight:
  - GPA (Grade Point Average) berkisar dari 0 hingga 4.0, sesuai dengan skala umum.
  - Median GPA berada di sekitar 2.0, menandakan sebagian besar siswa memiliki performa akademik rata-rata atau kurang dari baik.
  - Ada penyebaran yang cukup seimbang, tanpa outlier ekstrem.
  - Siswa dengan GPA di bawah 2.0 perlu diperhatikan lebih lanjut (bisa berkaitan dengan waktu belajar, absensi, atau dukungan orang tua).

#### Membuat Pie Chart kolom GradeClass

<p align="center">
  <img src="https://github.com/user-attachments/assets/02fefa30-149d-4e08-b7c3-b2fa9b2c500c" width="500"/>
</p>

🔹 Distribution of Student GradeClass:
- Berdasarkan hasil yang ditampilkan sebanyak 50.6% Siswa berada pada Grade F (Kelas dengan Prestasi Terendah) menjadi jumlah terbanyak, dan hanya sedikit sekitar 4.5% Siswa berada pada Grade A (Kelas dengan Prestasi Terbaik).
- Sedangkan siswa yang lainnya berada pada Grade B sekitar 11.2% Siswa, Grade C sekitar 16.3% Siswa dan Grade D sekitar 17.3% Siswa.

#### Membuat Histogram untuk Numerical Column

<p align="center">
  <img src="https://github.com/user-attachments/assets/c869662f-356f-41b9-915a-12ad2e61220c" width="500"/>
</p>

📊 Distribution of Student Age:
- Distribusi usia terlihat multimodal, dengan puncak di usia 15, 16, 17, dan 18 tahun.
- Hal ini menunjukkan bahwa siswa berasal dari berbagai tingkat atau kelas yang memiliki rentang usia relatif luas.
- Distribusi tidak normal dan menunjukkan pola siklis, kemungkinan karena jumlah siswa di tiap tingkat/kelas hampir merata.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a0fd4c50-1282-4c1d-b2d1-f5c3ce45c452" width="500"/>
</p>

⏳ Distribution of Weekly Study Time:
- Distribusi waktu belajar mingguan mendekati normal dengan puncak di sekitar 8 - 10 jam/minggu.
- Artinya, sebagian besar siswa belajar kurang lebih 1 - 1.5 jam per hari.
- Masih ada sebagian kecil siswa yang belajar di bawah 5 jam atau di atas 15 jam, menunjukkan adanya variasi motivasi atau kebiasaan belajar.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fa87c5eb-b799-4246-95db-0d9828ec7d2c" width="500"/>
</p>

📅 Distribution of Student Absences:
- Distribusi relatif merata, tetapi ada sedikit puncak pada beberapa hari tertentu (sekitar 0, 5, 10, dan 20 hari).
- Ini bisa berarti tidak ada pola absen yang dominan, namun ada kelompok siswa yang sangat rajin (absen sedikit) dan kelompok dengan tingkat absen cukup tinggi (hingga 25–30 hari).
- Hal ini bisa menjadi indikator potensi masalah kehadiran atau disiplin siswa.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1a76a720-2f02-4dc9-9296-91eecebe56d7" width="500"/>
</p>

🏅 Distribution of Student GPA:
- Distribusi prestasi menunjukkan bentuk kurva normal, dengan puncak di sekitar nilai 2.
- Ini menandakan bahwa sebagian besar siswa memiliki prestasi rata-rata, sementara hanya sedikit yang sangat rendah (mendekati 0) atau sangat tinggi (mendekati 4).
- Ini bisa berarti program pengajaran memiliki pengaruh merata terhadap siswa, namun ada peluang peningkatan pada siswa dengan nilai tertinggi atau terendah.

✨ Kesimpulan Umum
- Sebagian besar fitur numerik menunjukkan distribusi yang relatif seimbang atau normal, kecuali usia yang terklasifikasi jelas berdasarkan tingkat kelas.
- Fitur seperti absensi dan waktu belajar bisa dijadikan variabel prediktor penting untuk model klasifikasi prestasi siswa.
- Insight ini mendukung eksplorasi selanjutnya dalam modeling, misalnya klasifikasi siswa berprestasi tinggi dan rendah berdasarkan waktu belajar, absen, dan dukungan lain.

### Multivariate Analysis EDA
#### Ananlisis data pada fitur numerik `StudyTimeWeekly` dengan `GPA`

<p align="center">
  <img src="https://github.com/user-attachments/assets/4629642c-7725-446b-9d5c-3a43ce2b6cb5" width="500"/>
</p>

📉 Impact of Study Time Every Week on GPA:
- Sumbu X: Waktu Belajar (jam/minggu)
- Sumbu Y: GPA
- Garis Merah: Garis regresi linier

Insight:
- Terdapat korelasi positif lemah: saat waktu belajar meningkat, GPA cenderung meningkat juga.
- Kemiringan garis regresi positif, tetapi sangat landai artinya, tambahan waktu belajar memberikan pengaruh kecil terhadap peningkatan GPA.
- Sebaran data sangat menyebar, menunjukkan banyak variabel lain yang memengaruhi GPA selain waktu belajar.

#### Ananlisis data pada fitur numerik `Absence` dengan `GPA`

<p align="center">
  <img src="https://github.com/user-attachments/assets/29d1ab68-9901-493a-92ec-bf2a132a34a5" width="500"/>
</p>

📉 Impact of Absence on GPA:
- Sumbu X: Jumlah Absen
- Sumbu Y: GPA
- Garis Merah: Garis regresi linier

Insight:
- Terdapat korelasi negatif kuat: semakin banyak absen, semakin rendah GPA.
- Kemiringan garis regresi negatif tajam, menandakan hubungan yang signifikan.
- Data cukup konsisten menurun dari kiri ke kanan - semakin sering siswa tidak hadir, prestasinya cenderung menurun secara konsisten.

#### Ananlisis data pada fitur kategori `Tutoring` dengan `Grade Class`

<p align="center">
  <img src="https://github.com/user-attachments/assets/4c94f030-3647-4902-837c-3674a59e0a1a" width="500"/>
</p>

📊 Comparison of Tutoring on GPA:
- Tanpa bimbingan (No) menunjukkan jumlah siswa yang lebih banyak mendapatkan Grade F, D, dan C.
- Dengan bimbingan (Yes), distribusi siswa bergeser ke nilai yang lebih baik, dan jumlah siswa dengan Grade F berkurang signifikan.
- Kesimpulan: Bimbingan belajar cenderung berdampak positif terhadap pencapaian nilai siswa.

#### Ananlisis data pada fitur kategori `Gender` dengan  `Grade Class`

<p align="center">
  <img src="https://github.com/user-attachments/assets/2f94dc5e-6e77-42f5-9a3c-b351c3134e53" width="500"/>
</p>

📊 Comparison of Tutoring on GPA:
- Tanpa bimbingan (No) menunjukkan jumlah siswa yang lebih banyak mendapatkan Grade F, D, dan C.
- Dengan bimbingan (Yes), distribusi siswa bergeser ke nilai yang lebih baik, dan jumlah siswa dengan Grade F berkurang signifikan.
- Kesimpulan: Bimbingan belajar cenderung berdampak positif terhadap pencapaian nilai siswa.

#### Ananlisis data pada fitur kategori kegiatan non akademik `Extracurricular`, `Sports`, `Music`, `Volunteering` dengan `GPA`

<p align="center">
  <img src="https://github.com/user-attachments/assets/3f955da7-b363-4446-97a9-642a72f4c434" width="500"/>
</p>

📊 Impact of Extracurricular on GPA:
- Siswa yang mengikuti aktivitas ekstrakurikuler memiliki rata-rata GPA lebih tinggi dibandingkan mereka yang tidak ikut.
- Peningkatan ini menunjukkan bahwa kegiatan di luar akademik seperti organisasi, klub, atau kegiatan komunitas dapat berdampak positif terhadap performa belajar.

Kesimpulan:
- Siswa yang aktif secara sosial memiliki keterampilan manajemen waktu yang lebih baik.
- Extracurricular membangun soft skills seperti tanggung jawab, kerjasama, dan kepemimpinan.

<p align="center">
  <img src="https://github.com/user-attachments/assets/594351ad-957b-4f72-9073-c575538bb43e" width="500"/>
</p>

📊 Impact of Sports Participation on GPA:
- Siswa yang terlibat dalam olahraga menunjukkan sedikit peningkatan GPA, meskipun tidak sebesar peningkatan dari extracurricular.
- Ini mengindikasikan bahwa aktivitas fisik memiliki dampak positif ringan terhadap akademik.

Kesimpulan:
  - Olahraga dapat membantu meningkatkan fokus, disiplin, dan kesehatan mental.
  - Namun, beban latihan yang tinggi mungkin juga mengurangi waktu belajar jika tidak dikelola dengan baik.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8e010bb8-ffe5-49f7-8881-2606dae4eeaa" width="500"/>
</p>

📊 Impact of Music on GPA:
- Siswa yang aktif di bidang musik (seperti bermain alat musik, paduan suara, band) menunjukkan peningkatan GPA yang signifikan dibandingkan yang tidak aktif di musik.
- Musik diyakini dapat mengaktifkan area otak yang berkaitan dengan konsentrasi, logika, dan kreativitas.

Kesimpulan:
- Pembelajaran musik melibatkan latihan rutin dan disiplin, yang dapat terbawa ke kebiasaan belajar.
- Musik juga mendukung perkembangan memori dan pemrosesan kognitif.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fb02c33e-e0a7-4c26-9f54-a9e59f3a354f" width="500"/>
</p>

📊 Impact of Volunteering Participation on GPA:
- Rata-rata GPA siswa yang melakukan kegiatan volunteering sedikit lebih tinggi dibandingkan dengan siswa yang tidak melakukan volunteering.
- Perbedaannya memang tidak sebesar pada aktivitas seperti music atau extracurricular, namun tren positif tetap terlihat.

Kesimpulan:
- Keterlibatan ini mendorong rasa makna dan motivasi intrinsik siswa, yang bisa meningkatkan komitmen terhadap pembelajaran.
- Siswa yang aktif dalam kegiatan sosial cenderung memiliki disiplin diri dan manajemen waktu yang lebih baik.

#### Ananlisis data pada fitur kategori `ParentalSupport` dengan `GradeClass`

<p align="center">
  <img src="https://github.com/user-attachments/assets/30a8550f-9828-4caf-984c-2cc48041f977" width="500"/>
</p>

📊 Impact of Parental Support on GPA
1. Tingkat Dukungan Orang Tua Rendah hingga Sedang:
  - Mayoritas siswa dengan dukungan orang tua "None", "Low", hingga "Moderate" mendapatkan nilai Grade F, yang jumlahnya sangat tinggi.
  - Hal ini menunjukkan bahwa kurangnya dukungan orang tua berkorelasi negatif terhadap performa akademik siswa.
2. Tingkat Dukungan Tinggi hingga Sangat Tinggi:
  - Ketika dukungan meningkat menjadi "High" atau "Very High", proporsi siswa yang mendapatkan nilai Grade A dan B meningkat.
  - Sementara jumlah siswa dengan Grade F menurun secara signifikan, terutama pada kategori "Very High".
3. Grade A Paling Banyak pada Dukungan "High":
  - Jumlah siswa dengan Grade A tertinggi berada pada kategori "High", bukan "Very High".
  - Ini mungkin menunjukkan bahwa dukungan berlebihan tidak selalu berkorelasi langsung dengan hasil akademik tertinggi, mungkin karena tekanan atau intervensi berlebihan.
4. Distribusi Merata pada Dukungan "Very High":
  - Pada tingkat "Very High", distribusi antar grade terlihat lebih seimbang, dengan penurunan tajam pada Grade F namun juga tanpa lonjakan signifikan pada Grade A.
  - Bisa diinterpretasikan bahwa dukungan sangat tinggi membantu menstabilkan performa siswa, mencegah kegagalan, tapi tidak selalu mendorong performa terbaik.

Kesimpulan:
- Dukungan orang tua memainkan peran penting dalam pencapaian akademik siswa.
- Dukungan yang rendah dikaitkan dengan risiko lebih tinggi mendapatkan nilai rendah (Grade D dan F).
- Dukungan yang cukup hingga tinggi dapat meningkatkan kemungkinan mendapatkan nilai yang lebih baik, namun dukungan yang terlalu tinggi belum tentu menjamin hasil maksimal.
- Strategi pendampingan yang seimbang dan suportif lebih efektif dibanding kontrol berlebihan.

#### Melihat korelasi variabel numerik dengan menggunakan `Heatmap`

<p align="center">
  <img src="https://github.com/user-attachments/assets/39a850ad-8c73-4381-b571-46c6fd4fa2f9" width="500"/>
</p>

📊 Numerical Variable Correlation Heatmap:
1. Absences vs GPA:

  🔵 Korelasi = -0.92 (sangat kuat negatif)
  - Artinya, semakin sering siswa absen, semakin rendah nilai GPA mereka.
  - Ini menunjukkan hubungan yang sangat signifikan bahwa kehadiran sangat memengaruhi prestasi akademik.

2. StudyTimeWeekly vs GPA:

  🟠 Korelasi = 0.18 (lemah positif)
  - Waktu belajar per minggu memiliki korelasi positif tapi lemah terhadap GPA.
  - Ini mengindikasikan bahwa lebih banyak belajar cenderung meningkatkan GPA, meskipun efeknya tidak terlalu besar secara statistik dalam data ini.

3. Age vs GPA:

  ⚪ Korelasi ≈ 0.00 (tidak signifikan)
  - Usia siswa tidak menunjukkan pengaruh berarti terhadap GPA.
  - Bisa disimpulkan bahwa usia bukan faktor utama dalam menentukan kinerja akademik pada kelompok data ini.

#### Melihat `Plot Scatter` yang Memiliki Nilai Korelasi Positif dan Negatif

<p align="center">
  <img src="https://github.com/user-attachments/assets/ba2b3ec5-eacc-44f3-a5c5-258b972a1afc" width="500"/>
</p>

Insight:
- Terdapat korelasi negatif kuat: semakin banyak absen, semakin rendah GPA.
- Kemiringan garis regresi negatif tajam, menandakan hubungan yang signifikan.
- Data cukup konsisten menurun dari kiri ke kanan - semakin sering siswa tidak hadir, prestasinya cenderung menurun secara konsisten.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1517fb54-6623-458d-988e-7ac30a02bd04" width="500"/>
</p>

Insight:
- Terdapat korelasi positif lemah: saat waktu belajar meningkat, GPA cenderung meningkat juga.
- Kemiringan garis regresi positif, tetapi sangat landai artinya, tambahan waktu belajar memberikan pengaruh kecil terhadap peningkatan GPA.
- Sebaran data sangat menyebar, menunjukkan banyak variabel lain yang memengaruhi GPA selain waktu belajar.

## Data Preparation
Pada tahap ini kita akan melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Ada beberapa tahap persiapan data perlu dilakukan, yaitu:

1. Cleaning Data.
2. Encoding Categorical Feature.
3. Data Spliting.

### Cleaning Data
Melakukan drop column `StudentID`, `Ethnicity` dan `ParentalEducation`.
```
Dataset After Cleaning :

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2392 entries, 0 to 2391
Data columns (total 12 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   Age              2392 non-null   int64  
 1   Gender           2392 non-null   object 
 2   StudyTimeWeekly  2392 non-null   float64
 3   Absences         2392 non-null   int64  
 4   Tutoring         2392 non-null   object 
 5   ParentalSupport  2392 non-null   object 
 6   Extracurricular  2392 non-null   object 
 7   Sports           2392 non-null   object 
 8   Music            2392 non-null   object 
 9   Volunteering     2392 non-null   object 
 10  GPA              2392 non-null   float64
 11  GradeClass       2392 non-null   object 
dtypes: float64(2), int64(2), object(8)
memory usage: 224.4+ KB
```

### Encoding Categorical Feature
Pada bagian ini, karena dataset fitur kategori kita sebelumnya sudah diubah dalam bentuk objek (string) pada tahap eksplorasi data analis maka kita perlu mengubah data kategori (yang berbentuk teks atau label) menjadi format numerik agar dapat diproses oleh algoritma machine learning.
Encoding Fitur Kategorikal dilakukan 3 bagian, yakni:
1. *Label Encoding* berfungsi untuk mengonversi nilai kategori menjadi angka integer (0 dan 1). Variabel yang akan diproses yakni
  a. `Tutoring` (Apakah siswa mengikuti bimbingan belajar?)
  b. `Extracurricular` (Apakah siswa mengikuti kegiatan ektrakulikuler?)
  c. `Sports` (Apakah siswa mengikuti kegiatan olahraga?
  d. `Music` (Apakah siswa mengikuti kegiatan musik?) 
  e. `Volunteering` (Apakah siswa mengikuti kegiatan sukarelaan?)
```
# Label Encoding
# Membuat list kolom-kolom kategorikal yang memiliki entri antara yes dan no
categorical_col = ["Tutoring", "Extracurricular", "Sports", "Music", "Volunteering"]

# Mengubah nilai yes menjadi 1 dan nilai no menjadi 0 pada seluruh kolom tersebut
for i in categorical_col:
    student_df[i] = student_df[i].map({"Yes": 1, "No": 0})
```
2. *One Hot Ecoding* berfungsi untuk mengubah setiap kategori menjadi kolom biner terpisah untuk data tidak terurut). Variabel yang akan diproses yakni `Gender`.
```
# One-hot Encoding
# Membentuk kolom dummy dari kolom Gender
data_encoded = pd.get_dummies(student_df[["Gender"]], drop_first = True)

# Menggabungkan data asli dengan data dummy yang telah dibuat
student_df = pd.concat([student_df, data_encoded], axis = 1)

# Menghapus kolom Gender
student_df.drop(columns = ["Gender"], inplace = True)
```
3. *Ordinal Encoding* berfungsi untuk memberikan nilai integer berdasarkan hierarki atau urutan kategori. Variabel yang akan diproses yakni `ParentalSupport`.

```
# Ordinal Encoding
# Mendefinisikan urutan encoding
encoding_mapping = {'Very High':4, 'High':3, 'Moderate':2, 'Low':1, 'None':0}

# Lakukan encoding
student_df['ParentalSupport'] = student_df['ParentalSupport'].map(encoding_mapping)

# Menampilkan 5 baris pertama dari data setelah dilakukan data preprocessing
student_df.head(100)
```
Output :
```
   Age  StudyTimeWeekly  Absences  Tutoring  ParentalSupport  Extracurricular  \
0   17        19.833723         7         1                2                0   
1   18        15.408756         0         0                1                0   
2   15         4.210570        26         0                2                0   
3   17        10.028829        14         0                3                1   
4   17         4.672495        17         1                3                0   

   Sports  Music  Volunteering       GPA GradeClass  Gender_Wanita  
0       0      1             0  2.929196    Grade C           True  
1       0      0             0  3.042915    Grade B          False  
2       0      0             0  0.112602    Grade F          False  
3       0      0             0  2.054218    Grade D           True  
4       0      0             0  1.288061    Grade F           True   
```

### Data Split
Menghapus kolom target GradeClass untuk variabel x, dan menetapkan GradeClass sebagai y target. Lalu membagi data menjadi 2 bagian, yaitu data training dan data testing dengan perbandingan 80% untuk data training dan 20% untuk data testing. 
```
# Menyiapkan fitur (X) dan target (y)
x = student_df.drop('GradeClass',axis=1)
y = student_df['GradeClass']  # Target

#  Membagi data menjadi 20% test size
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    stratify=y,
    shuffle=True,
    random_state=15
)
```
Dan didapat hasil sebagai berikut:
```
Ukuran x_train:  (1913, 11)
Ukuran x_test:  (479, 11)
Ukuran y_train:  (1913,)
Ukuran y_test:  (479,)
```
Lalu menggunakan Label Encoder:
```
# And the LabelEncoder for the target:
le = LabelEncoder()

# Melakukan fitting terhadap data training dan mentransformasikan data training dan testing
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
```

## Modeling
Pada bagian ini, saya akan menggunakan 4 model machine learning untuk menguji dan membandingkan beberapa akurasi dari model, dan akan melakukan evaluasi terhadap model dengan akurasi terbaik untuk dapat digunakan memprediksi prestasi siswa.

1. Model `Random Forest`
2. Model `XGBoost`
3. Model `SVM`
4. Model `Naive Bayes`

### Model Random Forest
Pertama saya menggunakan algoritma ensemble yang sangat populer, yaitu Random Forest, untuk melakukan prediksi prestasi siswa. Algoritma ini bekerja dengan membangun sejumlah pohon keputusan (decision trees) selama proses pelatihan, lalu menggabungkan hasil dari masing-masing pohon tersebut. Untuk klasifikasi, penggabungan dilakukan dengan metode voting, sedangkan untuk regresi digunakan rata-rata. Pendekatan ini terbukti efektif dalam meningkatkan akurasi prediksi sekaligus mengurangi risiko overfitting.

Saya mengimplementasikan Random Forest menggunakan `RandomForestClassifier` dari library `sklearn.ensemble`. Saya melatih model dengan data `x_train` dan `y_train`, lalu mengujinya menggunakan `x_test` dan `y_test`, yang merupakan data uji. Beberapa parameter penting yang saya atur dalam model ini antara lain: `n_estimators = 200`, yaitu jumlah pohon keputusan yang dibangun; `criterion = "entropy"`, yaitu fungsi yang digunakan untuk menilai kualitas pemisahan data; `max_depth = 10`, yaitu kedalaman maksimum dari masing-masing pohon; serta `random_state = 50`, yang berfungsi mengontrol nilai acak untuk memastikan hasil yang konsisten.

Kelebihan:
- Sangat akurat untuk prediksi Grade C (72), Grade D (73), dan terutama Grade F (239).
- Grade B juga diprediksi dengan cukup baik (47 benar, hanya 7 salah prediksi).

Kelemahan:
- Sedikit kesulitan dalam mengenali Grade A (13 benar, 8 salah), cukup tinggi untuk kategori ini.
- Ada kesalahan minor seperti Grade B - Grade F (4 kasus).

### Model XGBoost
Lanjut algoritma kedua yang saya gunakan adalah Extreme Gradient Boosting (XGBoost) karena dikenal sangat kuat untuk tugas klasifikasi dan regresi. XGBoost (Extreme Gradient Boosting) adalah algoritma boosting berbasis pohon keputusan yang membangun model secara iteratif. Pada setiap iterasi, XGBoost membuat pohon baru yang belajar untuk “memperbaiki” kesalahan residu dari model sebelumnya dengan meminimalkan fungsi loss (biasanya log-loss untuk klasifikasi) ditambah regularisasi untuk mencegah overfitting. Pembaruan bobot dilakukan menggunakan gradient descent pada loss function—sehingga setiap pohon baru diarahkan untuk menurunkan gradien loss secara optimal. Teknik ini menghasilkan model ensemble yang sangat kuat dengan kemampuan menangani missing value dan interaksi fitur kompleks secara otomatis. 

Saya mengimplementasikannya dengan `XGBClassifier` dari library `xgboost`, melatih model menggunakan `x_train` dan `y_train`, lalu mengujinya dengan `x_test` dan `y_test`. Dan saya mengatur beberapa parameter penting: `max_depth = 6`, `n_estimators = 125`, `random_state = 30`, `learning_rate = 0.01`, dan `n_jobs = -1` untuk memaksimalkan performa dan efisiensi model.

Kelebihan:
- Meningkatkan akurasi pada Grade A (16 benar, hanya 5 salah), lebih baik dibandingkan Random Forest.
- Konsisten sangat baik untuk Grade C (72), Grade D (73), dan Grade F (238).
- Grade B juga cukup akurat (48 benar).

Kelemahan:
- Hampir tidak ada, distribusi kesalahan sangat minim dan merata.

### Model SVM
Dan Algoritma ketiga yaitu, Support Vector Machine (SVM) dimana algoritma ini akan membangun sebuah hyperplane di ruang fitur berdimensi tinggi yang memaksimalkan margin, yaitu jarak terdekat antara hyperplane dengan data training terdekat (support vectors). Dengan kernel trick (mis. RBF), SVM dapat memetakan data non-linier ke ruang berdimensi lebih tinggi, di mana pemisahan linear menjadi mungkin. Pada prediksi, SVM menentukan sisi hyperplane mana sebuah titik data berada, sehingga secara efisien memisahkan kelas—cocok untuk dataset dengan pola batas keputusan yang kompleks. SVM merupakan algoritma yang efektif untuk klasifikasi, terutama dalam kasus dimana data memiliki struktur yang kompleks. 

Saya menggunakan `SVC` dari library `sklearn.svm` untuk melatih model dengan `x_train` dan `y_train`, lalu mengujinya dengan `x_test` dan `y_test`. Saya mengatur beberapa parameter penting: `kernel = "rbf"`, `gamma = "auto"`, `random_state = 50` untuk memastikan hasil yang konsisten.

Kelebihan:
- Masih mampu mengenali Grade F (229 benar) dan Grade C (56 benar) dengan baik.

Kelemahan:
- Performa untuk Grade A (5 benar dari 21) dan Grade B (31 benar dari 54) cukup buruk.
- Banyak kesalahan antar kelas tengah seperti Grade B - C, D - C, C - D.
- Banyak prediksi Grade D salah ke Grade F (16 kasus).

### Model Naive Bayes
Model keempat yang saya gunakan yaitu, Naive Bayes adalah classifier probabilistik yang menerapkan Teorema Bayes dengan asumsi independence antar fitur. Untuk setiap kelas `𝐶`, model menghitung probabilitas posterior `𝑃(𝐶∣𝑥)∝𝑃(𝐶)∏𝑖𝑃(𝑥𝑖∣𝐶)`, di mana `𝑃(𝐶)` adalah prior kelas dan `𝑃(𝑥𝑖∣𝐶)` bisa dihitung (misal dengan distribusi Gaussian untuk continuous data). Meskipun *naif* karena mengabaikan korelasi fitur, pendekatan ini sangat cepat, hemat memori, dan sering memberikan baseline yang kuat untuk klasifikasi teks atau data tabular berskala besar.

Naive Bayes. Meskipun namanya "naif", Naive Bayes telah terbukti efektif dalam banyak kasus klasifikasi. Saya menggunakan `GaussianNB` dari library `sklearn.naive_bayes` untuk melatih model dengan `x_train` dan `y_train`, lalu menguji dengan `x_test` dan `y_test`. Saya mengatur parameter `var_smoothing=1e-9` untuk mengatasi masalah numeriik.

Kelebihan:
- Cukup baik untuk prediksi Grade C (61) dan Grade D (65).

Kelemahan:
- Grade A (2 benar dari 21) dan Grade B (33 benar dari 54) jauh dari baik.
- Salah satu error terbesar adalah Grade F - Grade D (21 kasus), sangat tinggi.
- Banyak kebingungan antara Grade B dan Grade C.

### Model Terbaik

<p align="center">
  <img src="https://github.com/user-attachments/assets/a275fed6-04df-4d50-8dd5-136c6a155d69" width="500"/>
</p>

|index|Model|Accuracy|
|---|---|---|
|1|XGBoost|93\.32|
|0|Random Forest|92\.69|
|3|Naive Bayes|79\.33|
|2|SVM|78\.08|

Berdasarkan data di atas, model terbaik yang saya gunakan adalah XGBoost dengan akurasi 93.32%. Model ini menunjukkan performa yang sangat baik dalam memprediksi grade siswa dan memiliki kelebihan dalam memprediksi grade C dan D. Model ini juga memiliki kekurangan dalam memprediksi grade A dan B, tetapi masih lebih baik daripada model lainnya.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

### Penjelasan Confussion Matrix

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*IzN36IDL95ASZcV7g_KRUg.jpeg" width="500"/>
</p>

Confusion matrix adalah sebuah tabel yang digunakan untuk mengevaluasi kinerja model klasifikasi dengan membandingkan nilai prediksi dari model terhadap nilai aktual yang sebenarnya. Tabel ini menyajikan informasi dalam empat kategori utama:

1. True Positive (TP): Jumlah kasus positif yang berhasil diprediksi dengan benar sebagai positif oleh model.
2. False Positive (FP): Jumlah kasus negatif yang secara keliru diprediksi sebagai positif oleh model (dikenal juga sebagai Type I Error).
3. False Negative (FN): Jumlah kasus positif yang keliru diprediksi sebagai negatif oleh model (dikenal juga sebagai Type II Error).
4. True Negative (TN): Jumlah kasus negatif yang berhasil diprediksi dengan benar sebagai negatif.

Setelah membentuk confusion matrix, saya menggunakan empat metrik utama untuk mengevaluasi performa model klasifikasi:
- **Akurasi (Accuracy)**: Merupakan rasio antara jumlah prediksi yang benar (TP + TN) dengan total jumlah data (TP + TN + FP + FN).
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*XjVhud9BW7vq5J_fUprnLg.png" height="80"/>
</p>

- **Precision**: Merupakan rasio antara jumlah prediksi positif yang benar (TP) dengan jumlah prediksi positif secara keseluruhan (TP + FP).
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1240/1*DoGL8YNxBOwkX_gd9P_CEA.png" height="80"/>
</p>

- **Recall**: Merupakan rasio antara jumlah prediksi positif yang benar (TP) dengan jumlah kasus positif sebenarnya (TP + FN).
<p align="center">
  <img src="https://miro.medium.com/max/538/1*OV0hfgCStTI8hy6lAY1SdA.jpeg" height="80"/>
</p>

- **F1 Score**: Merupakan rata-rata dari precision dan recall. F1 score dapat dihitung dengan menggunakan rumus berikut:
<p align="center">
    <img src="https://github.com/user-attachments/assets/de176d91-a6b6-40a7-adc4-dd0d755eaa16" height="80"/>
</p>

### Implementasi Confussion Matrix
#### Confussion Matrix Random Forest
<p align="center">
    <img src="https://github.com/user-attachments/assets/b3ec8727-e38c-4da4-aea5-7867b1e361c7" width="500"/>
</p>

Dan berikut adalah `classification_report` model Random Forest:

```
Akurasi pada data uji: 92.69 %

Laporan Klasifikasi untuk Model Random Forest:
              precision    recall  f1-score   support

     Grade A       0.87      0.62      0.72        21
     Grade B       0.87      0.87      0.87        54
     Grade C       0.92      0.92      0.92        78
     Grade D       0.92      0.88      0.90        83
     Grade F       0.94      0.98      0.96       243

    accuracy                           0.93       479
   macro avg       0.91      0.86      0.88       479
weighted avg       0.93      0.93      0.93       479
```
Insight:
> Random Forest memberikan performa stabil dan akurat, terutama untuk kelas dengan jumlah data besar seperti Grade F. Namun masih ada sedikit kebingungan antar kelas yang berdekatan (A - B, D - C), meskipun tidak signifikan.

#### Confussion Matrix XGBoost

<p align="center">
    <img src="https://github.com/user-attachments/assets/d32222ed-2260-42d3-ae10-7e1af7436b7c" width="500"/>
</p>

Dan berikut adalah `classification_report` model XGBoost:
```
Akurasi pada data uji: 93.32 %

Laporan Klasifikasi untuk Model XGBoost:
              precision    recall  f1-score   support

     Grade A       0.89      0.76      0.82        21
     Grade B       0.92      0.89      0.91        54
     Grade C       0.94      0.92      0.93        78
     Grade D       0.91      0.88      0.90        83
     Grade F       0.94      0.98      0.96       243

    accuracy                           0.93       479
   macro avg       0.92      0.89      0.90       479
weighted avg       0.93      0.93      0.93       479
```
Insight:
> XGBoost memberikan hasil paling stabil dan presisi tinggi di antara semua model. Hampir tidak ada label yang benar-benar membingungkan model, menandakan pemisahan fitur yang efektif. Ini menandakan XGBoost kemungkinan adalah model terbaik dari keempatnya untuk kasus ini.

#### Confussion Matrix SVM

<p align="center">
    <img src="https://github.com/user-attachments/assets/fd98189a-0ef7-4c98-975c-81a711a38839" width="500"/>
</p>

Dan berikut adalah `classification_report` model SVM:
```
Akurasi pada data uji: 78.08 %

Laporan Klasifikasi untuk Model SVM:
              precision    recall  f1-score   support

     Grade A       0.50      0.24      0.32        21
     Grade B       0.61      0.57      0.59        54
     Grade C       0.67      0.72      0.70        78
     Grade D       0.68      0.64      0.66        83
     Grade F       0.89      0.94      0.92       243

    accuracy                           0.78       479
   macro avg       0.67      0.62      0.64       479
weighted avg       0.77      0.78      0.77       479
```
Insight:
> SVM tampaknya mengalami overlap antar kelas tengah (B-C-D). Hal ini mengindikasikan bahwa decision boundary SVM tidak bekerja optimal di dataset ini—kemungkinan karena distribusi data tidak linier atau kurang terpisah dengan jelas.

#### Confussion Matrix Naive Bayes

<p align="center">
    <img src="https://github.com/user-attachments/assets/429a97b8-dc9f-45ae-a7f3-9f533588f25e" width="500"/>
</p>

Dan berikut adalah `classification_report` model Naive Bayes:
```
Akurasi pada data uji: 79.33 %

Laporan Klasifikasi untuk Model Naive Bayes:
              precision    recall  f1-score   support

     Grade A       0.67      0.10      0.17        21
     Grade B       0.56      0.61      0.58        54
     Grade C       0.69      0.78      0.73        78
     Grade D       0.68      0.78      0.73        83
     Grade F       0.94      0.90      0.92       243

    accuracy                           0.79       479
   macro avg       0.71      0.63      0.63       479
weighted avg       0.80      0.79      0.79       479
```
Insight:
> Naive Bayes tampaknya tidak cocok untuk dataset ini, mungkin karena asumsi independensi antar fitur yang tidak terpenuhi. Model ini terlihat sangat keliru dalam memetakan Grade F dan Grade A. Ini juga bisa berarti data memiliki fitur yang saling tergantung atau tidak cocok dengan distribusi probabilistik Naive Bayes.

<!-- **Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja. -->

## Kesimpulan
### 1. Faktor yang berpengaruh terhadap Student Grade Class

<p align="center">
    <img src="https://github.com/user-attachments/assets/83ae436a-53cb-4920-a7a5-4d7b612a318d" width="500"/>
</p>

- GPA adalah prediktor paling kuat untuk menentukan prestasi siswa ke depan.
- Faktor lain seperti dukungan orang tua, waktu belajar, dan absensi juga berpengaruh, namun jauh lebih kecil. Digunakan sebagai fitur tambahan untuk meningkatkan akurasi model.
- Aktivitas luar sekolah dan demografi seperti usia & jenis kelamin tidak memberi pengaruh besar dalam model prediksi ini.

### 2. Menggunakan XGBoost sebagai model terbaik untuk menampilkan Inference
Membuat function infer_grade_class untuk memprediksi grade class berdasarkan input data yang diberikan. Function ini menggunakan model XGBoost yang telah dilatih sebelumnya.
```
def infer_grade_class(model, label_encoder):
    """
    Fungsi untuk melakukan prediksi GradeClass siswa.
    model: trained classifier (sudah fit pada X yang berurutan seperti di bawah)
    label_encoder: LabelEncoder yang dipakai untuk encode target GradeClass
    """

    print("Masukkan data berikut untuk prediksi GradeClass siswa:\n")

    # 1. Input numerik
    age               = int(  input("Umur siswa (Tahun): ") )
    study_time_weekly = float(input("Jam belajar per minggu (StudyTimeWeekly): ") )
    absences          = int(  input("Jumlah absensi (Absences): ") )
    gpa               = float(input("GPA (0.0 – 4.0): ") )

    # 2. Input kategorikal biner / ordinal
    tutoring          = input("Mengikuti bimbingan belajar? (1:Ya, 0:Tidak): ").strip()
    parental_support  = int(  input("Skala dukungan orang tua (misal 1–3): ") )
    extracurricular    = input("Ekstrakurikuler? (1:Ya, 0:Tidak): ").strip()
    sports            = input("Ikut kegiatan olahraga? (1:Ya, 0:Tidak): ").strip()
    music             = input("Ikut kegiatan musik? (1:Ya, 0:Tidak): ").strip()
    volunteering      = input("Ikut kegiatan sosial/volunteering? (1:Ya, 0:Tidak): ").strip()
    gender_input      = input("Jenis kelamin (Pria/Wanita): ").strip().lower()

    # 3. Mapping ke integer sesuai kolom X pada model
    tutoring       = 1 if tutoring in ('1','ya','yes','y') else 0
    extracurricular = 1 if extracurricular in ('1','ya','yes','y') else 0
    sports         = 1 if sports in ('1','ya','yes','y') else 0
    music          = 1 if music in ('1','ya','yes','y') else 0
    volunteering   = 1 if volunteering in ('1','ya','yes','y') else 0
    gender_wanita  = 1 if gender_input in ('female','wanita','f') else 0

    # 4. Susun fitur sesuai urutan X.columns:
    fitur_input = np.array([[
        age, study_time_weekly, absences, tutoring, parental_support,
        extracurricular, sports, music, volunteering, gpa, gender_wanita
    ]])

    # 5. Prediksi (menghasilkan encoded label, misal 0–4)
    pred_encoded = model.predict(fitur_input)

    # 6. Inverse-transform untuk dapat kelas asli (angka/string)
    pred_raw = label_encoder.inverse_transform(pred_encoded)[0]

    # 7. Mapping ke letter grade (jika label_encoder memberikan angka 0–4):
    letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

    # _jika_ label_encoder.classes_ sudah berisi ['F','D','C','B','A'],
    # maka pred_raw == 'B' langsung; kalau numeric, pakai mapping:
    if isinstance(pred_raw, (int, np.integer)):
        grade_letter = letter_map.get(int(pred_raw), str(pred_raw))
    else:
        # pred_raw mungkin string seperti '3' atau 'B'
        grade_letter = pred_raw if pred_raw in letter_map.values() else str(pred_raw)

    print(f"\nPrediksi GradeClass siswa adalah: {grade_letter}")
    return grade_letter
```
Lanjut memanggil function infer_grade_class() dengan xgb model sebagai parameter dan label encode:
```
infer_grade_class(xgb_model, le)
```
Lalu hasil yang didapat sebagai berikut:
```
Masukkan data berikut untuk prediksi GradeClass siswa:

Umur siswa (Tahun): 18
Jam belajar per minggu (StudyTimeWeekly): 9
Jumlah absensi (Absences): 0
GPA (0.0 – 4.0): 3.9
Mengikuti bimbingan belajar? (1:Ya, 0:Tidak): 0
Skala dukungan orang tua (misal 1–3): 2
Ekstrakurikuler? (1:Ya, 0:Tidak): 0
Ikut kegiatan olahraga? (1:Ya, 0:Tidak): 0
Ikut kegiatan musik? (1:Ya, 0:Tidak): 0
Ikut kegiatan sosial/volunteering? (1:Ya, 0:Tidak): 0
Jenis kelamin (Pria/Wanita): Pria

Prediksi GradeClass siswa adalah: Grade A
Grade A
```

Insight:

📌 Ringkasan Input Siswa

| Fitur                         | Nilai | Interpretasi                                 |
|-------------------------------|-------|-----------------------------------------------|
| **Umur**                      | 18    | Usia tipikal siswa akhir SMA                  |
| **Jam belajar/minggu**        | 9     | Di atas rata-rata → dedikasi tinggi           |
| **Absensi**                   | 0     | Tidak pernah bolos → sangat disiplin          |
| **GPA**                       | 3.9   | Sangat tinggi (skala 0–4)                     |
| **Bimbingan belajar (bimbel)**| 0     | Belajar mandiri                               |
| **Dukungan orang tua**        | 2     | Cukup positif                                 |
| **Ekstrakurikuler, Olahraga, Musik, Sosial** | 0 | Fokus pada akademik (tidak ikut kegiatan) |
| **Jenis kelamin**             | Pria  | Netral (tidak mempengaruhi prediksi signifikan) |

---
🔍 Faktor Berpengaruh

1. **GPA 3.9**  
   Fitur paling dominan; nilai hampir maksimal → korelasi kuat dengan Grade A.

2. **Absensi = 0**  
   Disiplin tinggi → sinyal positif untuk performa akademik.

3. **Jam belajar 9 jam/minggu**  
   Terletak di kuantil atas distribusi study time → mendukung prediksi Grade A.

4. **Tidak ikut kegiatan non-akademik**  
   Model memprioritaskan variabel akademik (GPA, absensi, study time) daripada ektrakurikuler.

5. **Tidak bimbel**  
   Menunjukkan bahwa bimbel bukan penentu utama—GPA dan jam belajar sudah cukup.
---
✅ Konsistensi dengan Evaluasi Model

- **XGBoost** memiliki **accuracy 93.32%** (tertinggi), dan Grade A dideteksi dengan akurasi **76.2%** (16/21) pada confusion matrix.
- Kombinasi **GPA tinggi**, **absensi nol**, dan **study time intensif** sesuai pola “siswa berprestasi” di dataset.
---

## Reference
1. Ambarita, M. N., Nasution, M., & Ah, R. M. (2024). Analisis prediksi prestasi siswa UPTD SD Negeri 30 Aek Batu dalam machine learning dengan metode Naive Bayes. Jurnal Informatika, 5(1), 45–57.
2. Mentari, P., & Nurhaeka. (2024). Prediksi prestasi siswa SMA Negeri 1 Muntok berdasarkan motivasi belajar, disiplin, dan status sosial-ekonomi keluarga. Jurnal Kesatria, 10(1), 12–20.
3. Murad, D. F., Wijanarko, B. D., Murad, S. A., & Windyasari, V. S. (2023). Pengukuran prestasi belajar mahasiswa berdasarkan prediksi nilai menggunakan General Linear Model. Jurnal Sistem Informasi Bisnis, 13(2), 135–142.
4. Prasad, K., Singh, R., & Sharma, S. (2024). Student performance prediction using machine learning algorithms. International Journal of Distributed Sensor Networks, 2024, Article 987654.
