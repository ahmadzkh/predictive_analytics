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

### Sumber Data
Pada proyek ini, kita menggunakan **Students Performance Dataset** yang diambil dari Kaggle. Dataset ini berisi data akademik dan demografis siswa sekolah menengah atas, dengan **1.000 baris** dan **12 kolom** fitur, antara lain usia, jam belajar per minggu, jumlah absensi, nilai GPA, partisipasi dalam bimbingan belajar, dukungan orang tua, dan keterlibatan dalam berbagai kegiatan ekstrakurikuler. Anda dapat mengunduh dataset lengkap di:  
[Students Performance Dataset – Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset/data)

### Target: **GradeClass**

Variabel target **GradeClass** dikonstruksi dari nilai akhir (GPA) siswa dan dikelompokkan menjadi lima kategori, yaitu:

- **Grade A**: Siswa dengan GPA ≥ 3.7  
- **Grade B**: Siswa dengan GPA ≥ 3.0 dan < 3.7  
- **Grade C**: Siswa dengan GPA ≥ 2.3 dan < 3.0  
- **Grade D**: Siswa dengan GPA ≥ 1.7 dan < 2.3  
- **Grade F**: Siswa dengan GPA < 1.7  

Kelima kategori inilah yang menjadi **target prediksi** model klasifikasi di proyek ini.  

## Exploratory Data Analysis (EDA)
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


### Unvariative Analysis EDA
#### Distribusi Categorical Column menggunakan Bar Plot

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447391920-1de4fd14-0fe6-450d-8a12-99ce0ac40997.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjY5ODQsIm5iZiI6MTc0ODIyNjY4NCwicGF0aCI6Ii83NTc3MjY1OS80NDczOTE5MjAtMWRlNGZkMTQtMGZlNi00NTBkLThhMTItOTljZTBhYzQwOTk3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyMzEyNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ2MmVhMDRiZjFkZjcxZmFmYzVjZGEzZDYzN2E5ZDExNDZhZWU0MTFjYzlmNmI3MzExNmNhYmMzOTFiZTBhNGImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.VIIzEY_c8alrCEfqWkFUNkNi996KtBLcyiiT23Vr9rw" width="1000"/>
</p>

📊 Distribution of Student Gender:

Kategori:
  - Wanita
  - Pria

Insight:
  - Jumlah siswa wanita lebih banyak daripada pria, meskipun tidak terlalu jauh perbedaannya.
  - Ini mengindikasikan distribusi gender cukup seimbang, namun ada sedikit dominan wanita.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447392151-73f4cd06-5c04-4757-b45c-1ab9c58a12ad.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjcxNTAsIm5iZiI6MTc0ODIyNjg1MCwicGF0aCI6Ii83NTc3MjY1OS80NDczOTIxNTEtNzNmNGNkMDYtNWMwNC00NzU3LWI0NWMtMWFiOWM1OGExMmFkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyMzQxMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTg1OGZkYzljNDY3NWY0NzFiZWE1MDhkYTJlNGRkZjkyMzA5ZTNjYjRlNWJiMTFiMDVjZDk3MTFlZjBkZjM0YjkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.FKwITMfCPUAlnE8dUeSfGT2gj2NxMktBNcdxRfLMWbU" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447392239-1f38eb8f-b927-4ec5-9bc1-38d5ea881dac.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjcyMjAsIm5iZiI6MTc0ODIyNjkyMCwicGF0aCI6Ii83NTc3MjY1OS80NDczOTIyMzktMWYzOGViOGYtYjkyNy00ZWM1LTliYzEtMzhkNWVhODgxZGFjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyMzUyMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTE5ODQ5ODkxODk0MTU3MWVjYzZkZWIyNjVlYjJlMWVlMGE4YWE3MzJkNjNlMGViOWJkZjJiZTQ1NGVlMWIxNGImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mSkAIofc77o3ZanyTQtY8QSjos8lQMF7Gmqi3E4oAOM" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447392310-5bb96b6f-7a5e-43ec-9a50-34ad2fd50f7f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjcyODAsIm5iZiI6MTc0ODIyNjk4MCwicGF0aCI6Ii83NTc3MjY1OS80NDczOTIzMTAtNWJiOTZiNmYtN2E1ZS00M2VjLTlhNTAtMzRhZDJmZDUwZjdmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyMzYyMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTYxZWI2YzAyZjg4YzlmNWQ4MjNkZTBlNWJmMDk4NmNmYzRlYzk1NDJjYzJjMmIxYzlkZmI5NDkyZmUzZTM0YjcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.sB5HBt3lNLOQ5_E-Lk2ql4qNMEUUe4_XYN4sbFxsNlo" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447392415-3779c3ec-53a0-47c0-b26a-30c6e413544f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjgxMDMsIm5iZiI6MTc0ODIyNzgwMywicGF0aCI6Ii83NTc3MjY1OS80NDczOTI0MTUtMzc3OWMzZWMtNTNhMC00N2MwLWIyNmEtMzBjNmU0MTM1NDRmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyNTAwM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA3YTc0NDY0NGE4MjhkODM3MTJmMTEzZmYwNzU0MWNlMzI4ODkyYjVlZGE1MGM2YzY0ZjkwMDJmN2Y5ZmM4OTcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.yHBnQJSZc3Jz1w11rYKxyDj2EGUmRHbKL6wq7hRU-qk" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447392482-5554dc6c-c96f-4999-a6ce-79a2ed0e51e2.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjgxNDAsIm5iZiI6MTc0ODIyNzg0MCwicGF0aCI6Ii83NTc3MjY1OS80NDczOTI0ODItNTU1NGRjNmMtYzk2Zi00OTk5LWE2Y2UtNzlhMmVkMGU1MWUyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyNTA0MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZiYzYyMjE0YzdiMmM1ZmZlOWI4NmQzNjEwYTM0YmJjNzQ2ZDUwOWRlYWRjYTE2ZTQwNTdhMzQ5ODA2NjI1NmMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.F31cq8CnqqJhJaEGsJ-2li1Y0cVqhcS5rWxZJuM7HF0" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447392544-262d226d-b49e-4650-a56a-f2462211268a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjgxODMsIm5iZiI6MTc0ODIyNzg4MywicGF0aCI6Ii83NTc3MjY1OS80NDczOTI1NDQtMjYyZDIyNmQtYjQ5ZS00NjUwLWE1NmEtZjI0NjIyMTEyNjhhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyNTEyM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWU5MDgxM2EyMmMwMDJhMmY4MTBjNDhmNTkzMmMwNjhjZTBhNGQ0YWQ2NWVjODQzMmIzODc0MjU5YTNkMzdkMDkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.wtETv3jsT9pHKkkgh_vdOFrVnlcZMhAK-twXV-poGmM" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447392605-d12890b0-6d16-4314-8482-6b71ffd7e999.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjgyMTQsIm5iZiI6MTc0ODIyNzkxNCwicGF0aCI6Ii83NTc3MjY1OS80NDczOTI2MDUtZDEyODkwYjAtNmQxNi00MzE0LTg0ODItNmI3MWZmZDdlOTk5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyNTE1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWVkMTkzNzU1NzBmYjNkNzg1YzM5ZGRjNTYwMTUwYzNmNjAwNzhiZmE4NDUyYTc3NmJhMGFjZTczMDNhNDM5MGYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.KWFqXWzYLv9yJif8Le0WDMLcz6VrXXOi3Z92IAV6ahA" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447392649-d4dc8037-9200-448d-91d1-3e43a7d0013c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjgyNjUsIm5iZiI6MTc0ODIyNzk2NSwicGF0aCI6Ii83NTc3MjY1OS80NDczOTI2NDktZDRkYzgwMzctOTIwMC00NDhkLTkxZDEtM2U0M2E3ZDAwMTNjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyNTI0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTAyZDg3NTFlZWFlZWNlNjI3ZWRlYTkzNDgwNWUxYmY3YzNhYjM3M2ViMTUxOTljZjIyM2RhMTlhMDQzOTIxZjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.raOcaTSMwha3eSCtT9lk51kxf6jR1_J1FCuWWshub8Q" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447392691-f36b38fd-184b-41b0-85b1-f71b82e16e17.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjgzMzksIm5iZiI6MTc0ODIyODAzOSwicGF0aCI6Ii83NTc3MjY1OS80NDczOTI2OTEtZjM2YjM4ZmQtMTg0Yi00MWIwLTg1YjEtZjcxYjgyZTE2ZTE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyNTM1OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWVlN2RiYmQyZjk2NDdmODYzNjVkMmQ5MTM3MGJkZjE5ZjU1YWU3ZTA1NWVmYTlkYThkNWQ3ZmYwOTZkM2U5NDUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.hfXkudSk8z_BH7nblWJEoEBmiuPNgPpxyrO43qXtv4M" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447392914-6751beb7-5c10-42af-8acf-b8898bfd2eba.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjgzOTcsIm5iZiI6MTc0ODIyODA5NywicGF0aCI6Ii83NTc3MjY1OS80NDczOTI5MTQtNjc1MWJlYjctNWMxMC00MmFmLThhY2YtYjg4OThiZmQyZWJhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyNTQ1N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTYxOTgyZTNkYjlkMjQ4MTk0MTc1ZDRjOTE2NGYyNjAyZGI0Yjk0OGYyYTU4MzEwOGU1NmI0YTc1Mzg1NTg5MDgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.hRjFdQgyYYQZYKgiA9Fkm78txRaEhqNZHbIuRJqu6cY" width="1000"/>
</p>

📊 Box Plot of Student Age:

Insight:
  - Rentang usia siswa adalah dari 15 hingga 18 tahun.
  - Mayoritas siswa berusia antara 15,5 hingga 17 tahun, dengan median di sekitar 16 tahun.
  - Tidak terdapat outlier, menandakan distribusi usia cukup normal untuk siswa SMA.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447392992-0086a488-cb84-4b01-910a-451d6caeceb8.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjg1NDUsIm5iZiI6MTc0ODIyODI0NSwicGF0aCI6Ii83NTc3MjY1OS80NDczOTI5OTItMDA4NmE0ODgtY2I4NC00YjAxLTkxMGEtNDUxZDZjYWVjZWI4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyNTcyNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPThlNjVlOGVlMTlmMWZmOTYyNTFkMDU3OTZkYzUyZTVkNGQ1NGMyYmYwNzM0OTc5NTg1YzRlYzk4ZTVjYjAxMDYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.zpX2VNvz_ZoC5Y4bjs0g6RZbG2Fb5uOYe2KBdkZlt4g" width="1000"/>
</p>

📊 Box Plot of Weekly Study Time:

Insight:
  - Rentang waktu belajar mingguan berkisar antara 0 hingga 20 jam.
  - Median berada sekitar 10 jam/minggu.
  - Hampir semua nilai berada dalam rentang interkuartil, tidak terlihat outlier.
  - Ini mengindikasikan bahwa sebagian besar siswa belajar sekitar 1-2 jam per hari secara konsisten.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447393060-571982de-98eb-41d0-8512-44bb6ba590f2.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjg1NzYsIm5iZiI6MTc0ODIyODI3NiwicGF0aCI6Ii83NTc3MjY1OS80NDczOTMwNjAtNTcxOTgyZGUtOThlYi00MWQwLTg1MTItNDRiYjZiYTU5MGYyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyNTc1NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTgzOGE2ODRkYmZhZDc2OTFhYzM3OTViYjgxMzAxMDg5NzY5ZmM5YmU4NWEzYTJiZjA1NGIxN2VlZTZmNmJjNjgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.DcVFnciWhSPK8y9w0Vj9l9sFlnz972Ae0xDfTNZoKMk" width="1000"/>
</p>

📊 Box Plot of Student Absences:

Insight:
  - Ketidakhadiran siswa bervariasi antara 0 hingga hampir 30 kali.
  - Median berada di sekitar 15 kali absen, dengan distribusi cukup merata. -
  - Tidak ada outlier ekstrem, tapi ada siswa dengan ketidakhadiran yang cukup tinggi (>25 kali).
  - Ini bisa menjadi indikator penting: siswa dengan banyak absen kemungkinan memiliki GPA lebih rendah atau keterlibatan yang minim.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447393144-3979469a-498d-4f0d-aadd-45d8bcf327d3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjg2MjUsIm5iZiI6MTc0ODIyODMyNSwicGF0aCI6Ii83NTc3MjY1OS80NDczOTMxNDQtMzk3OTQ2OWEtNDk4ZC00ZjBkLWFhZGQtNDVkOGJjZjMyN2QzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyNTg0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWMwNDdiNTIyYTY1ZjE3MzBjNTI2ZWEwOTYzZGU1OWMyMjI1Y2RkYmM2NzNlNDQ5NWQ2NmRiYWU1MDU1NTg3Y2EmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.QSDXIuxHCKnqe39Q1QWwsbpBesO9AD8liqNjRe4QVC0" width="1000"/>
</p>

📊 Box Plot of Student GPA:

Insight:
  - GPA (Grade Point Average) berkisar dari 0 hingga 4.0, sesuai dengan skala umum.
  - Median GPA berada di sekitar 2.0, menandakan sebagian besar siswa memiliki performa akademik rata-rata atau kurang dari baik.
  - Ada penyebaran yang cukup seimbang, tanpa outlier ekstrem.
  - Siswa dengan GPA di bawah 2.0 perlu diperhatikan lebih lanjut (bisa berkaitan dengan waktu belajar, absensi, atau dukungan orang tua).

#### Membuat Pie Chart kolom GradeClass

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447393189-94d8c098-513e-48b4-a990-2b6ce725b959.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMjg2NDUsIm5iZiI6MTc0ODIyODM0NSwicGF0aCI6Ii83NTc3MjY1OS80NDczOTMxODktOTRkOGMwOTgtNTEzZS00OGI0LWE5OTAtMmI2Y2U3MjViOTU5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAyNTkwNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA3Y2VmMDkyMjFkMzkzNzQ2MWYwMjBkNjJiZmVjYWQxZTU3MmM4YjY3MjFlZjY2ZjEwZjg5ODM5YzgwZDE3ZGQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.JfehEudZ4231KYqM_N3H_AoaHjujcuz3Iy_pfTasnQw" width="1000"/>
</p>

🔹 Distribution of Student GradeClass:
- Berdasarkan hasil yang ditampilkan sebanyak 50.6% Siswa berada pada Grade F (Kelas dengan Prestasi Terendah) menjadi jumlah terbanyak, dan hanya sedikit sekitar 4.5% Siswa berada pada Grade A (Kelas dengan Prestasi Terbaik).
- Sedangkan siswa yang lainnya berada pada Grade B sekitar 11.2% Siswa, Grade C sekitar 16.3% Siswa dan Grade D sekitar 17.3% Siswa.

#### Membuat Histogram untuk Numerical Column

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447399651-e9c0b2ba-00b0-4752-88b9-ad186e126ea1.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzEyNDcsIm5iZiI6MTc0ODIzMDk0NywicGF0aCI6Ii83NTc3MjY1OS80NDczOTk2NTEtZTljMGIyYmEtMDBiMC00NzUyLTg4YjktYWQxODZlMTI2ZWExLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAzNDIyN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTU1YWMyYTRlNzliMzA2N2I2YThhMmJjZjhiYzIxMDA5NTgzYjZmMmVkMzMyNzhmYWI5OWQzOTUyOWI1MmU5MmImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.dgKuGWApAb-cuNNO_qX6MSwTZpHSGjxNY5UuM4F4Z1k" width="1000"/>
</p>

📊 Distribution of Student Age:
- Distribusi usia terlihat multimodal, dengan puncak di usia 15, 16, 17, dan 18 tahun.
- Hal ini menunjukkan bahwa siswa berasal dari berbagai tingkat atau kelas yang memiliki rentang usia relatif luas.
- Distribusi tidak normal dan menunjukkan pola siklis, kemungkinan karena jumlah siswa di tiap tingkat/kelas hampir merata.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447399769-61913fce-25a9-44cc-a5c3-576f7acdf792.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzEyODIsIm5iZiI6MTc0ODIzMDk4MiwicGF0aCI6Ii83NTc3MjY1OS80NDczOTk3NjktNjE5MTNmY2UtMjVhOS00NGNjLWE1YzMtNTc2ZjdhY2RmNzkyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAzNDMwMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTAyNjI2ZTY3NzQwODIxMTI0OTgyZWNiNjRjMDAyZjBlZTJmNzAyYzdmOThiMWE5MzBjZTU2MWZiM2JiZmFhNGUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.5b7UyCP_gxdPZcK2f_wQQAmLeoVH_efuKiG51szrcpc" width="1000"/>
</p>

⏳ Distribution of Weekly Study Time:
- Distribusi waktu belajar mingguan mendekati normal dengan puncak di sekitar 8 - 10 jam/minggu.
- Artinya, sebagian besar siswa belajar kurang lebih 1 - 1.5 jam per hari.
- Masih ada sebagian kecil siswa yang belajar di bawah 5 jam atau di atas 15 jam, menunjukkan adanya variasi motivasi atau kebiasaan belajar.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447399830-d25860cf-899f-468a-9484-3e6637d270a3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzEzMTEsIm5iZiI6MTc0ODIzMTAxMSwicGF0aCI6Ii83NTc3MjY1OS80NDczOTk4MzAtZDI1ODYwY2YtODk5Zi00NjhhLTk0ODQtM2U2NjM3ZDI3MGEzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAzNDMzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI5NGY5ZDI4YTRhY2JhNzNmZGI5M2JkYmMyYmJmYzM1MWUzYWQ2ZWUwZTJjNzMyODg2NGQ4NDRjNDljNjBiN2UmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.C4SCGiiCcON86TRJIxJ8qiZjYXNs7vXc6HrS4uS-U_0" width="1000"/>
</p>

📅 Distribution of Student Absences:
- Distribusi relatif merata, tetapi ada sedikit puncak pada beberapa hari tertentu (sekitar 0, 5, 10, dan 20 hari).
- Ini bisa berarti tidak ada pola absen yang dominan, namun ada kelompok siswa yang sangat rajin (absen sedikit) dan kelompok dengan tingkat absen cukup tinggi (hingga 25–30 hari).
- Hal ini bisa menjadi indikator potensi masalah kehadiran atau disiplin siswa.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447399901-92b0cbc1-542f-4f28-b85e-1f4b25fb6e23.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzEzNjMsIm5iZiI6MTc0ODIzMTA2MywicGF0aCI6Ii83NTc3MjY1OS80NDczOTk5MDEtOTJiMGNiYzEtNTQyZi00ZjI4LWI4NWUtMWY0YjI1ZmI2ZTIzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAzNDQyM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTFkN2IxNDBmZTM4NWRmNzcxNDRhZGE0NjQ2NzFhZmM4NmFlOGZjNGMxOGNhNjA1NzE2MTYyMmM1M2ZmOTY1NWMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mcQYliaGmHlxhZ4HrgHbkQu4VawQK_OxX0_mVJJ2Ncg" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447399942-8b0b42d9-fc98-490a-ac2b-99607b395654.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzE0ODksIm5iZiI6MTc0ODIzMTE4OSwicGF0aCI6Ii83NTc3MjY1OS80NDczOTk5NDItOGIwYjQyZDktZmM5OC00OTBhLWFjMmItOTk2MDdiMzk1NjU0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAzNDYyOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTQxNWY0MDBkOGNlNDdjNTYwNjIwZThmNDRmZDU4ZjMxM2M4ZjFmYTE1YjRmMDU5NzE0M2IxNTkwODIyMjJiODcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Bi_gD-zoCnJ1j-FKEhnheyhyLksafJ3RoDK2wDMauY4" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447400023-4f2e1ce9-3b35-44b8-b86b-feb89bd758b9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzE1MjIsIm5iZiI6MTc0ODIzMTIyMiwicGF0aCI6Ii83NTc3MjY1OS80NDc0MDAwMjMtNGYyZTFjZTktM2IzNS00NGI4LWI4NmItZmViODliZDc1OGI5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAzNDcwMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTFmMjVjMmE3OTdmOGQ0Y2E2Y2Y1NGRjNzRmYjk5MDIxMWI3NDI4MGY1OTg1MTQ4M2NkMGQwODc5YjcxODkwYzcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.6rHOJnWdZbeVE98YFdjctIfRW2xAyXqVesWKRjbJaAo" width="1000"/>
</p>

📉 Impact of Absence on GPA:
- Sumbu X: Jumlah Absen
- Sumbu Y: GPA
- Garis Merah: Garis regresi linier

Insight:
- Terdapat korelasi negatif kuat: semakin banyak absen, semakin rendah GPA.
- Kemiringan garis regresi negatif tajam, menandakan hubungan yang signifikan.
- Data cukup konsisten menurun dari kiri ke kanan - semakin sering siswa tidak hadir, prestasinya cenderung menurun secara konsisten.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447400023-4f2e1ce9-3b35-44b8-b86b-feb89bd758b9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzE1MjIsIm5iZiI6MTc0ODIzMTIyMiwicGF0aCI6Ii83NTc3MjY1OS80NDc0MDAwMjMtNGYyZTFjZTktM2IzNS00NGI4LWI4NmItZmViODliZDc1OGI5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDAzNDcwMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTFmMjVjMmE3OTdmOGQ0Y2E2Y2Y1NGRjNzRmYjk5MDIxMWI3NDI4MGY1OTg1MTQ4M2NkMGQwODc5YjcxODkwYzcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.6rHOJnWdZbeVE98YFdjctIfRW2xAyXqVesWKRjbJaAo" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447411281-b1f61485-3073-47f5-8c88-a1845a2568a1.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzMyMTcsIm5iZiI6MTc0ODIzMjkxNywicGF0aCI6Ii83NTc3MjY1OS80NDc0MTEyODEtYjFmNjE0ODUtMzA3My00N2Y1LThjODgtYTE4NDVhMjU2OGExLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA0MTUxN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWIxODJiNmQzMWEzMTQxNTNlOTU2YTZkZGM5MjUzODRjNmJiYWZkYzUyMjQ3YjUyNjFjMzBjM2Q4NjhiZGI4ZWMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.cUpYE1H2Ny0NoT0tYEGNmGIZaqs8xAQLtoRKdXZ-bec" width="1000"/>
</p>

📊 Comparison of Tutoring on GPA:
- Tanpa bimbingan (No) menunjukkan jumlah siswa yang lebih banyak mendapatkan Grade F, D, dan C.
- Dengan bimbingan (Yes), distribusi siswa bergeser ke nilai yang lebih baik, dan jumlah siswa dengan Grade F berkurang signifikan.
- Kesimpulan: Bimbingan belajar cenderung berdampak positif terhadap pencapaian nilai siswa.

#### Ananlisis data pada fitur kategori `Gender` dengan  `Grade Class`

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447412071-1e5f09d1-b424-43c9-b677-4f6f002dc670.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzM1MjYsIm5iZiI6MTc0ODIzMzIyNiwicGF0aCI6Ii83NTc3MjY1OS80NDc0MTIwNzEtMWU1ZjA5ZDEtYjQyNC00M2M5LWI2NzctNGY2ZjAwMmRjNjcwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA0MjAyNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJjODE5MjZiMjk5M2RjNmE3MmVlNzYyNjM4MWE5NmFlNDg4MGI1ZTdlOGZkNzBkNzdhNjMzZTE1ZmZlYTg3MDEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.H1Od5mqXo1KC6zY9rbp1rpBSdTF-I6AEr7JOQ_ZAdNY" width="1000"/>
</p>

📊 Comparison of Tutoring on GPA:
- Tanpa bimbingan (No) menunjukkan jumlah siswa yang lebih banyak mendapatkan Grade F, D, dan C.
- Dengan bimbingan (Yes), distribusi siswa bergeser ke nilai yang lebih baik, dan jumlah siswa dengan Grade F berkurang signifikan.
- Kesimpulan: Bimbingan belajar cenderung berdampak positif terhadap pencapaian nilai siswa.

#### Ananlisis data pada fitur kategori kegiatan non akademik `Extracurricular`, `Sports`, `Music`, `Volunteering` dengan `GPA`

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447412253-a4a52d4f-e607-491e-8844-6a18be41846a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzM1OTEsIm5iZiI6MTc0ODIzMzI5MSwicGF0aCI6Ii83NTc3MjY1OS80NDc0MTIyNTMtYTRhNTJkNGYtZTYwNy00OTFlLTg4NDQtNmExOGJlNDE4NDZhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA0MjEzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJmZmFiNmNjNzkxZDAyMGZmM2ZlOWI2ODgyYWE5NDMwYjdhOGExZTY3ZjI1ZTQ0ZmI1YjE2ZWIyOGExNGEwNmYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.HPUmcC9OIBmmXraSyc3zwkEXj6Rlwr432Z7ddzDsQqs" width="1000"/>
</p>

📊 Impact of Extracurricular on GPA:
- Siswa yang mengikuti aktivitas ekstrakurikuler memiliki rata-rata GPA lebih tinggi dibandingkan mereka yang tidak ikut.
- Peningkatan ini menunjukkan bahwa kegiatan di luar akademik seperti organisasi, klub, atau kegiatan komunitas dapat berdampak positif terhadap performa belajar.

Kesimpulan:
- Siswa yang aktif secara sosial memiliki keterampilan manajemen waktu yang lebih baik.
- Extracurricular membangun soft skills seperti tanggung jawab, kerjasama, dan kepemimpinan.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447412675-ed0ea1ac-6585-428a-9076-afcca0346bd4.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzM3MjcsIm5iZiI6MTc0ODIzMzQyNywicGF0aCI6Ii83NTc3MjY1OS80NDc0MTI2NzUtZWQwZWExYWMtNjU4NS00MjhhLTkwNzYtYWZjY2EwMzQ2YmQ0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA0MjM0N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNjNzRjNzk4YmYzODZkMzQ0MGE2ZTM2Y2IyMTUyYjQ1ZmQ4M2ZhNTdkZjAyMTUzMDk2NTMyNmU1NWVkYThjNGMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Y_koI9xLOPSUJY19ebuWAGK9wOaJt8iXB7aERneIR-s" width="1000"/>
</p>

📊 Impact of Sports Participation on GPA:
- Siswa yang terlibat dalam olahraga menunjukkan sedikit peningkatan GPA, meskipun tidak sebesar peningkatan dari extracurricular.
- Ini mengindikasikan bahwa aktivitas fisik memiliki dampak positif ringan terhadap akademik.

Kesimpulan:
  - Olahraga dapat membantu meningkatkan fokus, disiplin, dan kesehatan mental.
  - Namun, beban latihan yang tinggi mungkin juga mengurangi waktu belajar jika tidak dikelola dengan baik.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447412823-fbd229af-f8e4-46c4-bf41-53cdcfe674dc.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzM3NzAsIm5iZiI6MTc0ODIzMzQ3MCwicGF0aCI6Ii83NTc3MjY1OS80NDc0MTI4MjMtZmJkMjI5YWYtZjhlNC00NmM0LWJmNDEtNTNjZGNmZTY3NGRjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA0MjQzMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI4YjhkZGVlMmU3NzQ2ZDgzMGZkNGMyNGE2NzYyNzNmNDM3ZGVhZDVhMTkzYTFkMDUyOGNkNmU0Y2RiNzg2NTgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.m7kg6NqzCE2TMbLQ6ktfdd35N6hDO-S7QJ8eI0LglLs" width="1000"/>
</p>

📊 Impact of Music on GPA:
- Siswa yang aktif di bidang musik (seperti bermain alat musik, paduan suara, band) menunjukkan peningkatan GPA yang signifikan dibandingkan yang tidak aktif di musik.
- Musik diyakini dapat mengaktifkan area otak yang berkaitan dengan konsentrasi, logika, dan kreativitas.

Kesimpulan:
- Pembelajaran musik melibatkan latihan rutin dan disiplin, yang dapat terbawa ke kebiasaan belajar.
- Musik juga mendukung perkembangan memori dan pemrosesan kognitif.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447412968-39b0d4b0-3ee3-47f7-ace7-e7ed017173ff.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzM4MTQsIm5iZiI6MTc0ODIzMzUxNCwicGF0aCI6Ii83NTc3MjY1OS80NDc0MTI5NjgtMzliMGQ0YjAtM2VlMy00N2Y3LWFjZTctZTdlZDAxNzE3M2ZmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA0MjUxNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTk3YTIxMjhiYzQ4MjE2YzQ1M2FhMTY4ZDY0ODQ0NjA4ZDM4MGI5ZWE2NWViY2IyYmI3ODA0ZjZhMDY3ZmIzODQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.EPFPoKEtV9-gcxtJ5QRjplK_5I7MC2wAg5Ye8QcSXYA" width="1000"/>
</p>

📊 Impact of Volunteering Participation on GPA:
- Rata-rata GPA siswa yang melakukan kegiatan volunteering sedikit lebih tinggi dibandingkan dengan siswa yang tidak melakukan volunteering.
- Perbedaannya memang tidak sebesar pada aktivitas seperti music atau extracurricular, namun tren positif tetap terlihat.

Kesimpulan:
- Keterlibatan ini mendorong rasa makna dan motivasi intrinsik siswa, yang bisa meningkatkan komitmen terhadap pembelajaran.
- Siswa yang aktif dalam kegiatan sosial cenderung memiliki disiplin diri dan manajemen waktu yang lebih baik.

#### Ananlisis data pada fitur kategori `ParentalSupport` dengan `GradeClass`

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447413868-f682991c-e26d-4476-b41f-79783bce389a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzQxMjEsIm5iZiI6MTc0ODIzMzgyMSwicGF0aCI6Ii83NTc3MjY1OS80NDc0MTM4NjgtZjY4Mjk5MWMtZTI2ZC00NDc2LWI0MWYtNzk3ODNiY2UzODlhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA0MzAyMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTczOWJhMDBmYTVjZjE5OGQ1NGUxZmM4ZmRiODdiOTY2ODk4YTE2NTY0MDVlMzhiZjM0ZjBjNDFmMDBlNjdjZDYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.e30U7xkhGwn0J_8vQl_gLZUd5xDV0kWzQwFDPEgxydQ" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447414361-2a1f2155-12bd-4c0a-9bab-faa34a83a2a6.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzQyODYsIm5iZiI6MTc0ODIzMzk4NiwicGF0aCI6Ii83NTc3MjY1OS80NDc0MTQzNjEtMmExZjIxNTUtMTJiZC00YzBhLTliYWItZmFhMzRhODNhMmE2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA0MzMwNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBhZTlmZTYxOWUxMmNiOGY3MDhmYWFjMTg1N2Q0YTczOGQ1NDgzMDVhYjhkOWY3MzI5ODNiYTVlNDUyZGNjYzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.8GcVIt1Z41bt0PpQATMaoZfGrutQgm3uOL7GdmXTFig" width="1000"/>
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
  <img src="https://private-user-images.githubusercontent.com/75772659/447414559-0327a20e-7127-429a-aa7f-83a3fb401b87.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzQzNTMsIm5iZiI6MTc0ODIzNDA1MywicGF0aCI6Ii83NTc3MjY1OS80NDc0MTQ1NTktMDMyN2EyMGUtNzEyNy00MjlhLWFhN2YtODNhM2ZiNDAxYjg3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA0MzQxM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTZjNmY4YWQ5YTJlMjY5MDQyYjRiZjg3ZjdlNjM4MmNkMGZiOWIyNmYzMzJlZjNiMzUzODM1MDQ5YTc3NWYxYzQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Nilq-5m7VXrLA73gLXZB12ZC55MoZ-hfEw72jARaCDs" width="1000"/>
</p>

Insight:
- Terdapat korelasi negatif kuat: semakin banyak absen, semakin rendah GPA.
- Kemiringan garis regresi negatif tajam, menandakan hubungan yang signifikan.
- Data cukup konsisten menurun dari kiri ke kanan - semakin sering siswa tidak hadir, prestasinya cenderung menurun secara konsisten.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447415162-1aa5c8b9-30ec-463d-a55c-502d62c3aa66.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzQ1MzksIm5iZiI6MTc0ODIzNDIzOSwicGF0aCI6Ii83NTc3MjY1OS80NDc0MTUxNjItMWFhNWM4YjktMzBlYy00NjNkLWE1NWMtNTAyZDYyYzNhYTY2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA0MzcxOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNmNzI2ZDgxZjgzYmJkMjk1YzQ0MWI0YmMyNTRiYjNjZDZkYWQwNDIwOTM0ZjA5NDMxMTI1NzI3ZDQ3ZDA5MDQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.V5T-5ttfXfTmUE99wrIlaQ-bRLWJsU2JCuuR4ZS2d_Q" width="1000"/>
</p>

Insight:
- Terdapat korelasi positif lemah: saat waktu belajar meningkat, GPA cenderung meningkat juga.
- Kemiringan garis regresi positif, tetapi sangat landai artinya, tambahan waktu belajar memberikan pengaruh kecil terhadap peningkatan GPA.
- Sebaran data sangat menyebar, menunjukkan banyak variabel lain yang memengaruhi GPA selain waktu belajar.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

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
Lanjut algoritma kedua yang saya gunakan adalah Extreme Gradient Boosting (XGBoost) karena dikenal sangat kuat untuk tugas klasifikasi dan regresi. Saya mengimplementasikannya dengan `XGBClassifier` dari library `xgboost`, melatih model menggunakan `x_train` dan `y_train`, lalu mengujinya dengan `x_test` dan `y_test`.

Saya mengatur beberapa parameter penting: `max_depth = 6`, `n_estimators = 125`, `random_state = 30`, `learning_rate = 0.01`, dan `n_jobs = -1` untuk memaksimalkan performa dan efisiensi model.

Kelebihan:
- Meningkatkan akurasi pada Grade A (16 benar, hanya 5 salah), lebih baik dibandingkan Random Forest.
- Konsisten sangat baik untuk Grade C (72), Grade D (73), dan Grade F (238).
- Grade B juga cukup akurat (48 benar).

Kelemahan:
- Hampir tidak ada, distribusi kesalahan sangat minim dan merata.



### Model SVM
Saya juga menggunakan Support Vector Machine (SVM) sebagai model ketiga. SVM merupakan algoritma yang efektif untuk klasifikasi, terutama dalam kasus dimana data memiliki struktur yang kompleks. Saya menggunakan `SVC` dari library `sklearn.svm` untuk melatih model dengan `x_train` dan `y_train`, lalu mengujinya dengan `x_test` dan `y_test`.

Saya mengatur beberapa parameter penting: `kernel = "rbf"`, `gamma = "auto"`, `random_state = 50` untuk memastikan hasil yang konsisten.

Kelebihan:
- Masih mampu mengenali Grade F (229 benar) dan Grade C (56 benar) dengan baik.

Kelemahan:
- Performa untuk Grade A (5 benar dari 21) dan Grade B (31 benar dari 54) cukup buruk.
- Banyak kesalahan antar kelas tengah seperti Grade B - C, D - C, C - D.
- Banyak prediksi Grade D salah ke Grade F (16 kasus).

### Model Naive Bayes
Model keempat yang saya gunakan adalah Naive Bayes. Meskipun namanya "naif", Naive Bayes telah terbukti efektif dalam banyak kasus klasifikasi . Saya menggunakan `GaussianNB` dari library `sklearn.naive_bayes` untuk melatih model dengan `x_train` dan `y_train`, lalu menguji dengan `x_test` dan `y_test`. Saya mengatur parameter `var_smoothing=1e-9` untuk mengatasi masalah numeriik.

Kelebihan:
- Cukup baik untuk prediksi Grade C (61) dan Grade D (65).

Kelemahan:
- Grade A (2 benar dari 21) dan Grade B (33 benar dari 54) jauh dari baik.
- Salah satu error terbesar adalah Grade F - Grade D (21 kasus), sangat tinggi.
- Banyak kebingungan antara Grade B dan Grade C.

### Model Terbaik

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/75772659/447427933-dbac394b-9e12-4dc3-996d-b9a7a025c6ec.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzgyNTksIm5iZiI6MTc0ODIzNzk1OSwicGF0aCI6Ii83NTc3MjY1OS80NDc0Mjc5MzMtZGJhYzM5NGItOWUxMi00ZGMzLTk5NmQtYjlhN2EwMjVjNmVjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA1MzkxOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWVmNmMwY2YxNzc1MWI2MmE0NzU0MzMzMjQ0NzUwNWRjYTRkZmYwMTRjNTNkZGYyYTY3YmRlMGUwYjIzNGU0NzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Bo4xkcikbUa__Ur6llfi8XkKsMiZpLWWK9mkn0c2xlE" width="1000"/>
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
    <img src="https://private-user-images.githubusercontent.com/75772659/447432682-ee6b86fa-c3dc-44f6-b0b2-d5348100a246.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzkzNjMsIm5iZiI6MTc0ODIzOTA2MywicGF0aCI6Ii83NTc3MjY1OS80NDc0MzI2ODItZWU2Yjg2ZmEtYzNkYy00NGY2LWIwYjItZDUzNDgxMDBhMjQ2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA1NTc0M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlYjQ1YmZjMmJlYjA1ZTcxZWNkMWVmMGFlNGM5MGI1ZDM5NWMyMjJmYjQwOGYyNTg2NmEwZjFmNmI0ODBkNDQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Oji8XotJK_ViJ6ZwvF_DOg5R_4Ohexg8TQrdndhKMm8" width="500"/>
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
    <img src="https://private-user-images.githubusercontent.com/75772659/447434093-e6b92dc5-5ca3-4c4d-9b71-433215830214.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzk3MjEsIm5iZiI6MTc0ODIzOTQyMSwicGF0aCI6Ii83NTc3MjY1OS80NDc0MzQwOTMtZTZiOTJkYzUtNWNhMy00YzRkLTliNzEtNDMzMjE1ODMwMjE0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA2MDM0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWEzMGIyNWMzN2QxZjY2Y2JlNmQ2OGMzMThhNzEwYzkzOGY0MjlhMjU1OGIwOWZkYzA3NWQxZWI0NjhlYWE3ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.nHHXR3RBtH5HsnVmE7PgCsnVOk3ngjcTxSR1Gyxg3Rk" width="500"/>
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
    <img src="https://private-user-images.githubusercontent.com/75772659/447434519-38583f9f-7292-4f4f-b3c3-fbe011429038.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzk4MTgsIm5iZiI6MTc0ODIzOTUxOCwicGF0aCI6Ii83NTc3MjY1OS80NDc0MzQ1MTktMzg1ODNmOWYtNzI5Mi00ZjRmLWIzYzMtZmJlMDExNDI5MDM4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA2MDUxOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTY4YjRlYWE0NTE0Yzc0YzNlMDJiNjU4ZWIwNGZmNDgxNWFjZDQ1MjJiNDBjZjQxMTJmMzhiYTljYTY1MDhlODgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Vkcr0_BRnemWvtTe0L6SAlYIkSnBLZtB4sgQCMrYL2Q" width="500"/>
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
    <img src="https://private-user-images.githubusercontent.com/75772659/447434916-f1635d71-3d42-4bf9-912a-bb54b3159516.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyMzk5MTUsIm5iZiI6MTc0ODIzOTYxNSwicGF0aCI6Ii83NTc3MjY1OS80NDc0MzQ5MTYtZjE2MzVkNzEtM2Q0Mi00YmY5LTkxMmEtYmI1NGIzMTU5NTE2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA2MDY1NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTY0OWRlMzNjMTM5NDA5MmUzZjYxOTNjZWU2YzQxMjQzYWExMTJhYjRmNmRkMzllMzUwYTY4ZjMwODE0ZjBkNDUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.SUIQa_g7cJTHip2yl9kfhYYVFP95gU_NZGrt_Ihe-08" width="500"/>
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
    <img src="https://private-user-images.githubusercontent.com/75772659/447445189-13576eb4-4b00-4cf8-949a-2736e02ddcd6.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyNDIwODcsIm5iZiI6MTc0ODI0MTc4NywicGF0aCI6Ii83NTc3MjY1OS80NDc0NDUxODktMTM1NzZlYjQtNGIwMC00Y2Y4LTk0OWEtMjczNmUwMmRkY2Q2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA2NDMwN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTE1NjZiZTE3ZGQ1MGU4ZTI2MmQ5MDVhMDBmZmQxZDA3ZWI5MmI1ZmQ0NjE4NGZmYTczZmE0MGFmNjY1YTU1NTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.4VHC5B8uaABK3fAB_8rahgHEVk7dbvEDnm_N8CkGzzc" width="500"/>
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