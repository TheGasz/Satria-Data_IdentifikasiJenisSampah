# Analisis Kemiripan Visual Antar Kelas
## Big Data Challenge Satria Data 2026

## 1. Ringkasan Metode

Analisis dilakukan menggunakan embedding dari model pretrained ResNet18 (tanpa layer klasifikasi terakhir) terhadap 500 sampel citra per kelas (total 1500 citra) dari folder train. Embedding kemudian diproyeksikan menggunakan TSNE untuk visualisasi dua dimensi, dan dicari pasangan nearest neighbor lintas kelas untuk menemukan bukti konkret kemiripan visual.

## 2. Hasil Jarak Rata Rata Embedding Antar Kelas

| Kelas A | Kelas B | Jarak Rata Rata |
|---|---|---|
| Electronic | Organic | 10.50 |
| Recyclable | Organic | 7.65 |
| Recyclable | Electronic | 7.52 |

Interpretasi: Electronic dan Organic adalah pasangan kelas yang paling mudah dibedakan karena jaraknya paling besar. Recyclable berada di posisi tengah, cukup dekat dengan kedua kelas lain, sehingga berpotensi menjadi sumber kesalahan klasifikasi terbanyak.

## 3. Hasil Visualisasi TSNE

Tiga gerombolan (cluster) terlihat terbentuk dengan cukup jelas, yaitu Organic di sisi kiri, Recyclable di tengah, dan Electronic di sisi kanan. Ini menunjukkan bahwa fitur dari model pretrained sudah mampu menangkap perbedaan umum antar kelas.

Namun ditemukan dua area tumpang tindih (overlap) yang perlu diperhatikan:

1. Overlap antara Recyclable dan Electronic pada area tengah plot, jumlahnya cukup banyak.
2. Overlap kecil antara sebagian titik Recyclable dengan Organic di sisi kiri plot.

## 4. Hasil Nearest Neighbor Lintas Kelas

Dari 15 pasangan gambar dengan embedding paling berdekatan, ditemukan pola sebagai berikut.

Pola pertama, seluruh pasangan yang muncul adalah kombinasi Recyclable dengan Electronic atau Recyclable dengan Organic. Tidak ada satu pun pasangan Electronic dengan Organic, konsisten dengan hasil matriks jarak rata rata pada bagian 2.

Pola kedua, gambar dengan nama file battery (battery_8, battery_9, battery_48, battery_254, battery_284) muncul berulang kali dan selalu berpasangan dengan gambar Recyclable seperti R_3047, R_2948, R_6498, R_8114. Pola ini menunjukkan bahwa foto baterai secara visual mirip dengan sebagian foto sampah Recyclable, kemungkinan karena bentuk silindris dan warna metalik yang menyerupai kaleng.

Pola ketiga, beberapa pasangan Recyclable dengan Organic juga muncul seperti R_198 dengan O_1129, R_4755 dengan beberapa gambar Organic (O_8465, O_8369, O_8414), dan R_695 dengan O_5709. Kemunculan file R_4755 sebanyak tiga kali menunjukkan gambar ini punya karakteristik visual yang ambigu.

## 5. Kesimpulan

Secara umum ketiga kelas sudah cukup terpisah secara visual, namun ditemukan overlap yang nyata dan konsisten antara kelas Recyclable dan Electronic, khususnya pada objek baterai. Overlap antara Recyclable dan Organic juga ditemukan namun dengan intensitas lebih rendah pada sampel yang dianalisis.

## 6. Solusi Yang Direkomendasikan

### Solusi 1: Verifikasi Manual Label

Buka langsung file file yang teridentifikasi sebagai pasangan nearest neighbor lintas kelas, terutama battery_8, battery_9, battery_48, battery_254, battery_284, serta R_3047, R_2948, R_6498, R_4755, R_198, R_695. Tujuannya memastikan tidak ada kesalahan label pada dataset sebelum melanjutkan ke tahap modeling.

### Solusi 2: Pertahankan Resolusi Input Yang Cukup Tinggi

Karena kemiripan antara baterai dan kaleng bersumber dari bentuk umum silindris dan warna metalik, model perlu menangkap detail halus seperti tulisan pada baterai, simbol kutub positif dan negatif, atau tekstur logo merek. Resolusi input yang terlalu kecil berisiko menghilangkan detail pembeda tersebut.

### Solusi 3: Augmentasi Yang Berfokus Pada Bentuk Dan Tekstur

Terapkan color jitter dan random grayscale saat training, agar model tidak hanya mengandalkan warna metalik sebagai patokan, melainkan juga belajar dari bentuk dan tekstur objek secara keseluruhan.

### Solusi 4: Finetuning Model Pretrained Secara Penuh

Gunakan arsitektur pretrained seperti ResNet atau EfficientNet yang di finetune secara penuh, bukan sekadar dipakai sebagai feature extractor beku. Finetuning memungkinkan model menyesuaikan representasi fitur secara spesifik untuk membedakan baterai dari kaleng Recyclable, sesuatu yang tidak bisa dilakukan oleh feature extractor generik hasil pretraining ImageNet.

### Solusi 5: Evaluasi Confusion Matrix Setelah Training Awal

Setelah model awal dilatih, periksa confusion matrix pada data validasi. Jika recall kelas Electronic banyak salah diprediksi sebagai Recyclable akibat kasus baterai, pertimbangkan menambah bobot loss pada sampel yang mirip tersebut, atau menambah variasi data baterai dalam kondisi berbeda agar model punya lebih banyak contoh untuk belajar detail pembedanya.

### Solusi 6: Pantau Overlap Recyclable Dengan Organic Secara Berkala

Karena jumlah overlap antara Recyclable dan Organic lebih sedikit dibanding Recyclable dengan Electronic, penanganan khusus di awal belum diperlukan. Cukup dipantau melalui confusion matrix pada tahap evaluasi model, dan baru ditangani lebih lanjut apabila kesalahan klasifikasinya signifikan.

## 7. Prioritas Tindak Lanjut

1. Verifikasi manual label pada pasangan gambar yang teridentifikasi (Solusi 1), karena ini langkah termurah dan bisa langsung dilakukan.
2. Menentukan resolusi input training yang memadai (Solusi 2).
3. Menyiapkan pipeline augmentasi (Solusi 3) dan strategi finetuning (Solusi 4) sebelum training model utama.
4. Menjadikan evaluasi confusion matrix (Solusi 5 dan Solusi 6) sebagai bagian rutin setelah setiap iterasi training untuk memantau apakah overlap antar kelas benar benar berkurang.
