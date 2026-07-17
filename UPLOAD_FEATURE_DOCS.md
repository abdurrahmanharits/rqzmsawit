# Dokumentasi Fitur Upload Data Spatial

## Overview

Dashboard RQZM-Sawit sekarang mendukung **upload file data spatial custom** selain file bawaan (`data-okta.gpkg`). Pengguna dapat mengupload:

- **Shapefile Package** (`.zip` berisi `.shp`, `.shx`, `.dbf`, `.prj`, dll)
- **GeoPackage** (`.gpkg` - single atau multiple layer)

## Fitur

### 1. **Sumber Data Fleksibel**
Di sidebar, pengguna dapat memilih:
- **"File Bawaan (data-okta.gpkg)"** → Menggunakan file default di `db/data-okta.gpkg`
- **"Upload Custom (ZIP/GPKG)"** → Upload file sendiri

### 2. **Deteksi Multi-Layer Otomatis**
Jika file GPKG memiliki beberapa layer, aplikasi akan:
- Menampilkan daftar layer dalam dropdown
- Pengguna memilih layer mana yang ingin diproses
- Otomatis memilih layer pertama jika tidak ada pilihan

### 3. **Validasi Data Otomatis**
Sebelum diproses, aplikasi mengecek:
- File valid (format ZIP atau GPKG)
- Ada geometry column (feature data)
- Data tidak kosong
- CRS compatibility

### 4. **Seamless Integration**
Setelah data dimuat, semua fitur dashboard tetap bekerja:
- Deteksi kolom numerik & kategori otomatis
- Filter berdasarkan kategori & rentang nilai
- Visualisasi peta interaktif
- Pembentukan zona RQZM
- Download hasil sebagai CSV

## Cara Menggunakan

### Upload Shapefile (ZIP)

1. Di sidebar, pilih **"Upload Custom (ZIP/GPKG)"**
2. Upload file `.zip` yang berisi shapefile components:
   ```
   mydata.zip
   ├── mydata.shp
   ├── mydata.shx
   ├── mydata.dbf
   ├── mydata.prj (opsional)
   └── mydata.cpg (opsional)
   ```
3. Aplikasi otomatis akan mendeteksi layer `mydata`
4. Klik tombol **"🚀 Proses"** untuk memuat data
5. Dashboard akan menampilkan data dan siap untuk dianalisis

### Upload GeoPackage (GPKG)

1. Di sidebar, pilih **"Upload Custom (ZIP/GPKG)"**
2. Upload file `.gpkg`
3. Jika ada multiple layer, pilih layer dari dropdown
4. Klik tombol **"🚀 Proses"** untuk memuat data

### Kembali ke File Bawaan

1. Di sidebar, pilih **"File Bawaan (data-okta.gpkg)"**
2. Dashboard akan reload dengan data default

## Implementasi Teknis

### File Dimodifikasi
- `page/okta_sawit.py`

### Helper Functions Ditambah

#### `_extract_zipfile(zip_bytes: bytes, extract_dir: Path)`
Ekstrak ZIP file ke temporary directory menggunakan Python `zipfile` module.

#### `_get_shapefile_layers(zip_or_dir: Path) -> list[str]`
Dapatkan daftar layer shapefile dalam directory (list nama `.shp` files).

#### `_get_gpkg_layers(gpkg_path: Path) -> list[str]`
Dapatkan daftar layer dalam GeoPackage dengan query SQLite `gpkg_contents`.

#### `load_data_from_upload(file_bytes, filename, selected_layer) -> tuple`
Main function untuk membaca uploaded file:
- Support format: `.zip` (shapefile) dan `.gpkg`
- Return: `(df, geojson_obj, center, error_msg, actual_layer_used)`
- Gunakan `geopandas.read_file()` untuk parsing
- Auto-convert columns numeric, tambah `_x_utm`, `_y_utm` centroid

### UI Components

**Sidebar Section: "📊 Sumber Data"**
```python
# Radio button untuk pilih sumber
data_source = st.sidebar.radio("Pilih sumber data:", [...])

# Jika upload custom:
st.sidebar.file_uploader(type=["zip", "gpkg"])
st.sidebar.selectbox("Pilih layer:", layers)
st.sidebar.button("🚀 Proses", use_container_width=True)
```

## Error Handling

Aplikasi menampilkan pesan error yang informatif jika:
- File upload bukan ZIP/GPKG: `"Format file harus .zip (shapefile) atau .gpkg"`
- Tidak ada shapefile dalam ZIP: `"Tidak ada file .shp dalam ZIP"`
- GPKG kosong atau tidak valid: `"Tidak ada layer dalam GPKG"` atau `"GeoDataFrame kosong"`
- Tidak ada geometry: `"Tidak ada geometry column"`
- Error saat parsing: `"Error membaca file: ..."`

## Testing

Semua function telah ditest dengan:
1. **Test 1**: Membaca Shapefile dari ZIP ✓
2. **Test 2**: Membaca GeoPackage tunggal ✓
3. **Test 3**: Membaca multi-layer GeoPackage ✓

## Notes & Limitations

- **Temporary Files**: Uploaded files disimpan ke temporary directory dan otomatis dihapus setelah session selesai
- **CRS Handling**: Jika CRS tidak kompatibel, aplikasi akan fallback ke default (tidak akan crash)
- **Layer Selection**: Jika user upload GPKG tapi tidak memilih layer, sistem otomatis memilih layer pertama
- **Size Limit**: Streamlit default 200MB file upload limit (dapat dikonfigurasi)
- **Performance**: Untuk shapefile besar, ekstraksi ZIP dan parsing geopandas mungkin memakan waktu

## Future Enhancements

Kemungkinan improvement:
- Preview data sebelum "Proses" (show first few rows)
- Support CSV dengan kolom geometry WKT/GeoJSON
- Validasi CRS compatibility
- Cache upload results untuk navigasi antar halaman
- Export hasil ke different spatial format
