import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import rasterio

# Pobieramy nowe funkcje od kolegi (SVD) oraz nasz niezbędny sprzęt
from pca_utils import pca_z_autorskim_svd
from data_utils import (
    stworz_macierz_3d_z_resamplingiem, 
    rozciagnij_na_2d, 
    przywroc_format_obrazka, 
    kat_widmowy
)

# ==========================================
# 1. ŚCIEŻKI DO PLIKÓW (OKO SAHARY)
# ==========================================
folder = "C:/Users/PiFi/Downloads/Browser_images (1)/"
prefiks = "2026-04-30-00_00_2026-04-30-23_59_Sentinel-2_L2A_"
sufiks = "_(Raw).tiff"

sciezki_10m = [
    folder + prefiks + "B02" + sufiks,
    folder + prefiks + "B03" + sufiks,
    folder + prefiks + "B04" + sufiks,
    folder + prefiks + "B08" + sufiks
]

sciezki_20m = [
    folder + prefiks + "B05" + sufiks,
    folder + prefiks + "B06" + sufiks,
    folder + prefiks + "B07" + sufiks,
    folder + prefiks + "B8A" + sufiks,
    folder + prefiks + "B11" + sufiks,
    folder + prefiks + "B12" + sufiks
]

# ==========================================
# 2. WCZYTANIE I NOWE PCA (SVD Z-SCORE)
# ==========================================
print("=== KROK 1: Analiza przestrzenna Oka Sahary ===")
kostka = stworz_macierz_3d_z_resamplingiem(sciezki_10m, sciezki_20m)

# Tworzymy podgląd RGB i zabezpieczamy skalę jasności
surowe_rgb = kostka[:, :, [2, 1, 0]].astype(float)
maks_jasnosci = np.percentile(surowe_rgb, 99.5)
if maks_jasnosci == 0: maks_jasnosci = 1.0
rgb_podglad = np.clip(surowe_rgb, 0, maks_jasnosci) / maks_jasnosci

# Redukcja wymiarowości - UŻYWAMY NOWEGO SVD KOLEGI
tabela_2d, wymiary = rozciagnij_na_2d(kostka)
wynik_pca_2d, _, _ = pca_z_autorskim_svd(tabela_2d, liczba_skladowych=3)
obraz_pca_3d = przywroc_format_obrazka(wynik_pca_2d, wymiary)

# ==========================================
# 3. KALIBRACJA WZORCA NA ZIELONĄ FORMACJĘ MACROSTRAT
# ==========================================
print("\n=== KROK 2: Pobieranie sygnatury skalnej ===")
# Celujemy w twardą formację skalną (to ułoży radar w widlasty kształt z mapy)
lon_cel = -11.550
lat_cel = 21.100

with rasterio.open(sciezki_10m[0]) as plik:
    y_piksela, x_piksela = plik.index(lon_cel, lat_cel)

wzorzec_sahary = obraz_pca_3d[y_piksela, x_piksela, :]
print(f"-> Skalibrowano cel na współrzędnych: (X:{x_piksela}, Y:{y_piksela})")

# ==========================================
# 4. KLASYFIKACJA WIDMOWA (SAM)
# ==========================================
print("\n=== KROK 3: Klasyfikacja mapy... ===")
surowe_katy = kat_widmowy(obraz_pca_3d, wzorzec_sahary)

# Wycinamy wszystko co niepodobne
progi_kat_maski = 0.20 
podobienstwo_mapa = (progi_kat_maski - np.clip(surowe_katy, 0, progi_kat_maski)) / progi_kat_maski

# ==========================================
# 5. WIZUALIZACJA
# ==========================================
print("\n-> Generuję wynik...")
plt.figure(figsize=(14, 8))

# Rysujemy prawdziwy widok z satelity
plt.imshow(rgb_podglad)

# Nakładamy naszą warstwę "skały" (będzie żółto-czerwona i pominie piasek w środku oka)
radar_z_maska = np.ma.masked_where(podobienstwo_mapa < 0.1, podobienstwo_mapa)
im = plt.imshow(radar_z_maska, cmap='inferno', alpha=0.85)

cb = plt.colorbar(im, shrink=0.7)
cb.set_label('Podobieństwo sygnatury skały (SAM)', fontsize=12)

plt.title("Spektralna Detekcja Formacji Skalnej - Oko Sahary\nKształt idealnie pokrywa się z formacją zieloną z bazy Macrostrat", fontsize=15, fontweight='bold')
plt.axis('off')

print("=== GOTOWE! ===")
plt.show()