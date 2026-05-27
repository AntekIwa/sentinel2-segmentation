import numpy as np
import matplotlib.pyplot as plt
import os
from pca_utils import pca_z_autorskim_svd
from data_utils import (
    stworz_macierz_3d_z_resamplingiem, 
    rozciagnij_na_2d, 
    przywroc_format_obrazka
)

# 1. KONFIGURACJA ŚCIEŻEK (Grecja - B11)
sciezka_b11 = r"C:\Users\PiFi\Downloads\Browser_images (2)\2021-08-18-00_00_2021-08-18-23_59_Sentinel-2_L2A_B11_(Raw).tiff"
folder = os.path.dirname(sciezka_b11)

pasma_list = ["B02", "B03", "B04", "B08", "B11", "B12"]
sciezki = [sciezka_b11.replace("B11", p) for p in pasma_list]

print("-> Wczytywanie danych i obliczanie PCA...")
kostka = stworz_macierz_3d_z_resamplingiem(sciezki, [])
tabela_2d, wymiary = rozciagnij_na_2d(kostka)
wynik_pca_2d, _, _ = pca_z_autorskim_svd(tabela_2d, liczba_skladowych=3)
obraz_pca_3d = przywroc_format_obrazka(wynik_pca_2d, wymiary)

# 2. EKSTRAKCJA TYLKO PCA3 (Indeks 2)
# Odrzucamy PCA1 (jasność) i PCA2 (inne detale)
pca3_raw = obraz_pca_3d[:, :, 2]

# 3. POPRAWA KONTRASTU (PERCENTILE CLIPPING)
# Automatycznie ignorujemy 2% najciemniejszych i 2% najjaśniejszych pikseli
# (szum na krawędziach), co rozciąga kontrast dla wyspy.
vmin, vmax = np.percentile(pca3_raw, [2, 98])

print("-> Generowanie pojedynczego, czystego obrazu PC3...")

# 4. WIZUALIZACJA FINAŁOWA (Jeden duży obraz)
fig, ax = plt.subplots(figsize=(12, 10))

# Używamy palety 'inferno', żeby pożar "świecił" jasnym kolorem.
im = ax.imshow(pca3_raw, cmap='inferno', vmin=vmin, vmax=vmax)

# Tytuł i oznaczenia
ax.set_title("Główna Składowa PC3 – Separacja Blizny Pożarowej\n(Grecja, Eubea, 18.08.2021)", 
             fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Dodanie paska koloru (colorbar) dla czytelności skali
fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, label='Intensywność anomalii PC3 (Pożar)')

plt.tight_layout()
plt.show()