import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def stworz_macierz_3d_z_resamplingiem(sciezki_10m, sciezki_20m):
    """Wczytuje kanały 10m i 20m, naciąga te mniejsze do siatki 10m i składa w kostkę 3D."""
    pasma = []
    
    with rasterio.open(sciezki_10m[0]) as plik:
        docelowa_wysokosc = plik.height
        docelowa_szerokosc = plik.width
        print(f"-> Baza (10m) ma wymiary: {docelowa_wysokosc} x {docelowa_szerokosc}")

    for sciezka in sciezki_10m:
        with rasterio.open(sciezka) as plik:
            pasma.append(plik.read(1))
            
    for sciezka in sciezki_20m:
        with rasterio.open(sciezka) as plik:
            pasmo_powiekszone = plik.read(
                1,
                out_shape=(docelowa_wysokosc, docelowa_szerokosc),
                resampling=Resampling.bilinear
            )
            pasma.append(pasmo_powiekszone)
            
    return np.dstack(pasma)


def rozciagnij_na_2d(macierz_3d):
    """Spłaszcza kostkę 3D do płaskiej tabeli 2D dla algorytmu PCA."""
    wysokosc, szerokosc, liczba_pasm = macierz_3d.shape
    macierz_2d = macierz_3d.reshape((wysokosc * szerokosc, liczba_pasm))
    return macierz_2d, (wysokosc, szerokosc, liczba_pasm)


def przywroc_format_obrazka(macierz_2d, oryginalny_ksztalt):
    """Zwraca wynikowi PCA oryginalny kształt mapy trójwymiarowej."""
    wysokosc, szerokosc, _ = oryginalny_ksztalt
    liczba_skladowych = macierz_2d.shape[1] 
    return macierz_2d.reshape((wysokosc, szerokosc, liczba_skladowych))


def zaawansowana_normalizacja_kontrastu(obraz_pca, dolny_percentyl=2, gorny_percentyl=98):
    """Poprawia kontrast obrazu odcinając skrajne wartości percentylowe (np. kopalnię)."""
    print("-> Poprawiam kontrast obrazu metodą percentylową...")
    gotowa_mapa = np.zeros_like(obraz_pca, dtype=np.float32)
    
    for i in range(obraz_pca.shape[2]):
        kanal = obraz_pca[:, :, i]
        min_val = np.percentile(kanal, dolny_percentyl)  
        max_val = np.percentile(kanal, gorny_percentyl) 
        
        kanal_znormalizowany = ((kanal - min_val) / (max_val - min_val)) * 255
        gotowa_mapa[:, :, i] = np.clip(kanal_znormalizowany, 0, 255)
        
    return gotowa_mapa.astype(np.uint8)

def wyswietl_mape(mapa, cmap=None):
    plt.figure(figsize=(12, 9))
    plt.imshow(mapa, cmap=cmap)
    plt.axis('off')
    plt.show()