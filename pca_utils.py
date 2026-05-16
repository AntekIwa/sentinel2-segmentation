import numpy as np

def wlasne_pca(dane, liczba_skladowych=3, iteracje=100):
    np.random.seed(42)
    print("-> Centrowanie danych (obliczanie średniej)...")
    srednia = np.mean(dane, axis=0)
    X = dane - srednia
    
    print("-> Tworzenie macierzy kowariancji (X.T @ X)...")
    C = np.dot(X.T, X)
    
    liczba_cech = X.shape[1]
    wektory_wlasne = []
    
    print("-> Uruchamiam Metodę Potęgową...")
    for k in range(liczba_skladowych):
        # 1. Losujemy wektor startowy o długości równej liczbie kanałów
        v = np.random.rand(liczba_cech)
        # Normalizujemy go (robimy z niego wektor jednostkowy bez użycia linalg.norm)
        v = v / np.sqrt(np.sum(v**2))
        
        # 2. Pętla Metody Potęgowej - wektor "obraca się" w stronę największej wariancji
        for _ in range(iteracje):
            v_nowy = np.dot(C, v)
            norma = np.sqrt(np.sum(v_nowy**2))
            
            if norma < 1e-9: # Zabezpieczenie przed dzieleniem przez zero
                break
            v = v_nowy / norma
            
        # 3. Obliczamy wartość własną (lambda) dla znalezionego wektora
        # Wzór: lambda = v^T @ C @ v
        wartosc_wlasna = np.dot(v.T, np.dot(C, v))
        
        # Zapisujemy znalezioną oś (Wektor Własny)
        wektory_wlasne.append(v)
        
        # 4. DEFLACJA (Metoda Hotellinga): Usuwamy znalezioną składową z macierzy C.
        # Dzięki temu w kolejnym obrocie pętli 'for' metoda potęgowa znajdzie 
        # KOLEJNĄ, prostopadłą oś (PC2, potem PC3).
        C = C - wartosc_wlasna * np.outer(v, v)
        print(f" Pomyślnie wyciągnięto składową PC{k+1}")
        
    # Składamy nasze wektory w macierz przejścia
    V_custom = np.array(wektory_wlasne)
    
    print("-> Matematyczne rzutowanie danych na nowe osie...")
    # Rzutujemy nasze wycentrowane dane na wyliczone ręcznie osie
    nowe_dane = np.dot(X, V_custom.T)
    
    return nowe_dane