import numpy as np

def wlasne_pca(dane, liczba_skladowych=3, iteracje=100):
    np.random.seed(42)
    print("-> Centrowanie danych (obliczanie średniej)...")
    srednia = np.mean(dane, axis=0)
    X = dane - srednia
    
    print("-> Tworzenie macierzy kowariancji (X.T @ X)...")
    C = np.dot(X.T, X)
    
    # NOWOŚĆ: Całkowita wariancja to suma przekątnej (ślad) oryginalnej macierzy kowariancji
    wariancja_total = np.trace(C)
    
    liczba_cech = X.shape[1]
    wektory_wlasne = []
    wartosci_wlasne = []  # NOWOŚĆ: Lista na nasze wartości własne (lambdy)
    
    print("-> Uruchamiam Metodę Potęgową...")
    for k in range(liczba_skladowych):
        # 1. Losujemy wektor startowy
        v = np.random.rand(liczba_cech)
        v = v / np.sqrt(np.sum(v**2))
        
        # 2. Pętla Metody Potęgowej
        for _ in range(iteracje):
            v_nowy = np.dot(C, v)
            norma = np.sqrt(np.sum(v_nowy**2))
            
            if norma < 1e-9: 
                break
            v = v_nowy / norma
            
        # 3. Obliczamy wartość własną (lambda)
        wartosc_wlasna = np.dot(v.T, np.dot(C, v))
        
        # Zapisujemy wektor i wartość własną
        wektory_wlasne.append(v)
        wartosci_wlasne.append(wartosc_wlasna)  # Zapisujemy lambdę do listy
        
        # 4. DEFLACJA (Metoda Hotellinga)
        C = C - wartosc_wlasna * np.outer(v, v)
        print(f"   Pomyślnie wyciągnięto składową PC{k+1}")
        
    V_custom = np.array(wektory_wlasne)
    
    print("-> Matematyczne rzutowanie danych na nowe osie...")
    nowe_dane = np.dot(X, V_custom.T)
    
    # NOWOŚĆ: Zwracamy wszystkie trzy elementy wymagane do Scree Plot
    return nowe_dane, wartosci_wlasne, wariancja_total
