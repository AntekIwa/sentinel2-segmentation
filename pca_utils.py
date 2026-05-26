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

def autorskie_svd(X, liczba_skladowych=3, iteracje=100):
    """
    Rozkład SVD (X = U * Sigma * V^T) napisany całkowicie od zera.
    Wykorzystuje metodę potęgową z deflacją Hotellinga.
    """
    # Zamrażamy losowość dla powtarzalności wyników
    np.random.seed(42)
    
    # 1. Obliczamy macierz relacji kolumn (X^T * X)
    C = np.dot(X.T, X)
    liczba_cech = X.shape[1]
    
    wektory_V = []
    wartosci_sigma = []
    
    # Kopia do deflacji, żeby nie nadpisywać oryginalnego C
    C_def = C.copy()
    
    print("-> [SVD] Wyliczanie prawych wektorów osobliwych (V) i wartości osobliwych (Sigma)...")
    for k in range(liczba_skladowych):
        # Inicjalizacja losowego wektora i jego normalizacja
        v = np.random.rand(liczba_cech)
        v = v / np.sqrt(np.sum(v**2))
        
        # Metoda potęgowa (szukanie dominującego kierunku)
        for _ in range(iteracje):
            v_nowy = np.dot(C_def, v)
            norma = np.sqrt(np.sum(v_nowy**2))
            if norma < 1e-9:
                break
            v = v_nowy / norma
            
        # Wyliczenie wartości własnej (lambda)
        lambda_k = np.dot(v.T, np.dot(C_def, v))
        
        # Wartość osobliwa (Sigma) to pierwiastek kwadratowy z wartości własnej
        sigma_k = np.sqrt(max(lambda_k, 0))
        
        wektory_V.append(v)
        wartosci_sigma.append(sigma_k)
        
        # Deflacja: "wycinamy" z macierzy znalezioną informację, by w kolejnej pętli szukać nowej osi
        C_def = C_def - lambda_k * np.outer(v, v)
        
    V = np.array(wektory_V).T
    Sigma = np.array(wartosci_sigma)
    
    print("-> [SVD] Obliczanie lewostronnych wektorów osobliwych (U)...")
    wektory_U = []
    for i in range(liczba_skladowych):
        # Uzywamy definicji: u_i = (X * v_i) / sigma_i
        if Sigma[i] > 1e-9:
            u = np.dot(X, V[:, i]) / Sigma[i]
        else:
            u = np.zeros(X.shape[0])
        wektory_U.append(u)
        
    U = np.array(wektory_U).T
    
    # Zwracamy pełną geometrię SVD
    return U, Sigma, V.T


def pca_z_autorskim_svd(dane, liczba_skladowych=3):
    """
    Potok PCA wykorzystujący nasze autorskie SVD oraz Standaryzację Z-score,
    aby zbalansować wariancję (zapobiec pożeraniu 99% przez PC1).
    """
    print("-> [PCA-SVD] Centrowanie i standaryzacja danych (Z-score)...")
    srednia = np.mean(dane, axis=0)
    odchylenie = np.std(dane, axis=0)
    
    # Standaryzacja: X = (dane - średnia) / odchylenie
    # Dodajemy 1e-8, aby zabezpieczyć się przed dzieleniem przez równe zero
    X = (dane - srednia) / (odchylenie + 1e-8)
    
    # Uruchamiamy nasz silnik matematyczny
    U, Sigma, Vt = autorskie_svd(X, liczba_skladowych=liczba_skladowych)
    
    print("-> [PCA-SVD] Wyliczanie informacji wyjaśnionej (Scree Plot prep)...")
    # Z macierzy X zrobiliśmy Z-score, więc całkowita wariancja to po prostu liczba cech (pasm)
    calkowita_wariancja = X.shape[1] 
    
    # Lambda = Sigma^2 / n
    n_probek = X.shape[0]
    wariancje_skladowych = (Sigma ** 2) / n_probek
    
    print("-> [PCA-SVD] Rzutowanie danych na nowe składowe (X * V)...")
    # Przesypanie milionów pikseli na nowe osie PC1, PC2, PC3
    nowe_dane = np.dot(X, Vt.T)
    
    return nowe_dane, wariancje_skladowych, calkowita_wariancja