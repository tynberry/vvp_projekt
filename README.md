# Projekt VVP 2024/2025 Letní semestr

Zadání lze nalézt v souboru [ZADÁNÍ.md](ZADÁNÍ.md).

## Vlastnosti

Knihovna projektu zprostředkovává funkce pro generování Mandelbrotovy a Juliovy množiny.
Dále dovoluje převedení matice iterací před divergencí
do barevné matice pomocí barevných map knihovny `matplotlib` a histogramu počtu iterací.
Na konec knihovna nabízí vizualizační aplikaci na bázi `pygame`, která dovoluje interaktivní
průchod množinami.

Příklady použití knihovny jsou v Jupyter Notebooku `./examples/examples.ipynb`.

## Kompilace

Projekt lze zkompilovat klasicky pomocí příkazu `pip`.
Zde je uveden příklad pro vytvoření virtuálního prostředí
pomocí příkazu `uv`. Balíčky potřebné pro sestavení
se stáhnou spolu s `uv pip install .`.

```bash
# vytvoření virtuálního prostředí
uv venv -p 3.13
# přepnutí do virtuálního prostředí
source ./venv/bin/activate
# instalace balíčku 
uv pip install .
# případně 
uv pip install mandelbrot
```

## Poznámky

Projekt byl vytvořen pro předmět VVP na VŠB-TUO v letním semestru 2024/2025.
