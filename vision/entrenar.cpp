#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <ctime>

#ifdef _WIN32
  #include <windows.h>
  #include <dirent.h>
#else
  #include <dirent.h>
#endif

#include "lector_bmp.h"
#include "features.h"
#include "perceptron.h"
#include "segmentacion.h"

using namespace std;

vector<string> listarBMP(const string& carpeta) {
    vector<string> archivos;
    DIR* dir = opendir(carpeta.c_str());
    if (!dir) return archivos;
    struct dirent* entrada;
    while ((entrada = readdir(dir)) != nullptr) {
        string nombre = entrada->d_name;
        if (nombre.size() > 4 && nombre.substr(nombre.size() - 4) == ".bmp")
            archivos.push_back(carpeta + "/" + nombre);
    }
    closedir(dir);
    return archivos;
}

// Calcula min/max de cada feature en el dataset
void calcularRangos(const vector<vector<float>>& Xs,
                    vector<float>& minVal, vector<float>& maxVal) {
    int n = Xs[0].size();
    minVal.assign(n,  1e9f);
    maxVal.assign(n, -1e9f);
    for (const auto& x : Xs)
        for (int i = 0; i < n; i++) {
            minVal[i] = min(minVal[i], x[i]);
            maxVal[i] = max(maxVal[i], x[i]);
        }
}

// Normaliza un vector usando los rangos calculados
vector<float> normalizar(const vector<float>& x,
                          const vector<float>& minVal,
                          const vector<float>& maxVal) {
    vector<float> xn(x.size());
    for (int i = 0; i < (int)x.size(); i++) {
        float rango = maxVal[i] - minVal[i];
        xn[i] = (rango > 1e-6f) ? (x[i] - minVal[i]) / rango : 0.0f;
    }
    return xn;
}

// Guarda los rangos en un archivo para usarlos al clasificar
void guardarRangos(const vector<float>& minVal,
                   const vector<float>& maxVal,
                   const string& ruta) {
    ofstream f(ruta);
    f << minVal.size() << "\n";
    for (float v : minVal) f << v << "\n";
    for (float v : maxVal) f << v << "\n";
}

int main() {
    srand((unsigned)time(nullptr));

    vector<string> categorias = {"top", "pantalon", "vestido", "falda", "short", "otro"};
    string carpetaBase  = "dataset";
    string rutaPesos    = "pesos.txt";
    string rutaRangos   = "rangos.txt";
    int    epocas       = 200;
    float  tasa         = 0.01f;
    int    numFeatures  = 31;

    vector<vector<float>> Xs;
    vector<string>        Ys;
    ExtractorFeatures extractor;

    cout << "=== CARGANDO DATASET ===" << endl;
    for (const string& cat : categorias) {
        string carpeta = carpetaBase + "/" + cat;
        vector<string> archivos = listarBMP(carpeta);
        if (archivos.empty()) {
            cout << "  [!] Sin imagenes en: " << carpeta << endl;
            continue;
        }
        int cargadas = 0;
        for (const string& ruta : archivos) {
            ImagenBMP img;
            if (!img.cargar(ruta)) continue;
            Features f = extractor.extraer(img);
            Xs.push_back(f.toVector());
            Ys.push_back(cat);
            cargadas++;
        }
        cout << "  " << cat << ": " << cargadas << " imagenes" << endl;
    }

    if (Xs.empty()) {
        cout << "Error: no se cargaron imagenes." << endl;
        return 1;
    }
    cout << "Total: " << Xs.size() << " imagenes" << endl;

    // Calcular y guardar rangos para normalizar al clasificar
    vector<float> minVal, maxVal;
    calcularRangos(Xs, minVal, maxVal);
    guardarRangos(minVal, maxVal, rutaRangos);

    // Normalizar todo el dataset
    vector<vector<float>> XsNorm;
    for (const auto& x : Xs)
        XsNorm.push_back(normalizar(x, minVal, maxVal));

    cout << "Features normalizadas. Rangos guardados en " << rutaRangos << endl << endl;

    ClasificadorPrendas clasificador;
    for (const string& cat : categorias)
        clasificador.agregar(cat, numFeatures, tasa);

    cout << "=== ENTRENANDO (" << epocas << " epocas) ===" << endl;
    for (int epoca = 1; epoca <= epocas; epoca++) {
        vector<int> indices(XsNorm.size());
        for (int i = 0; i < (int)indices.size(); i++) indices[i] = i;
        for (int i = (int)indices.size()-1; i > 0; i--) {
            int j = rand() % (i+1);
            swap(indices[i], indices[j]);
        }
        for (int idx : indices)
            clasificador.entrenar(XsNorm[idx], Ys[idx]);

        if (epoca % 25 == 0 || epoca == 1) {
            int correctos = 0;
            for (int i = 0; i < (int)XsNorm.size(); i++)
                if (clasificador.clasificar(XsNorm[i]) == Ys[i]) correctos++;
            float prec = 100.0f * correctos / XsNorm.size();
            cout << "  Epoca " << epoca << "/" << epocas
                 << "  ->  precision: " << prec << "%" << endl;
        }
    }

    if (clasificador.guardarPesos(rutaPesos))
        cout << "\nPesos guardados en: " << rutaPesos << endl;
    else
        cout << "\n[!] Error al guardar pesos." << endl;

    return 0;
}