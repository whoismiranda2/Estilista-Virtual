#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <string>

using namespace std;

// Un perceptron binario: responde 1 si la imagen ES su categoria, 0 si no
class Perceptron {
public:
    string categoria;       // "top", "pantalon", etc.
    vector<float> pesos;    // un peso por feature
    float bias;
    float tasa;             // tasa de aprendizaje

    Perceptron(const string& cat, int numFeatures, float tasaAprendizaje = 0.01f);

    // Predice: 1 si es su categoria, 0 si no
    int predecir(const vector<float>& x) const;

    // Salida continua (antes del umbral) — util para comparar entre perceptrones
    float salida(const vector<float>& x) const;

    // Entrena con un ejemplo: x = features, etiquetaCorrecta = 1 o 0
    void entrenar(const vector<float>& x, int etiquetaCorrecta);

    // Guardar y cargar pesos en archivo
    void guardar(ofstream& archivo) const;
    void cargar(ifstream& archivo);
};

// Conjunto de perceptrones: uno por categoria
// Al clasificar, gana el de mayor salida continua
class ClasificadorPrendas {
public:
    vector<Perceptron> perceptrones;

    void agregar(const string& categoria, int numFeatures, float tasa = 0.01f);

    // Devuelve la categoria ganadora
    string clasificar(const vector<float>& x) const;

    // Entrena todos con un ejemplo etiquetado
    void entrenar(const vector<float>& x, const string& etiqueta);

    // Guardar/cargar todos los pesos
    bool guardarPesos(const string& ruta) const;
    bool cargarPesos(const string& ruta);
};

#endif