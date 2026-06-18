#include "perceptron.h"
#include <fstream>
#include <numeric>
#include <cstdlib>
#include <iostream>

using namespace std;

// -------------------------------------------------------
// Perceptron
// -------------------------------------------------------

Perceptron::Perceptron(const string& cat, int numFeatures, float tasaAprendizaje)
    : categoria(cat), bias(0.0f), tasa(tasaAprendizaje)
{
    // Pesos iniciales pequeños aleatorios entre -0.1 y 0.1
    pesos.resize(numFeatures);
    for (float& w : pesos)
        w = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
}

// Suma ponderada + bias
float Perceptron::salida(const vector<float>& x) const {
    if (x.size() < pesos.size()) return 0.0f;
    float suma = bias;
    for (int i = 0; i < (int)pesos.size(); i++)
        suma += pesos[i] * x[i];
    return suma;
}

// Funcion escalon: 1 si suma > 0, 0 si no
int Perceptron::predecir(const vector<float>& x) const {
    return salida(x) > 0.0f ? 1 : 0;
}

// Regla delta: w = w + alpha * (y - y_pred) * x
void Perceptron::entrenar(const vector<float>& x, int etiquetaCorrecta) {
    if (x.size() < pesos.size()) return;
    int prediccion = predecir(x);
    int error = etiquetaCorrecta - prediccion;

    if (error != 0) {
        for (int i = 0; i < (int)pesos.size(); i++)
            pesos[i] += tasa * error * x[i];
        bias += tasa * error;
    }
}

void Perceptron::guardar(ofstream& archivo) const {
    archivo << categoria << "\n";
    archivo << pesos.size() << "\n";
    for (float w : pesos)
        archivo << w << "\n";
    archivo << bias << "\n";
}

void Perceptron::cargar(ifstream& archivo) {
    int n;
    archivo >> categoria >> n;
    pesos.resize(n);
    for (float& w : pesos)
        archivo >> w;
    archivo >> bias;
}

// -------------------------------------------------------
// ClasificadorPrendas
// -------------------------------------------------------

void ClasificadorPrendas::agregar(const string& categoria,
                                   int numFeatures, float tasa) {
    perceptrones.emplace_back(categoria, numFeatures, tasa);
}

// Gana el perceptron con mayor salida continua (no solo 0/1)
string ClasificadorPrendas::clasificar(const vector<float>& x) const {
    if (perceptrones.empty()) return "desconocido";

    int ganador = 0;
    float maxSalida = perceptrones[0].salida(x);

    for (int i = 1; i < (int)perceptrones.size(); i++) {
        float s = perceptrones[i].salida(x);
        if (s > maxSalida) {
            maxSalida = s;
            ganador = i;
        }
    }
    return perceptrones[ganador].categoria;
}

// Entrena cada perceptron: el de la etiqueta recibe 1, los demas 0
void ClasificadorPrendas::entrenar(const vector<float>& x,
                                    const string& etiqueta) {
    for (Perceptron& p : perceptrones)
        p.entrenar(x, (p.categoria == etiqueta) ? 1 : 0);
}

bool ClasificadorPrendas::guardarPesos(const string& ruta) const {
    ofstream archivo(ruta);
    if (!archivo.is_open()) return false;

    archivo << perceptrones.size() << "\n";
    for (const Perceptron& p : perceptrones)
        p.guardar(archivo);

    return true;
}

bool ClasificadorPrendas::cargarPesos(const string& ruta) {
    ifstream archivo(ruta);
    if (!archivo.is_open()) return false;

    int n;
    archivo >> n;
    perceptrones.resize(n, Perceptron("", 0));
    for (Perceptron& p : perceptrones)
        p.cargar(archivo);

    return true;
}