#ifndef LECTOR_BMP_H
#define LECTOR_BMP_H

#include <vector>
#include <string>

using namespace std;

/*
Estructura Pixel
Representa un pixel en formato RGB
*/
struct Pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

/*
Clase ImagenBMP
Encargada de:
- Leer archivos BMP
- Guardar los pixeles en memoria
*/
class ImagenBMP
{
public:

    int ancho;
    int alto;

    vector<std::vector<Pixel>> pixeles;

    bool cargar(const string& ruta);

    //Guardar la imagen generada en escala de grises
    bool guardarGrises(
        const string& ruta,
        const vector<vector<unsigned char>>& gris
    );

    bool guardarBMP(
        const string& ruta,
        const vector<vector<Pixel>>& data
    );
};

#endif