#ifndef COLORIMETRIA_H
#define COLORIMETRIA_H

#include "lector_bmp.h"
#include "segmentacion.h"
#include <string>
#include <vector>

using namespace std;

struct ColorHSV {
    float h; // 0-360
    float s; // 0-255
    float v; // 0-255
};

struct ResultadoColorimetria {
    string estacion;       // "primavera", "verano", "otono", "invierno"
    string temperatura;    // "calido" o "frio"
    string valor;          // "claro" u "oscuro"
    float  tono_piel_h;
    float  tono_piel_s;
    float  tono_piel_v;
    int    pixeles_piel;   // cuántos píxeles de piel se detectaron
    float  piel_r;    // valor R promedio de píxeles de piel
    float  piel_g;    // valor G promedio
    float  piel_b;    // valor B promedio
};

class AnalizadorColorimetria {
public:
    // Convierte un píxel RGB a HSV
    ColorHSV rgbAhsv(unsigned char r, unsigned char g, unsigned char b);

    // Verifica si un color HSV corresponde a piel humana
    bool esPiel(const ColorHSV& c);

    // Analiza la imagen y devuelve la estación y paleta recomendada
    ResultadoColorimetria analizar(const ImagenBMP& img);

    // Devuelve los colores recomendados para cada estación
    vector<string> paletaRecomendada(const string& estacion);
};

#endif
