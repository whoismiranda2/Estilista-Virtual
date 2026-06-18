// vision/features.h
#pragma once
#include "lector_bmp.h"
#include <vector>

struct Features {
    // 1. Geometría
    float proporcion;       // alto / ancho

    // 2. Densidad por tercios (píxeles oscuros en cada zona)
    float densidad_arriba;  // tercio superior
    float densidad_centro;  // tercio medio
    float densidad_abajo;   // tercio inferior

    // 3. Centro de masa vertical (0.0 = arriba, 1.0 = abajo)
    float centro_masa_y;

    // 4. Simetría horizontal (0.0 = perfecta, 1.0 = muy asimétrica)
    float asimetria;

    // 5. Varianza de columnas (detecta piernas separadas de pantalones)
    float varianza_columnas;

    // 6. Histograma simplificado (8 bins por canal = 24 valores)
    float hist_r[8];
    float hist_g[8];
    float hist_b[8];

    // Helper: convierte todo a vector para el perceptrón
    std::vector<float> toVector() const;
};

class ExtractorFeatures {
public:
    Features extraer(const ImagenBMP& img);

private:
    float calcularDensidad(const ImagenBMP& img, int filaInicio, int filaFin);
    void  calcularHistograma(const ImagenBMP& img, float* hr, float* hg, float* hb);
};