#ifndef SEGMENTACION_H
#define SEGMENTACION_H

#include "lector_bmp.h"
#include <vector>

/*
---------------------------------------------------------
Clase Segmentador
Encargada de separar la prenda del fondo
usando similitud de color
---------------------------------------------------------
*/

class Segmentador
{
public:

    std::vector<std::vector<int>> mascara;

    void segmentar(ImagenBMP& imagen, Pixel colorDominante, int umbral);

    // Segmenta asumiendo fondo blanco
    void segmentarFondoBlanco(const ImagenBMP& imagen, int umbral = 240);

    // Detecta el color de fondo por las esquinas y segmenta por diferencia
    void segmentarPorFondo(const ImagenBMP& imagen, int umbral = 20);

    // Segmenta con umbral adaptativo basado en la varianza del fondo
    // Más robusto para fondos con sombras o gradientes
    void segmentarAdaptativo(const ImagenBMP& imagen, float factor = 3.5f, int umbralMin = 18);

    // Recorta al bounding box de la mascara (con margen opcional en px)
    ImagenBMP recortar(const ImagenBMP& imagen, int margen = 25);

    // Detecta y elimina marcos/bordes uniformes alrededor de la imagen
    // antes de segmentar (resuelve imágenes con marcas de agua o bordes
    // de catálogo que confunden la segmentación)
    ImagenBMP eliminarMarco(const ImagenBMP& imagen, int umbralMarco = 30);
};

#endif