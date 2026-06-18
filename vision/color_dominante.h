#ifndef COLOR_DOMINANTE_H
#define COLOR_DOMINANTE_H

#include "lector_bmp.h"

/*
Clase ColorDominante
Calcula el color promedio de la imagen (imagen total)
*/

class ColorDominante
{
public:

    int r;
    int g;
    int b;

    void calcular(const ImagenBMP& imagen);

    void imprimir();

};

#endif