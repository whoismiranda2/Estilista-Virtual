#ifndef HISTOGRAMA_H
#define HISTOGRAMA_H

#include "lector_bmp.h"

/*
Clase Histograma
Calcula histogramas RGB de una imagen BMP
*/

class Histograma
{
public:

    int histR[256];
    int histG[256];
    int histB[256];
    int histGray[256];

    void calcular(const ImagenBMP& imagen);

    void calcularAcumulado();

    void imprimir();

    void imprimirJSON();

};

#endif