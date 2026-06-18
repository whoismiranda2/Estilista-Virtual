#ifndef ESCALA_GRISES_H
#define ESCALA_GRISES_H

#include "lector_bmp.h"
#include <vector>

/*
Clase EscalaGrises
Convierte una imagen RGB a escala de grises
*/

class EscalaGrises
{
public:

    std::vector<std::vector<unsigned char>> gris;

    void convertir(const ImagenBMP& imagen);
};

#endif