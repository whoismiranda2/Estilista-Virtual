#ifndef RGB_HSV_H
#define RGB_HSV_H

#include "lector_bmp.h"
#include "hsv.h"
#include <vector>

/*
---------------------------------------------------------
Clase RGB_HSV
Convierte los pixeles de una imagen RGB a HSV
---------------------------------------------------------
*/

class RGB_HSV
{
public:

    std::vector<std::vector<HSV>> imagenHSV;

    void convertir(const ImagenBMP& imagen);

};

#endif