#include "rgb_hsv.h"
#include <algorithm>

/*
---------------------------------------------------------
Función: convertir
Convierte todos los pixeles de RGB a HSV
---------------------------------------------------------
*/

void RGB_HSV::convertir(const ImagenBMP& imagen)
{
    imagenHSV.resize(imagen.alto, std::vector<HSV>(imagen.ancho));

    for(int y = 0; y < imagen.alto; y++)
    {
        for(int x = 0; x < imagen.ancho; x++)
        {
            Pixel p = imagen.pixeles[y][x];

            float r = p.r / 255.0;
            float g = p.g / 255.0;
            float b = p.b / 255.0;

            float maximo = std::max({r,g,b});
            float minimo = std::min({r,g,b});
            float delta = maximo - minimo;

            HSV hsv;

            // Hue
            if(delta == 0)
                hsv.h = 0;

            else if(maximo == r)
                hsv.h = 60 * ( (g - b) / delta );

            else if(maximo == g)
                hsv.h = 60 * ( 2 + (b - r) / delta );

            else
                hsv.h = 60 * ( 4 + (r - g) / delta );

            if(hsv.h < 0)
                hsv.h += 360;

            // Saturation
            if(maximo == 0)
                hsv.s = 0;
            else
                hsv.s = delta / maximo;

            // Value
            hsv.v = maximo;

            imagenHSV[y][x] = hsv;
        }
    }
}