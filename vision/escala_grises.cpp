#include "escala_grises.h"

/*
Función: convertir
Convierte cada pixel RGB a escala de grises usando promedio
*/

void EscalaGrises::convertir(const ImagenBMP& imagen)
{
    gris.resize(imagen.alto, std::vector<unsigned char>(imagen.ancho));

    for(int y = 0; y < imagen.alto; y++)
    {
        for(int x = 0; x < imagen.ancho; x++)
        {
            Pixel p = imagen.pixeles[y][x];

            unsigned char valorGris = (p.r + p.g + p.b)/3;

            
            //unsigned char valorGris = ();

            gris[y][x] = valorGris;
        }
    }
}

//metodos numericos: newton rampson, serie de taylor
//segmentar cpn hisograma (hacer operaciones)