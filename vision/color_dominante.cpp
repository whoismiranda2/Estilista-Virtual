#include "color_dominante.h"
#include <iostream>

/*
Función: calcular
Obtiene el color promedio de la imagen
aún considera el fondo como parte de la prenda, es un promedio global
*/

void ColorDominante::calcular(const ImagenBMP& imagen)
{
    long sumaR = 0;
    long sumaG = 0;
    long sumaB = 0;

    int total = imagen.ancho * imagen.alto;

    for(int y = 0; y < imagen.alto; y++)
    {
        for(int x = 0; x < imagen.ancho; x++)
        {
            Pixel p = imagen.pixeles[y][x];

            sumaR += p.r;
            sumaG += p.g;
            sumaB += p.b;
        }
    }

    r = sumaR / total;
    g = sumaG / total;
    b = sumaB / total;
}

/*
Función: imprimir
Muestra el color dominante
*/

void ColorDominante::imprimir()
{
    std::cout << "\nColor dominante aproximado:\n";

    std::cout << "R: " << r << "\n";
    std::cout << "G: " << g << "\n";
    std::cout << "B: " << b << "\n";

    if(r > g && r > b)
        std::cout << "Dominante rojizo\n";

    if(b > r && b > g)
        std::cout << "Dominante azulado\n";

    if(g > r && g > b)
        std::cout << "Dominante verdoso\n";
}