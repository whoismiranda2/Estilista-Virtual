#ifndef CUERPO_H
#define CUERPO_H

#include "lector_bmp.h"
#include "segmentacion.h"
#include <string>
#include <vector>

using namespace std;

struct MedidasCuerpo {
    float ancho_hombros = 0;
    float ancho_cintura = 0;
    float ancho_cadera  = 0;
    float ratio_ch  = 0;
    float ratio_cah = 0;
    string tipo;

    // Fila y bordes detectados (coordenadas sobre la imagen recortada)
    int fila_hombros = -1, izq_hombros = 0, der_hombros = 0;
    int fila_cintura = -1, izq_cintura = 0, der_cintura = 0;
    int fila_cadera  = -1, izq_cadera  = 0, der_cadera  = 0;
};

class AnalizadorCuerpo {
public:
    MedidasCuerpo analizar(const ImagenBMP& img);

    // Dibuja las líneas de medición sobre la imagen recortada y la guarda
    void marcar(const MedidasCuerpo& m, const string& rutaSalida);

private:
    ImagenBMP _recortada;   // imagen recortada con fondo blanco (guardada para marcar)

    string clasificar(float ratio_ch, float ratio_cah);

    // Dibuja una línea horizontal coloreada con marcas en los extremos
    void dibujarLinea(ImagenBMP& img, int fila, int izq, int der,
                      Pixel color, int grosor = 3);
};

#endif
