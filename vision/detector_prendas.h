#ifndef DETECTOR_PRENDAS_H
#define DETECTOR_PRENDAS_H

#include "lector_bmp.h"
#include "segmentacion.h"
#include "features.h"
#include "perceptron.h"
#include "colorimetria.h"
#include <vector>
#include <string>

using namespace std;

struct PrendaDetectada {
    string categoria;
    float  score;
    int    zona_ini;
    int    zona_fin;
    string ruta_imagen;  // ruta de la imagen recortada guardada
};

class DetectorPrendas {
public:
    vector<float> minVal;
    vector<float> maxVal;

    vector<PrendaDetectada> detectar(const ImagenBMP& img,
                                     ClasificadorPrendas& clasificador,
                                     const string& carpetaTemp = "static");

private:
    ImagenBMP extraerZonaRopa(const ImagenBMP& img,
                               const vector<vector<int>>& mascara,
                               int filaIni, int filaFin);
    float densidadTotal(const Features& f);
};

#endif
