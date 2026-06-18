#include "histograma.h"
#include <iostream>

using namespace std;

void Histograma::calcular(const ImagenBMP& imagen)
{
    for(int i = 0; i < 256; i++)
        histR[i] = histG[i] = histB[i] = histGray[i] = 0;

    for(int y = 0; y < imagen.alto; y++)
    {
        for(int x = 0; x < imagen.ancho; x++)
        {
            Pixel p = imagen.pixeles[y][x];

            histR[p.r]++;
            histG[p.g]++;
            histB[p.b]++;

            int gray = (p.r + p.g + p.b) / 3;
            histGray[gray]++;
        }
    }
}

void Histograma::calcularAcumulado()
{
    for(int i = 1; i < 256; i++)
    {
        histR[i] += histR[i-1];
        histG[i] += histG[i-1];
        histB[i] += histB[i-1];
        histGray[i] += histGray[i-1];
    }
}

static void imprimirArray(int* hist)
{
    cout << "[";
    for(int i = 0; i < 256; i++){
        cout << hist[i];
        if(i < 255) cout << ",";
    }
    cout << "]";
}

void Histograma::imprimirJSON()
{
    cout << "{";

    cout << "\"r\":";
    imprimirArray(histR);
    cout << ",\"g\":";
    imprimirArray(histG);
    cout << ",\"b\":";
    imprimirArray(histB);
    cout << ",\"gray\":";
    imprimirArray(histGray);

    cout << "}";

    cout.flush();
}