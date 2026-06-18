#include "lector_bmp.h"
#include <fstream>
#include <iostream>

using namespace std;
/*
Función: cargar
Lee un archivo BMP y guarda los pixeles en memoria
*/

bool ImagenBMP::cargar(const string& ruta)
{
    ifstream archivo(ruta, ios::binary);

    if(!archivo)
    {
        //cout << "No se pudo abrir la imagen\n";
        return false;
    }

    //HEADER BMP 

    unsigned char header[54];
    archivo.read((char*)header, 54);

    ancho = *(int*)&header[18];
    alto  = *(int*)&header[22];

    int offsetDatos = *(int*)&header[10];
    int bitsPorPixel = *(short*)&header[28];

    if(bitsPorPixel != 24)  //se asegura de que el bmp sea de 24 bits - 3bytes rojo, 3bytes verde, 3bytes azul
    {
        //cout << "Solo se soportan BMP de 24 bits\n";
        return false;
    }

    //CALCULO DEL PADDING
    /*
    como las filas deben ser múltiplos de 4 (bits) se agregan bits de relleno en caso de que no los tenga
    dichos bits, se ignoran/saltan al fin de la fila
    */

    int padding = (4 - (ancho * 3) % 4) % 4;

    pixeles.resize(alto, vector<Pixel>(ancho));

    archivo.seekg(offsetDatos, ios::beg);

    //LECTURA DE PIXELES

    for(int y = alto - 1; y >= 0; y--)
    {
        for(int x = 0; x < ancho; x++)
        {
            unsigned char b = archivo.get();
            unsigned char g = archivo.get();
            unsigned char r = archivo.get();

            Pixel p;
            p.r = r;
            p.g = g;
            p.b = b;

            pixeles[y][x] = p;
        }

        archivo.ignore(padding);
    }

    archivo.close();

    //cout << "Imagen cargada correctamente\n";
    //cout << "Ancho: " << ancho << "\n";
    //cout << "Alto: " << alto << "\n";

    return true;
}

/*
Función: guardarGrises
Guarda una imagen BMP en escala de grises
*/

bool ImagenBMP::guardarGrises(
    const string& ruta,
    const vector<vector<unsigned char>>& gris)
{
    ofstream archivo(ruta, ios::binary);

    if(!archivo)
        return false;

    int padding = (4 - (ancho * 3) % 4) % 4;

    int tamArchivo = 54 + (3 * ancho + padding) * alto;

    unsigned char encabezado[54] = {0};

    encabezado[0] = 'B';
    encabezado[1] = 'M';

    *(int*)&encabezado[2] = tamArchivo;
    *(int*)&encabezado[10] = 54;
    *(int*)&encabezado[14] = 40;
    *(int*)&encabezado[18] = ancho;
    *(int*)&encabezado[22] = alto;
    encabezado[26] = 1;
    encabezado[28] = 24;

    archivo.write((char*)encabezado, 54);

    for(int y = alto - 1; y >= 0; y--)
    {
        for(int x = 0; x < ancho; x++)
        {
            unsigned char g = gris[y][x];

            unsigned char pixel[3];

            pixel[0] = g;
            pixel[1] = g;
            pixel[2] = g;

            archivo.write((char*)pixel, 3);
        }

        unsigned char relleno[3] = {0,0,0};
        archivo.write((char*)relleno, padding);
    }

    archivo.close();

    return true;
}

bool ImagenBMP::guardarBMP(
    const string& ruta,
    const vector<vector<Pixel>>& data)
{
    ofstream archivo(ruta, std::ios::binary);

    if(!archivo)
        return false;

    int padding = (4 - (ancho * 3) % 4) % 4;

    int tamArchivo = 54 + (3 * ancho + padding) * alto;

    unsigned char encabezado[54] = {0};

    encabezado[0] = 'B';
    encabezado[1] = 'M';

    *(int*)&encabezado[2] = tamArchivo;
    *(int*)&encabezado[10] = 54;
    *(int*)&encabezado[14] = 40;
    *(int*)&encabezado[18] = ancho;
    *(int*)&encabezado[22] = alto;
    encabezado[26] = 1;
    encabezado[28] = 24;

    archivo.write((char*)encabezado, 54);

    //ESCRITURA DE PIXELES (RGB REAL)
    for(int y = alto - 1; y >= 0; y--)
    {
        for(int x = 0; x < ancho; x++)
        {
            Pixel p = data[y][x];

            unsigned char pixel[3];

            pixel[0] = p.b; // BMP usa BGR
            pixel[1] = p.g;
            pixel[2] = p.r;

            archivo.write((char*)pixel, 3);
        }

        unsigned char relleno[3] = {0,0,0};
        archivo.write((char*)relleno, padding);
    }

    archivo.close();

    return true;
}