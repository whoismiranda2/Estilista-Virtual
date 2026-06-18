#include "segmentacion.h"
#include <cmath>

/*
---------------------------------------------------------
Función segmentar

Separa los pixeles que tienen color parecido
al color dominante
---------------------------------------------------------
*/

void Segmentador::segmentar(ImagenBMP& imagen, Pixel colorDominante, int umbral)
{
    mascara.resize(imagen.alto, std::vector<int>(imagen.ancho));

    for(int y = 0; y < imagen.alto; y++)
    {
        for(int x = 0; x < imagen.ancho; x++)
        {
            Pixel p = imagen.pixeles[y][x];

            int distancia =
                abs(p.r - colorDominante.r) +
                abs(p.g - colorDominante.g) +
                abs(p.b - colorDominante.b);

            if(distancia < umbral)
            {
                mascara[y][x] = 1; // pertenece a prenda
            }
            else
            {
                mascara[y][x] = 0; // fondo
            }
        }
    }
}

// Segmenta asumiendo fondo blanco: píxel es prenda si su brillo < umbral
void Segmentador::segmentarFondoBlanco(const ImagenBMP& imagen, int umbral)
{
    mascara.assign(imagen.alto, std::vector<int>(imagen.ancho, 0));

    for (int y = 0; y < imagen.alto; y++)
        for (int x = 0; x < imagen.ancho; x++) {
            const Pixel& p = imagen.pixeles[y][x];
            int brillo = (p.r + p.g + p.b) / 3;
            mascara[y][x] = (brillo < umbral) ? 1 : 0;
        }
}

// Detecta el color de fondo muestreando las 4 esquinas (10x10 px)
// y marca como prenda todo píxel suficientemente diferente a ese fondo
void Segmentador::segmentarPorFondo(const ImagenBMP& imagen, int umbral)
{
    int H = imagen.alto, W = imagen.ancho;
    int muestra = std::min(10, std::min(H, W) / 4);

    // Promediar color de las 4 esquinas
    long sumR = 0, sumG = 0, sumB = 0, count = 0;
    for (int y = 0; y < muestra; y++)
        for (int x = 0; x < muestra; x++) {
            // esquina superior izquierda
            sumR += imagen.pixeles[y][x].r;
            sumG += imagen.pixeles[y][x].g;
            sumB += imagen.pixeles[y][x].b;
            // esquina superior derecha
            sumR += imagen.pixeles[y][W-1-x].r;
            sumG += imagen.pixeles[y][W-1-x].g;
            sumB += imagen.pixeles[y][W-1-x].b;
            // esquina inferior izquierda
            sumR += imagen.pixeles[H-1-y][x].r;
            sumG += imagen.pixeles[H-1-y][x].g;
            sumB += imagen.pixeles[H-1-y][x].b;
            // esquina inferior derecha
            sumR += imagen.pixeles[H-1-y][W-1-x].r;
            sumG += imagen.pixeles[H-1-y][W-1-x].g;
            sumB += imagen.pixeles[H-1-y][W-1-x].b;
            count += 4;
        }

    Pixel fondo;
    fondo.r = (unsigned char)(sumR / count);
    fondo.g = (unsigned char)(sumG / count);
    fondo.b = (unsigned char)(sumB / count);

    // Marcar como prenda si la distancia al fondo supera el umbral
    mascara.assign(H, std::vector<int>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const Pixel& p = imagen.pixeles[y][x];
            int distancia = std::abs(p.r - fondo.r)
                          + std::abs(p.g - fondo.g)
                          + std::abs(p.b - fondo.b);
            mascara[y][x] = (distancia > umbral) ? 1 : 0;
        }
}
// Segmentación adaptativa: calcula umbral basado en la desviación
// estándar del color de las esquinas. Funciona mejor con fondos
// que tienen sombras, gradientes o no son blanco puro.
void Segmentador::segmentarAdaptativo(const ImagenBMP& imagen,
                                       float factor, int umbralMin)
{
    int H = imagen.alto, W = imagen.ancho;
    int muestra = std::min(20, std::min(H, W) / 4);

    // Recolectar píxeles de las 4 esquinas (muestra x muestra cada una)
    std::vector<int> mR, mG, mB;
    for (int y = 0; y < muestra; y++)
        for (int x = 0; x < muestra; x++) {
            auto add = [&](int yy, int xx) {
                mR.push_back(imagen.pixeles[yy][xx].r);
                mG.push_back(imagen.pixeles[yy][xx].g);
                mB.push_back(imagen.pixeles[yy][xx].b);
            };
            add(y,       x);        // sup izq
            add(y,       W-1-x);    // sup der
            add(H-1-y,   x);        // inf izq
            add(H-1-y,   W-1-x);    // inf der
        }

    // Media del fondo
    float medR = 0, medG = 0, medB = 0;
    int n = (int)mR.size();
    for (int i = 0; i < n; i++) { medR += mR[i]; medG += mG[i]; medB += mB[i]; }
    medR /= n; medG /= n; medB /= n;

    // Desviación estándar del fondo
    float varR = 0, varG = 0, varB = 0;
    for (int i = 0; i < n; i++) {
        varR += (mR[i] - medR) * (mR[i] - medR);
        varG += (mG[i] - medG) * (mG[i] - medG);
        varB += (mB[i] - medB) * (mB[i] - medB);
    }
    float stdR = std::sqrt(varR / n);
    float stdG = std::sqrt(varG / n);
    float stdB = std::sqrt(varB / n);

    // Umbral adaptativo = factor * (std promedio) + umbralMin
    // Para fondo muy uniforme (std~5): umbral ~30
    // Para fondo con sombras (std~15): umbral ~70 → más permisivo
    float stdMedia = (stdR + stdG + stdB) / 3.0f;
    int umbral = (int)(factor * stdMedia) + umbralMin;
    umbral = std::min(umbral, 80); // tope para no sobre-segmentar

    // Segmentar usando distancia euclidiana al color promedio del fondo
    mascara.assign(H, std::vector<int>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const Pixel& p = imagen.pixeles[y][x];
            float d = std::abs(p.r - medR)
                    + std::abs(p.g - medG)
                    + std::abs(p.b - medB);
            mascara[y][x] = (d > umbral) ? 1 : 0;
        }

    // Erosión simple (2 pasadas) para limpiar ruido del fondo
    for (int iter = 0; iter < 2; iter++) {
        std::vector<std::vector<int>> tmp = mascara;
        for (int y = 1; y < H-1; y++)
            for (int x = 1; x < W-1; x++) {
                if (mascara[y][x] == 0) continue;
                // Si algún vecino en cruz es fondo, marcar como fondo
                if (mascara[y-1][x] == 0 || mascara[y+1][x] == 0 ||
                    mascara[y][x-1] == 0 || mascara[y][x+1] == 0)
                    tmp[y][x] = 0;
            }
        mascara = tmp;
    }

    // Dilatación (3 pasadas) para recuperar bordes del cuerpo
    for (int iter = 0; iter < 3; iter++) {
        std::vector<std::vector<int>> tmp = mascara;
        for (int y = 1; y < H-1; y++)
            for (int x = 1; x < W-1; x++) {
                if (mascara[y][x] == 1) continue;
                if (mascara[y-1][x] == 1 || mascara[y+1][x] == 1 ||
                    mascara[y][x-1] == 1 || mascara[y][x+1] == 1)
                    tmp[y][x] = 1;
            }
        mascara = tmp;
    }
}

ImagenBMP Segmentador::recortar(const ImagenBMP& imagen, int margen)
{
    int H = imagen.alto, W = imagen.ancho;

    // Encontrar límites
    int minY = H, maxY = 0, minX = W, maxX = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (mascara[y][x] == 1) {
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
            }

    // Si no se encontró nada, devolver imagen original
    if (minY > maxY || minX > maxX)
        return imagen;

    // Aplicar margen con clamp a los bordes
    minY = std::max(0, minY - margen);
    maxY = std::min(H - 1, maxY + margen);
    minX = std::max(0, minX - margen);
    maxX = std::min(W - 1, maxX + margen);

    // Construir imagen recortada
    ImagenBMP resultado;
    resultado.alto  = maxY - minY + 1;
    resultado.ancho = maxX - minX + 1;
    resultado.pixeles.resize(resultado.alto,
                             std::vector<Pixel>(resultado.ancho));

    for (int y = minY; y <= maxY; y++)
        for (int x = minX; x <= maxX; x++)
            resultado.pixeles[y - minY][x - minX] = imagen.pixeles[y][x];

    return resultado;
}

ImagenBMP Segmentador::eliminarMarco(const ImagenBMP& imagen, int umbralMarco)
{
    int H = imagen.alto, W = imagen.ancho;
    int limIzq = 0, limDer = W - 1, limSup = 0, limInf = H - 1;

    // Columnas del borde izquierdo: avanzar mientras sean marco
    for (int x = 0; x < W / 4; x++) {
        int marcoPx = 0;
        for (int y = 0; y < H; y++) {
            const Pixel& p = imagen.pixeles[y][x];
            int distBlanco = std::abs(p.r - 255) + std::abs(p.g - 255) + std::abs(p.b - 255);
            if (distBlanco > umbralMarco && distBlanco < umbralMarco * 5)
                marcoPx++;
        }
        if (marcoPx > H * 0.40f)
            limIzq = x + 1;
        else
            break;
    }

    // Columnas del borde derecho
    for (int x = W - 1; x >= 3 * W / 4; x--) {
        int marcoPx = 0;
        for (int y = 0; y < H; y++) {
            const Pixel& p = imagen.pixeles[y][x];
            int distBlanco = std::abs(p.r - 255) + std::abs(p.g - 255) + std::abs(p.b - 255);
            if (distBlanco > umbralMarco && distBlanco < umbralMarco * 5)
                marcoPx++;
        }
        if (marcoPx > H * 0.40f)
            limDer = x - 1;
        else
            break;
    }

    // Filas del borde superior
    for (int y = 0; y < H / 4; y++) {
        int marcoPx = 0;
        for (int x = 0; x < W; x++) {
            const Pixel& p = imagen.pixeles[y][x];
            int distBlanco = std::abs(p.r - 255) + std::abs(p.g - 255) + std::abs(p.b - 255);
            if (distBlanco > umbralMarco && distBlanco < umbralMarco * 5)
                marcoPx++;
        }
        if (marcoPx > W * 0.40f)
            limSup = y + 1;
        else
            break;
    }

    // Filas del borde inferior
    for (int y = H - 1; y >= 3 * H / 4; y--) {
        int marcoPx = 0;
        for (int x = 0; x < W; x++) {
            const Pixel& p = imagen.pixeles[y][x];
            int distBlanco = std::abs(p.r - 255) + std::abs(p.g - 255) + std::abs(p.b - 255);
            if (distBlanco > umbralMarco && distBlanco < umbralMarco * 5)
                marcoPx++;
        }
        if (marcoPx > W * 0.40f)
            limInf = y - 1;
        else
            break;
    }

    // Sin marco detectado: devolver imagen original sin copiar
    if (limIzq == 0 && limDer == W - 1 && limSup == 0 && limInf == H - 1)
        return imagen;

    // Recortar al área interior
    ImagenBMP resultado;
    resultado.alto  = limInf - limSup + 1;
    resultado.ancho = limDer - limIzq + 1;
    resultado.pixeles.resize(resultado.alto,
                             std::vector<Pixel>(resultado.ancho));

    for (int y = limSup; y <= limInf; y++)
        for (int x = limIzq; x <= limDer; x++)
            resultado.pixeles[y - limSup][x - limIzq] = imagen.pixeles[y][x];

    return resultado;
}