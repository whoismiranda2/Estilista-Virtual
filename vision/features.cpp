#include "features.h"
#include "segmentacion.h"
#include <cmath>
#include <algorithm>

// Aplana todos los valores en un vector de 31 floats
vector<float> Features::toVector() const {
    vector<float> v;
    v.push_back(proporcion);
    v.push_back(densidad_arriba);
    v.push_back(densidad_centro);
    v.push_back(densidad_abajo);
    v.push_back(centro_masa_y);
    v.push_back(asimetria);
    v.push_back(varianza_columnas);
    for (int i = 0; i < 8; i++) v.push_back(hist_r[i]);
    for (int i = 0; i < 8; i++) v.push_back(hist_g[i]);
    for (int i = 0; i < 8; i++) v.push_back(hist_b[i]);
    return v; // total: 31 valores
}

// Fraccion de pixeles "con contenido" (no blancos) en un rango de filas
float ExtractorFeatures::calcularDensidad(const ImagenBMP& img,
                                          int filaInicio, int filaFin) {
    int total = 0, oscuros = 0;
    for (int y = filaInicio; y < filaFin; y++) {
        for (int x = 0; x < img.ancho; x++) {
            const Pixel& p = img.pixeles[y][x];
            int gris = (p.r + p.g + p.b) / 3;
            if (gris < 245) oscuros++;
            total++;
        }
    }
    return (total > 0) ? (float)oscuros / total : 0.0f;
}

// Histograma normalizado de 8 bins por canal
void ExtractorFeatures::calcularHistograma(const ImagenBMP& img,
                                           float* hr, float* hg, float* hb) {
    int cr[8] = {}, cg[8] = {}, cb[8] = {};
    int contados = 0;
    for (int y = 0; y < img.alto; y++)
        for (int x = 0; x < img.ancho; x++) {
            const Pixel& p = img.pixeles[y][x];
            if ((p.r + p.g + p.b) / 3 >= 245) continue;
            cr[p.r / 32]++;
            cg[p.g / 32]++;
            cb[p.b / 32]++;
            contados++;
        }
    if (contados == 0) contados = 1;
    for (int i = 0; i < 8; i++) {
        hr[i] = (float)cr[i] / contados;
        hg[i] = (float)cg[i] / contados;
        hb[i] = (float)cb[i] / contados;
    }
}

// Funcion principal de extraccion
Features ExtractorFeatures::extraer(const ImagenBMP& img) {

    // Recortar al bounding box antes de extraer features
    Segmentador seg;
    seg.segmentarPorFondo(img);
    ImagenBMP recortada = seg.recortar(img);

    // Trabajar sobre la imagen recortada
    const ImagenBMP& imagen = (recortada.alto > 10 && recortada.ancho > 10)
                               ? recortada : img;

    Features f;
    int H = imagen.alto, W = imagen.ancho;
    int t1 = H / 3, t2 = 2 * H / 3;

    // 1. Proporcion alto/ancho
    f.proporcion = (W > 0) ? (float)H / W : 1.0f;

    // 2. Densidad por tercios
    f.densidad_arriba = calcularDensidad(imagen, 0,  t1);
    f.densidad_centro = calcularDensidad(imagen, t1, t2);
    f.densidad_abajo  = calcularDensidad(imagen, t2, H);

    // 3. Centro de masa vertical
    float suma_y = 0, suma_total = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const Pixel& p = imagen.pixeles[y][x];
            float peso = (255.0f - (p.r + p.g + p.b) / 3.0f) / 255.0f;
            suma_y     += y * peso;
            suma_total += peso;
        }
    f.centro_masa_y = (suma_total > 0) ? (suma_y / suma_total) / H : 0.5f;

    // 4. Asimetria horizontal
    float diff = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W / 2; x++) {
            int gi = (imagen.pixeles[y][x].r +
                      imagen.pixeles[y][x].g +
                      imagen.pixeles[y][x].b) / 3;
            int gd = (imagen.pixeles[y][W-1-x].r +
                      imagen.pixeles[y][W-1-x].g +
                      imagen.pixeles[y][W-1-x].b) / 3;
            diff += abs(gi - gd);
        }
    f.asimetria = diff / (255.0f * H * (W / 2 + 1));

    // 5. Varianza de densidad por columna
    vector<float> dens_col(W);
    for (int x = 0; x < W; x++) {
        int oscuros = 0;
        for (int y = 0; y < H; y++) {
            int gris = (imagen.pixeles[y][x].r +
                        imagen.pixeles[y][x].g +
                        imagen.pixeles[y][x].b) / 3;
            if (gris < 245) oscuros++;
        }
        dens_col[x] = (float)oscuros / H;
    }
    float media = 0;
    for (float d : dens_col) media += d;
    media /= W;
    float var = 0;
    for (float d : dens_col) var += (d - media) * (d - media);
    f.varianza_columnas = var / W;

    // 6. Histograma por canal
    calcularHistograma(imagen, f.hist_r, f.hist_g, f.hist_b);

    return f;
}