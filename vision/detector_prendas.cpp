#include "detector_prendas.h"
#include <algorithm>
#include <sstream>

using namespace std;

float DetectorPrendas::densidadTotal(const Features& f) {
    return f.densidad_arriba + f.densidad_centro + f.densidad_abajo;
}

ImagenBMP DetectorPrendas::extraerZonaRopa(const ImagenBMP& img,
                                            const vector<vector<int>>& mascara,
                                            int filaIni, int filaFin) {
    int H = filaFin - filaIni;
    int W = img.ancho;
    AnalizadorColorimetria ac;

    // Paso 1: crear imagen con fondo blanco, solo píxeles de ropa
    ImagenBMP zona;
    zona.alto  = H;
    zona.ancho = W;
    zona.pixeles.assign(H, vector<Pixel>(W, {255, 255, 255}));

    for (int y = filaIni; y < filaFin && y < img.alto; y++)
        for (int x = 0; x < W; x++) {
            if (mascara[y][x] == 0) continue;  // fondo
            const Pixel& p = img.pixeles[y][x];
            ColorHSV c = ac.rgbAhsv(p.r, p.g, p.b);
            if (ac.esPiel(c)) continue;         // piel
            // Verificar que no sea gris muy claro (residuos de sombra)
            int brillo = (p.r + p.g + p.b) / 3;
            if (brillo > 210) continue;         // demasiado claro = residuo
            zona.pixeles[y - filaIni][x] = p;
        }

    // Paso 2: recortar al bounding box de los píxeles de ropa
    int minY = H, maxY = 0, minX = W, maxX = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const Pixel& p = zona.pixeles[y][x];
            if (p.r < 230 || p.g < 230 || p.b < 230) {
                minY = min(minY, y); maxY = max(maxY, y);
                minX = min(minX, x); maxX = max(maxX, x);
            }
        }

    // Si no hay contenido suficiente, devolver zona vacía pequeña
    if (minY >= maxY || minX >= maxX || (maxY-minY) < 20 || (maxX-minX) < 20) {
        ImagenBMP vacia;
        vacia.alto = 10; vacia.ancho = 10;
        vacia.pixeles.assign(10, vector<Pixel>(10, {255,255,255}));
        return vacia;
    }

    // Añadir margen de 15px
    int margen = 15;
    minY = max(0,   minY - margen);
    maxY = min(H-1, maxY + margen);
    minX = max(0,   minX - margen);
    maxX = min(W-1, maxX + margen);

    // Paso 2.5: rellenar huecos pequeños dentro de la prenda (morfología)
    // Para cada píxel blanco, si tiene vecinos de ropa en todas las direcciones
    // lo convierte en el color promedio de sus vecinos
    for (int iter = 0; iter < 2; iter++) {
        for (int y = 1; y < H-1; y++)
            for (int x = 1; x < W-1; x++) {
                Pixel& p = zona.pixeles[y][x];
                if (p.r < 230) continue; // ya es ropa
                // Contar vecinos con contenido
                int vecinos = 0;
                int sr=0, sg=0, sb=0;
                for (int dy=-1; dy<=1; dy++)
                    for (int dx=-1; dx<=1; dx++) {
                        if (dy==0 && dx==0) continue;
                        const Pixel& v = zona.pixeles[y+dy][x+dx];
                        if (v.r < 230) { vecinos++; sr+=v.r; sg+=v.g; sb+=v.b; }
                    }
                if (vecinos >= 5) {
                    p.r = sr/vecinos; p.g = sg/vecinos; p.b = sb/vecinos;
                }
            }
    }

    // Paso 3: construir imagen recortada
    ImagenBMP recortada;
    recortada.alto  = maxY - minY + 1;
    recortada.ancho = maxX - minX + 1;
    recortada.pixeles.assign(recortada.alto,
                             vector<Pixel>(recortada.ancho, {255,255,255}));
    for (int y = minY; y <= maxY; y++)
        for (int x = minX; x <= maxX; x++)
            recortada.pixeles[y-minY][x-minX] = zona.pixeles[y][x];

    return recortada;
}

vector<PrendaDetectada> DetectorPrendas::detectar(
        const ImagenBMP& img,
        ClasificadorPrendas& clasificador,
        const string& carpetaTemp) {

    vector<PrendaDetectada> resultado;

    // Segmentar y recortar
    Segmentador seg;
    seg.segmentarPorFondo(img, 20);
    ImagenBMP recortada = seg.recortar(img, 5);
    int Hr = recortada.alto;

    Segmentador seg2;
    seg2.segmentarPorFondo(recortada, 20);

    int inicio = (int)(Hr * 0.15f);
    int fin    = (int)(Hr * 0.92f);
    int rango  = fin - inicio;

    // Zona top: 15%-52% con solapamiento
    int ini_top = inicio;
    int fin_top = inicio + (int)(rango * 0.46f);  // top: 10%-46%
    int ini_inf = inicio + (int)(rango * 0.43f);  // inferior: 43%-78%
    int fin_inf = inicio + (int)(rango * 0.78f);  // solapamiento 3%

    ExtractorFeatures ef;

    // --- Analizar zona top ---
    ImagenBMP zonaTop = extraerZonaRopa(recortada, seg2.mascara, ini_top, fin_top);
    Features fTop = ef.extraer(zonaTop);
    float densTop = densidadTotal(fTop);

    if (densTop > 0.25f) {
        vector<float> vTop = fTop.toVector();
        // Normalizar features antes de clasificar
        if (!minVal.empty() && minVal.size() == vTop.size()) {
            for (int i = 0; i < (int)vTop.size(); i++) {
                float rango = maxVal[i] - minVal[i];
                vTop[i] = (rango > 1e-6f) ? (vTop[i] - minVal[i]) / rango : 0.0f;
            }
        }
        string catTop = clasificador.clasificar(vTop);

        // Guardar imagen recortada limpia de la prenda
        string rutaTop = carpetaTemp + "/temp_prenda_0.bmp";
        if (zonaTop.alto > 10 && zonaTop.ancho > 10)
            zonaTop.guardarBMP(rutaTop, zonaTop.pixeles);

        PrendaDetectada pd;
        pd.categoria   = catTop;
        pd.score       = densTop;
        pd.zona_ini    = ini_top;
        pd.zona_fin    = fin_top;
        pd.ruta_imagen = "/static/temp_prenda_0.bmp";
        resultado.push_back(pd);

        // Si es vestido, no analizar zona inferior
        if (catTop == "vestido") return resultado;
    }

    // --- Analizar zona inferior ---
    ImagenBMP zonaInf = extraerZonaRopa(recortada, seg2.mascara, ini_inf, fin_inf);
    Features fInf = ef.extraer(zonaInf);
    float densInf = densidadTotal(fInf);

    if (densInf > 0.25f) {
        vector<float> vInf = fInf.toVector();
        if (!minVal.empty() && minVal.size() == vInf.size()) {
            for (int i = 0; i < (int)vInf.size(); i++) {
                float rango = maxVal[i] - minVal[i];
                vInf[i] = (rango > 1e-6f) ? (vInf[i] - minVal[i]) / rango : 0.0f;
            }
        }
        string catInf = clasificador.clasificar(vInf);

        // No repetir la misma categoría que el top
        if (!resultado.empty() && catInf == resultado[0].categoria)
            catInf = "pantalon"; // fallback razonable

        string rutaInf = carpetaTemp + "/temp_prenda_1.bmp";
        if (zonaInf.alto > 10 && zonaInf.ancho > 10)
            zonaInf.guardarBMP(rutaInf, zonaInf.pixeles);

        PrendaDetectada pd;
        pd.categoria   = catInf;
        pd.score       = densInf;
        pd.zona_ini    = ini_inf;
        pd.zona_fin    = fin_inf;
        pd.ruta_imagen = "/static/temp_prenda_1.bmp";
        resultado.push_back(pd);
    }

    return resultado;
}
