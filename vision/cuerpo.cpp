#include "cuerpo.h"
#include <algorithm>
#include <vector>
#include <climits>
#include <cmath>

using namespace std;


string AnalizadorCuerpo::clasificar(float ratio_ch, float ratio_cah) {
    if (ratio_cah < 0.90f) return "triangulo_invertido";
    if (ratio_cah > 1.10f) return "triangulo";
    if (ratio_ch  < 0.75f) return "reloj_de_arena";
    if (ratio_ch  > 0.90f) return "ovalo";
    return "rectangulo";
}

void AnalizadorCuerpo::dibujarLinea(ImagenBMP& img, int fila,
                                    int izq, int der,
                                    Pixel color, int grosor) {
    int H = img.alto, W = img.ancho;
    int mitad = grosor / 2;
    for (int dy = -mitad; dy <= mitad; dy++) {
        int y = fila + dy;
        if (y < 0 || y >= H) continue;
        for (int x = izq; x <= der && x < W; x++)
            img.pixeles[y][x] = color;
    }
    // Marcas verticales en los extremos
    int tickAlto = 11;
    for (int dy = -(tickAlto / 2); dy <= tickAlto / 2; dy++) {
        int y = fila + dy;
        if (y < 0 || y >= H) continue;
        for (int dx = 0; dx < 3 && izq + dx < W; dx++)
            img.pixeles[y][izq + dx] = color;
        for (int dx = 0; dx < 3 && der - dx >= 0; dx++)
            img.pixeles[y][der - dx] = color;
    }
}

MedidasCuerpo AnalizadorCuerpo::analizar(const ImagenBMP& img) {
    MedidasCuerpo m;

    Segmentador segMarco;
    ImagenBMP sinMarco = segMarco.eliminarMarco(img);

    Segmentador seg1;
    seg1.segmentarAdaptativo(sinMarco);
    ImagenBMP recortada = seg1.recortar(sinMarco, 20);
    int Hr = recortada.alto, Wr = recortada.ancho;

    Segmentador seg2;
    seg2.segmentarAdaptativo(recortada);
    const auto& mascara = seg2.mascara;

    // Guardar con fondo blanco para marcar después
    _recortada = recortada;
    for (int y = 0; y < Hr; y++)
        for (int x = 0; x < Wr; x++)
            if (mascara[y][x] == 0)
                _recortada.pixeles[y][x] = {255, 255, 255};

    // Perfil de anchos por máscara de segmentación (para localizar zonas)
    vector<int> izqs(Hr, -1), ders(Hr, -1), anchos(Hr, 0);
    for (int y = 0; y < Hr; y++) {
        for (int x = 0;      x < Wr;  x++) if (mascara[y][x]) { izqs[y] = x; break; }
        for (int x = Wr - 1; x >= 0;  x--) if (mascara[y][x]) { ders[y] = x; break; }
        if (izqs[y] != -1) anchos[y] = ders[y] - izqs[y] + 1;
    }

    // Suavizado fuerte (ventana=17) para ignorar pelo, dedos y ruido
    const int V = 17;
    vector<float> suave(Hr, 0.0f);
    for (int y = 0; y < Hr; y++) {
        float s = 0; int n = 0;
        for (int d = -(V / 2); d <= V / 2; d++) {
            int yy = y + d;
            if (yy >= 0 && yy < Hr) { s += anchos[yy]; n++; }
        }
        suave[y] = n > 0 ? s / n : 0.0f;
    }

    // Rango activo del cuerpo
    int inicio = 0, fin = Hr - 1;
    for (int y = 0;      y < Hr;  y++) if (anchos[y] > 0) { inicio = y; break; }
    for (int y = Hr - 1; y >= 0;  y--) if (anchos[y] > 0) { fin   = y; break; }
    int rango = fin - inicio;
    if (rango < 20) { m.tipo = "rectangulo"; return m; }

    // Perfil capado: suprime protuberancias de brazos/codos
    int ctxDelta = max(10, (int)(rango * 0.20f));
    vector<float> capado(Hr);
    for (int y = 0; y < Hr; y++) {
        int y1 = max(0,      y - ctxDelta);
        int y2 = min(Hr - 1, y + ctxDelta);
        float ctx = (suave[y1] + suave[y2]) / 2.0f;
        capado[y] = min(suave[y], ctx * 1.10f);
    }

    // -----------------------------------------------------------------------
    // COLOR DE FONDO: muestreo de las 4 esquinas de la imagen recortada.
    // El recorte añade 20 px de margen, por lo que las esquinas son siempre
    // fondo. Se usa la media RGB como referencia.
    // -----------------------------------------------------------------------
    int cornerSize = max(4, min(12, min(Hr, Wr) / 10));
    long bgR = 0, bgG = 0, bgB = 0, bgN = 0;
    for (int sy = 0; sy < cornerSize; sy++) {
        for (int sx = 0; sx < cornerSize; sx++) {
            int coords[4][2] = {
                {sy,        sx},
                {sy,        Wr - 1 - sx},
                {Hr-1-sy,   sx},
                {Hr-1-sy,   Wr - 1 - sx}
            };
            for (auto& c : coords) {
                int cy = c[0], cx = c[1];
                if (cy >= 0 && cy < Hr && cx >= 0 && cx < Wr) {
                    bgR += recortada.pixeles[cy][cx].r;
                    bgG += recortada.pixeles[cy][cx].g;
                    bgB += recortada.pixeles[cy][cx].b;
                    bgN++;
                }
            }
        }
    }
    float fgR = bgN ? (float)bgR / bgN : 255.0f;
    float fgG = bgN ? (float)bgG / bgN : 255.0f;
    float fgB = bgN ? (float)bgB / bgN : 255.0f;
    const float BG_THRESH = 40.0f;

    // esCuerpo: devuelve true si el pixel difiere del fondo (distancia RGB > 40)
    auto esCuerpo = [&](int y, int x) -> bool {
        if (y < 0 || y >= Hr || x < 0 || x >= Wr) return false;
        const Pixel& p = recortada.pixeles[y][x];
        float dr = (float)p.r - fgR;
        float dg = (float)p.g - fgG;
        float db = (float)p.b - fgB;
        return sqrtf(dr*dr + dg*dg + db*db) > BG_THRESH;
    };

    // -----------------------------------------------------------------------
    // bordesEnFila: escanea desde los bordes de la imagen hacia el centro
    // dentro de la ventana horizontal [xMin, xMax].
    // Promedia sobre una ventana de ±half filas para estabilidad.
    // xMin/xMax permiten limitar la búsqueda excluyendo los brazos.
    // -----------------------------------------------------------------------
    auto bordesEnFila = [&](int fila, int windowSize, int& izq, int& der,
                             int xMin, int xMax) {
        int half = windowSize / 2;
        long long sumIzq = 0, sumDer = 0, n = 0;
        for (int yy = fila - half; yy <= fila + half; yy++) {
            if (yy < 0 || yy >= Hr) continue;
            int li = -1;
            for (int x = xMin; x <= xMax; x++) {
                if (esCuerpo(yy, x)) { li = x; break; }
            }
            int ri = -1;
            for (int x = xMax; x >= xMin; x--) {
                if (esCuerpo(yy, x)) { ri = x; break; }
            }
            if (li != -1 && ri != -1 && ri > li) {
                sumIzq += li; sumDer += ri; n++;
            }
        }
        if (n > 0) { izq = (int)(sumIzq / n); der = (int)(sumDer / n); }
        else       { izq = xMin;               der = xMax; }
    };

    // -----------------------------------------------------------------------
    // bordesEnFilaMask: fallback que usa la máscara de segmentación
    // (escaneo desde el centro hacia los bordes). Se activa si la validación
    // anatómica falla tras la detección por color.
    // -----------------------------------------------------------------------
    auto bordesEnFilaMask = [&](int fila, int windowSize, int& izq, int& der) {
        int half = windowSize / 2;
        long long sumIzq = 0, sumDer = 0, n = 0;
        for (int yy = fila - half; yy <= fila + half; yy++) {
            if (yy < 0 || yy >= Hr || izqs[yy] == -1) continue;
            int cx = (izqs[yy] + ders[yy]) / 2;
            int li = cx;
            for (int x = cx; x >= 0; x--) {
                if (mascara[yy][x] == 0) { li = x + 1; break; }
                if (x == 0) li = 0;
            }
            int ri = cx;
            for (int x = cx; x < Wr; x++) {
                if (mascara[yy][x] == 0) { ri = x - 1; break; }
                if (x == Wr - 1) ri = Wr - 1;
            }
            if (ri > li) { sumIzq += li; sumDer += ri; n++; }
        }
        if (n > 0) { izq = (int)(sumIzq / n); der = (int)(sumDer / n); }
        else       { izq = 0;                  der = Wr - 1; }
    };

    // -----------------------------------------------------------------------
    // CUELLO: mínimo suavizado en [8 %, 20 %] del cuerpo
    // -----------------------------------------------------------------------
    int cuello = inicio + (int)(rango * 0.14f);
    float minCuello = 1e9f;
    for (int y = inicio + (int)(rango * 0.08f);
             y < inicio + (int)(rango * 0.20f); y++) {
        if (suave[y] < minCuello) { minCuello = suave[y]; cuello = y; }
    }

    // -----------------------------------------------------------------------
    // HOMBROS (shoulder_y): máximo en zona [cuello, 28 %]
    // Sin límite lateral: los hombros pueden ser el punto más ancho.
    // -----------------------------------------------------------------------
    int fin_H = inicio + (int)(rango * 0.28f);
    int fH = cuello;
    float maxH = 0;
    for (int y = cuello; y <= fin_H; y++)
        if (suave[y] > maxH) { maxH = suave[y]; fH = y; }

    m.fila_hombros = fH;
    bordesEnFila(fH, 9, m.izq_hombros, m.der_hombros, 0, Wr - 1);
    m.ancho_hombros = max(1.0f, (float)(m.der_hombros - m.izq_hombros + 1));

    // -----------------------------------------------------------------------
    // CADERA DE REFERENCIA (hip_y): máximo en zona [58 %, 80 %]
    // Solo se usa como ancla para interpolación; no se publica directamente.
    // -----------------------------------------------------------------------
    int ini_HIP = max(fH + (int)(rango * 0.28f), inicio + (int)(rango * 0.58f));
    int fin_HIP = inicio + (int)(rango * 0.80f);
    int fHIP = ini_HIP;
    float maxHIP = 0;
    for (int y = ini_HIP; y <= fin_HIP && y < Hr; y++)
        if (suave[y] > maxHIP) { maxHIP = suave[y]; fHIP = y; }

    int distSH = fHIP - fH;
    if (distSH < 5) distSH = (int)(rango * 0.40f);

    // -----------------------------------------------------------------------
    // CINTURA y CADERA: posiciones fijas por interpolación
    // -----------------------------------------------------------------------
    int fC   = fH + (int)(0.55f * distSH);
    fC   = max(fH + 2, min(fC,   fHIP - 2));

    int fCAD = fH + (int)(1.10f * distSH);
    fCAD = max(fC  + 2, min(fCAD, fin  - 1));

    // -----------------------------------------------------------------------
    // FRONTERA DE CODOS: el punto de máximo ancho capado en [shoulder, waist)
    // representa la zona de los codos. Se usa como límite lateral para las
    // mediciones de cintura y cadera (× 1.1 de margen).
    // -----------------------------------------------------------------------
    int codo_xMin = 0, codo_xMax = Wr - 1;
    {
        float maxEl = 0;
        int   rowEl = fH;
        for (int y = fH; y < fC && y < Hr; y++) {
            if (capado[y] > maxEl && izqs[y] >= 0) {
                maxEl = capado[y];
                rowEl = y;
            }
        }
        if (izqs[rowEl] >= 0 && ders[rowEl] > izqs[rowEl]) {
            int cx = (izqs[rowEl] + ders[rowEl]) / 2;
            int hw = (int)((ders[rowEl] - izqs[rowEl]) * 0.5f * 1.1f);
            codo_xMin = max(0,      cx - hw);
            codo_xMax = min(Wr - 1, cx + hw);
        }
    }

    // -----------------------------------------------------------------------
    // CINTURA: detección por color limitada a la frontera de codos
    // -----------------------------------------------------------------------
    m.fila_cintura = fC;
    bordesEnFila(fC, 9, m.izq_cintura, m.der_cintura, codo_xMin, codo_xMax);
    m.ancho_cintura = max(1.0f, (float)(m.der_cintura - m.izq_cintura + 1));

    // -----------------------------------------------------------------------
    // CADERA BAJA: detección por color limitada a la frontera de codos
    // -----------------------------------------------------------------------
    m.fila_cadera = fCAD;
    bordesEnFila(fCAD, 9, m.izq_cadera, m.der_cadera, codo_xMin, codo_xMax);
    m.ancho_cadera = max(1.0f, (float)(m.der_cadera - m.izq_cadera + 1));

    // -----------------------------------------------------------------------
    // VALIDACIÓN ANATÓMICA: cintura < hombros Y cintura < cadera.
    // Si no se cumple, la detección por color capturó brazos u otro ruido;
    // se descarta y se re-mide con la máscara de segmentación.
    // -----------------------------------------------------------------------
    bool valido = (m.ancho_cintura < m.ancho_hombros &&
                   m.ancho_cintura < m.ancho_cadera);
    if (!valido) {
        bordesEnFilaMask(fC,   9, m.izq_cintura, m.der_cintura);
        m.ancho_cintura = max(1.0f, (float)(m.der_cintura - m.izq_cintura + 1));

        bordesEnFilaMask(fCAD, 9, m.izq_cadera,  m.der_cadera);
        m.ancho_cadera  = max(1.0f, (float)(m.der_cadera  - m.izq_cadera  + 1));

        // Pinza de último recurso para garantizar la invariante
        if (m.ancho_cintura >= m.ancho_hombros)
            m.ancho_cintura = m.ancho_hombros * 0.90f;
        if (m.ancho_cintura >= m.ancho_cadera)
            m.ancho_cintura = m.ancho_cadera  * 0.90f;
    }

    m.ratio_ch  = m.ancho_cintura / m.ancho_hombros;
    m.ratio_cah = m.ancho_cadera  / m.ancho_hombros;
    m.tipo = clasificar(m.ratio_ch, m.ratio_cah);

    return m;
}

void AnalizadorCuerpo::marcar(const MedidasCuerpo& m, const string& rutaSalida) {
    ImagenBMP img = _recortada;

    Pixel verde   = {93,  202, 165};
    Pixel morado  = {167, 139, 192};
    Pixel naranja = {232, 168, 124};

    if (m.fila_hombros >= 0)
        dibujarLinea(img, m.fila_hombros, m.izq_hombros, m.der_hombros, verde);
    if (m.fila_cintura >= 0)
        dibujarLinea(img, m.fila_cintura, m.izq_cintura, m.der_cintura, morado);
    if (m.fila_cadera >= 0)
        dibujarLinea(img, m.fila_cadera,  m.izq_cadera,  m.der_cadera,  naranja);

    img.guardarBMP(rutaSalida, img.pixeles);
}
