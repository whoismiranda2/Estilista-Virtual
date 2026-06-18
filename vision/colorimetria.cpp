#include "colorimetria.h"
#include <algorithm>
#include <cmath>

using namespace std;

ColorHSV AnalizadorColorimetria::rgbAhsv(unsigned char r,
                                          unsigned char g,
                                          unsigned char b) {
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;

    float cmax = max({rf, gf, bf});
    float cmin = min({rf, gf, bf});
    float delta = cmax - cmin;

    ColorHSV c;
    c.v = cmax * 255.0f;

    if (cmax < 1e-6f) { c.s = 0; c.h = 0; return c; }
    c.s = (delta / cmax) * 255.0f;

    if (delta < 1e-6f) { c.h = 0; return c; }

    if (cmax == rf)
        c.h = 60.0f * fmod((gf - bf) / delta, 6.0f);
    else if (cmax == gf)
        c.h = 60.0f * ((bf - rf) / delta + 2.0f);
    else
        c.h = 60.0f * ((rf - gf) / delta + 4.0f);

    if (c.h < 0) c.h += 360.0f;
    return c;
}

bool AnalizadorColorimetria::esPiel(const ColorHSV& c) {
    // Excluir grises neutros (S < 25) aunque caigan en rango de H cálido
    // Los grises tienen saturación baja independientemente del tono
    bool tono_calido = (c.h <= 50.0f || c.h >= 340.0f);
    return tono_calido && c.s >= 25.0f && c.s <= 180.0f && c.v >= 70.0f;
}

ResultadoColorimetria AnalizadorColorimetria::analizar(const ImagenBMP& img) {
    ResultadoColorimetria res;

    // Segmentar fondo y recortar
    Segmentador seg;
    seg.segmentarPorFondo(img, 20);
    ImagenBMP recortada = seg.recortar(img, 5);
    const ImagenBMP& imagen = (recortada.alto > 10) ? recortada : img;

    int H = imagen.alto, W = imagen.ancho;

    // Zona de rostro: entre 5% y 20% superior de la silueta
    int ini = (int)(H * 0.05f);
    int fin = (int)(H * 0.20f);

    double sumH = 0, sumS = 0, sumV = 0;
    double sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    for (int y = ini; y < fin; y++)
        for (int x = 0; x < W; x++) {
            const Pixel& p = imagen.pixeles[y][x];
            ColorHSV c = rgbAhsv(p.r, p.g, p.b);
            if (esPiel(c)) {
                sumH += c.h;
                sumS += c.s;
                sumV += c.v;
                sumR += p.r; sumG += p.g; sumB += p.b;
                count++;
            }
        }

    if (count < 50) {
        // Muy pocos píxeles de piel detectados — ampliar zona
        sumH = sumS = sumV = 0;
        sumR = sumG = sumB = 0;
        count = 0;
        ini = 0; fin = (int)(H * 0.35f);
        for (int y = ini; y < fin; y++)
            for (int x = 0; x < W; x++) {
                const Pixel& p = imagen.pixeles[y][x];
                ColorHSV c = rgbAhsv(p.r, p.g, p.b);
                if (esPiel(c)) {
                    sumH += c.h; sumS += c.s; sumV += c.v;
                    sumR += p.r; sumG += p.g; sumB += p.b;
                    count++;
                }
            }
    }

    res.pixeles_piel = count;

    if (count == 0) {
        res.estacion    = "indefinido";
        res.temperatura = "indefinido";
        res.valor       = "indefinido";
        res.tono_piel_h = res.tono_piel_s = res.tono_piel_v = 0;
        res.piel_r = res.piel_g = res.piel_b = 0;
        return res;
    }

    res.tono_piel_h = (float)(sumH / count);
    res.tono_piel_s = (float)(sumS / count);
    res.tono_piel_v = (float)(sumV / count);
    res.piel_r = (float)(sumR / count);
    res.piel_g = (float)(sumG / count);
    res.piel_b = (float)(sumB / count);

    float R = res.piel_r;
    float G = res.piel_g;
    float B = res.piel_b;

    // Temperatura: basada en la diferencia R-B (cálido=dorado, frío=rosado)
    // y confirmada con G-B como señal secundaria
    float diff_rb = R - B;
    float diff_gb = G - B;
    float diff_rg = R - G;

    int puntos_calido = 0;
    if (diff_rb > 60.0f) puntos_calido += 2;
    else if (diff_rb > 45.0f) puntos_calido += 1;
    if (diff_gb > 22.0f) puntos_calido += 1;
    bool calido = (puntos_calido >= 2);
    res.temperatura = calido ? "calido" : "frio";

    // Profundidad: R-G alto indica piel con más pigmentación/profundidad
    // Separa primavera (clara y luminosa) de otoño (cálida y profunda)
    bool profundo = (diff_rg > 42.0f);

    // Valor: luminancia perceptual (más precisa que V de HSV)
    float lum = R * 0.299f + G * 0.587f + B * 0.114f;
    bool claro = (lum > 155.0f);
    res.valor = claro ? "claro" : "oscuro";

    // Clasificación final
    // primavera = cálido + claro + no profundo (piel luminosa dorada)
    // otoño     = cálido + oscuro O profundo   (piel terrosa/bronce)
    // verano    = frío   + claro               (piel rosada/neutra clara)
    // invierno  = frío   + oscuro              (piel fría y profunda)
    if (calido && claro && !profundo)       res.estacion = "primavera";
    else if (calido)                        res.estacion = "otono";
    else if (claro)                         res.estacion = "verano";
    else                                    res.estacion = "invierno";

    return res;
}

vector<string> AnalizadorColorimetria::paletaRecomendada(const string& estacion) {
    if (estacion == "primavera")
        return {"#F4A460","#FFD700","#98FB98","#87CEEB","#FFA07A","#FFFACD","#F08080"};
    if (estacion == "verano")
        return {"#DDA0DD","#B0C4DE","#F0F8FF","#E6E6FA","#FFB6C1","#C0C0C0","#AFEEEE"};
    if (estacion == "otono")
        return {"#8B4513","#D2691E","#DAA520","#556B2F","#8B0000","#CD853F","#A0522D"};
    // invierno
    return {"#000000","#FFFFFF","#DC143C","#00008B","#006400","#4B0082","#C0C0C0"};
}
