#include <iostream>
#include <fstream>
#include "lector_bmp.h"
#include "histograma.h"
#include "color_dominante.h"
#include "escala_grises.h"
#include "rgb_hsv.h"
#include "features.h"
#include "segmentacion.h"
#include "perceptron.h"
#include "cuerpo.h"
#include "colorimetria.h"
#include "detector_prendas.h"

using namespace std;

int main(int argc, char* argv[])
{
    if(argc < 3) return 1;

    string ruta = argv[1];
    string op   = argv[2];

    ImagenBMP imagen;
    if(!imagen.cargar(ruta)) return 1;

    // --- Histogramas ---
    if(op == "hist") {
        Histograma h;
        h.calcular(imagen);
        h.imprimirJSON();
        return 0;
    }
    else if(op == "hist_acum") {
        Histograma h;
        h.calcular(imagen);
        h.calcularAcumulado();
        h.imprimirJSON();
        return 0;
    }

    // --- Escala de grises ---
    else if(op == "grises") {
        EscalaGrises eg;
        eg.convertir(imagen);
        imagen.guardarGrises("static/procesada.bmp", eg.gris);
        return 0;
    }

    // --- Ajuste de color ---
    else if(op == "color") {
        if(argc < 6) return 1;
        int rShift = atoi(argv[3]);
        int gShift = atoi(argv[4]);
        int bShift = atoi(argv[5]);
        for(int y = 0; y < imagen.alto; y++)
            for(int x = 0; x < imagen.ancho; x++) {
                Pixel& p = imagen.pixeles[y][x];
                p.r = min(255, max(0, p.r + rShift));
                p.g = min(255, max(0, p.g + gShift));
                p.b = min(255, max(0, p.b + bShift));
            }
        imagen.guardarBMP("static/procesada.bmp", imagen.pixeles);
        return 0;
    }

    // --- Segmentar: elimina el fondo y guarda solo la prenda ---
    else if(op == "segmentar") {
        Segmentador seg;
        seg.segmentarPorFondo(imagen);

        // Poner fondo blanco donde la mascara es 0
        for(int y = 0; y < imagen.alto; y++)
            for(int x = 0; x < imagen.ancho; x++)
                if(seg.mascara[y][x] == 0) {
                    imagen.pixeles[y][x].r = 255;
                    imagen.pixeles[y][x].g = 255;
                    imagen.pixeles[y][x].b = 255;
                }

        // Recortar al bounding box de la prenda
        ImagenBMP recortada = seg.recortar(imagen);
        recortada.guardarBMP("static/procesada.bmp", recortada.pixeles);
        return 0;
    }

    // --- Features (debug) ---
    else if(op == "features") {
        ExtractorFeatures ef;
        Features f = ef.extraer(imagen);
        vector<float> v = f.toVector();

        Segmentador seg;
        seg.segmentarPorFondo(imagen);
        ImagenBMP recortada = seg.recortar(imagen);

        cout << "=== FEATURES DE LA IMAGEN ===" << endl;
        cout << "Dimensiones originales: " << imagen.ancho << " x " << imagen.alto << endl;
        cout << "Dimensiones recortadas: " << recortada.ancho << " x " << recortada.alto << endl;
        cout << endl;
        cout << "[Geometria]" << endl;
        cout << "  proporcion (alto/ancho): " << f.proporcion << endl;
        cout << "[Densidad por tercios]" << endl;
        cout << "  arriba: " << f.densidad_arriba << endl;
        cout << "  centro: " << f.densidad_centro << endl;
        cout << "  abajo : " << f.densidad_abajo  << endl;
        cout << "[Espaciales]" << endl;
        cout << "  centro de masa Y: " << f.centro_masa_y << endl;
        cout << "  asimetria:        " << f.asimetria << endl;
        cout << "  varianza cols:    " << f.varianza_columnas << endl;
        cout << "[Histograma R] ";
        for(int i = 0; i < 8; i++) cout << f.hist_r[i] << " ";
        cout << endl;
        cout << "[Histograma G] ";
        for(int i = 0; i < 8; i++) cout << f.hist_g[i] << " ";
        cout << endl;
        cout << "[Histograma B] ";
        for(int i = 0; i < 8; i++) cout << f.hist_b[i] << " ";
        cout << endl;
        cout << "Vector completo (" << v.size() << " valores):" << endl;
        for(int i = 0; i < (int)v.size(); i++)
            cout << "  x[" << i << "] = " << v[i] << endl;
        return 0;
    }

    // --- Clasificar ---
    else if(op == "clasificar") {
        ClasificadorPrendas clasificador;
        if(!clasificador.cargarPesos("pesos.txt")) {
            cout << "otro" << endl;
            return 0;
        }

        // Cargar rangos de normalizacion
        vector<float> minVal, maxVal;
        ifstream fRangos("rangos.txt");
        if(fRangos.is_open()) {
            int n; fRangos >> n;
            minVal.resize(n); maxVal.resize(n);
            for(float& v : minVal) fRangos >> v;
            for(float& v : maxVal) fRangos >> v;
        }

        ExtractorFeatures ef;
        Features f = ef.extraer(imagen);
        vector<float> v = f.toVector();

        // Normalizar si se cargaron los rangos
        if(!minVal.empty()) {
            for(int i = 0; i < (int)v.size(); i++) {
                float rango = maxVal[i] - minVal[i];
                v[i] = (rango > 1e-6f) ? (v[i] - minVal[i]) / rango : 0.0f;
            }
        }

        string categoria = clasificador.clasificar(v);
        cout << categoria << endl;
        return 0;
    }
    else if(op == "cuerpo") {
        AnalizadorCuerpo ac;
        MedidasCuerpo m = ac.analizar(imagen);
        ac.marcar(m, "static/cuerpo_marcado.bmp");

        // Salida JSON para que Flask la parsee
        cout << "{";
        cout << "\"tipo\":\"" << m.tipo << "\",";
        cout << "\"hombros\":" << m.ancho_hombros << ",";
        cout << "\"cintura\":"  << m.ancho_cintura  << ",";
        cout << "\"cadera\":"   << m.ancho_cadera   << ",";
        cout << "\"ratio_ch\":"  << m.ratio_ch  << ",";
        cout << "\"ratio_cah\":" << m.ratio_cah;
        cout << "}" << endl;
        return 0;
    }

    else if(op == "colorimetria") {
        AnalizadorColorimetria ac;
        ResultadoColorimetria r = ac.analizar(imagen);
        vector<string> paleta = ac.paletaRecomendada(r.estacion);
        cout << "{";
        cout << "\"estacion\":\"" << r.estacion << "\",";
        cout << "\"temperatura\":\"" << r.temperatura << "\",";
        cout << "\"valor\":\"" << r.valor << "\",";
        cout << "\"h\":" << r.tono_piel_h << ",";
        cout << "\"s\":" << r.tono_piel_s << ",";
        cout << "\"v\":" << r.tono_piel_v << ",";
        cout << "\"pixeles_piel\":" << r.pixeles_piel << ",";
        cout << "\"piel_r\":" << r.piel_r << ",";
        cout << "\"piel_g\":" << r.piel_g << ",";
        cout << "\"piel_b\":" << r.piel_b << ",";
        cout << "\"paleta\":[";
        for (int i = 0; i < (int)paleta.size(); i++) {
            cout << "\"" << paleta[i] << "\"";
            if (i < (int)paleta.size()-1) cout << ",";
        }
        cout << "]}";
        cout << endl;
        return 0;
    }

    else if(op == "detectar_prendas") {
        ClasificadorPrendas clasificador;
        if(!clasificador.cargarPesos("pesos.txt")) {
            cout << "{\"error\":\"sin modelo\"}" << endl;
            return 1;
        }
        vector<float> minV, maxV;
        ifstream fRangos("rangos.txt");
        if(fRangos.is_open()) {
            int n; fRangos >> n;
            minV.resize(n); maxV.resize(n);
            for(float& v : minV) fRangos >> v;
            for(float& v : maxV) fRangos >> v;
        }
        DetectorPrendas dp;
        dp.minVal = minV;
        dp.maxVal = maxV;
        vector<PrendaDetectada> prendas = dp.detectar(imagen, clasificador, "static");
        cout << "{\"prendas\":[";
        for (int i = 0; i < (int)prendas.size(); i++) {
            const auto& p = prendas[i];
            cout << "{"
                 << "\"categoria\":\"" << p.categoria    << "\","
                 << "\"score\":"       << p.score         << ","
                 << "\"ruta\":\""      << p.ruta_imagen   << "?t=" << i << "\""
                 << "}";
            if (i < (int)prendas.size()-1) cout << ",";
        }
        cout << "]}" << endl;
        return 0;
    }

    return 0;
}