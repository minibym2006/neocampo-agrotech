"""
modules/vision_system.py
Sistema de visão computacional para detecção de pragas no NeoCampo.

Suporta três modos, tentados nesta ordem:
  1. YOLO real       — ultralytics YOLOv8 com modelo customizado ou pré-treinado
  2. OpenCV clássico — análise de cor/textura HSV sem rede neural (leve, offline)
  3. Mock realista   — simulação com variação temporal e estatísticas por cultura

Variáveis de ambiente (opcionais):
  VISION_MODO         : "yolo" | "opencv" | "mock" | "auto"  (padrão: auto)
  VISION_MODELO_PATH  : caminho para weights YOLO (.pt)  ex: models/pragas_yolo.pt
  VISION_CONFIANCA_MIN: limiar mínimo de confiança para aceitar detecção (padrão: 0.45)
  VISION_CLASSES      : classes de pragas separadas por vírgula (padrão: lista interna)
  VISION_DISPOSITIVO  : "cpu" | "cuda" | "mps"  (padrão: auto-detecta)
  VISION_SALVAR_RESULT: "1" salva imagem anotada em outputs/visao/  (padrão: 0)
"""

import os
import logging
import random
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Configurações ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODO             = os.getenv("VISION_MODO", "auto").lower()
MODELO_PATH      = os.getenv("VISION_MODELO_PATH", "models/pragas_yolo.pt")
CONFIANCA_MIN    = float(os.getenv("VISION_CONFIANCA_MIN", "0.45"))
DISPOSITIVO      = os.getenv("VISION_DISPOSITIVO", "auto").lower()
SALVAR_RESULTADO = os.getenv("VISION_SALVAR_RESULT", "0") == "1"
OUTPUT_DIR       = Path("outputs/visao")

_CLASSES_PADRAO = [
    "lagarta-do-cartucho",
    "percevejo-marrom",
    "mosca-branca",
    "pulgao",
    "trips",
    "acaro-rajado",
    "broca-da-cana",
    "cigarrinha",
]

CLASSES_PRAGAS: list[str] = (
    os.getenv("VISION_CLASSES", "").split(",")
    if os.getenv("VISION_CLASSES")
    else _CLASSES_PADRAO
)

# Cache do modelo YOLO em memória
_modelo_yolo_cache = None


# ── Utilitários internos ─────────────────────────────────────────────────────

def _resolver_dispositivo() -> str:
    """Detecta automaticamente CPU, CUDA ou Apple MPS."""
    if DISPOSITIVO != "auto":
        return DISPOSITIVO
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _montar_payload(
    praga_detectada: bool,
    confianca: float,
    classe: Optional[str],
    bbox: Optional[list],
    n_deteccoes: int,
    modo_usado: str,
    imagem_anotada: Optional[np.ndarray] = None,
    tempo_ms: float = 0.0,
    alertas: Optional[list] = None,
) -> dict:
    """Normaliza o retorno para um formato único independente do modo."""
    return {
        # Resultado principal
        "praga_detectada":  praga_detectada,
        "confianca":        round(float(confianca), 4),
        "classe_principal": classe,
        "n_deteccoes":      n_deteccoes,
        "bounding_box":     bbox,                  # [x1, y1, x2, y2] ou None

        # Recomendação automática
        "severidade":       _classificar_severidade(confianca, n_deteccoes) if praga_detectada else "nenhuma",
        "acao_recomendada": _recomendar_acao(confianca, n_deteccoes, classe) if praga_detectada else "Monitoramento rotineiro",

        # Metadados
        "modo":             modo_usado,
        "tempo_inferencia_ms": round(tempo_ms, 1),
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "imagem_anotada":   imagem_anotada,        # np.ndarray ou None
        "alertas":          alertas or [],
    }


def _classificar_severidade(confianca: float, n: int) -> str:
    if confianca >= 0.80 or n >= 3:
        return "alta"
    if confianca >= 0.60 or n >= 2:
        return "media"
    return "baixa"


def _recomendar_acao(confianca: float, n: int, classe: Optional[str]) -> str:
    sev = _classificar_severidade(confianca, n)
    acoes = {
        "alta":  f"🔴 Aplicação imediata de defensivo específico para {classe or 'praga identificada'}. Isolar talhão.",
        "media": f"🟠 Monitorar por 48h e preparar defensivo para {classe or 'praga identificada'}.",
        "baixa": "🟡 Aumentar frequência de inspeção para confirmar infestação.",
    }
    return acoes.get(sev, "Consultar agrônomo.")


def _salvar_imagem_anotada(imagem: np.ndarray, prefixo: str = "resultado") -> Optional[Path]:
    """Salva imagem anotada em disco se VISION_SALVAR_RESULT=1."""
    if not SALVAR_RESULTADO or imagem is None:
        return None
    try:
        import cv2
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho = OUTPUT_DIR / f"{prefixo}_{ts}.jpg"
        cv2.imwrite(str(caminho), imagem)
        logger.info("Visão: imagem anotada salva em '%s'", caminho)
        return caminho
    except Exception as e:
        logger.warning("Visão: falha ao salvar imagem — %s", e)
        return None


# ── Modo 1: YOLO real ────────────────────────────────────────────────────────

def _carregar_modelo_yolo():
    """Carrega (e faz cache) do modelo YOLOv8."""
    global _modelo_yolo_cache
    if _modelo_yolo_cache is not None:
        return _modelo_yolo_cache

    from ultralytics import YOLO  # pip install ultralytics

    modelo_path = Path(MODELO_PATH)
    if not modelo_path.exists():
        # Fallback: usa yolov8n pré-treinado no COCO como demonstração
        logger.warning(
            "Visão: modelo '%s' não encontrado. "
            "Baixando yolov8n (COCO) como demonstração — "
            "substitua por um modelo treinado em pragas agrícolas.",
            MODELO_PATH,
        )
        modelo_path = "yolov8n.pt"

    device = _resolver_dispositivo()
    modelo = YOLO(str(modelo_path))
    modelo.to(device)
    _modelo_yolo_cache = modelo
    logger.info("YOLO carregado — dispositivo: %s", device)
    return modelo


def _detectar_yolo(imagem: Optional[np.ndarray]) -> Optional[dict]:
    """
    Executa inferência YOLOv8 e retorna payload normalizado.
    Retorna None se ultralytics não estiver instalado ou ocorrer erro.
    """
    try:
        import time
        import cv2

        modelo = _carregar_modelo_yolo()

        # Se não veio imagem, cria frame sintético para demonstração
        if imagem is None:
            imagem = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        resultados = modelo(imagem, conf=CONFIANCA_MIN, verbose=False)
        tempo_ms = (time.perf_counter() - t0) * 1000

        deteccoes = resultados[0].boxes
        n = len(deteccoes) if deteccoes is not None else 0

        if n == 0:
            img_anotada = resultados[0].plot()
            _salvar_imagem_anotada(img_anotada, "limpo")
            return _montar_payload(False, 0.0, None, None, 0, "yolo",
                                   img_anotada, tempo_ms)

        # Maior confiança entre as detecções
        confis = deteccoes.conf.cpu().numpy()
        idx_max = int(confis.argmax())
        conf_max = float(confis[idx_max])

        # Nome da classe
        cls_id = int(deteccoes.cls.cpu().numpy()[idx_max])
        nomes = modelo.names
        classe = CLASSES_PRAGAS[cls_id] if cls_id < len(CLASSES_PRAGAS) else nomes.get(cls_id, f"classe_{cls_id}")

        # Bounding box [x1, y1, x2, y2]
        bbox_raw = deteccoes.xyxy.cpu().numpy()[idx_max]
        bbox = [round(float(v), 1) for v in bbox_raw]

        img_anotada = resultados[0].plot()
        _salvar_imagem_anotada(img_anotada, "praga")

        return _montar_payload(True, conf_max, classe, bbox, n, "yolo",
                               img_anotada, tempo_ms)

    except ImportError:
        logger.debug("ultralytics não instalado — modo YOLO indisponível.")
        return None
    except Exception as e:
        logger.warning("YOLO: erro na inferência — %s", e)
        return None


# ── Modo 2: OpenCV clássico ──────────────────────────────────────────────────

def _detectar_opencv(imagem: Optional[np.ndarray]) -> Optional[dict]:
    """
    Análise de pragas por cor e textura HSV com OpenCV.
    Detecta regiões amareladas/acastanhadas associadas a infestação.
    Leve, funciona sem GPU e sem modelo treinado.
    """
    try:
        import cv2
        import time

        if imagem is None:
            # Frame sintético com mancha simulada
            imagem = np.full((480, 640, 3), [60, 120, 60], dtype=np.uint8)
            # Insere mancha amarelada (simula praga) com 30% de chance
            if random.random() < 0.30:
                cx, cy = random.randint(100, 540), random.randint(80, 400)
                cv2.ellipse(imagem, (cx, cy), (random.randint(20,60), random.randint(15,40)),
                            0, 0, 360, (30, 180, 200), -1)

        t0 = time.perf_counter()
        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

        # Faixas HSV que indicam doença/praga em folhas
        mascaras = []
        # Amarelamento (clorose, pulgões)
        mascaras.append(cv2.inRange(hsv, np.array([20, 80, 80]),  np.array([35, 255, 255])))
        # Marrom/ferrugem (fungos, ácaros)
        mascaras.append(cv2.inRange(hsv, np.array([5,  60, 40]),  np.array([20, 200, 180])))
        # Branco/prateado (mosca-branca, trips)
        mascaras.append(cv2.inRange(hsv, np.array([0,  0,  200]), np.array([180, 30, 255])))

        mascara_total = mascaras[0]
        for m in mascaras[1:]:
            mascara_total = cv2.bitwise_or(mascara_total, m)

        # Morfologia para remover ruído
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mascara_total = cv2.morphologyEx(mascara_total, cv2.MORPH_OPEN, kernel)

        # Contornos suspeitos
        contornos, _ = cv2.findContours(mascara_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_sig = [c for c in contornos if cv2.contourArea(c) > 300]

        area_total_img = imagem.shape[0] * imagem.shape[1]
        area_afetada   = sum(cv2.contourArea(c) for c in contornos_sig)
        pct_afetada    = area_afetada / area_total_img

        tempo_ms = (time.perf_counter() - t0) * 1000

        # Limiar: >1% da área afetada indica possível praga
        if pct_afetada < 0.01 or len(contornos_sig) == 0:
            return _montar_payload(False, 0.0, None, None, 0, "opencv", None, tempo_ms)

        # Confiança estimada pela proporção de área afetada
        confianca = float(np.clip(pct_afetada * 8, CONFIANCA_MIN, 0.95))
        n = len(contornos_sig)

        # Bounding box do maior contorno
        x, y, w, h = cv2.boundingRect(max(contornos_sig, key=cv2.contourArea))
        bbox = [x, y, x + w, y + h]

        # Classifica pela cor dominante
        mascara_contorno = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mascara_contorno, contornos_sig, -1, 255, -1)
        h_vals = hsv[:, :, 0][mascara_contorno > 0]
        h_medio = float(np.median(h_vals)) if len(h_vals) else 0
        if 20 <= h_medio <= 35:
            classe = "pulgao"
        elif 5 <= h_medio < 20:
            classe = "acaro-rajado"
        else:
            classe = "mosca-branca"

        # Anota imagem
        img_anotada = imagem.copy()
        cv2.drawContours(img_anotada, contornos_sig, -1, (0, 0, 255), 2)
        cv2.rectangle(img_anotada, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(img_anotada, f"{classe} {confianca:.0%}",
                    (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        _salvar_imagem_anotada(img_anotada, "opencv_praga")

        return _montar_payload(True, confianca, classe, bbox, n, "opencv",
                               img_anotada, tempo_ms,
                               alertas=[f"Área foliar afetada: {pct_afetada:.1%}"])

    except ImportError:
        logger.debug("opencv-python não instalado — modo OpenCV indisponível.")
        return None
    except Exception as e:
        logger.warning("OpenCV: erro na análise — %s", e)
        return None


# ── Modo 3: Mock realista ─────────────────────────────────────────────────────

def _detectar_mock() -> dict:
    """
    Simulação com comportamento realista:
      - 25% de chance base de detectar praga
      - Confiança com distribuição beta (evita sempre 0.94)
      - Classe sorteada da lista de pragas reais
      - Bounding box aleatório plausível
      - Tempo de inferência simulado
    """
    import time

    t0 = time.perf_counter()
    time.sleep(random.uniform(0.05, 0.15))  # simula latência de rede/GPU
    tempo_ms = (time.perf_counter() - t0) * 1000

    praga = random.random() < 0.25

    if not praga:
        return _montar_payload(False, 0.0, None, None, 0, "mock (simulação)", None, tempo_ms)

    # Confiança com distribuição beta — mais realista que valor fixo
    confianca = float(np.clip(np.random.beta(a=8, b=2), CONFIANCA_MIN, 0.98))
    n         = random.choices([1, 2, 3, 4], weights=[50, 30, 15, 5])[0]
    classe    = random.choice(CLASSES_PRAGAS)

    # Bounding box plausível numa imagem 640x480
    x1 = random.randint(50, 400)
    y1 = random.randint(50, 300)
    x2 = x1 + random.randint(60, 200)
    y2 = y1 + random.randint(40, 150)
    bbox = [x1, y1, min(x2, 639), min(y2, 479)]

    alertas = []
    if n >= 3:
        alertas.append(f"🔴 {n} focos detectados — infestação disseminada.")

    return _montar_payload(True, confianca, classe, bbox, n,
                           "mock (simulação)", None, tempo_ms, alertas)


# ── Ponto de entrada público ──────────────────────────────────────────────────

def detectar_pragas_mock(
    imagem: Optional[Union[np.ndarray, str, Path]] = None,
) -> dict:
    """
    Detecta pragas em uma imagem de lavoura.

    Parâmetros
    ----------
    imagem : np.ndarray (BGR), caminho str/Path para arquivo, ou None.
             Se None, usa câmera sintética (mock) ou frame interno (YOLO/OpenCV).

    Retorna dict com:
        praga_detectada, confianca, classe_principal, n_deteccoes,
        bounding_box, severidade, acao_recomendada,
        modo, tempo_inferencia_ms, timestamp, imagem_anotada, alertas
    """
    # Carrega imagem se veio como caminho
    img_array: Optional[np.ndarray] = None
    if isinstance(imagem, (str, Path)):
        try:
            import cv2
            img_array = cv2.imread(str(imagem))
            if img_array is None:
                raise ValueError(f"Não foi possível abrir '{imagem}'.")
            logger.debug("Visão: imagem carregada de '%s' (%s)", imagem, img_array.shape)
        except ImportError:
            logger.warning("opencv-python não disponível para carregar imagem de disco.")
        except Exception as e:
            logger.warning("Visão: erro ao carregar imagem — %s", e)
    elif isinstance(imagem, np.ndarray):
        img_array = imagem

    # Tenta modos na ordem configurada
    resultado = None

    if MODO in ("yolo", "auto"):
        resultado = _detectar_yolo(img_array)

    if resultado is None and MODO in ("opencv", "auto"):
        resultado = _detectar_opencv(img_array)

    if resultado is None:
        if MODO not in ("yolo", "opencv"):
            logger.debug("Visão: usando mock.")
        else:
            logger.warning("Visão: modo '%s' falhou — usando mock.", MODO)
        resultado = _detectar_mock()

    logger.info(
        "Visão [%s]: praga=%s conf=%.2f classe=%s t=%.0fms",
        resultado["modo"],
        resultado["praga_detectada"],
        resultado["confianca"],
        resultado["classe_principal"] or "—",
        resultado["tempo_inferencia_ms"],
    )
    return resultado


def status_visao() -> dict:
    """
    Informa quais backends estão disponíveis no ambiente atual.
    Útil para exibir no dashboard de status do sistema.
    """
    yolo_ok, opencv_ok = False, False

    try:
        import ultralytics  # noqa: F401
        yolo_ok = True
    except ImportError:
        pass

    try:
        import cv2  # noqa: F401
        opencv_ok = True
    except ImportError:
        pass

    dispositivo = _resolver_dispositivo() if yolo_ok else "—"

    return {
        "yolo_disponivel":   yolo_ok,
        "opencv_disponivel": opencv_ok,
        "dispositivo":       dispositivo,
        "modelo_path":       MODELO_PATH,
        "classes":           CLASSES_PRAGAS,
        "modo_ativo":        MODO,
        "confianca_min":     CONFIANCA_MIN,
    }