"""
modules/iot_service.py
Leitura de sensores IoT para o NeoCampo.

Suporta três modos de coleta, tentados nesta ordem:
  1. MQTT  — broker local ou em nuvem (ex: HiveMQ, Mosquitto)
  2. HTTP  — endpoint REST de um gateway IoT (ex: ESP32 com API própria)
  3. Simulação — dados realistas com deriva temporal (fallback sempre disponível)

Variáveis de ambiente (todas opcionais — sem elas, usa simulação):
  IOT_MODO            : "mqtt" | "http" | "simulacao"  (padrão: auto-detecta)
  IOT_MQTT_BROKER     : ex "broker.hivemq.com"
  IOT_MQTT_PORT       : padrão 1883
  IOT_MQTT_TOPIC      : ex "neocampo/sensores/loteA"
  IOT_MQTT_USUARIO    : usuário MQTT (opcional)
  IOT_MQTT_SENHA      : senha MQTT (opcional)
  IOT_HTTP_URL        : ex "http://192.168.1.50/sensores"
  IOT_HTTP_TOKEN      : Bearer token para autenticação HTTP (opcional)
  IOT_HTTP_TIMEOUT    : segundos de timeout HTTP (padrão 5)
  IOT_LOTE            : identificador do lote (padrão "A")
"""

import os
import json
import random
import logging
import math
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configurações via ambiente ───────────────────────────────────────────────
MODO            = os.getenv("IOT_MODO", "auto").lower()
MQTT_BROKER     = os.getenv("IOT_MQTT_BROKER", "")
MQTT_PORT       = int(os.getenv("IOT_MQTT_PORT", "1883"))
MQTT_TOPIC      = os.getenv("IOT_MQTT_TOPIC", "neocampo/sensores")
MQTT_USUARIO    = os.getenv("IOT_MQTT_USUARIO", "")
MQTT_SENHA      = os.getenv("IOT_MQTT_SENHA", "")
HTTP_URL        = os.getenv("IOT_HTTP_URL", "")
HTTP_TOKEN      = os.getenv("IOT_HTTP_TOKEN", "")
HTTP_TIMEOUT    = int(os.getenv("IOT_HTTP_TIMEOUT", "5"))
LOTE            = os.getenv("IOT_LOTE", "A")

# Limites agronômicos de referência
LIMITES = {
    "umidade":     {"min": 0.0,  "max": 100.0, "ideal_min": 50.0, "ideal_max": 75.0},
    "ph":          {"min": 4.0,  "max": 9.0,   "ideal_min": 5.5,  "ideal_max": 7.0},
    "temperatura": {"min": 0.0,  "max": 50.0,  "ideal_min": 18.0, "ideal_max": 32.0},
    "luminosidade":{"min": 0.0,  "max": 12000, "ideal_min": 4000, "ideal_max": 9000},
    "condutividade":{"min": 0.0, "max": 3.0,   "ideal_min": 0.5,  "ideal_max": 2.0},
}

# Estado interno da simulação (deriva entre chamadas)
_estado_sim: dict = {}


# ── Utilitários ──────────────────────────────────────────────────────────────

def _clamp(valor: float, minv: float, maxv: float) -> float:
    return max(minv, min(maxv, valor))


def _classificar(chave: str, valor: float) -> str:
    """Retorna 'OK', 'BAIXO' ou 'ALTO' baseado nos limites ideais."""
    lim = LIMITES.get(chave, {})
    if not lim:
        return "OK"
    if valor < lim["ideal_min"]:
        return "BAIXO"
    if valor > lim["ideal_max"]:
        return "ALTO"
    return "OK"


def _status_bomba(umidade: float) -> str:
    """Lógica de acionamento da bomba com histerese (evita liga/desliga rápido)."""
    if umidade < 45.0:
        return "LIGADA"
    if umidade > 65.0:
        return "DESLIGADA"
    # Zona de histerese: mantém estado anterior se existir
    return _estado_sim.get("bomba_anterior", "DESLIGADA")


def _montar_payload(raw: dict) -> dict:
    """
    Normaliza qualquer dict de sensor para o formato padrão do NeoCampo,
    calcula classificações e injeta metadados.
    """
    umidade      = _clamp(float(raw.get("umidade",      raw.get("humidity",     55.0))), 0, 100)
    ph           = _clamp(float(raw.get("ph",           raw.get("pH",           6.5))),  4, 9)
    temperatura  = _clamp(float(raw.get("temperatura",  raw.get("temperature",  25.0))), 0, 50)
    luminosidade = _clamp(float(raw.get("luminosidade", raw.get("light",        6000))), 0, 12000)
    condutiv     = _clamp(float(raw.get("condutividade",raw.get("ec",           1.2))),  0, 3)

    bomba = _status_bomba(umidade)
    _estado_sim["bomba_anterior"] = bomba

    return {
        # Leituras brutas
        "umidade":            round(umidade, 2),
        "ph":                 round(ph, 2),
        "temperatura":        round(temperatura, 2),
        "luminosidade":       round(luminosidade, 1),
        "condutividade":      round(condutiv, 3),
        # Controle
        "bomba":              bomba,
        # Classificações agronômicas
        "status_umidade":     _classificar("umidade", umidade),
        "status_ph":          _classificar("ph", ph),
        "status_temperatura": _classificar("temperatura", temperatura),
        # Metadados
        "lote":               LOTE,
        "timestamp":          datetime.now().isoformat(timespec="seconds"),
        "fonte":              raw.get("_fonte", "desconhecida"),
        "alertas":            _gerar_alertas(umidade, ph, temperatura),
    }


def _gerar_alertas(umidade: float, ph: float, temperatura: float) -> list[str]:
    """Retorna lista de strings de alerta para condições fora do ideal."""
    alertas = []
    if umidade < LIMITES["umidade"]["ideal_min"]:
        alertas.append(f"⚠️ Umidade baixa ({umidade}%) — irrigação recomendada")
    if umidade > LIMITES["umidade"]["ideal_max"]:
        alertas.append(f"⚠️ Umidade alta ({umidade}%) — risco de encharcamento")
    if ph < LIMITES["ph"]["ideal_min"]:
        alertas.append(f"⚠️ pH ácido ({ph}) — considere calagem")
    if ph > LIMITES["ph"]["ideal_max"]:
        alertas.append(f"⚠️ pH alcalino ({ph}) — verificar adubação")
    if temperatura > LIMITES["temperatura"]["ideal_max"]:
        alertas.append(f"🌡️ Temperatura elevada ({temperatura}°C)")
    return alertas


# ── Modo MQTT ────────────────────────────────────────────────────────────────

def _ler_mqtt() -> Optional[dict]:
    """
    Lê uma mensagem do broker MQTT com timeout de 6 segundos.
    Retorna o payload como dict ou None em caso de falha.
    """
    try:
        import paho.mqtt.client as mqtt  # pip install paho-mqtt
    except ImportError:
        logger.debug("paho-mqtt não instalado — modo MQTT indisponível.")
        return None

    resultado: dict = {}

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(MQTT_TOPIC)
        else:
            logger.warning("MQTT: falha na conexão (rc=%s)", rc)

    def on_message(client, userdata, msg):
        try:
            resultado.update(json.loads(msg.payload.decode()))
            resultado["_fonte"] = "mqtt"
        except json.JSONDecodeError:
            logger.error("MQTT: payload não é JSON válido.")
        finally:
            client.disconnect()

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    if MQTT_USUARIO:
        client.username_pw_set(MQTT_USUARIO, MQTT_SENHA)

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=10)
        client.loop_start()
        import time
        time.sleep(6)
        client.loop_stop()
    except Exception as e:
        logger.warning("MQTT: erro de conexão com '%s': %s", MQTT_BROKER, e)
        return None

    return resultado if resultado else None


# ── Modo HTTP ────────────────────────────────────────────────────────────────

def _ler_http() -> Optional[dict]:
    """
    Faz GET no gateway IoT HTTP e retorna o JSON como dict.
    Retorna None em caso de timeout, erro HTTP ou JSON inválido.
    """
    try:
        import requests  # pip install requests
    except ImportError:
        logger.debug("requests não instalado — modo HTTP indisponível.")
        return None

    headers = {"Accept": "application/json"}
    if HTTP_TOKEN:
        headers["Authorization"] = f"Bearer {HTTP_TOKEN}"

    try:
        resp = requests.get(HTTP_URL, headers=headers, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        data["_fonte"] = "http"
        return data

    except requests.exceptions.Timeout:
        logger.warning("HTTP IoT: timeout após %ss em '%s'", HTTP_TIMEOUT, HTTP_URL)
    except requests.exceptions.HTTPError as e:
        logger.warning("HTTP IoT: erro %s em '%s'", e.response.status_code, HTTP_URL)
    except requests.exceptions.ConnectionError:
        logger.warning("HTTP IoT: não foi possível conectar a '%s'", HTTP_URL)
    except (ValueError, KeyError) as e:
        logger.warning("HTTP IoT: resposta inválida — %s", e)

    return None


# ── Simulação realista ───────────────────────────────────────────────────────

def _ler_simulacao() -> dict:
    """
    Gera leituras simuladas com:
      - Deriva suave entre chamadas (random walk)
      - Variação circadiana da temperatura e luminosidade
      - Correlação realista: pH influenciado pela umidade
    """
    hora = datetime.now().hour + datetime.now().minute / 60.0

    # Inicializa estado na primeira chamada
    if not _estado_sim.get("inicializado"):
        _estado_sim.update({
            "umidade":     random.uniform(50, 70),
            "ph":          random.uniform(5.8, 6.8),
            "condutiv":    random.uniform(0.8, 1.6),
            "inicializado": True,
        })

    # Deriva suave (random walk limitado)
    def deriva(chave: str, amplitude: float, minv: float, maxv: float) -> float:
        novo = _estado_sim[chave] + random.gauss(0, amplitude)
        novo = _clamp(novo, minv, maxv)
        _estado_sim[chave] = novo
        return novo

    umidade    = deriva("umidade",  1.5, 20.0, 95.0)
    ph         = deriva("ph",       0.05, 4.5, 8.5)
    condutiv   = deriva("condutiv", 0.03, 0.2, 2.8)

    # Variação circadiana: temperatura e luminosidade seguem o ciclo do dia
    temp_base  = 22 + 10 * math.sin(math.pi * (hora - 6) / 12)
    temp       = _clamp(temp_base + random.gauss(0, 0.8), 5.0, 45.0)
    lum_base   = max(0, 8000 * math.sin(math.pi * (hora - 6) / 12))
    lum        = _clamp(lum_base + random.gauss(0, 200), 0, 12000)

    return {
        "umidade":      umidade,
        "ph":           ph,
        "temperatura":  temp,
        "luminosidade": lum,
        "condutividade": condutiv,
        "_fonte":       "simulacao",
    }


# ── Ponto de entrada público ─────────────────────────────────────────────────

def ler_sensores_iot() -> dict:
    """
    Lê sensores IoT e retorna payload normalizado.

    Tenta os modos na ordem configurada:
      "mqtt"      → só MQTT
      "http"      → só HTTP
      "simulacao" → só simulação
      "auto"      → MQTT → HTTP → simulação (fallback automático)

    O dict retornado sempre contém:
        umidade, ph, temperatura, luminosidade, condutividade,
        bomba, status_umidade, status_ph, status_temperatura,
        lote, timestamp, fonte, alertas
    """
    raw: Optional[dict] = None

    if MODO == "mqtt" or (MODO == "auto" and MQTT_BROKER):
        logger.debug("IoT: tentando MQTT (%s:%s)...", MQTT_BROKER, MQTT_PORT)
        raw = _ler_mqtt()
        if raw:
            logger.info("IoT: dados recebidos via MQTT")

    if raw is None and (MODO == "http" or (MODO == "auto" and HTTP_URL)):
        logger.debug("IoT: tentando HTTP (%s)...", HTTP_URL)
        raw = _ler_http()
        if raw:
            logger.info("IoT: dados recebidos via HTTP")

    if raw is None:
        if MODO not in ("mqtt", "http"):
            logger.debug("IoT: usando simulação.")
        else:
            logger.warning("IoT: modo '%s' falhou — usando simulação como fallback.", MODO)
        raw = _ler_simulacao()

    return _montar_payload(raw)


def status_conexao() -> dict:
    """
    Retorna informações sobre o modo de coleta configurado.
    Útil para exibir no dashboard de status do sistema.
    """
    mqtt_ok = bool(MQTT_BROKER)
    http_ok = bool(HTTP_URL)

    if MODO == "mqtt":
        modo_ativo = "MQTT" if mqtt_ok else "Simulação (MQTT não configurado)"
    elif MODO == "http":
        modo_ativo = "HTTP" if http_ok else "Simulação (HTTP não configurado)"
    elif MODO == "auto":
        if mqtt_ok:
            modo_ativo = f"Auto → MQTT ({MQTT_BROKER})"
        elif http_ok:
            modo_ativo = f"Auto → HTTP ({HTTP_URL})"
        else:
            modo_ativo = "Auto → Simulação (nenhum endpoint configurado)"
    else:
        modo_ativo = "Simulação"

    return {
        "modo":         modo_ativo,
        "lote":         LOTE,
        "mqtt_broker":  MQTT_BROKER or "—",
        "http_url":     HTTP_URL or "—",
        "simulacao":    not (mqtt_ok or http_ok),
    }