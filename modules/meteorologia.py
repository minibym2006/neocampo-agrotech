"""
modules/meteorologia.py
Planejamento de plantio e coleta de dados climáticos para o NeoCampo.

Clima — tenta as fontes nesta ordem:
  1. Open-Meteo  — API gratuita, sem chave, geocoding incluído
  2. OpenWeather — API gratuita com chave (OPENWEATHER_API_KEY)
  3. Mock        — retorno estático para desenvolvimento offline

Variáveis de ambiente (opcionais):
  OPENWEATHER_API_KEY  : chave da API OpenWeatherMap
  CLIMA_CIDADE_PADRAO  : cidade padrão (padrão: "Santa Cruz do Rio Pardo")
  CLIMA_TIMEOUT        : timeout HTTP em segundos (padrão: 6)
"""

import os
import math
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configurações ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENWEATHER_KEY  = os.getenv("OPENWEATHER_API_KEY", "")
CIDADE_PADRAO    = os.getenv("CLIMA_CIDADE_PADRAO", "Santa Cruz da Conceição")
TIMEOUT          = int(os.getenv("CLIMA_TIMEOUT", "6"))

# Base completa de municípios de SP com coordenadas
from modules.municipios_sp import (
    buscar_municipio,
    listar_municipios,
    buscar_municipios_por_prefixo,
    REGIOES_SP,
    MUNICIPIOS_SP,
)

# ── Tabelas agronômicas ──────────────────────────────────────────────────────

# kg de insumo por m² · por tipo de insumo
INSUMOS_POR_CULTURA: dict[str, dict] = {
    "Soja": {
        "semente_kg_m2":    0.0060,   # 60 kg/ha
        "fertilizante_kg_m2": 0.0180, # 180 kg/ha NPK
        "defensivo_l_m2":   0.0008,
    },
    "Milho": {
        "semente_kg_m2":    0.0250,   # 250 kg/ha (densidade de plantio)
        "fertilizante_kg_m2": 0.0400,
        "defensivo_l_m2":   0.0012,
    },
    "Cana-de-açúcar": {
        "semente_kg_m2":    0.0000,   # muda vegetativa (não aplica kg)
        "fertilizante_kg_m2": 0.0600,
        "defensivo_l_m2":   0.0015,
    },
    "Café": {
        "semente_kg_m2":    0.0000,
        "fertilizante_kg_m2": 0.0350,
        "defensivo_l_m2":   0.0010,
    },
    "Trigo": {
        "semente_kg_m2":    0.0180,
        "fertilizante_kg_m2": 0.0250,
        "defensivo_l_m2":   0.0007,
    },
    "Algodão": {
        "semente_kg_m2":    0.0040,
        "fertilizante_kg_m2": 0.0300,
        "defensivo_l_m2":   0.0020,
    },
}

# Ciclo vegetativo estimado em dias
CICLO_DIAS: dict[str, int] = {
    "Soja":           120,
    "Milho":          130,
    "Cana-de-açúcar": 360,
    "Café":           270,
    "Trigo":          110,
    "Algodão":        180,
}

# Produtividade média em kg/m²
PRODUTIVIDADE_KG_M2: dict[str, float] = {
    "Soja":           0.30,   # ~3 t/ha
    "Milho":          0.85,   # ~8,5 t/ha
    "Cana-de-açúcar": 8.00,   # ~80 t/ha
    "Café":           0.25,
    "Trigo":          0.35,
    "Algodão":        0.40,
}


# ── Cálculo de plantio ───────────────────────────────────────────────────────

def calcular_plantio(
    largura: float,
    comprimento: float,
    cultura: str,
    preco_venda_kg: float = 0.0,
) -> dict:
    """
    Calcula todos os parâmetros de planejamento para um talhão.

    Parâmetros
    ----------
    largura, comprimento : dimensões em metros
    cultura              : deve estar em INSUMOS_POR_CULTURA
    preco_venda_kg       : preço de mercado (R$/kg) para estimativa de receita

    Retorna dict com:
        area_total, area_hectares, cultura,
        insumo_necessario (total kg), detalhes_insumos,
        producao_estimada_kg, receita_estimada_brl,
        ciclo_dias, data_colheita_estimada,
        alertas
    """
    # Validação
    if largura <= 0 or comprimento <= 0:
        raise ValueError(
            f"Largura e comprimento devem ser positivos "
            f"(recebido: {largura} x {comprimento})."
        )
    cultura_norm = _normalizar_cultura(cultura)
    if cultura_norm not in INSUMOS_POR_CULTURA:
        culturas_disp = ", ".join(INSUMOS_POR_CULTURA.keys())
        raise ValueError(
            f"Cultura '{cultura}' não cadastrada. "
            f"Disponíveis: {culturas_disp}"
        )

    area_m2  = largura * comprimento
    area_ha  = area_m2 / 10_000
    tab      = INSUMOS_POR_CULTURA[cultura_norm]

    # Insumos detalhados
    semente      = round(area_m2 * tab["semente_kg_m2"], 2)
    fertilizante = round(area_m2 * tab["fertilizante_kg_m2"], 2)
    defensivo    = round(area_m2 * tab["defensivo_l_m2"], 3)
    total_insumo = round(semente + fertilizante, 2)

    # Produção e receita estimadas
    producao     = round(area_m2 * PRODUTIVIDADE_KG_M2.get(cultura_norm, 0), 2)
    receita      = round(producao * preco_venda_kg, 2) if preco_venda_kg > 0 else 0.0

    # Data de colheita estimada
    ciclo        = CICLO_DIAS.get(cultura_norm, 120)
    from datetime import timedelta
    data_colheita = (datetime.now() + timedelta(days=ciclo)).strftime("%d/%m/%Y")

    # Alertas automáticos
    alertas = []
    if area_ha > 50:
        alertas.append("⚠️ Área grande — considere dividir em talhões menores para manejo.")
    if area_m2 < 1000:
        alertas.append("ℹ️ Área pequena — verifique a viabilidade econômica do plantio.")

    return {
        "area_total":             round(area_m2, 2),
        "area_hectares":          round(area_ha, 4),
        "cultura":                cultura_norm,
        "insumo_necessario":      total_insumo,
        "detalhes_insumos": {
            "semente_kg":         semente,
            "fertilizante_kg":    fertilizante,
            "defensivo_litros":   defensivo,
        },
        "producao_estimada_kg":   producao,
        "receita_estimada_brl":   receita,
        "ciclo_dias":             ciclo,
        "data_colheita_estimada": data_colheita,
        "alertas":                alertas,
    }


def _normalizar_cultura(cultura: str) -> str:
    """Tolerante a variações de escrita (ex: 'cana' → 'Cana-de-açúcar')."""
    mapa = {
        "cana":           "Cana-de-açúcar",
        "cana de acucar": "Cana-de-açúcar",
        "cana-de-acucar": "Cana-de-açúcar",
        "cana de açúcar": "Cana-de-açúcar",
        "soja":           "Soja",
        "milho":          "Milho",
        "cafe":           "Café",
        "café":           "Café",
        "trigo":          "Trigo",
        "algodao":        "Algodão",
        "algodão":        "Algodão",
    }
    return mapa.get(cultura.strip().lower(), cultura.strip())


def culturas_disponiveis() -> list[str]:
    """Retorna a lista de culturas cadastradas no sistema."""
    return sorted(INSUMOS_POR_CULTURA.keys())


# ── Dados climáticos ─────────────────────────────────────────────────────────

def buscar_clima(cidade: str = "") -> dict:
    """
    Busca dados climáticos em tempo real para qualquer município de SP.

    Prioridade:
      1. Coordenadas locais do banco de 645 municípios (sem chamada de geocoding)
      2. Open-Meteo geocoding como fallback para cidades fora da lista
      3. OpenWeather (se OPENWEATHER_API_KEY configurada)
      4. Mock com variação circadiana

    Retorna dict com:
        temp, sensacao_termica, umidade, condicao, vento_kmh,
        pressao_hpa, indice_uv, cidade, regiao, latitude, longitude,
        fonte, timestamp
    """
    cidade = cidade.strip() or CIDADE_PADRAO

    # Tenta encontrar coordenadas na base local (mais rápido, sem geocoding)
    municipio = buscar_municipio(cidade)
    if municipio:
        resultado = _clima_por_coordenadas(
            municipio["lat"], municipio["lon"], municipio["nome"], municipio["regiao"]
        )
        if resultado:
            logger.info("Clima obtido via Open-Meteo (coords locais) para '%s'", cidade)
            return resultado

    # Fallback: geocoding via Open-Meteo (cidades não cadastradas)
    resultado = _clima_open_meteo(cidade)
    if resultado:
        logger.info("Clima obtido via Open-Meteo (geocoding) para '%s'", cidade)
        return resultado

    # Fallback: OpenWeather
    if OPENWEATHER_KEY:
        resultado = _clima_openweather(cidade)
        if resultado:
            logger.info("Clima obtido via OpenWeather para '%s'", cidade)
            return resultado

    logger.warning("APIs climáticas indisponíveis — usando mock para '%s'", cidade)
    return _clima_mock(cidade)


def _clima_por_coordenadas(lat: float, lon: float, nome: str, regiao: str = "") -> Optional[dict]:
    """
    Busca clima diretamente pelas coordenadas via Open-Meteo.
    Mais rápido que geocoding pois pula a primeira chamada de API.
    """
    try:
        import requests
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":  lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m,surface_pressure,uv_index"
                ),
                "wind_speed_unit": "kmh",
                "timezone": "America/Sao_Paulo",
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        c = resp.json().get("current", {})
        return {
            "temp":               round(c.get("temperature_2m", 0), 1),
            "sensacao_termica":   round(c.get("apparent_temperature", 0), 1),
            "umidade":            int(c.get("relative_humidity_2m", 0)),
            "condicao":           _wmo_para_texto(c.get("weather_code", 0)),
            "vento_kmh":          round(c.get("wind_speed_10m", 0), 1),
            "pressao_hpa":        round(c.get("surface_pressure", 0), 1),
            "indice_uv":          round(c.get("uv_index", 0), 1),
            "cidade":             nome,
            "regiao":             regiao,
            "latitude":           lat,
            "longitude":          lon,
            "fonte":              "Open-Meteo",
            "timestamp":          datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as e:
        logger.warning("Open-Meteo (coords): erro para '%s' — %s", nome, e)
        return None


def listar_cidades_favoritas() -> list[str]:
    """Retrocompatibilidade — retorna todos os 645 municípios de SP ordenados."""
    return listar_municipios()


def listar_cidades_por_regiao(regiao: str = "") -> list[str]:
    """Retorna municípios filtrados por mesorregião do IBGE."""
    return listar_municipios(regiao)


def autocomplete_cidade(prefixo: str, limite: int = 10) -> list[str]:
    """Retorna até `limite` municípios cujo nome começa com o prefixo digitado."""
    return buscar_municipios_por_prefixo(prefixo, limite)


def geocodificar_cidade(nome: str) -> Optional[dict]:
    """
    Valida e retorna coordenadas de uma cidade.
    Prioriza a base local; cai no geocoding da Open-Meteo se não encontrar.
    """
    # Tenta base local primeiro
    m = buscar_municipio(nome)
    if m:
        return {
            "nome":      m["nome"],
            "estado":    "SP",
            "pais":      "Brasil",
            "latitude":  m["lat"],
            "longitude": m["lon"],
            "regiao":    m["regiao"],
        }

    # Geocoding externo como fallback
    try:
        import requests
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": nome, "count": 1, "language": "pt", "format": "json"},
            timeout=TIMEOUT,
        )
        geo.raise_for_status()
        resultados = geo.json().get("results")
        if not resultados:
            return None
        r = resultados[0]
        return {
            "nome":      r.get("name", nome),
            "estado":    r.get("admin1", ""),
            "pais":      r.get("country", ""),
            "latitude":  r.get("latitude"),
            "longitude": r.get("longitude"),
            "regiao":    "",
        }
    except Exception as e:
        logger.warning("Geocoding: erro para '%s' — %s", nome, e)
        return None


def _clima_open_meteo(cidade: str) -> Optional[dict]:
    """
    Open-Meteo: geocoding + forecast em duas chamadas.
    Totalmente gratuito, sem chave de API.
    Docs: https://open-meteo.com/
    """
    try:
        import requests

        # 1. Geocoding
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": cidade, "count": 1, "language": "pt", "format": "json"},
            timeout=TIMEOUT,
        )
        geo.raise_for_status()
        geo_data = geo.json()
        resultados = geo_data.get("results")
        if not resultados:
            logger.warning("Open-Meteo: cidade '%s' não encontrada no geocoding.", cidade)
            return None

        lugar = resultados[0]
        lat, lon = lugar["latitude"], lugar["longitude"]
        nome_cidade = lugar.get("name", cidade)

        # 2. Clima atual
        clima = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":  lat,
                "longitude": lon,
                "current":   (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m,surface_pressure,uv_index"
                ),
                "wind_speed_unit": "kmh",
                "timezone": "America/Sao_Paulo",
            },
            timeout=TIMEOUT,
        )
        clima.raise_for_status()
        c = clima.json().get("current", {})

        return {
            "temp":               round(c.get("temperature_2m", 0), 1),
            "sensacao_termica":   round(c.get("apparent_temperature", 0), 1),
            "umidade":            int(c.get("relative_humidity_2m", 0)),
            "condicao":           _wmo_para_texto(c.get("weather_code", 0)),
            "vento_kmh":          round(c.get("wind_speed_10m", 0), 1),
            "pressao_hpa":        round(c.get("surface_pressure", 0), 1),
            "indice_uv":          round(c.get("uv_index", 0), 1),
            "cidade":             nome_cidade,
            "latitude":           lat,
            "longitude":          lon,
            "fonte":              "Open-Meteo",
            "timestamp":          datetime.now().isoformat(timespec="seconds"),
        }

    except Exception as e:
        logger.warning("Open-Meteo: erro — %s", e)
        return None


def _wmo_para_texto(code: int) -> str:
    """Converte código WMO (Open-Meteo) para descrição em português."""
    tabela = {
        0: "Céu limpo", 1: "Principalmente limpo", 2: "Parcialmente nublado",
        3: "Nublado", 45: "Névoa", 48: "Geada de névoa",
        51: "Garoa leve", 53: "Garoa moderada", 55: "Garoa intensa",
        61: "Chuva leve", 63: "Chuva moderada", 65: "Chuva forte",
        71: "Neve leve", 73: "Neve moderada", 75: "Neve intensa",
        80: "Chuvas esparsas", 81: "Chuvas moderadas", 82: "Chuvas fortes",
        95: "Trovoada", 96: "Trovoada com granizo", 99: "Trovoada intensa",
    }
    return tabela.get(code, f"Código {code}")


def _clima_openweather(cidade: str) -> Optional[dict]:
    """
    OpenWeather Current Weather API v2.5.
    Requer OPENWEATHER_API_KEY.
    Docs: https://openweathermap.org/current
    """
    try:
        import requests

        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q":     cidade,
                "appid": OPENWEATHER_KEY,
                "units": "metric",
                "lang":  "pt_br",
            },
            timeout=TIMEOUT,
        )

        if resp.status_code == 401:
            logger.error("OpenWeather: chave de API inválida.")
            return None
        if resp.status_code == 404:
            logger.warning("OpenWeather: cidade '%s' não encontrada.", cidade)
            return None
        resp.raise_for_status()

        d = resp.json()
        return {
            "temp":             round(d["main"]["temp"], 1),
            "sensacao_termica": round(d["main"]["feels_like"], 1),
            "umidade":          d["main"]["humidity"],
            "condicao":         d["weather"][0]["description"].capitalize(),
            "vento_kmh":        round(d["wind"]["speed"] * 3.6, 1),
            "pressao_hpa":      d["main"]["pressure"],
            "indice_uv":        None,
            "cidade":           d["name"],
            "latitude":         d["coord"]["lat"],
            "longitude":        d["coord"]["lon"],
            "fonte":            "OpenWeather",
            "timestamp":        datetime.now().isoformat(timespec="seconds"),
        }

    except Exception as e:
        logger.warning("OpenWeather: erro — %s", e)
        return None


def _clima_mock(cidade: str) -> dict:
    """Retorno estático para desenvolvimento sem acesso a APIs."""
    hora = datetime.now().hour
    temp_base = 22 + 8 * math.sin(math.pi * (hora - 6) / 12)
    return {
        "temp":             round(temp_base, 1),
        "sensacao_termica": round(temp_base - 2, 1),
        "umidade":          65,
        "condicao":         "Parcialmente nublado (simulado)",
        "vento_kmh":        12.0,
        "pressao_hpa":      1013.0,
        "indice_uv":        None,
        "cidade":           cidade,
        "latitude":         None,
        "longitude":        None,
        "fonte":            "mock",
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
    }


def listar_cidades_favoritas() -> list[str]:
    """Retorna lista de nomes das cidades pré-configuradas para o seletor do app."""
    return [c["nome"] for c in CIDADES_FAVORITAS]


def geocodificar_cidade(nome: str) -> Optional[dict]:
    """
    Busca coordenadas de qualquer cidade via Open-Meteo geocoding.
    Útil para validar cidades digitadas manualmente pelo usuário.

    Retorna dict com 'nome', 'estado', 'pais', 'latitude', 'longitude'
    ou None se não encontrada.
    """
    try:
        import requests
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": nome, "count": 1, "language": "pt", "format": "json"},
            timeout=TIMEOUT,
        )
        geo.raise_for_status()
        resultados = geo.json().get("results")
        if not resultados:
            return None
        r = resultados[0]
        return {
            "nome":      r.get("name", nome),
            "estado":    r.get("admin1", ""),
            "pais":      r.get("country", ""),
            "latitude":  r.get("latitude"),
            "longitude": r.get("longitude"),
        }
    except Exception as e:
        logger.warning("Geocoding: erro para '%s' — %s", nome, e)
        return # --- VARIÁVEIS GLOBAIS NECESSÁRIAS PARA O APP.PY ---

CIDADE_PADRAO = "Santa Cruz da Conceição"

CIDADES_FAVORITAS = [
    {"nome": "Santa Cruz da Conceição"},
    {"nome": "Leme"},
    {"nome": "Pirassununga"},
    {"nome": "Piracicaba"}
]

# ... (Mantenha todo o resto do seu código gigante daqui para baixo intacto!) ...