"""
modules/database.py
Camada de acesso a dados para o NeoCampo.

Melhorias em relação à versão original:
  - Context manager (with) garante fechamento da conexão mesmo em caso de erro
  - Caminho do banco configurável via variável de ambiente DB_PATH
  - Migrações automáticas: adiciona colunas novas sem recriar a tabela
  - Validação de entrada antes de inserir
  - Tratamento de exceções com logging estruturado
  - Funções extras: buscar_por_cultura, deletar_plantio, resumo_estatistico
"""

import os
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Caminho do banco: usa variável de ambiente ou padrão local
DB_PATH = os.getenv("NEOCAMPO_DB_PATH", "database.db")

# Esquema atual — adicione colunas aqui para migrações automáticas
COLUNAS_ESPERADAS = {
    "id":           "INTEGER PRIMARY KEY AUTOINCREMENT",
    "data":         "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    "cultura":      "TEXT NOT NULL",
    "area":         "REAL NOT NULL",
    "insumo":       "REAL NOT NULL",
    "cidade":       "TEXT",
    "temperatura":  "REAL",
    "observacoes":  "TEXT",          # coluna nova — migrada automaticamente
    "lote":         "TEXT",          # coluna nova — migrada automaticamente
}


# ── Gerenciador de conexão ───────────────────────────────────────────────────

@contextmanager
def _conexao():
    """
    Context manager que abre, entrega e fecha a conexão SQLite com segurança.
    Faz rollback automático em caso de exceção.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # acesso por nome de coluna
    conn.execute("PRAGMA journal_mode=WAL") # melhor performance em leituras concorrentes
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Criação e migração do banco ──────────────────────────────────────────────

def criar_banco() -> None:
    """
    Cria a tabela `plantios` se não existir e aplica migrações
    automáticas para colunas adicionadas ao esquema.
    """
    colunas_ddl = ",\n            ".join(
        f"{nome} {tipo}" for nome, tipo in COLUNAS_ESPERADAS.items()
    )
    ddl_criar = f"""
        CREATE TABLE IF NOT EXISTS plantios (
            {colunas_ddl}
        )
    """

    with _conexao() as conn:
        conn.execute(ddl_criar)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_plantios_data
            ON plantios (data DESC)
        """)
        _migrar_colunas(conn)

    logger.info("Banco de dados inicializado em '%s'", DB_PATH)


def _migrar_colunas(conn: sqlite3.Connection) -> None:
    """Adiciona colunas do esquema que ainda não existem na tabela."""
    cursor = conn.execute("PRAGMA table_info(plantios)")
    colunas_existentes = {row["name"] for row in cursor.fetchall()}

    for nome, tipo in COLUNAS_ESPERADAS.items():
        if nome not in colunas_existentes:
            # PRIMARY KEY não pode ser adicionada via ALTER TABLE
            if "PRIMARY KEY" in tipo.upper():
                continue
            # Remove DEFAULT da definição para ALTER TABLE (SQLite limitação)
            tipo_simples = tipo.split(" DEFAULT ")[0]
            conn.execute(f"ALTER TABLE plantios ADD COLUMN {nome} {tipo_simples}")
            logger.info("Migração: coluna '%s' adicionada à tabela plantios", nome)


# ── Escrita ──────────────────────────────────────────────────────────────────

def salvar_plantio(
    cultura: str,
    area: float,
    insumo: float,
    cidade: str = "",
    temperatura: float = 0.0,
    observacoes: str = "",
    lote: str = "",
) -> int:
    """
    Insere um registro de plantio no banco.

    Retorna o id do registro inserido.
    Levanta ValueError se os dados obrigatórios forem inválidos.
    """
    # Validação
    if not cultura or not cultura.strip():
        raise ValueError("O campo 'cultura' é obrigatório.")
    if area <= 0:
        raise ValueError(f"Área deve ser maior que zero (recebido: {area}).")
    if insumo < 0:
        raise ValueError(f"Insumo não pode ser negativo (recebido: {insumo}).")

    with _conexao() as conn:
        cursor = conn.execute(
            """
            INSERT INTO plantios (cultura, area, insumo, cidade, temperatura, observacoes, lote)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cultura.strip(),
                round(float(area), 4),
                round(float(insumo), 4),
                cidade.strip() if cidade else "",
                round(float(temperatura), 2),
                observacoes.strip() if observacoes else "",
                lote.strip() if lote else "",
            ),
        )
        novo_id = cursor.lastrowid

    logger.info(
        "Plantio salvo — id=%s cultura=%s área=%.2f m² lote=%s",
        novo_id, cultura, area, lote or "—",
    )
    return novo_id


def deletar_plantio(plantio_id: int) -> bool:
    """
    Remove um registro pelo id.
    Retorna True se alguma linha foi deletada, False se não encontrado.
    """
    with _conexao() as conn:
        cursor = conn.execute("DELETE FROM plantios WHERE id = ?", (plantio_id,))
        deletado = cursor.rowcount > 0

    if deletado:
        logger.info("Plantio id=%s removido.", plantio_id)
    else:
        logger.warning("Tentativa de deletar id=%s — não encontrado.", plantio_id)

    return deletado


# ── Leitura ──────────────────────────────────────────────────────────────────

def listar_historico(
    limite: int = 50,
    cultura: Optional[str] = None,
    lote: Optional[str] = None,
) -> pd.DataFrame:
    """
    Recupera os plantios mais recentes.

    Parâmetros
    ----------
    limite  : máximo de registros retornados (padrão 50).
    cultura : filtra por cultura específica (opcional).
    lote    : filtra por lote específico (opcional).
    """
    query = "SELECT * FROM plantios WHERE 1=1"
    params: list = []

    if cultura:
        query += " AND cultura = ?"
        params.append(cultura.strip())
    if lote:
        query += " AND lote = ?"
        params.append(lote.strip())

    query += " ORDER BY data DESC LIMIT ?"
    params.append(limite)

    with _conexao() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if not df.empty:
        # Formata a coluna de data para exibição
        df["data"] = pd.to_datetime(df["data"]).dt.strftime("%d/%m/%Y %H:%M")
        df = df.rename(columns={
            "id":          "ID",
            "data":        "Data",
            "cultura":     "Cultura",
            "area":        "Área (m²)",
            "insumo":      "Insumo (kg)",
            "cidade":      "Cidade",
            "temperatura": "Temp (°C)",
            "observacoes": "Observações",
            "lote":        "Lote",
        })

    return df


def buscar_por_id(plantio_id: int) -> Optional[dict]:
    """Retorna um plantio específico como dicionário, ou None se não encontrado."""
    with _conexao() as conn:
        cursor = conn.execute("SELECT * FROM plantios WHERE id = ?", (plantio_id,))
        row = cursor.fetchone()

    return dict(row) if row else None


# ── Análise / agregações ─────────────────────────────────────────────────────

def resumo_estatistico() -> dict:
    """
    Retorna métricas agregadas para o dashboard:
      - total_registros
      - area_total_m2
      - insumo_total_kg
      - culturas (lista de culturas distintas)
      - ultimo_plantio (timestamp)
    """
    with _conexao() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)           AS total,
                COALESCE(SUM(area), 0)   AS area_total,
                COALESCE(SUM(insumo), 0) AS insumo_total,
                MAX(data)          AS ultimo
            FROM plantios
        """).fetchone()

        culturas = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT cultura FROM plantios ORDER BY cultura"
            ).fetchall()
        ]

    return {
        "total_registros":  row["total"],
        "area_total_m2":    round(row["area_total"], 2),
        "insumo_total_kg":  round(row["insumo_total"], 2),
        "culturas":         culturas,
        "ultimo_plantio":   row["ultimo"] or "—",
    }