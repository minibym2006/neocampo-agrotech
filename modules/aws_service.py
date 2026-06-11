"""
modules/aws_service.py
Integração com AWS SNS para disparo de alertas via SMS, Email e Push.

Variáveis de ambiente necessárias (configure em .env ou nos segredos do Streamlit):
    AWS_ACCESS_KEY_ID       — chave de acesso IAM
    AWS_SECRET_ACCESS_KEY   — chave secreta IAM
    AWS_REGION              — ex: "us-east-1" ou "sa-east-1"
    AWS_SNS_TOPIC_ARN       — ARN do tópico SNS (ex: arn:aws:sns:sa-east-1:123456:neocampo-alertas)
    AWS_SNS_PHONE_NUMBER    — número para SMS direto, formato E.164 (ex: +5519900000000) — opcional
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Tenta importar boto3; se não instalado, avisa e usa modo simulado ────────
try:
    import boto3
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
        NoCredentialsError,
        EndpointResolutionError,
    )
    BOTO3_DISPONIVEL = True
except ImportError:
    BOTO3_DISPONIVEL = False
    logger.warning("boto3 não instalado. Rodando em modo simulado. Execute: pip install boto3")

# ── Tenta carregar .env automaticamente se python-dotenv estiver disponível ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Configurações lidas do ambiente ─────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("AWS_REGION", "sa-east-1")
AWS_SNS_TOPIC_ARN     = os.getenv("AWS_SNS_TOPIC_ARN")
AWS_SNS_PHONE_NUMBER  = os.getenv("AWS_SNS_PHONE_NUMBER")


def _criar_cliente_sns():
    """
    Cria e retorna um cliente SNS autenticado.
    Levanta RuntimeError se as credenciais não estiverem configuradas.
    """
    if not BOTO3_DISPONIVEL:
        raise RuntimeError("boto3 não está instalado.")

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise RuntimeError(
            "Credenciais AWS não configuradas. "
            "Defina AWS_ACCESS_KEY_ID e AWS_SECRET_ACCESS_KEY no ambiente."
        )

    return boto3.client(
        "sns",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def enviar_alerta_aws(
    mensagem: str,
    assunto: str = "Alerta NeoCampo",
    severidade: str = "INFO",
    canal: str = "topico",        # "topico" | "sms" | "ambos"
) -> str:
    """
    Envia um alerta via AWS SNS.

    Parâmetros
    ----------
    mensagem   : Corpo do alerta.
    assunto    : Assunto (visível em assinantes de email).
    severidade : "INFO", "ATENCAO" ou "URGENTE" — prefixado na mensagem.
    canal      : "topico" publica no tópico SNS configurado;
                 "sms" envia SMS direto ao número em AWS_SNS_PHONE_NUMBER;
                 "ambos" faz os dois.

    Retorna
    -------
    String de confirmação com MessageId ou descrição do erro/simulação.
    """
    if not mensagem or not mensagem.strip():
        return "⚠️ Mensagem vazia — alerta não enviado."

    prefixo = {"INFO": "ℹ️", "ATENCAO": "⚠️", "URGENTE": "🔴"}.get(severidade.upper(), "📢")
    ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    mensagem_completa = f"[{severidade.upper()}] {prefixo} {mensagem}\n\nNeoCampo · {ts}"

    # ── Modo simulado ──────────────────────────────────────────────────────
    if not BOTO3_DISPONIVEL or not AWS_ACCESS_KEY_ID:
        logger.info("[SIMULADO] Alerta AWS SNS: %s", mensagem_completa)
        return (
            f"[SIMULADO] Alerta '{severidade}' registrado com sucesso.\n"
            f"Canal: {canal} · {ts}\n"
            f"Mensagem: {mensagem}"
        )

    # ── Envio real ─────────────────────────────────────────────────────────
    resultados = []
    erros = []

    try:
        cliente = _criar_cliente_sns()

        # Publicar no tópico SNS (alcança todos os assinantes)
        if canal in ("topico", "ambos"):
            if not AWS_SNS_TOPIC_ARN:
                erros.append("AWS_SNS_TOPIC_ARN não configurado — publicação no tópico ignorada.")
            else:
                resp = cliente.publish(
                    TopicArn=AWS_SNS_TOPIC_ARN,
                    Message=mensagem_completa,
                    Subject=assunto,
                )
                msg_id = resp.get("MessageId", "—")
                resultados.append(f"Tópico SNS ✅ (MessageId: {msg_id})")
                logger.info("SNS tópico OK — MessageId: %s", msg_id)

        # Enviar SMS direto
        if canal in ("sms", "ambos"):
            numero = AWS_SNS_PHONE_NUMBER
            if not numero:
                erros.append("AWS_SNS_PHONE_NUMBER não configurado — SMS ignorado.")
            else:
                resp_sms = cliente.publish(
                    PhoneNumber=numero,
                    Message=mensagem_completa,
                    MessageAttributes={
                        "AWS.SNS.SMS.SMSType": {
                            "DataType": "String",
                            "StringValue": "Transactional",  # alta prioridade
                        }
                    },
                )
                msg_id_sms = resp_sms.get("MessageId", "—")
                resultados.append(f"SMS ✅ → {numero} (MessageId: {msg_id_sms})")
                logger.info("SNS SMS OK — MessageId: %s", msg_id_sms)

    except NoCredentialsError:
        msg = "Credenciais AWS inválidas ou não encontradas."
        logger.error(msg)
        return f"❌ Erro de autenticação AWS: {msg}"

    except ClientError as e:
        codigo = e.response["Error"]["Code"]
        detalhe = e.response["Error"]["Message"]
        logger.error("ClientError AWS [%s]: %s", codigo, detalhe)
        return f"❌ Erro AWS ({codigo}): {detalhe}"

    except (BotoCoreError, EndpointResolutionError) as e:
        logger.error("BotoCoreError: %s", e)
        return f"❌ Erro de conexão com AWS: {e}"

    except Exception as e:
        logger.exception("Erro inesperado ao enviar alerta AWS")
        return f"❌ Erro inesperado: {e}"

    # Monta resposta final
    linhas = resultados + [f"⚠️ {e}" for e in erros]
    if not linhas:
        return "⚠️ Nenhum canal foi acionado. Verifique as variáveis de ambiente."

    return "\n".join(linhas)


def verificar_conexao_aws() -> dict:
    """
    Verifica se a conexão com a AWS está funcional.
    Útil para exibir status no dashboard.

    Retorna dict com chaves: 'ok' (bool), 'mensagem' (str).
    """
    if not BOTO3_DISPONIVEL:
        return {"ok": False, "mensagem": "boto3 não instalado (pip install boto3)"}

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        return {"ok": False, "mensagem": "Credenciais AWS não configuradas no ambiente"}

    try:
        cliente = _criar_cliente_sns()
        cliente.list_topics()  # chamada leve para validar credenciais
        return {"ok": True, "mensagem": f"AWS SNS conectado ({AWS_REGION})"}

    except NoCredentialsError:
        return {"ok": False, "mensagem": "Credenciais AWS inválidas"}

    except ClientError as e:
        return {"ok": False, "mensagem": f"Erro AWS: {e.response['Error']['Code']}"}

    except Exception as e:
        return {"ok": False, "mensagem": f"Erro de conexão: {e}"}