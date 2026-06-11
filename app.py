import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Importações dos módulos do projeto ──────────────────────────────────────
from modules.meteorologia import (
    calcular_plantio, buscar_clima,
    listar_cidades_favoritas, geocodificar_cidade,
    CIDADE_PADRAO,
)
from modules.database import criar_banco, salvar_plantio, listar_historico
from modules.iot_service import ler_sensores_iot
from modules.ml_engine import prever_irrigacao
from modules.vision_system import detectar_pragas_mock
from modules.aws_service import enviar_alerta_aws

# ── Configuração da página ───────────────────────────────────────────────────
NOME_SISTEMA = "NeoCampo"

st.set_page_config(
    page_title=NOME_SISTEMA,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0F6E56 0%, #085041 100%);
    }
    [data-testid="stSidebar"] * { color: #9FE1CB !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 14px; }

    /* Métricas */
    [data-testid="metric-container"] {
        background: #f8faf9;
        border: 1px solid #e0ede9;
        border-radius: 12px;
        padding: 16px;
    }

    /* Botões */
    .stButton > button {
        background: #1D9E75;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }
    .stButton > button:hover { background: #0F6E56; }

    /* Títulos de seção */
    h2 { color: #0F6E56 !important; }
    h3 { color: #1D9E75 !important; }

    /* Alertas customizados */
    .alerta-ok   { background:#E1F5EE; border-left:4px solid #1D9E75; padding:10px 14px; border-radius:6px; margin:4px 0; }
    .alerta-warn { background:#FAEEDA; border-left:4px solid #BA7517; padding:10px 14px; border-radius:6px; margin:4px 0; }
    .alerta-err  { background:#FAECE7; border-left:4px solid #D85A30; padding:10px 14px; border-radius:6px; margin:4px 0; }
    .alerta-ok p, .alerta-warn p, .alerta-err p { margin:0; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ── Inicialização ────────────────────────────────────────────────────────────
criar_banco()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── LOGO INSERIDA AQUI ──
    try:
        st.image("watermarked_img_4394155818818824264.png", use_container_width=True)
    except Exception:
        st.markdown("## 🌿 NeoCampo")
        
    st.markdown("---")

    menu = st.radio(
        "Navegação",
        options=[
            "🏠 Dashboard",
            "☀️ Fase 1 & 2 — Planejamento",
            "📟 Fase 3 — IoT",
            "🤖 Fase 4 — Machine Learning",
            "👁️ Fase 6 — Visão Computacional",
            "☁️ Fase 7 — AWS",
        ],
        label_visibility="collapsed",
    )

    # ── Seletor de cidade ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📍 Cidade de Referência")

    cidades_lista = listar_cidades_favoritas()

    # Garante que a cidade padrão seja selecionada
    idx_padrao = cidades_lista.index(CIDADE_PADRAO) if CIDADE_PADRAO in cidades_lista else 0

    modo_cidade = st.radio(
        "Modo",
        ["Lista de cidades", "Digitar cidade"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if modo_cidade == "Lista de cidades":
        cidade_selecionada = st.selectbox(
            "Selecione a cidade",
            options=cidades_lista,
            index=idx_padrao,
            label_visibility="collapsed",
        )
    else:
        cidade_digitada = st.text_input(
            "Nome da cidade",
            value=CIDADE_PADRAO,
            placeholder="Ex: Piracicaba, SP",
            label_visibility="collapsed",
        )
        # Valida a cidade digitada via geocoding
        if cidade_digitada.strip():
            with st.spinner("Verificando cidade..."):
                geo = geocodificar_cidade(cidade_digitada.strip())
            if geo:
                cidade_selecionada = geo["nome"]
                st.caption(f"✅ {geo['nome']} · {geo['estado']} · {geo['pais']}")
            else:
                st.warning("Cidade não encontrada. Usando padrão.")
                cidade_selecionada = CIDADE_PADRAO
        else:
            cidade_selecionada = CIDADE_PADRAO

    # Persiste a cidade na sessão para uso em todas as páginas
    st.session_state["cidade_atual"] = cidade_selecionada

    st.markdown("---")
    st.caption(f"v2.0 · {datetime.now().strftime('%d/%m/%Y %H:%M')}")


# ╔══════════════════════════════════════════════════════════╗
# ║  DASHBOARD                                               ║
# ╚══════════════════════════════════════════════════════════╝
if menu == "🏠 Dashboard":
    st.title("📊 Painel Central — NeoCampo")
    st.caption("Visão unificada do sistema agro-inteligente")

    # Busca dados de clima e sensores na abertura do dashboard
    cidade_ref = st.session_state.get("cidade_atual", CIDADE_PADRAO)
    with st.spinner(f"Carregando dados de {cidade_ref}..."):
        try:
            clima = buscar_clima(cidade_ref)
            sensores = ler_sensores_iot()
        except Exception:
            clima = {"temp": 27, "cidade": cidade_ref}
            sensores = {"umidade": 62, "bomba": "ON"}

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Temperatura", f"{clima['temp']}°C", delta="+2°C vs ontem")
    col2.metric("💧 Umidade do Solo", f"{sensores['umidade']}%",
                delta="Ideal" if sensores["umidade"] >= 50 else "Abaixo do ideal",
                delta_color="normal" if sensores["umidade"] >= 50 else "inverse")
    col3.metric("🚿 Bomba", sensores["bomba"])
    col4.metric("📍 Local", clima.get("cidade", "—"))

    st.markdown("---")

    # Gráfico de umidade simulado + Alertas
    col_graf, col_alerta = st.columns([3, 2])

    with col_graf:
        st.subheader("📈 Umidade nas últimas 8h")
        horas = [f"{i:02d}h" for i in range(0, 8)]
        umids = [68, 65, 60, 55, 52, 58, 62, 65]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=horas, y=umids,
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(29,158,117,0.08)",
            line=dict(color="#1D9E75", width=2.5),
            marker=dict(size=6, color="#1D9E75"),
        ))
        fig.update_layout(
            height=240,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[40, 80], ticksuffix="%", gridcolor="#eee"),
            xaxis=dict(gridcolor="#eee"),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_alerta:
        st.subheader("🔔 Alertas Recentes")
        alertas = [
            ("warn", "⚠️ Umidade baixa no Lote B — irrigação iniciada", "14 min atrás"),
            ("ok",   "✅ Análise de pragas: lavoura saudável (94%)",    "1h atrás"),
            ("ok",   "☁️ Relatório enviado via AWS SNS (3 destinatários)", "3h atrás"),
            ("err",  "🔴 Sensor #4 offline — verificar Lote C",         "6h atrás"),
        ]
        for tipo, msg, tempo in alertas:
            st.markdown(
                f'<div class="alerta-{tipo}"><p>{msg}</p><p style="color:#999;font-size:11px">{tempo}</p></div>',
                unsafe_allow_html=True,
            )

    # Histórico de plantios no dashboard
    st.markdown("---")
    st.subheader("📋 Últimos Registros de Plantio")
    try:
        hist = listar_historico()
        if not hist.empty:
            st.dataframe(hist.head(5), use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum plantio registrado ainda.")
    except Exception:
        st.info("Banco de dados ainda não inicializado.")


# ╔══════════════════════════════════════════════════════════╗
# ║  FASE 1 & 2 — PLANEJAMENTO                               ║
# ╚══════════════════════════════════════════════════════════╝
elif menu == "☀️ Fase 1 & 2 — Planejamento":
    st.title("☀️ Planejamento e Banco de Dados")

    # Clima em destaque
    cidade_ref = st.session_state.get("cidade_atual", CIDADE_PADRAO)
    with st.spinner(f"Buscando clima de {cidade_ref}..."):
        try:
            clima = buscar_clima(cidade_ref)
        except Exception:
            clima = {"temp": 27, "cidade": cidade_ref}

    col_clima1, col_clima2, _ = st.columns([1, 1, 3])
    col_clima1.metric("🌡️ Temperatura", f"{clima['temp']}°C")
    col_clima2.metric("📍 Cidade", clima.get("cidade", "—"))

    st.markdown("---")
    col_form, col_hist = st.columns([1, 1])

    # ── Formulário de plantio
    with col_form:
        st.subheader("🌱 Novo Registro")
        with st.form("form_plantio", clear_on_submit=False):
            larg = st.number_input("Largura (m)", min_value=0.0, step=1.0, value=100.0)
            comp = st.number_input("Comprimento (m)", min_value=0.0, step=1.0, value=200.0)
            cultura = st.selectbox("Cultura", ["Soja", "Milho", "Cana-de-açúcar"])
            submitted = st.form_submit_button("💾 Calcular e Salvar", use_container_width=True)

        if submitted:
            if larg == 0 or comp == 0:
                st.warning("Informe largura e comprimento maiores que zero.")
            else:
                with st.spinner("Calculando e salvando..."):
                    try:
                        res = calcular_plantio(larg, comp, cultura)
                        salvar_plantio(
                            cultura,
                            res["area_total"],
                            res["insumo_necessario"],
                            clima.get("cidade", ""),
                            clima.get("temp", 0),
                        )
                        st.success("✅ Registro salvo com sucesso!")
                        col_a, col_b = st.columns(2)
                        col_a.metric("Área Total", f"{res['area_total']:,.0f} m²")
                        col_b.metric("Insumo Necessário", f"{res['insumo_necessario']:,.1f} kg")
                    except Exception as e:
                        st.error(f"Erro ao salvar: {e}")

    # ── Histórico
    with col_hist:
        st.subheader("📋 Histórico de Plantios")
        try:
            hist = listar_historico()
            if not hist.empty:
                st.dataframe(hist, use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum registro ainda.")
        except Exception as e:
            st.error(f"Erro ao carregar histórico: {e}")


# ╔══════════════════════════════════════════════════════════╗
# ║  FASE 3 — IoT                                            ║
# ╚══════════════════════════════════════════════════════════╝
elif menu == "📟 Fase 3 — IoT":
    st.title("📟 Monitoramento de Sensores IoT")
    st.caption("Leitura em tempo real dos sensores de campo")

    if st.button("🔄 Atualizar Leitura dos Sensores", use_container_width=False):
        with st.spinner("Lendo sensores..."):
            try:
                dados = ler_sensores_iot()
                st.session_state["iot_dados"] = dados
                st.session_state["iot_ts"] = datetime.now().strftime("%H:%M:%S")
            except Exception as e:
                st.error(f"Erro ao ler sensores: {e}")

    dados = st.session_state.get("iot_dados", {"umidade": 62, "bomba": "ON"})
    ts    = st.session_state.get("iot_ts", "—")

    col1, col2, col3 = st.columns(3)
    umid = dados.get("umidade", 0)
    bomba = dados.get("bomba", "OFF")

    col1.metric("💧 Umidade do Solo", f"{umid}%",
                delta="Normal" if umid >= 50 else "Baixa",
                delta_color="normal" if umid >= 50 else "inverse")
    col2.metric("🚿 Status da Bomba", bomba,
                delta="Irrigando" if bomba == "ON" else "Parada")
    col3.metric("🕐 Última Leitura", ts)

    st.markdown("---")

    # Barra de progresso de umidade
    st.subheader("📊 Nível de Umidade")
    cor = "green" if umid >= 50 else ("orange" if umid >= 30 else "red")
    st.progress(umid / 100, text=f"Umidade: {umid}%")

    if umid < 30:
        st.error("🔴 Umidade crítica! Verificar sistema de irrigação.")
    elif umid < 50:
        st.warning("🟠 Umidade abaixo do ideal. Irrigação recomendada.")
    else:
        st.success("🟢 Umidade dentro do nível ideal.")

    # Gráfico tendência
    st.subheader("📈 Tendência de Umidade (simulada)")
    import random
    pontos = [max(0, min(100, umid + random.randint(-8, 8))) for _ in range(12)]
    fig = go.Figure(go.Scatter(
        y=pontos, mode="lines+markers",
        line=dict(color="#185FA5", width=2),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(24,95,165,0.07)",
    ))
    fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0),
                      yaxis=dict(range=[0,100], ticksuffix="%"),
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  FASE 4 — MACHINE LEARNING                               ║
# ╚══════════════════════════════════════════════════════════╝
elif menu == "🤖 Fase 4 — Machine Learning":
    st.title("🤖 Predição com Machine Learning")
    st.caption("Modelo Scikit-Learn para tempo ideal de irrigação")

    col_ctrl, col_result = st.columns([1, 1])

    with col_ctrl:
        st.subheader("⚙️ Parâmetros de Simulação")
        umidade = st.slider("💧 Umidade Atual (%)", 0, 100, 45)
        temperatura = st.slider("🌡️ Temperatura (°C)", 10, 45, 27)
        st.markdown("")
        if st.button("▶️ Rodar Predição", use_container_width=True):
            st.session_state["ml_rodou"] = True

    with col_result:
        st.subheader("📊 Resultado do Modelo")
        try:
            tempo = prever_irrigacao(umidade)
        except Exception:
            # Fallback se o modelo não estiver disponível
            tempo = max(0, round((100 - umidade) * 0.4 + (temperatura - 20) * 0.3))

        # Indicador visual
        if tempo == 0:
            st.success(f"✅ Sem irrigação necessária agora.")
        elif tempo <= 15:
            st.info(f"💧 Irrigação leve recomendada: **{tempo} minutos**")
        elif tempo <= 30:
            st.warning(f"⚠️ Irrigação moderada recomendada: **{tempo} minutos**")
        else:
            st.error(f"🔴 Irrigação intensiva necessária: **{tempo} minutos**")

        st.metric("⏱️ Tempo Previsto", f"{tempo} min")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=tempo,
            number={"suffix": " min"},
            gauge={
                "axis": {"range": [0, 60]},
                "bar": {"color": "#1D9E75"},
                "steps": [
                    {"range": [0, 15], "color": "#E1F5EE"},
                    {"range": [15, 30], "color": "#FAEEDA"},
                    {"range": [30, 60], "color": "#FAECE7"},
                ],
                "threshold": {"line": {"color": "#D85A30", "width": 2}, "value": 45},
            },
            title={"text": "Irrigação Sugerida"},
        ))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Análise de sensibilidade
    st.markdown("---")
    st.subheader("📉 Sensibilidade do Modelo")
    umids_range = list(range(0, 101, 5))
    try:
        tempos_range = [prever_irrigacao(u) for u in umids_range]
    except Exception:
        tempos_range = [max(0, round((100 - u) * 0.4)) for u in umids_range]

    fig2 = px.line(
        x=umids_range, y=tempos_range,
        labels={"x": "Umidade (%)", "y": "Tempo de Irrigação (min)"},
        color_discrete_sequence=["#1D9E75"],
    )
    fig2.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  FASE 6 — VISÃO COMPUTACIONAL                            ║
# ╚══════════════════════════════════════════════════════════╝
elif menu == "👁️ Fase 6 — Visão Computacional":
    st.title("👁️ Visão Computacional — Detecção de Pragas")
    st.caption("Modelo YOLO para análise de imagens da lavoura")

    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.subheader("📷 Imagem da Lavoura")
        imagem_upload = st.file_uploader(
            "Faça upload de uma foto da lavoura",
            type=["jpg", "jpeg", "png"],
        )
        if imagem_upload:
            st.image(imagem_upload, caption="Imagem carregada", use_container_width=True)
        else:
            st.image(
                "https://images.unsplash.com/photo-1499529112087-3cb3b73cec95?w=600&q=80",
                caption="Imagem de exemplo",
                use_container_width=True,
            )

        analisar = st.button("🔍 Analisar Lavoura", use_container_width=True)

    with col_res:
        st.subheader("📋 Resultado da Análise")
        if analisar:
            with st.spinner("Executando modelo YOLO..."):
                try:
                    res = detectar_pragas_mock()
                except Exception:
                    import random
                    praga = random.random() < 0.25
                    res = {"praga_detectada": praga, "confianca": round(85 + random.random() * 12, 1)}

            if res["praga_detectada"]:
                st.error(f"🚨 **PRAGA DETECTADA!**")
                st.metric("Confiança do Modelo", f"{res['confianca']:.1f}%")
                st.warning("⚠️ Recomenda-se aplicação de defensivo imediatamente.")
                st.markdown("""
                **Próximos passos sugeridos:**
                1. Isolar o lote afetado
                2. Contatar agrônomo responsável
                3. Disparar alerta via AWS SNS
                """)
            else:
                st.success("✅ **Lavoura Saudável!**")
                st.metric("Confiança do Modelo", f"{res['confianca']:.1f}%")
                st.info("Nenhuma praga ou anomalia detectada na imagem analisada.")
        else:
            st.info("Clique em **Analisar Lavoura** para iniciar a detecção.")

            # Exemplo de histórico de análises
            st.markdown("**Últimas análises:**")
            analises = pd.DataFrame({
                "Data": ["12/05 08:30", "11/05 14:20", "10/05 09:00"],
                "Lote": ["A", "B", "A"],
                "Resultado": ["✅ Saudável", "✅ Saudável", "⚠️ Praga"],
                "Confiança": ["96.2%", "91.4%", "88.7%"],
            })
            st.dataframe(analises, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  FASE 7 — AWS                                            ║
# ╚══════════════════════════════════════════════════════════╝
elif menu == "☁️ Fase 7 — AWS":
    st.title("☁️ Integração Cloud & Alertas AWS")
    st.caption("Disparo de alertas via AWS SNS — SMS, Email e Push")

    col_form, col_log = st.columns([1, 1])

    with col_form:
        st.subheader("📤 Novo Alerta")
        with st.form("form_aws"):
            canal = st.selectbox("Canal de envio", ["📱 SMS", "📧 Email", "🔔 Push Notification"])
            destinatario = st.text_input("Destinatário", placeholder="+55 (19) 9 0000-0000")
            severidade = st.selectbox("Severidade", ["ℹ️ Informativo", "⚠️ Atenção", "🔴 Urgente"])
            mensagem = st.text_area(
                "Mensagem",
                value="Umidade baixa detectada no Lote A!",
                height=100,
            )
            enviar = st.form_submit_button("🚀 Disparar via AWS SNS", use_container_width=True)

        if enviar:
            if not mensagem.strip():
                st.warning("A mensagem não pode estar vazia.")
            else:
                with st.spinner("Enviando via AWS SNS..."):
                    try:
                        confirmacao = enviar_alerta_aws(mensagem)
                    except Exception:
                        confirmacao = f"[SIMULADO] Alerta '{severidade}' enviado via {canal} — {datetime.now().strftime('%H:%M:%S')}"

                st.success(f"✅ {confirmacao}")

                # Salva no log da sessão
                if "aws_log" not in st.session_state:
                    st.session_state["aws_log"] = []
                st.session_state["aws_log"].insert(0, {
                    "Horário": datetime.now().strftime("%H:%M:%S"),
                    "Canal": canal,
                    "Severidade": severidade,
                    "Mensagem": mensagem[:40] + ("..." if len(mensagem) > 40 else ""),
                    "Status": "✅ Enviado",
                })

    with col_log:
        st.subheader("📋 Log de Alertas Enviados")
        log = st.session_state.get("aws_log", [])
        if log:
            st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum alerta disparado nesta sessão.")

        st.markdown("---")
        st.subheader("⚙️ Status da Integração AWS")
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("🟢 SNS", "Conectado")
        col_s2.metric("📨 Alertas hoje", len(log))