# app_streamlit.py

# ... (imports e configurações de página ficam iguais)

# ===== Estado =====
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []
if "mode" not in st.session_state:
    st.session_state.mode = "Nuvem"
if "api_key" not in st.session_state:
    st.session_state.api_key = SETTINGS.openai_api_key

# ===== Sidebar: Config =====
st.sidebar.header("Configuração")
mode = st.sidebar.radio("Backend", ["Nuvem", "Local"], index=(0 if st.session_state.mode=="Nuvem" else 1))
st.session_state.mode = mode

# Embeddings/LLM backend
if mode == "Nuvem":
    # Adiciona o campo de entrada para a chave da API
    api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", key="api_key_input")
    
    # Usa a chave inserida pelo usuário, se existir, senão usa a do arquivo .env
    api_key = api_key_input or st.session_state.api_key
    
    # Inicializa as classes com a chave
    emb = EmbeddingsCloud(api_key=api_key)
    llm = LLMCloud(api_key=api_key)

    if not api_key:
        st.sidebar.warning("Insira sua chave de API para usar o modo Nuvem.")
else:
    emb = EmbeddingsLocal()  # all-MiniLM-L6-v2
    llm = LLMLocal()         # Ollama (opcional)

# ... (o resto do código permanece igual)
