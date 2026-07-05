import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import scipy.stats as stats
except Exception:
    stats = None

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except Exception:
    XGBOOST_OK = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

st.set_page_config(
    page_title="Analítica E-commerce - BPA Grupo 2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #12355B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.02rem;
        color: #4F5D75;
        margin-bottom: 1.4rem;
    }
    .step-card {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid #E6EAF0;
        background: #FFFFFF;
        box-shadow: 0 2px 10px rgba(18,53,91,0.05);
        margin-bottom: 0.8rem;
    }
    .ok-box {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #EAF7EF;
        color: #174A2A;
        border: 1px solid #B7E0C2;
    }
    .warn-box {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #FFF5DF;
        color: #664A00;
        border: 1px solid #FFE0A3;
    }
    .small-note {
        color: #5E6B78;
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E6EAF0;
        padding: 0.8rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(18,53,91,0.04);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Utilidades generales
# -----------------------------

def fig_to_streamlit(fig):
    st.pyplot(fig, clear_figure=True)


def read_input_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato no soportado. Use CSV, XLSX o XLS.")


def normalize_loaded_columns(df_raw):
    """Respeta el punto inicial del notebook: trabajar con las 20 columnas originales."""
    df = df_raw.copy()
    cols_originales = [
        'Row ID', 'Order ID', 'Order Date', 'Ship Mode', 'Customer ID',
        'Customer Name', 'Segment', 'Country', 'City', 'State',
        'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category',
        'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit'
    ]

    faltantes = [col for col in cols_originales if col not in df.columns]
    if faltantes:
        raise ValueError(
            "El archivo cargado no contiene las columnas originales requeridas: "
            + ", ".join(faltantes)
        )

    df = df[cols_originales].copy()

    columnas_renombrar = {
        'Order ID': 'order_id',
        'Order Date': 'order_date',
        'Ship Mode': 'ship_mode',
        'Customer ID': 'customer_id',
        'Customer Name': 'customer_name',
        'Postal Code': 'postal_code',
        'Product ID': 'product_id',
        'Sub-Category': 'sub_category',
        'Product Name': 'product_name'
    }
    df.rename(columns=columnas_renombrar, inplace=True)
    return df


def clean_data(df_loaded):
    """Replica la limpieza principal del notebook de Colab."""
    log = []
    df = df_loaded.copy()
    shape_inicial = df.shape
    log.append(f"Shape inicial tras selección de columnas: {shape_inicial}")

    # order_date debe ser datetime
    df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True, format='mixed', errors='coerce')

    columnas_castear = {'Segment': 'category', 'Category': 'category', 'Region': 'category'}
    for col, dtype in columnas_castear.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # Row ID reindexado
    df['Row ID'] = range(1, len(df) + 1)

    duplicados_antes = int(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.append(f"Duplicados eliminados: {duplicados_antes}")

    # Eliminar filas con más del 40% de nulos: conservar al menos 60% de columnas no nulas
    antes = len(df)
    thresh_filas = int(df.shape[1] * 0.6)
    df.dropna(thresh=thresh_filas, inplace=True)
    log.append(f"Filas eliminadas por más de 40% de nulos: {antes - len(df)}")

    df_num = df.select_dtypes(include='number').copy()
    df_cat = df.select_dtypes(exclude='number').copy()

    # Corrección de typos - Segment
    anterior = ['Consumr','CONSUMER','consumer','Consummer','Corporrate',
                'Corp.','CORPORATE','Coorporate','HomeOffice','Home-Office',
                'HOME OFFICE','Hom Office']
    actual = ['Consumer','Consumer','Consumer','Consumer','Corporate',
              'Corporate','Corporate','Corporate','Home Office','Home Office',
              'Home Office','Home Office']
    if 'Segment' in df_cat.columns:
        df_cat['Segment'] = df_cat['Segment'].replace(anterior, actual)

    # Corrección de typos - Category
    anterior = ['OFFICE SUPPLIES','Office Supply','Office Suplies','Útiles de Oficina',
                'FURNITURE','Furniturre','Furntiture','Muebles',
                'TECHNOLOGY','Tech','Techonology','Technologgy','Tecnología']
    actual = ['Office Supplies','Office Supplies','Office Supplies','Office Supplies',
              'Furniture','Furniture','Furniture','Furniture',
              'Technology','Technology','Technology','Technology','Technology']
    if 'Category' in df_cat.columns:
        df_cat['Category'] = df_cat['Category'].replace(anterior, actual)

    # Corrección de typos - Ship Mode
    anterior = ['Standard-Class','Std Class','Standart Class',
                'Second-Class','Secound Class','2nd Class',
                'First-Class','Firsst Class','1st Class']
    actual = ['Standard Class','Standard Class','Standard Class',
              'Second Class','Second Class','Second Class',
              'First Class','First Class','First Class']
    if 'ship_mode' in df_cat.columns:
        df_cat['ship_mode'] = df_cat['ship_mode'].replace(anterior, actual)

    # Region
    if 'Region' in df_cat.columns:
        df_cat['Region'] = df_cat['Region'].astype(str).str.strip().str.title()
        df_cat['Region'] = df_cat['Region'].replace('Nan', np.nan)

    # Normalizar strings
    cols_texto = ['customer_name', 'City', 'State', 'Country', 'product_name', 'sub_category']
    for col in cols_texto:
        if col in df_cat.columns:
            df_cat[col] = df_cat[col].astype(str).str.strip().str.title()
            df_cat[col] = df_cat[col].replace('Nan', np.nan)

    # IDs críticos
    ids_criticos = [c for c in ['order_id', 'customer_id', 'product_id'] if c in df_cat.columns]
    antes = len(df_cat)
    df_cat.dropna(subset=ids_criticos, inplace=True)
    log.append(f"Filas eliminadas por IDs críticos nulos: {antes - len(df_cat)}")

    # Imputación con moda
    for col in ['ship_mode', 'Segment', 'Region', 'Category']:
        if col in df_cat.columns:
            moda = df_cat[col].mode(dropna=True)
            if not moda.empty:
                df_cat[col] = df_cat[col].fillna(moda.iloc[0])

    # Sincronizar indices numéricos con categóricos
    df_num = df.select_dtypes(include='number').copy()
    df_num = df_num.loc[df_cat.index]

    # Corregir Quantity y Discount
    if 'Quantity' in df_num.columns:
        df_num['Quantity'] = df_num['Quantity'].abs()
    if 'Discount' in df_num.columns:
        df_num['Discount'] = df_num['Discount'].apply(lambda x: x / 100 if pd.notnull(x) and x > 1 else x)

    # Imputación numérica por mediana
    for col in ['Sales', 'Profit', 'Quantity', 'Discount']:
        if col in df_num.columns:
            df_num[col] = df_num[col].fillna(df_num[col].median())

    # Outliers - winsorización al percentil 98 para Sales y Profit
    outlier_info = {}
    for col in ['Sales', 'Profit']:
        if col in df_num.columns:
            q1 = df_num[col].quantile(0.25)
            q3 = df_num[col].quantile(0.75)
            iqr = q3 - q1
            min_threshold = q1 - 1.5 * iqr
            max_threshold = q3 + 1.5 * iqr
            cantidad_outliers = int(((df_num[col] < min_threshold) | (df_num[col] > max_threshold)).sum())
            tope_alto = round(np.percentile(df_num[col], 98), 2)
            df_num.loc[df_num[col] > tope_alto, col] = tope_alto
            outlier_info[col] = {
                'Q1': q1,
                'Q3': q3,
                'IQR': iqr,
                'Límite inferior': min_threshold,
                'Límite superior': max_threshold,
                'Outliers detectados': cantidad_outliers,
                'Tope P98 aplicado': tope_alto
            }

    df_cat = df_cat.reset_index(drop=True)
    df_num = df_num.reset_index(drop=True)
    df_cat_num = pd.concat([df_cat, df_num], axis=1)

    log.append(f"Shape final df_cat: {df_cat.shape}")
    log.append(f"Shape final df_num: {df_num.shape}")
    log.append(f"Shape final consolidado: {df_cat_num.shape}")

    return df_cat, df_num, df_cat_num, pd.DataFrame(outlier_info).T, log


def build_transformation(df_cat_num):
    df = df_cat_num.copy()
    df_cat = df.select_dtypes(exclude='number').copy()
    df_num = df.select_dtypes(include='number').copy()

    target = 'Profit'
    df_target = df_num[[target]].copy()

    columnas_cat_transformar = ['ship_mode', 'Segment', 'Country', 'Region', 'Category', 'sub_category']
    columnas_cat_transformar = [c for c in columnas_cat_transformar if c in df_cat.columns]

    df_cat_dummy_base = df_cat[columnas_cat_transformar].copy()
    for col in df_cat_dummy_base.columns:
        df_cat_dummy_base[col] = df_cat_dummy_base[col].astype(str)
    df_cat_dummy_base = df_cat_dummy_base.fillna("Sin dato")
    df_cat_dummy = pd.get_dummies(df_cat_dummy_base, drop_first=True, dtype='int64')

    columnas_num_transformar = ['Sales', 'Quantity', 'Discount']
    columnas_num_transformar = [c for c in columnas_num_transformar if c in df_num.columns]
    mms = MinMaxScaler()
    df_mms_array = mms.fit_transform(df_num[columnas_num_transformar])
    df_mms = pd.DataFrame(df_mms_array, columns=columnas_num_transformar, index=df_num.index)

    df_final_transformacion = pd.concat(
        [df_cat_dummy.reset_index(drop=True), df_mms.reset_index(drop=True), df_target.reset_index(drop=True)],
        axis=1
    )
    return df_final_transformacion, columnas_cat_transformar, columnas_num_transformar


def run_kmeans(df_cat_num):
    df_model = df_cat_num.copy()
    df_productos = df_model.groupby('product_name').agg(
        Frecuencia=('order_id', 'nunique'),
        Volumen=('Quantity', 'sum')
    ).reset_index()

    scaler = MinMaxScaler()
    sc = df_productos[['Frecuencia', 'Volumen']]
    sc_scaled = scaler.fit_transform(sc)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_productos['Cluster_ID'] = kmeans.fit_predict(sc_scaled)

    cluster_resumen = df_productos.groupby('Cluster_ID').agg({
        'Frecuencia': 'mean',
        'Volumen': 'mean',
        'product_name': 'count'
    }).rename(columns={'product_name': 'Cantidad de Productos'})

    # Mapeo usado en el notebook
    dicc_rotacion = {0: 'Alta', 2: 'Media', 1: 'Baja'}
    df_productos['Nivel_Rotacion'] = df_productos['Cluster_ID'].map(dicc_rotacion)

    rango_k = range(2, 11)
    inercia, silueta, calinski, davies = [], [], [], []
    for k in rango_k:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(sc_scaled)
        etiquetas = kmeans_temp.labels_
        inercia.append(kmeans_temp.inertia_)
        silueta.append(silhouette_score(sc_scaled, etiquetas))
        calinski.append(calinski_harabasz_score(sc_scaled, etiquetas))
        davies.append(davies_bouldin_score(sc_scaled, etiquetas))

    metricas_df = pd.DataFrame({
        'valor de K': list(rango_k),
        'Codo': inercia,
        'Silueta': silueta,
        'calinsky': calinski,
        'davies': davies
    }).set_index('valor de K')

    df_model = df_model.merge(df_productos[['product_name', 'Nivel_Rotacion']], on='product_name', how='left')
    return df_model, df_productos, cluster_resumen, metricas_df, sc_scaled


def build_time_series_and_xgboost(df_model, optimize=False):
    if not XGBOOST_OK:
        return None, None, None, None, "La librería xgboost no está instalada. Instale xgboost para ejecutar esta sección."

    df_model = df_model.copy()
    df_model['volumen_total'] = df_model['Quantity'] * df_model['Sales']
    df_model['impacto_descuento_real'] = df_model['volumen_total'] * df_model['Discount']
    df_model['order_date'] = pd.to_datetime(df_model['order_date'], errors='coerce')
    df_model = df_model.dropna(subset=['order_date'])
    df_model['periodo_diario'] = df_model['order_date'].dt.to_period('D')

    df_ts = df_model.groupby('periodo_diario').agg(
        Profit=('Profit', 'sum'),
        volumen_total=('volumen_total', 'sum'),
        impacto_descuento_real=('impacto_descuento_real', 'sum'),
        Descuento_Promedio=('Discount', 'mean')
    ).reset_index()
    df_ts['Fecha'] = df_ts['periodo_diario'].dt.to_timestamp()

    def obtener_segmento_dominante(grupo):
        return grupo['Nivel_Rotacion'].mode()[0] if not grupo['Nivel_Rotacion'].mode().empty else 'Desconocido'

    segmentos_diarios = df_model.groupby('periodo_diario').apply(obtener_segmento_dominante).reset_index(name='Segmento_Dominante')
    df_ts = pd.merge(df_ts, segmentos_diarios, on='periodo_diario')
    df_ts = pd.get_dummies(df_ts, columns=['Segmento_Dominante'], drop_first=True, dtype=int)

    df_ts['mes'] = df_ts['Fecha'].dt.month
    df_ts['dia_del_mes'] = df_ts['Fecha'].dt.day
    df_ts['dia_de_la_semana'] = df_ts['Fecha'].dt.dayofweek
    df_ts['es_fin_de_semana'] = df_ts['dia_de_la_semana'].apply(lambda x: 1 if x >= 5 else 0)
    df_ts = df_ts.sort_values('Fecha').drop(columns=['periodo_diario']).reset_index(drop=True)

    columnas_excluir = ['Profit']
    if 'Fecha' in df_ts.columns:
        columnas_excluir.append('Fecha')

    X = df_ts.drop(columns=columnas_excluir)
    y = df_ts['Profit']

    dias_test = 90
    if len(df_ts) <= dias_test + 10:
        dias_test = max(5, int(len(df_ts) * 0.25))
    train_size = len(df_ts) - dias_test

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    modelo_XGBoost = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=10, random_state=42)
    modelo_XGBoost.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_XGBoost = modelo_XGBoost.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_XGBoost)
    mse = mean_squared_error(y_test, y_pred_XGBoost)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_XGBoost)

    results = pd.DataFrame([['XGBoost Regressor', mae, rmse, r2]],
                           columns=['Model', 'MAE (Error Absoluto)', 'RMSE (Error Cuadrático)', 'R2 Score'])

    df_comparar = X_test.copy()
    df_comparar['Fecha'] = df_ts['Fecha'].iloc[train_size:].values
    df_comparar['y_test_XGBoost'] = y_test.values
    df_comparar['y_pred_XGBoost'] = y_pred_XGBoost

    best_params = None
    if optimize and len(X_train) >= 30:
        random_parameters = {
            'n_estimators': [100, 150, 200, 300, 400],
            'max_depth': [3, 5, 7, 10, 12],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 1, 5, 10],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        n_splits = min(5, max(2, len(X_train) // 25))
        random_search = RandomizedSearchCV(
            estimator=XGBRegressor(random_state=42),
            param_distributions=random_parameters,
            cv=TimeSeriesSplit(n_splits=n_splits),
            scoring='r2',
            n_iter=20,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        modelo_opt = XGBRegressor(**random_search.best_params_, random_state=42)
        modelo_opt.fit(X_train, y_train)
        y_pred_opt = modelo_opt.predict(X_test)
        mae_opt = mean_absolute_error(y_test, y_pred_opt)
        rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
        r2_opt = r2_score(y_test, y_pred_opt)
        results.loc[len(results)] = ['XGBoost Regressor Optimizado', mae_opt, rmse_opt, r2_opt]
        df_comparar['y_pred_XGBoost_Opt'] = y_pred_opt

    return df_ts, results, df_comparar, best_params, None


def dataframe_to_excel_bytes(sheets):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=True if df.index.name else False)
    return output.getvalue()

# -----------------------------
# Interfaz
# -----------------------------

st.markdown('<div class="main-title">📊 Analítica avanzada de E-commerce</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Carga el Excel sucio del proyecto y ejecuta automáticamente limpieza, EDA, transformación, K-Means y modelo predictivo.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("Sube el archivo original sucio usado en el proyecto.")
    uploaded_file = st.file_uploader("Archivo Excel o CSV", type=["xlsx", "xls", "csv"])
    st.divider()
    ejecutar_optimizacion = st.checkbox(
        "Ejecutar optimización XGBoost",
        value=False,
        help="Activa RandomizedSearchCV. Puede tardar más, pero replica la sección de optimización."
    )
    mostrar_tablas = st.checkbox("Mostrar tablas detalladas", value=True)
    st.divider()
    st.caption("Proyecto: Business Predictive Analytics - Grupo 2")

if uploaded_file is None:
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="step-card">
            <h3>¿Cómo usar la aplicación?</h3>
            <ol>
                <li>Carga el Excel sucio del e-commerce.</li>
                <li>Presiona <b>Ejecutar análisis completo</b>.</li>
                <li>Revisa los gráficos, indicadores y modelos.</li>
                <li>Descarga los resultados finales en Excel.</li>
            </ol>
            <p class="small-note">La aplicación reproduce el flujo del notebook de Google Colab, pero en una interfaz web.</p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

try:
    df_raw = read_input_file(uploaded_file)
    st.markdown('<div class="ok-box">Archivo cargado correctamente. Revise la vista previa y ejecute el análisis completo.</div>', unsafe_allow_html=True)
    st.subheader("Vista previa del archivo cargado")
    st.dataframe(df_raw.head(10), use_container_width=True)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

if st.button("🚀 Ejecutar análisis completo", type="primary", use_container_width=True):
    progress = st.progress(0, text="Preparando archivo...")
    try:
        with st.spinner("Por favor espere. Se está ejecutando el flujo completo del notebook..."):
            df_loaded = normalize_loaded_columns(df_raw)
            progress.progress(10, text="Columnas originales seleccionadas y renombradas...")

            df_cat, df_num, df_cat_num, outlier_info, cleaning_log = clean_data(df_loaded)
            progress.progress(35, text="Limpieza, imputación y tratamiento de outliers completados...")

            df_final_transformacion, columnas_cat_transformar, columnas_num_transformar = build_transformation(df_cat_num)
            progress.progress(55, text="Transformación de datos completada...")

            df_model, df_productos, cluster_resumen, metricas_df, sc_scaled = run_kmeans(df_cat_num)
            progress.progress(75, text="Modelo K-Means ejecutado...")

            df_ts, xgb_results, df_comparar, best_params, xgb_error = build_time_series_and_xgboost(
                df_model,
                optimize=ejecutar_optimizacion
            )
            progress.progress(95, text="Modelo de serie de tiempo evaluado...")
            progress.progress(100, text="Análisis finalizado.")

        st.success("Análisis completo ejecutado correctamente.")
        st.session_state['results'] = {
            'df_loaded': df_loaded,
            'df_cat': df_cat,
            'df_num': df_num,
            'df_cat_num': df_cat_num,
            'outlier_info': outlier_info,
            'cleaning_log': cleaning_log,
            'df_final_transformacion': df_final_transformacion,
            'df_model': df_model,
            'df_productos': df_productos,
            'cluster_resumen': cluster_resumen,
            'metricas_df': metricas_df,
            'df_ts': df_ts,
            'xgb_results': xgb_results,
            'df_comparar': df_comparar,
            'best_params': best_params,
            'xgb_error': xgb_error,
            'columnas_cat_transformar': columnas_cat_transformar,
            'columnas_num_transformar': columnas_num_transformar,
        }
    except Exception as e:
        st.error(f"Ocurrió un error durante la ejecución: {e}")
        st.stop()

if 'results' not in st.session_state:
    st.info("Presiona el botón de ejecución para generar todos los resultados.")
    st.stop()

r = st.session_state['results']
df_loaded = r['df_loaded']
df_cat = r['df_cat']
df_num = r['df_num']
df_cat_num = r['df_cat_num']
outlier_info = r['outlier_info']
df_final_transformacion = r['df_final_transformacion']
df_model = r['df_model']
df_productos = r['df_productos']
cluster_resumen = r['cluster_resumen']
metricas_df = r['metricas_df']
df_ts = r['df_ts']
xgb_results = r['xgb_results']
df_comparar = r['df_comparar']
best_params = r['best_params']
xgb_error = r['xgb_error']

# Métricas superiores
st.subheader("Resumen general del procesamiento")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filas archivo original", f"{len(df_loaded):,}")
col2.metric("Filas base limpia", f"{len(df_cat_num):,}")
col3.metric("Variables consolidadas", f"{df_cat_num.shape[1]:,}")
col4.metric("Productos segmentados", f"{len(df_productos):,}")

with st.expander("Ver bitácora de limpieza"):
    for item in r['cleaning_log']:
        st.write("-", item)

# Tabs de resultados
(tab_limpieza, tab_eda, tab_transformacion, tab_kmeans, tab_xgb, tab_descarga) = st.tabs([
    "1. Limpieza de datos",
    "2. EDA e insights",
    "3. Transformación",
    "4. K-Means",
    "5. Serie de tiempo",
    "6. Descargas"
])

with tab_limpieza:
    st.header("1. Calidad y limpieza de datos")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Nulos finales en variables categóricas**")
        st.dataframe(df_cat.isnull().sum().sort_values(ascending=False).to_frame("nulos"), use_container_width=True)
    with c2:
        st.markdown("**Nulos finales en variables numéricas**")
        st.dataframe(df_num.isnull().sum().sort_values(ascending=False).to_frame("nulos"), use_container_width=True)

    st.markdown("**Tratamiento de outliers aplicado**")
    st.dataframe(outlier_info, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    df_num['Sales'].plot.box(vert=False, ax=ax)
    ax.set_title('Boxplot de Sales')
    fig_to_streamlit(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    df_num['Profit'].plot.box(vert=False, ax=ax)
    ax.set_title('Boxplot de Profit')
    fig_to_streamlit(fig)

    if mostrar_tablas:
        st.markdown("**Base limpia consolidada**")
        st.dataframe(df_cat_num.head(50), use_container_width=True)

with tab_eda:
    st.header("2. Análisis exploratorio de datos")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Frecuencia de pedidos por Category**")
        fig, ax = plt.subplots(figsize=(8, 4))
        df_cat['Category'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Frecuencia de pedidos por Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Cantidad de registros')
        ax.tick_params(axis='x', rotation=0)
        fig_to_streamlit(fig)
    with colB:
        st.markdown("**Frecuencia de pedidos por Segment**")
        fig, ax = plt.subplots(figsize=(8, 4))
        df_cat['Segment'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Frecuencia de pedidos por Segment')
        ax.set_xlabel('Segment')
        ax.set_ylabel('Cantidad de registros')
        ax.tick_params(axis='x', rotation=0)
        fig_to_streamlit(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Histograma de Sales**")
        fig, ax = plt.subplots(figsize=(8, 4))
        df_num['Sales'].plot.hist(bins=30, ax=ax)
        ax.set_title('Histograma de Sales')
        ax.set_xlabel('Sales')
        ax.set_ylabel('Frecuencia')
        fig_to_streamlit(fig)
    with c2:
        st.markdown("**Histograma de Profit**")
        fig, ax = plt.subplots(figsize=(8, 4))
        df_num['Profit'].plot.hist(bins=30, ax=ax)
        ax.set_title('Histograma de Profit')
        ax.set_xlabel('Profit')
        ax.set_ylabel('Frecuencia')
        fig_to_streamlit(fig)

    st.markdown("**Ventas y utilidad acumulada por Category**")
    ventas_utilidad_categoria = df_cat_num.groupby('Category')[['Sales', 'Profit']].sum().sort_values('Sales', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ventas_utilidad_categoria.plot(kind='bar', ax=ax)
    ax.set_title('Ventas y utilidad por Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Monto acumulado')
    ax.tick_params(axis='x', rotation=0)
    fig_to_streamlit(fig)

    st.markdown("**Relación entre Sales y Profit**")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_cat_num['Sales'], df_cat_num['Profit'], alpha=0.5, s=30)
    ax.set_title('Gráfico de dispersión Sales vs Profit')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Profit')
    fig_to_streamlit(fig)

    st.markdown("**Distribución de Profit por Category**")
    fig, ax = plt.subplots(figsize=(10, 5))
    if sns is not None:
        sns.boxplot(data=df_cat_num, x='Category', y='Profit', hue='Category', legend=False, ax=ax)
    else:
        df_cat_num.boxplot(column='Profit', by='Category', ax=ax)
    ax.set_title('Distribución de Profit por Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Profit')
    fig_to_streamlit(fig)

    st.markdown("**Matriz de correlación de variables numéricas**")
    matriz_correlacion = df_num[['Sales', 'Quantity', 'Discount', 'Profit']].corr()
    fig, ax = plt.subplots(figsize=(9, 5))
    if sns is not None:
        sns.heatmap(matriz_correlacion, center=0, vmin=-1, vmax=1, linewidths=0.5, annot=True, ax=ax)
    else:
        cax = ax.matshow(matriz_correlacion, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(range(len(matriz_correlacion.columns)))
        ax.set_xticklabels(matriz_correlacion.columns)
        ax.set_yticks(range(len(matriz_correlacion.index)))
        ax.set_yticklabels(matriz_correlacion.index)
    ax.set_title('Matriz de correlación de variables numéricas')
    fig_to_streamlit(fig)

    st.markdown("**Evolución mensual de Sales**")
    df_temp = df_cat_num.copy()
    df_temp['order_date'] = pd.to_datetime(df_temp['order_date'], errors='coerce')
    ventas_mensuales = df_temp.groupby(pd.Grouper(key='order_date', freq='ME'))['Sales'].sum().dropna()
    fig, ax = plt.subplots(figsize=(11, 4))
    ventas_mensuales.plot(ax=ax)
    ax.set_title('Evolución mensual de Sales')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Sales acumulado')
    fig_to_streamlit(fig)

    if mostrar_tablas:
        st.markdown("**Tablas de apoyo del EDA**")
        st.dataframe(ventas_utilidad_categoria, use_container_width=True)
        st.dataframe(matriz_correlacion, use_container_width=True)

with tab_transformacion:
    st.header("3. Transformación de datos")
    st.markdown("Se aplican variables dummy a las columnas categóricas seleccionadas y MinMaxScaler a Sales, Quantity y Discount, manteniendo Profit como variable objetivo.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Columnas categóricas transformadas", len(r['columnas_cat_transformar']))
    col2.metric("Columnas numéricas escaladas", len(r['columnas_num_transformar']))
    col3.metric("Dimensión final", f"{df_final_transformacion.shape[0]} x {df_final_transformacion.shape[1]}")
    st.dataframe(df_final_transformacion.head(50), use_container_width=True)

with tab_kmeans:
    st.header("4. Modelo K-Means: segmentación de productos")
    st.markdown("El modelo agrupa productos por **Frecuencia** de pedidos y **Volumen** vendido, usando tres clústeres: Alta, Media y Baja rotación.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Resumen por clúster**")
        st.dataframe(cluster_resumen, use_container_width=True)
    with col2:
        st.markdown("**Cantidad de productos por nivel de rotación**")
        st.dataframe(df_productos['Nivel_Rotacion'].value_counts().to_frame('Cantidad'), use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for nivel, grupo in df_productos.groupby('Nivel_Rotacion'):
        ax.scatter(grupo['Frecuencia'], grupo['Volumen'], label=nivel, alpha=0.65)
    ax.set_title('Segmentación K-Means por Frecuencia y Volumen')
    ax.set_xlabel('Frecuencia')
    ax.set_ylabel('Volumen')
    ax.legend(title='Nivel de rotación')
    fig_to_streamlit(fig)

    st.markdown("**Métricas para selección de K**")
    st.dataframe(metricas_df, use_container_width=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    metricas_df['Codo'].plot(marker='o', ax=axes[0,0])
    axes[0,0].set_title('Método del Codo')
    axes[0,0].set_xlabel('K')
    axes[0,0].set_ylabel('Inercia')
    metricas_df['Silueta'].plot(marker='o', ax=axes[0,1])
    axes[0,1].set_title('Método de Silueta')
    axes[0,1].set_xlabel('K')
    axes[0,1].set_ylabel('Silueta')
    metricas_df['davies'].plot(marker='o', ax=axes[1,0])
    axes[1,0].set_title('Davies-Bouldin')
    axes[1,0].set_xlabel('K')
    axes[1,0].set_ylabel('Índice')
    metricas_df['calinsky'].plot(marker='o', ax=axes[1,1])
    axes[1,1].set_title('Calinski-Harabasz')
    axes[1,1].set_xlabel('K')
    axes[1,1].set_ylabel('Índice')
    plt.tight_layout()
    fig_to_streamlit(fig)

    if mostrar_tablas:
        st.markdown("**Productos clasificados**")
        st.dataframe(df_productos.sort_values(['Nivel_Rotacion', 'Frecuencia'], ascending=[True, False]), use_container_width=True)

with tab_xgb:
    st.header("5. Modelo de serie de tiempo y XGBoost")
    if xgb_error:
        st.warning(xgb_error)
    else:
        st.markdown("El modelo predictivo usa la serie diaria de Profit y variables derivadas de volumen, descuento y calendario.")
        st.dataframe(xgb_results, use_container_width=True)
        if best_params is not None:
            st.markdown("**Mejores parámetros encontrados en optimización**")
            st.json(best_params)

        st.markdown("**Serie diaria de Profit**")
        fig, ax = plt.subplots(figsize=(11, 4))
        df_ts.plot(x='Fecha', y='Profit', ax=ax, legend=False)
        ax.set_title('Serie diaria de Profit')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Profit')
        fig_to_streamlit(fig)

        st.markdown("**Comparación de valores reales vs predicción XGBoost**")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(df_comparar['Fecha'], df_comparar['y_test_XGBoost'], label='Real')
        ax.plot(df_comparar['Fecha'], df_comparar['y_pred_XGBoost'], label='Predicción XGBoost')
        if 'y_pred_XGBoost_Opt' in df_comparar.columns:
            ax.plot(df_comparar['Fecha'], df_comparar['y_pred_XGBoost_Opt'], label='Predicción optimizada')
        ax.set_title('Profit real vs predicho')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Profit')
        ax.legend()
        fig_to_streamlit(fig)

        if mostrar_tablas:
            st.markdown("**Dataset diario para modelamiento**")
            st.dataframe(df_ts.head(100), use_container_width=True)
            st.markdown("**Tabla real vs predicción**")
            st.dataframe(df_comparar, use_container_width=True)

with tab_descarga:
    st.header("6. Descarga de resultados")
    sheets = {
        'base_limpia': df_cat_num,
        'transformacion': df_final_transformacion,
        'productos_kmeans': df_productos,
        'cluster_resumen': cluster_resumen,
        'metricas_k': metricas_df.reset_index(),
    }
    if df_ts is not None:
        sheets['serie_tiempo'] = df_ts
    if xgb_results is not None:
        sheets['metricas_xgboost'] = xgb_results
    if df_comparar is not None:
        sheets['real_vs_prediccion'] = df_comparar

    excel_bytes = dataframe_to_excel_bytes(sheets)
    st.download_button(
        label="📥 Descargar resultados completos en Excel",
        data=excel_bytes,
        file_name="resultados_analitica_ecommerce.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    st.markdown("""
    <div class="warn-box">
        <b>Nota:</b> la aplicación ejecuta el flujo completo desde el archivo sucio cargado. 
        Los resultados descargados incluyen base limpia, transformación, K-Means, métricas de K y modelo de serie de tiempo.
    </div>
    """, unsafe_allow_html=True)
