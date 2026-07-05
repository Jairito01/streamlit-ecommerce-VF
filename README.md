# Analítica avanzada de E-commerce - Streamlit

Aplicación web para ejecutar el flujo del notebook de Google Colab desde un archivo sucio de e-commerce.

## Funcionalidad

La app permite:

1. Cargar el Excel/CSV sucio del proyecto.
2. Ejecutar limpieza de datos.
3. Visualizar EDA con gráficos e indicadores.
4. Transformar variables categóricas y numéricas.
5. Ejecutar K-Means para segmentación de productos.
6. Evaluar métricas de selección de K.
7. Ejecutar modelo de serie de tiempo con XGBoost.
8. Descargar resultados completos en Excel.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Subir estos archivos a un repositorio de GitHub.
2. Entrar a Streamlit Community Cloud.
3. Crear una nueva app.
4. Seleccionar el repositorio.
5. Seleccionar `app.py` como archivo principal.
6. Presionar Deploy.

## Archivo de entrada esperado

El archivo cargado debe contener las 20 columnas originales del dataset:

- Row ID
- Order ID
- Order Date
- Ship Mode
- Customer ID
- Customer Name
- Segment
- Country
- City
- State
- Postal Code
- Region
- Product ID
- Category
- Sub-Category
- Product Name
- Sales
- Quantity
- Discount
- Profit
