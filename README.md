# Detector de Plagio con IA (NLP)

Detector de plagio y similitud textual que compara documentos o textos usando embeddings semanticos de Inteligencia Artificial (OpenAI).

## Descripcion

Esta aplicacion permite subir archivos PDF o pegar texto directamente para comparar dos documentos y calcular su nivel de similitud semantica mediante embeddings vectoriales y similitud de coseno. Es util para identificar posible plagio o coincidencias significativas entre textos academicos.

## Caracteristicas

- Carga de documentos PDF o ingreso de texto directo.
- Calculo de similitud semantica mediante embeddings de IA.
- Umbral de alerta configurable para deteccion de plagio.
- Visualizacion del porcentaje de similitud con barra de progreso.
- Interfaz simple e intuitiva construida con Streamlit.

## Tecnologias utilizadas

- Python
- Streamlit
- OpenAI API (embeddings)
- scikit-learn (similitud de coseno)
- PyPDF2
- python-dotenv

## Instalacion

1. Clonar el repositorio:
```
git clone https://github.com/Ja1ros/detector-plagio-nlp.git
cd detector-plagio-nlp
```

2. Instalar dependencias:
```
pip install -r requirements.txt
```

3. Configurar variables de entorno:
```
cp .env.example .env
```
Luego edita el archivo `.env` y agrega tu API Key de OpenAI.

## Uso

Ejecuta la aplicacion con:
```
streamlit run app.py
```

Luego abre tu navegador en `http://localhost:8501`, sube o pega los dos textos/documentos a comparar y analiza su nivel de similitud.

## Licencia

Este proyecto es de uso libre para fines educativos y de portafolio.
