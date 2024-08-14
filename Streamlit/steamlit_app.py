import streamlit as st
import requests
from opentelemetry.trace import Tracer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import environment_variables, metrics, trace
import logging
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from azure.monitor.opentelemetry import configure_azure_monitor

# URL de votre API Azure
API_URL = "https://p07.azurewebsites.net"

logger = logging.getLogger(__name__)



configure_azure_monitor(
    connection_string="InstrumentationKey=50e26b78-c13b-4662-ba12-6e9467939251")
tracer = trace.get_tracer(__name__)

#----------------------------------------------------------------------------------------

if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

st.title("Analyse de sentiment")

user_input = st.text_input("Entrez une phrase :")

# Fonction pour analyser le sentiment
def analyze_sentiment():
    with tracer.start_as_current_span("analyze_sentiment") as span:
        response = requests.post(f"{API_URL}/predict_sentiment", params={"text":user_input})
        st.session_state.sentiment = response.json()['sentiment']
        st.session_state.probability = response.json()['probability']
        span.set_attribute("text", user_input)
        span.set_attribute("predicted_sentiment", st.session_state.sentiment)
        span.set_attribute("probability", st.session_state.probability)
    st.session_state.feedback_given = False

# Bouton pour analyser
if st.button("Analyser"):
    analyze_sentiment()

# Affichage du résultat et des boutons de feedback
if st.session_state.sentiment is not None:
    st.write(f"Sentiment prédit : {st.session_state.sentiment} , Probabilité : {st.session_state.probability}")
    
    if not st.session_state.feedback_given:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Prédiction conforme"):
                logger.warning('GOOD PREDICTION')
                with tracer.start_as_current_span("prediction_feedback") as feedback_span:
                    feedback_span.set_attribute("feedback", "conforme")
                    feedback_span.set_attribute("text", user_input)
                    feedback_span.set_attribute("sentiment", st.session_state.sentiment)
                st.success("Merci pour votre retour !")
                st.session_state.feedback_given = True

        with col2:
            if st.button("Prédiction non conforme"):
                logger.warning('WRONG PREDICTION')
                with tracer.start_as_current_span("prediction_feedback") as feedback_span:
                    feedback_data = {"non_conforme"}
                    response = requests.post(f"{API_URL}/feedback", params={"feedback_error":feedback_data})
                    feedback_span.set_attribute("feedback", "non_conforme")
                    feedback_span.set_attribute("text", user_input)
                    feedback_span.set_attribute("sentiment", st.session_state.sentiment)
                st.error("Merci pour votre retour. Nous allons améliorer notre modèle.")
                st.session_state.feedback_given = True

