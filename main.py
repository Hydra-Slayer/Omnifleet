import json
import joblib
import pandas as pd
import numpy as np
import ssl
import asyncio
import boto3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder
from paho.mqtt import client as mqtt_client
from boto3.dynamodb.conditions import Key

app = FastAPI()

# --- 1. Load AI Model & Scaler ---
# Ensure these files are in the same directory as main.py on your EC2
model = joblib.load("car_anomaly_model.joblib")
scaler = joblib.load("scaler.joblib")
THRESHOLD = 0.07

# --- 2. AWS / MQTT / DynamoDB Config ---
MQTT_BROKER = "a1xcd9hlriueb2-ats.iot.ap-south-1.amazonaws.com"
PORT = 8883 
TOPIC = "car/telemetry"

REGION_NAME = 'ap-south-1'
TABLE_NAME = 'car_telematics_data'

# Certificate Paths (Ensure these exact filenames exist in your folder)
CA_PATH = "AmazonRootCA1.pem"
CERT_PATH = "407846cf0fe3a87133c7accf26e178a13e9300f7d9a121493b67092537a061ab-certificate.pem.crt"
KEY_PATH = "407846cf0fe3a87133c7accf26e178a13e9300f7d9a121493b67092537a061ab-private.pem.key"

# DynamoDB setup
dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
table = dynamodb.Table('car_telematics_data')

# Global state for the 3 Nodes (What the Website sees)
node_data = {
    "NODE1": {"live": {}, "status": "Healthy", "ai_score": 0},
    "NODE2": {"live": {}, "status": "Healthy", "ai_score": 0},
    "NODE3": {"live": {}, "status": "Healthy", "ai_score": 0}
}

# --- 3. AI Prediction Logic ---
def predict_anomaly(data):
    try:
        # Map incoming data to the specific 5 features expected by the model
        # Note: mapping 'vibration' from ESP32 to 'vibration_z' used in training
        input_row = pd.DataFrame([[
            data.get('coolant_temp', 0), 
            data.get('engine_temp', 0), 
            data.get('battery_temp', 0), 
            data.get('battery_voltage', 0), 
            data.get('vibration', 0)
        ]], columns=['coolant_temp', 'engine_temp', 'battery_temp', 'battery_voltage', 'vibration'])
        
        # Scale and Weight
        scaled = scaler.transform(input_row)
        scaled[:, 4] = scaled[:, 4] * 100  # 100x Vibration Weighting to catch the misfire
        
        # Get Anomaly Score (Decision Function)
        score = model.decision_function(scaled)[0]
        status = "Anomaly" if score < THRESHOLD else "Healthy"
        return status, score
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error", 0

# --- 4. MQTT Subscriber Logic ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to AWS IoT Core!")
        client.subscribe(TOPIC)
    else:
        print(f"Connection failed, result code: {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        # Force Node ID to uppercase to match our node_data keys
        node_id = str(payload.get("device_id", "")).upper()
        
        if node_id in node_data:
            status, score = predict_anomaly(payload)
            
            # Update the global state
            node_data[node_id]["live"] = payload
            node_data[node_id]["status"] = status
            node_data[node_id]["ai_score"] = round(score, 4)
            
            print(f"Processed {node_id}: {status} | Score: {score:.4f}")
    except Exception as e:
        print(f"MQTT Message Error: {e}")

# Initialize and Configure MQTT Client
client = mqtt_client.Client(client_id="EC2_Omnifleet_Subscriber")
client.on_connect = on_connect
client.on_message = on_message

client.tls_set(
    ca_certs=CA_PATH,
    certfile=CERT_PATH,
    keyfile=KEY_PATH,
    cert_reqs=ssl.CERT_REQUIRED,
    tls_version=ssl.PROTOCOL_TLSv1_2
)

# Start the MQTT background loop
client.connect(MQTT_BROKER, PORT, keepalive=60)
client.loop_start()

# --- 5. FastAPI Web Endpoints ---

@app.get("/")
async def get_dashboard():
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found on EC2</h1>", status_code=404)

@app.get("/history/{node_id}")
async def get_history(node_id: str):
    try:
        # Query last 50 points from DynamoDB for the selected node
        response = table.query(
            KeyConditionExpression=Key('device_id').eq(node_id.upper()),
            Limit=50, 
            ScanIndexForward=False
        )
        # jsonable_encoder solves the 'Decimal' type error from DynamoDB
        return jsonable_encoder(response.get('Items', []))
    except Exception as e:
        print(f"DynamoDB History Error: {e}")
        return []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Dashboard WebSocket Connected")
    try:
        while True:
            # Broadcast the latest state of all 3 nodes every second
            await websocket.send_json(node_data)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("Dashboard WebSocket Disconnected")