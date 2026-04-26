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

# --- 1. Load AI Model ---
model = joblib.load("car_anomaly_model.joblib")
scaler = joblib.load("scaler.joblib")
THRESHOLD = 0.07

# --- 2. AWS Configuration ---
MQTT_BROKER = "a1xcd9hlriueb2-ats.iot.ap-south-1.amazonaws.com"
PORT = 8883 
TOPIC = "car/telemetry"
REGION_NAME = 'ap-south-1'
TABLE_NAME = 'car_telematics_data'

CA_PATH = "AmazonRootCA1.pem"
CERT_PATH = "407846cf0fe3a87133c7accf26e178a13e9300f7d9a121493b67092537a061ab-certificate.pem.crt"
KEY_PATH = "407846cf0fe3a87133c7accf26e178a13e9300f7d9a121493b67092537a061ab-private.pem.key"

dynamodb = boto3.resource('dynamodb', region_name=REGION_NAME)
table = dynamodb.Table(TABLE_NAME)

node_data = {
    "NODE1": {"live": {}, "status": "Healthy", "ai_score": 0},
    "NODE2": {"live": {}, "status": "Healthy", "ai_score": 0},
    "NODE3": {"live": {}, "status": "Healthy", "ai_score": 0}
}

# --- 3. AI & Diagnostic Logic ---
def diagnose_anomaly(data, score):
    if score >= THRESHOLD:
        return "Healthy"
    
    et = data.get('engine_temp', 0)
    ct = data.get('coolant_temp', 0)
    bt = data.get('battery_temp', 0)
    bv = data.get('battery_voltage', 0)
    vz = data.get('vibration_z', 0)

    if vz > 15.0: return "Engine Misfire"
    if et > 105 and ct < 80: return "Thermostat Failure"
    if bt > 50 and bv < 12.5: return "Battery Cell Failure"
    if bv > 15.5: return "Alternator Overcharge"
    if et > 105 and ct > 100: return "Radiator Fan Failure"
    if et > 100 and vz > 11.5: return "Low Oil / Lubrication"
    if et < 30 and bv < 11.5: return "Cold Crank / Weak Start"
    return "Unknown Anomaly"

def predict_anomaly(data):
    try:
        input_row = pd.DataFrame([[
            data.get('coolant_temp', 0), data.get('engine_temp', 0), 
            data.get('battery_temp', 0), data.get('battery_voltage', 0), 
            data.get('vibration_z', 9.81)
        ]], columns=['coolant_temp', 'engine_temp', 'battery_temp', 'battery_voltage', 'vibration_z'])
        
        scaled = scaler.transform(input_row)
        scaled[:, 4] = scaled[:, 4] * 100 
        score = model.decision_function(scaled)[0]
        
        status = diagnose_anomaly(data, score)
        return status, score
    except Exception as e:
        print(f"AI Error: {e}")
        return "Error", 0

# --- 4. MQTT Client ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to AWS IoT Core!")
        client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        node_id = str(payload.get("device_id", "")).upper()
        if node_id in node_data:
            status, score = predict_anomaly(payload)
            node_data[node_id]["live"] = payload
            node_data[node_id]["status"] = status
            node_data[node_id]["ai_score"] = round(score, 4)
    except Exception as e:
        print(f"MQTT Error: {e}")

client = mqtt_client.Client(client_id="EC2_Omnifleet_Subscriber")
client.on_connect = on_connect
client.on_message = on_message
client.tls_set(ca_certs=CA_PATH, certfile=CERT_PATH, keyfile=KEY_PATH, 
               cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)

client.connect(MQTT_BROKER, PORT, keepalive=60)
client.loop_start()

# --- 5. Endpoints ---
@app.get("/")
async def get_dashboard():
    with open("index.html", "r") as f: return HTMLResponse(content=f.read())

@app.get("/history/{node_id}")
async def get_history(node_id: str):
    response = table.query(KeyConditionExpression=Key('device_id').eq(node_id.upper()), 
                           Limit=50, ScanIndexForward=False)
    return jsonable_encoder(response.get('Items', []))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(node_data)
            await asyncio.sleep(1)
    except WebSocketDisconnect: pass