import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from paho.mqtt import client as mqtt_client
import boto3
from boto3.dynamodb.conditions import Key
import asyncio

app = FastAPI()

# --- 1. Load AI Model & Scaler ---
model = joblib.load("car_anomaly_model.joblib")
scaler = joblib.load("scaler.joblib")
THRESHOLD = 0.07

# --- 2. AWS / MQTT Config ---
MQTT_BROKER = "a1xcd9hlriueb2-ats.iot.ap-south-1.amazonaws.com"
TOPIC = "car/telemetry"
# DynamoDB setup
dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
table = dynamodb.Table('car_telematics_data')

# Shared state for the 3 Nodes
node_data = {
    "NODE1": {"live": {}, "status": "Healthy"},
    "NODE2": {"live": {}, "status": "Healthy"},
    "NODE3": {"live": {}, "status": "Healthy"}
}

# --- 3. AI Prediction Logic ---
def predict_anomaly(data):
    # Order must match training: [cool, eng, batt_t, batt_v, vib]
    input_row = pd.DataFrame([[
        data['coolant_temp'], data['engine_temp'], 
        data['battery_temp'], data['battery_voltage'], data['vibration']
    ]], columns=['coolant_temp', 'engine_temp', 'battery_temp', 'battery_voltage', 'vibration_z'])
    
    scaled = scaler.transform(input_row)
    scaled[:, 4] = scaled[:, 4] * 100  # Your 100x Weighting
    score = model.decision_function(scaled)[0]
    return "Anomaly" if score < THRESHOLD else "Healthy", score

# --- 4. MQTT Subscriber ---
def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    node_id = payload.get("device_id")
    if node_id in node_data:
        status, score = predict_anomaly(payload)
        payload['status'] = status
        payload['ai_score'] = round(score, 4)
        node_data[node_id]["live"] = payload
        node_data[node_id]["status"] = status

client = mqtt_client.Client()
client.on_message = on_message
# client.connect(...) code goes here

# --- 5. API Endpoints ---
@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/history/{node_id}")
async def get_history(node_id: str):
    # Fetch last 50 records from DynamoDB
    response = table.query(
        KeyConditionExpression=Key('device_id').eq(node_id),
        Limit=50, ScanIndexForward=False
    )
    return response['Items']

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(node_data)
            await asyncio.sleep(1) # Update UI every second
    except WebSocketDisconnect:
        pass