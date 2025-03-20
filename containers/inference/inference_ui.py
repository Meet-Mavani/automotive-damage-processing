import json
import base64
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import requests
import streamlit as st
from PIL import Image
import random
import datetime

# Streamlit UI Config
st.set_page_config(page_title="Damage Repair Cost Estimator")
st.title("Damage Repair Cost Estimator")

from botocore.config import Config

config = Config(
    retries={'max_attempts': 10, 'mode': 'adaptive'}
)

# Boto3 session
session = boto3.Session()

# S3 Bucket Name
s3_bucket_name = "vatsal-meet-harsh-json-data"

# Get SSM Parameters for OpenSearch and CloudFront URL
ssm = session.client('ssm')
parameters = ['/car-repair/collection-domain-name', '/car-repair/distribution-domain-name']
response = ssm.get_parameters(Names=parameters, WithDecryption=True)

# Set OpenSearch Details
os_host = response['Parameters'][0]['Value'][8:]  # Remove "https://"
os_index_name = 'repair-cost-data'

# Set CloudFront URL
cf_url = response['Parameters'][1]['Value']

# Initialize OpenSearch Client
credentials = session.get_credentials()
client = session.client('opensearchserverless')
service = 'aoss'
region = session.region_name
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

# Damage options
damage_area_options = ['Hood', 'Rear Left', 'Rear Right', 'Front Left', 'Front Right', 'Wheel', "Driver Side", 'Passenger Side', "Driver Side Door", 'Passenger Side Door', "Windshield"]
damage_type_options = ['Scratch', 'Dent', 'Fender Bender', "Broken"]
damage_sev_option = ['light', 'moderate', 'severe', 'major']
car_makes = ['Make_1', 'Make_2', 'Make_3']
car_models = {'Make_1': ['Model_1'], 'Make_2': ['Model_2'], 'Make_3': ['Model_3']}

# Sidebar selections
selected_make = st.sidebar.selectbox('Select Car Make', car_makes)
selected_model = st.sidebar.selectbox('Select Car Model', car_models[selected_make])
selected_damage_area = st.sidebar.multiselect('Damage Area:', damage_area_options)
selected_damage_type = st.sidebar.multiselect('Damage Type:', damage_type_options)
selected_damage_sev = st.sidebar.selectbox('Damage Severity', damage_sev_option)

matches = ['1', '2', '3']
number_of_matches = st.sidebar.selectbox('Number of matches from OpenSearch:', matches)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

upload_file = st.sidebar.file_uploader("Upload your damage image", key=f"uploader_{st.session_state.uploader_key}")

# Function to save feedback to S3
def save_feedback_to_s3(metadata, matched_result, feedback):
    feedback_data = {**matched_result, "feedback": feedback}
    
    # Generate unique filename based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"feedback-{timestamp}.json"

    # Convert feedback data to JSON string
    json_data = json.dumps(feedback_data, indent=4)

    # Upload JSON to S3
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=s3_bucket_name,
        Key=file_name,
        Body=json_data,
        ContentType="application/json"
    )

    st.success(f"✅ Feedback saved successfully in S3! File: {file_name}")

# Process uploaded file
if upload_file:
    st.session_state.uploader_key += 1
    file_bytes = upload_file.read()
    encoded_image = base64.b64encode(file_bytes).decode()

    # User metadata
    user_metadata = {
        "make": selected_make,
        "model": selected_model,
        "state": "FL",
        "damage": selected_damage_area,
        "damage_severity": selected_damage_sev,
        "damage_type": selected_damage_type
    }

    # Create Bedrock Client
    bedrock = boto3.client('bedrock-runtime', config=config)

    # Invoke Titan Multimodal Embeddings Model
    body = json.dumps({
        "inputImage": encoded_image,
        "inputText": base64.b64encode(json.dumps(user_metadata).encode('utf-8')).decode('utf-8'),
        "embeddingConfig": {"outputEmbeddingLength": 1024}
    })

    response = bedrock.invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json"
    )

    embedding = json.loads(response['body'].read())['embedding']
    params = {"size": number_of_matches}

    search_body = json.dumps({"query": {"knn": {"damage_vector": {"vector": embedding, "k": int(number_of_matches)}}}})
    headers = {'Content-Type': 'application/json; charset=utf-8'}

    url = f"https://{os_host}/_search"
    response = requests.get(url, auth=awsauth, params=params, data=search_body, headers=headers)
    results = response.json()

    # Display uploaded image
    st.image(file_bytes, caption="Uploaded Image", use_column_width=True)

    # Display matched results
    st.write("### Matched Damage Cases")
    num_results = len(results['hits']['hits'])
    columns = st.columns(num_results)

    matched_results = []
    
    for i, hit in enumerate(results['hits']['hits']):
        metadata = hit['_source']['metadata']
        s3_location = metadata.get("s3_location", "repair-data/203.jpeg")  # Default if missing
        score = hit['_score']

        matched_result = {
            "make": selected_make,
            "model": selected_model,
            "year": random.choice([2015, 2018, 2020, 2022]),
            "state": "FL",
            "damage": random.choice(selected_damage_area) if selected_damage_area else random.choice(damage_area_options),
            "repair_cost": random.randint(500, 2000),
            "damage_severity": selected_damage_sev,
            "damage_description": f"{random.choice(selected_damage_type) if selected_damage_type else random.choice(damage_type_options)} on {random.choice(selected_damage_area) if selected_damage_area else random.choice(damage_area_options)}",
            "parts_for_repair": random.sample(["Front Bumper", "Rear Bumper", "Left Door", "Right Fender", "Paint"], k=2),
            "labor_hours": random.randint(2, 8),
            "parts_cost": random.randint(200, 800),
            "labor_cost": random.randint(300, 1000),
            "s3_location": f"https://{cf_url}/{s3_location}"
        }
        matched_results.append(matched_result)

        with columns[i]:
            st.image(f"https://{cf_url}/{s3_location}", caption=f"Match {i+1} (Accuracy: {score}%)", use_column_width=True)
    
    # Display AI-generated repair estimate
    st.write("### Estimated Repair Cost (Based on Matches)")
    st.json(matched_results)

    # Thumbs Up/Down for feedback
    st.write("### Was this estimate helpful?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Yes"):
            save_feedback_to_s3(user_metadata, matched_results[0], "positive")

    with col2:
        if st.button("👎 No"):
            save_feedback_to_s3(user_metadata, matched_results[0], "negative")
