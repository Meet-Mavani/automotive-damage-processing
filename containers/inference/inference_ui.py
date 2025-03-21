import json
import base64
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import requests
import streamlit as st
from PIL import Image
from io import BytesIO
import datetime

st.set_page_config(page_title="Damage Repair Cost Estimator")  # HTML title
st.title("Damage Repair Cost Estimator")  # Page title

from botocore.config import Config

config = Config(
    retries={'max_attempts': 10, 'mode': 'adaptive'}
)

# Boto3 session
session = boto3.Session()

# S3 Bucket Name for feedback storage
s3_bucket_name = "vatsal-meet-harsh-json-data"

# Get SSM Parameter values for OpenSearch and CloudFront URL
ssm = session.client('ssm')
parameters = ['/car-repair/collection-domain-name', '/car-repair/distribution-domain-name']
response = ssm.get_parameters(Names=parameters, WithDecryption=True)

# OpenSearch Details
os_host = response['Parameters'][0]['Value'][8:]  # Remove "https://"
os_index_name = 'repair-cost-data'

# CloudFront URL
cf_url = response['Parameters'][1]['Value']

# Initialize OpenSearch Client
credentials = session.get_credentials()
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

if upload_file:
    st.session_state.uploader_key += 1
    file_bytes = upload_file.read()
    encoded_image = base64.b64encode(file_bytes).decode()

    # **Step 1: Generate Metadata using Claude 3**
    bedrock = boto3.client('bedrock-runtime', config=config)
    prompt_description = "Instruction: Analyze the image and generate a JSON containing a short damage description."

    invoke_body = {
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': 1000,
        'messages': [{'role': 'user', 'content': [{"type": "image", "source": {"type": "base64", "data": encoded_image}}, {"type": "text", "text": prompt_description}]}]
    }

    response = bedrock.invoke_model(
        body=json.dumps(invoke_body),
        contentType='application/json',
        accept='application/json',
        modelId='anthropic.claude-3-haiku-20240307-v1:0'
    )

    generated_metadata = json.loads(response['body'].read())['content'][0]['text']

    # **Step 2: Generate Image Embeddings**
    embedding_body = json.dumps({
        "inputImage": encoded_image,
        "inputText": generated_metadata,
        "embeddingConfig": {"outputEmbeddingLength": 1024}
    })

    embedding_response = bedrock.invoke_model(
        body=embedding_body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json"
    )

    embedding_vector = json.loads(embedding_response['body'].read())['embedding']

    # **Step 3: Search in OpenSearch**
    search_body = json.dumps({"query": {"knn": {"damage_vector": {"vector": embedding_vector, "k": int(number_of_matches)}}}})
    response = requests.get(f"https://{os_host}/_search", auth=awsauth, data=search_body, headers={'Content-Type': 'application/json'})

    search_results = response.json()['hits']['hits']

    # **Step 4: Estimate Cost with Claude 3**
    repair_cost_prompt = f"Calculate repair cost based on matches: {search_results}"
    cost_response = bedrock.invoke_model(
        body=json.dumps({"messages": [{"role": "user", "content": [{"type": "text", "text": repair_cost_prompt}]}]}),
        contentType="application/json",
        accept="application/json",
        modelId="anthropic.claude-3-haiku-20240307-v1:0"
    )

    repair_estimate = json.loads(cost_response['body'].read())['content'][0]['text']

    # **Step 5: Display Matches and Feedback Option**
    num_results = len(search_results)
    columns = st.columns(num_results + 1)

    with columns[0]:
        st.write('Uploaded Image:')
        st.image(file_bytes)

    for i, hit in enumerate(search_results):
        metadata = hit['_source']['metadata']
        s3_location = metadata['s3_location']
        score = hit['_score']
        
        with columns[i + 1]:
            image_url = f'https://{cf_url}/{s3_location}'
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.write(f'Match {i + 1} (Score: {score})')
            st.image(img)

            # Thumbs Up/Down Feedback
            col1, col2 = st.columns(2)
            if col1.button(f"👍 Match {i + 1}", key=f"up_{i}"):
                save_feedback_to_s3(metadata, hit, "positive")
            if col2.button(f"👎 Match {i + 1}", key=f"down_{i}"):
                save_feedback_to_s3(metadata, hit, "negative")

    # **Step 6: Show Repair Estimate**
    st.subheader("Estimated Repair Cost")
    st.write(repair_estimate)
