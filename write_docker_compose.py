import re

def write_docker_compose(file_path, animal_ids, brain_layers, cloud_mode=False):
    with open(file_path, 'w') as file:
        file.write(make_compose_yaml_string(animal_ids, brain_layers, cloud_mode=cloud_mode))

def make_compose_yaml_string(animal_ids, brain_layers, cloud_mode=False):

    compose_string = "version: '3.8'\nservices:\n"
    for animal_id in animal_ids:
        for brain_layer in brain_layers:
            if cloud_mode:
                compose_string += make_cloud_service_string(animal_id, brain_layer)
            else:
                compose_string += make_local_service_string(animal_id, brain_layer)
    return compose_string

def make_cloud_service_string(animal_id, brain_layer):
    return f"""
    simulation-{animal_id}-{brain_layer}:
        image: spikeinterface/mountainsort5-base:latest
        volumes:
        - ./app:/app
        - ./MEArec:/MEArec
        - ./logs:/logs
        environment:
        - LOGGING_DIR=/logs/
        - CELL_MODELS_DIR=mearec/cell_models/{animal_id}/{brain_layer}/
        - OUTPUT_DIR=mearec/recordings/{animal_id}/{brain_layer}/
        - BUCKET_NAME=neuron-tracker-simulation
        - GOOGLE_APPLICATION_CREDENTIALS=app/source/config/google_application_cridentials.json
        - REDO_TEMPLATES=0
        - REDO_RECORDINGS=1
        - CLOUD_MODE=1
        command: bash -c "pip install --upgrade pip && apt-get update && apt-get install -y build-essential && pip install MEArec/ && pip install -e app/ && python /app/source/main_gcs.py && exit 0"
        ports:
        - "8011:8000"
"""

def make_local_service_string(animal_id, brain_layer):
    return f"""
    simulation-{animal_id}-{brain_layer}:
        image: spikeinterface/mountainsort5-base:latest
        volumes:
        - ./app:/app
        - ./MEArec:/MEArec
        - K:/ke/sta/data/mearec/cell_models/{animal_id}/{brain_layer}:/cell_models
        - K:/ke/sta/data/mearec/recordings/{animal_id}/{brain_layer}:/output
        - ./logs:/logs
        environment:
        - LOGGING_DIR=/logs/
        - CELL_MODELS_DIR=/cell_models/
        - OUTPUT_DIR=/output/
        - REDO_TEMPLATES=0
        - REDO_RECORDINGS=1
        - REGION_NUM={brain_layer[-1]}
        command: bash -c "pip install --upgrade pip && apt-get update && apt-get install -y build-essential && pip install -e MEArec/ && pip install -e app/ && python /app/source/main.py && exit 0"
        ports:
        - "80{animal_id[-1]}{brain_layer[-1]}:8000"
"""

if __name__ == '__main__':
    animal_ids = ['A1']#, 'A2', 'A3', 'A4', 'A5']
    brain_layers = ['L6']#['L1', 'L23', 'L4', 'L5', 'L6']
    write_docker_compose('docker-compose.yml', animal_ids, brain_layers, cloud_mode=False)