def write_docker_compose(file_path, animal_ids, brain_layers):
    with open(file_path, 'w') as file:
        file.write(make_compose_yaml_string(animal_ids, brain_layers))

def make_compose_yaml_string(animal_ids, brain_layers):

    compose_string = "version: '3.8'\nservices:\n"
    for animal_id in animal_ids:
        for brain_layer in brain_layers:
            compose_string += make_service_string(animal_id, brain_layer)
    return compose_string

def make_service_string(animal_id, brain_layer):
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
        command: bash -c "pip install --upgrade pip && apt-get update && apt-get install -y build-essential && pip install -e MEArec/ && pip install -e app/ && python /app/source/main.py && exit 0"
        ports:
        - "80{animal_id[-1]}{brain_layer[-1]}:8000"
"""

if __name__ == '__main__':
    animal_ids = ['A1']#, 'A2', 'A3', 'A4', 'A5']
    brain_layers = ['L1', 'L23', 'L4', 'L5', 'L6']
    write_docker_compose('docker-compose.yml', animal_ids, brain_layers)