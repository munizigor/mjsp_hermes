import sys
import json
import os

from docker_control import DockerController
"""
Example: python run_load_tests.py ../envs/config.gcp.glinerx_ministral.json
"""
print("Argumentos recebidos:", sys.argv)

# Handle constants and environment variables

proj_dir = "../.."
mandatory_keys = ["LOCAL_SQL_DB_PATH", "API_KEY", "HERMES_ADDR", "HERMES_PORT"]
env_path = f"{proj_dir}/.env"
env_vals = {
    rawline.split("=")[0]: rawline.split("=")[1].rstrip("\n")
    for rawline in open(env_path, "r").read().split("\n")
    if "=" in rawline
}
for key, value in env_vals.items():
    os.environ[key] = value

if not all(key in os.environ for key in mandatory_keys):
    print("❌ Missing mandatory keys in .env file")
    print("Missing:", [key for key in mandatory_keys if key not in os.environ])
    sys.exit(1)

print(env_vals)
api_key = os.environ["API_KEY"]
hermes_addr = os.environ["HERMES_ADDR"]
hermes_port = os.environ["HERMES_PORT"]
sqlite_path = f"{proj_dir}/{os.environ['LOCAL_SQL_DB_PATH']}/database.sqlite"

docker_ctrl = DockerController(proj_dir=proj_dir, env_vals=env_vals)

if len(sys.argv) > 1:
    config_path = sys.argv[1]
else:
    #uses default /config.json for testing on local setup
    config_path = f"{proj_dir}/config.json"

configs = json.load(open(config_path, "r"))
output_dir_base = f"{proj_dir}/{configs['test_output_dir']}"
header = f"X-Hermes-API-Key: {api_key}"
dataset_name = "fake-emergencies-br"
fake_calls_dataset_path = "pitagoras-alves/fake-emergencies-br"
hermes_url = f"http://{hermes_addr}:{hermes_port}"
hermes_headers = {"X-Hermes-API-Key": api_key}

print(f"Endereço Hermes: {hermes_addr}:{hermes_port}")
print(f"URL Hermes: {hermes_url}")
print(f"Caminho do dataset: {fake_calls_dataset_path}")

config_name = os.path.basename(config_path).replace("config.", "").replace(".json", "")