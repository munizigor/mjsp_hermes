nohup python3 src/start-interpretation.py >> logs/interpretation.txt 2>&1 &
fastapi dev --host 0.0.0.0 --port 8001 src/start-hermes.py >> logs/agent.txt