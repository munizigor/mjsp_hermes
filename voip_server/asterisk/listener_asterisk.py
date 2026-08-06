from asterisk.ami import AMIClient, EventListener
import requests

client = AMIClient(address='127.0.0.1', port=5038)
client.login(username='hermes', secret='senha_forte')

def on_event(event, **kwargs):
    if event.name in ['BridgeEnter', 'Hangup']:
        payload = {
            "event": event.name,
            "uniqueid": event.get('Uniqueid'),
            "linkedid": event.get('Linkedid'),
            "caller": event.get('CallerIDNum'),
            "callee": event.get('ConnectedLineNum'),
            "timestamp": event.get('Timestamp')
        }
        requests.post("https://backend-hermes/api/sip-event", json=payload)

listener = EventListener(on_event)
client.add_event_listener(listener)
