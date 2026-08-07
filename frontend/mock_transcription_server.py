import asyncio
import sys
from mock_servicer import serve

if __name__ == '__main__':
    if len(sys.argv) > 1:
        port_n = int(sys.argv[1])
    else:
        port_n = 50051
        
    asyncio.run(serve(port_n))
