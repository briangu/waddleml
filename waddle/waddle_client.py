import asyncio
import websockets
import argparse

async def listen_to_waddle_proxy(ws_url):
    """
    Connects to the WaddleProxy WebSocket and streams log data to stdout.
    """
    async with websockets.connect(ws_url) as websocket:
        try:
            while True:
                # Wait for a message from the WebSocket
                message = await websocket.recv()

                # Print the message (log entry) as a row in stdout
                print(message)
        except websockets.ConnectionClosed:
            print("Connection to WaddleProxy WebSocket closed.")
        except Exception as e:
            print(f"Error: {e}")

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="CLI client for WaddleProxy WebSocket to stream log events.")

    # WebSocket URL argument (default: ws://localhost:8765)
    parser.add_argument('--ws-url', type=str, default="ws://localhost:8765",
                        help="The WebSocket URL of the WaddleProxy server (default: ws://localhost:8765)")

    return parser.parse_args()

if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()

    # Run the event loop to listen for streaming data
    asyncio.run(listen_to_waddle_proxy(args.ws_url))
