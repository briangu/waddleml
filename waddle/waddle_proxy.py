import duckdb
import json
import asyncio
import websockets
import argparse
from aiohttp import web
from datetime import datetime

class WaddleProxy:
    def __init__(self, db_path, ws_sub_port, ws_source_url=None, enable_polling=False, polling_interval=60, enable_http=False, http_port=8080, static_dir=None):
        self.db_path = db_path
        self.ws_sub_port = ws_sub_port  # Port for WebSocket subscribers
        self.ws_source_url = ws_source_url  # URL to source data from
        self.enable_polling = enable_polling
        self.polling_interval = polling_interval
        self.enable_http = enable_http  # Serve HTTP if enabled
        self.http_port = http_port  # HTTP server port
        self.static_dir = static_dir  # Directory for serving static files (HTML, JS, CSS)
        self.last_poll_time = datetime.min.isoformat()  # Start polling from earliest time

        # Reuse a single read-only connection to the DuckDB database
        self.db_connection = duckdb.connect(database=db_path, read_only=True)

        # Set to manage connected WebSocket clients
        self.connected_clients = set()

    def get_new_logs(self, start_time, end_time):
        """
        Queries the DuckDB database for logs between start_time and end_time using the existing read-only connection.
        """
        query = """
        SELECT * FROM logs WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp ASC;
        """
        return self.db_connection.execute(query, [start_time, end_time]).fetchall()

    async def forward_logs_to_clients(self, logs):
        """
        Sends log data to all connected WebSocket clients.
        """
        log_entries = [
            {
                'timestamp': str(log[5]),    # Timestamp in ISO format
                'category': log[2],          # Category (e.g., model, gpu_system)
                'name': log[3],              # Name (e.g., loss, accuracy)
                'value': log[4],             # Value (e.g., 0.05 for loss)
                'step': log[1],              # Step (e.g., training step)
                'run_id': log[0]             # Run ID for distinguishing between experiments
            } for log in logs
        ]

        # Forward to connected WebSocket clients
        if self.connected_clients:
            message = json.dumps(log_entries)
            await asyncio.gather(*[client.send(message) for client in self.connected_clients])

    async def ws_source_handler(self):
        """
        If ws_source_url is set, this method listens for incoming logs from the source WebSocket (acting as a WaddleLogger).
        """
        if self.ws_source_url:
            async with websockets.connect(self.ws_source_url) as ws:
                async for message in ws:
                    log_entry = json.loads(message)
                    await self.forward_logs_to_clients([log_entry])

    async def poll_db_for_logs(self):
        """
        Periodically polls the DuckDB database for new logs and forwards them to clients.
        """
        while self.enable_polling:
            current_time = datetime.now().isoformat()  # Current time for querying logs
            logs = self.get_new_logs(self.last_poll_time, current_time)
            if logs:
                await self.forward_logs_to_clients(logs)
                self.last_poll_time = current_time  # Update the last poll time

            await asyncio.sleep(self.polling_interval)  # Poll every interval

    async def ws_client_handler(self, websocket, path):
        """
        Handles WebSocket connections for subscribers (clients).
        """
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                # Parse client message (can include 'start_time')
                data = json.loads(message)
                start_time = data.get('start_time', self.last_poll_time)

                # Poll logs from DB if polling is enabled
                if self.enable_polling:
                    logs = self.get_new_logs(start_time, datetime.now().isoformat())
                    await self.forward_logs_to_clients(logs)
        finally:
            self.connected_clients.remove(websocket)

    async def handle_index(self, request):
        """
        Serves the index HTML page.
        """
        if self.static_dir:
            with open(f'{self.static_dir}/index.html', 'r') as f:
                return web.Response(text=f.read(), content_type='text/html')
        return web.Response(text="HTML Directory not configured", status=404)

    def start_proxy(self):
        """
        Starts the proxy based on the configuration (polling or WebSocket sourcing).
        """
        loop = asyncio.get_event_loop()

        # If ws_source_url is provided, listen for WebSocket data
        if self.ws_source_url:
            print(f"Starting WebSocket source listener from {self.ws_source_url}...")
            loop.create_task(self.ws_source_handler())

        # If polling is enabled, poll the database
        if self.enable_polling:
            print("Starting database polling for new logs...")
            loop.create_task(self.poll_db_for_logs())

        # Start WebSocket server for subscribers
        print(f"Starting WebSocket server for subscribers on port {self.ws_sub_port}...")
        loop.run_until_complete(websockets.serve(self.ws_client_handler, "0.0.0.0", self.ws_sub_port))

        # Optionally start the HTTP server
        if self.enable_http:
            print(f"Starting HTTP server on port {self.http_port}...")
            app = web.Application()
            app.add_routes([web.get('/', self.handle_index)])

            if self.static_dir:
                app.router.add_static('/static', self.static_dir)

            web.run_app(app, port=self.http_port)

        loop.run_forever()

def main():
    parser = argparse.ArgumentParser(description="WaddleProxy: A proxy for forwarding and polling logs.")
    parser.add_argument('--db_path', type=str, default='ml_logs.duckdb', help='Path to DuckDB file')
    parser.add_argument('--ws_sub_port', type=int, default=8080, help='Port for WebSocket subscribers')
    parser.add_argument('--ws_source_url', type=str, help='WebSocket URL to receive source data from')
    parser.add_argument('--enable_polling', action='store_true', help='Enable database polling for logs')
    parser.add_argument('--polling_interval', type=int, default=60, help='Polling interval in seconds')
    parser.add_argument('--enable_http', action='store_true', help='Enable HTTP server for serving static pages')
    parser.add_argument('--http_port', type=int, default=8081, help='Port for HTTP server')
    parser.add_argument('--static_dir', type=str, help='Directory for serving static files (HTML, JS, CSS)')

    args = parser.parse_args()

    # Start the WaddleProxy with the parsed arguments
    proxy = WaddleProxy(
        db_path=args.db_path,
        ws_sub_port=args.ws_sub_port,
        ws_source_url=args.ws_source_url,
        enable_polling=args.enable_polling,
        polling_interval=args.polling_interval,
        enable_http=args.enable_http,
        http_port=args.http_port,
        static_dir=args.static_dir
    )
    proxy.start_proxy()

if __name__ == "__main__":
    main()
