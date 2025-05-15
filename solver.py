# Trigger Cloud Build CI/CD
import numpy as np
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from google.cloud import storage
import os

# WCCT Parameters
c, gamma, beta = 1.0, 0.2, 0.001

# Fetch JSON input from GCS
def fetch_input_json(bucket_name='waas-459615-wcct-data', blob_name='inputs/input.json'):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    return json.loads(content)

# Core WCCT solver
def run_wcct(grid=50, timesteps=100):
    u = np.zeros((grid, grid))
    v = np.zeros_like(u)
    S = np.zeros_like(u); S[grid//2, grid//2] = 1.0
    dt = 0.01; dx = dy = 1.0/grid

    def lap(Z):
        return ((np.roll(Z,1,0)-2*Z+np.roll(Z,-1,0))/dx**2 +
                (np.roll(Z,1,1)-2*Z+np.roll(Z,-1,1))/dy**2)

    for _ in range(timesteps):
        dudt = v
        dvdt = c**2 * lap(u) - gamma * u - beta * (u**3) + S
        u += dt * dudt
        v += dt * dvdt

    return u.tolist()

# HTTP handler
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Parse GCS input JSON (ignores query params for now)
            input_params = fetch_input_json()
            grid = int(input_params.get("grid", 50))
            timesteps = int(input_params.get("timesteps", 100))

            result = run_wcct(grid=grid, timesteps=timesteps)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'field': result}).encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

