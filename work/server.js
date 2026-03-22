import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import chokidar from 'chokidar';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT ?? 3000;

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

// Serve static files; disable caching for PDFs so the browser always fetches fresh
app.use(
  express.static(__dirname, {
    setHeaders(res, filePath) {
      if (filePath.endsWith('.pdf')) {
        res.set({
          'Cache-Control': 'no-store, no-cache, must-revalidate',
          Pragma: 'no-cache',
          Expires: '0',
        });
      }
    },
  })
);

// WebSocket: track connected clients and broadcast PDF-change events
const clients = new Set();

wss.on('connection', (ws) => {
  clients.add(ws);
  console.log(`[ws] client connected  (total: ${clients.size})`);

  ws.on('close', () => {
    clients.delete(ws);
    console.log(`[ws] client disconnected (total: ${clients.size})`);
  });
});

function broadcast(payload) {
  const data = JSON.stringify(payload);
  for (const client of clients) {
    if (client.readyState === client.OPEN) {
      client.send(data);
    }
  }
}

// Watch main.pdf; wait for the write to fully finish before notifying clients
chokidar
  .watch(path.join(__dirname, 'main.pdf'), {
    persistent: true,
    awaitWriteFinish: { stabilityThreshold: 300, pollInterval: 50 },
  })
  .on('change', () => {
    console.log('[pdf] change detected – notifying clients');
    broadcast({ type: 'pdf_updated', timestamp: Date.now() });
  });

server.listen(PORT, () => {
  console.log(`LaTeX preview server → http://localhost:${PORT}`);
});
