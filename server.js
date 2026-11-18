// Lightweight Node.js Web Server for Income Classification & Fairness Dashboard
// Department of Computer Science - University of Delhi
// Student: Ekta Sachdeva | Roll No: 24020570063 | Problem 24

const http = require('http');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

const PORT = 8080;
const PUBLIC_DIR = __dirname;

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
  let safePath = path.normalize(req.url.split('?')[0]);
  if (safePath === '/' || safePath === '') safePath = '/index.html';
  
  const filePath = path.join(PUBLIC_DIR, safePath);

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('404 Not Found');
      } else {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end('Internal Server Error: ' + err.code);
      }
    } else {
      const ext = path.extname(filePath).toLowerCase();
      res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' });
      res.end(content);
    }
  });
});

server.listen(PORT, () => {
  const url = `http://localhost:${PORT}/`;
  console.log("======================================================================");
  console.log("  INCOME CLASSIFICATION & GENDER-FAIRNESS AUDIT WEB DASHBOARD");
  console.log("  Department of Computer Science - University of Delhi");
  console.log("  Candidate: Ekta Sachdeva | Roll No: 24020570063");
  console.log(`  Serving at: ${url}`);
  console.log("======================================================================");

  // Automatically open browser in Windows
  exec(`start ${url}`);
});
