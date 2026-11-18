# Lightweight native PowerShell HTTP Server (Zero Dependencies)
# University of Delhi - Department of Computer Science
# Student: Ekta Sachdeva | Roll No: 24020570063 | Problem 24

param(
    [int]$Port = 8080
)

$webRoot = $PSScriptRoot
if (-not $webRoot) { $webRoot = Get-Location }

$listener = New-Object System.Net.HttpListener
$prefix = "http://localhost:$Port/"
$listener.Prefixes.Add($prefix)

try {
    $listener.Start()
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "  INCOME CLASSIFICATION & GENDER-FAIRNESS AUDIT WEB DASHBOARD" -ForegroundColor Green
    Write-Host "  Department of Computer Science - University of Delhi" -ForegroundColor Yellow
    Write-Host "  Candidate: Ekta Sachdeva | Roll: 24020570063" -ForegroundColor Yellow
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "Server running at: $prefix" -ForegroundColor White
    Write-Host "Press Ctrl+C to terminate the server.`n" -ForegroundColor Gray

    # Launch browser automatically
    Start-Process $prefix

    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response

        $relPath = $request.Url.LocalPath.TrimStart('/')
        if ([string]::IsNullOrWhiteSpace($relPath) -or $relPath -eq "/") {
            $relPath = "index.html"
        }

        $filePath = Join-Path $webRoot $relPath

        if (Test-Path $filePath -PathType Leaf) {
            $bytes = [System.IO.File]::ReadAllBytes($filePath)
            $response.ContentLength64 = $bytes.Length
            
            # Content-type detection
            if ($filePath.EndsWith(".html")) { $response.ContentType = "text/html; charset=utf-8" }
            elseif ($filePath.EndsWith(".css")) { $response.ContentType = "text/css" }
            elseif ($filePath.EndsWith(".js")) { $response.ContentType = "application/javascript" }
            elseif ($filePath.EndsWith(".png")) { $response.ContentType = "image/png" }
            else { $response.ContentType = "application/octet-stream" }

            $response.OutputStream.Write($bytes, 0, $bytes.Length)
        } else {
            $response.StatusCode = 404
            $errBytes = [System.Text.Encoding]::UTF8.GetBytes("404 Not Found")
            $response.OutputStream.Write($errBytes, 0, $errBytes.Length)
        }
        $response.Close()
    }
} catch {
    Write-Host "Server stopped or error occurred: $_" -ForegroundColor Red
} finally {
    $listener.Stop()
    $listener.Close()
}
