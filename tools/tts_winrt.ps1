# Genera archivos WAV con las voces de Windows (las mismas que usa el navegador).
# Uso: tts_winrt.ps1 -Voice "Sabina" -InFile textos.json -OutDir carpeta
#   textos.json = { "nombre_archivo": "texto a leer", ... }
param(
    [Parameter(Mandatory)] [string] $Voice,
    [Parameter(Mandatory)] [string] $InFile,
    [Parameter(Mandatory)] [string] $OutDir
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]

$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq "AsTask" -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type] $type) {
    $task = $asTask.MakeGenericMethod($type).Invoke($null, @($op))
    $task.Wait(-1) | Out-Null
    $task.Result
}

$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$synth.Voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices |
    Where-Object { $_.DisplayName -like "*$Voice*" } | Select-Object -First 1
if (-not $synth.Voice) { throw "No se encontró la voz $Voice" }

New-Item -ItemType Directory -Force $OutDir | Out-Null
$texts = Get-Content $InFile -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($p in $texts.PSObject.Properties) {
    $stream = Await ($synth.SynthesizeTextToStreamAsync($p.Value)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
    $reader = New-Object Windows.Storage.Streams.DataReader ($stream.GetInputStreamAt(0))
    $null = Await ($reader.LoadAsync([uint32]$stream.Size)) ([uint32])
    $bytes = New-Object byte[] $stream.Size
    $reader.ReadBytes($bytes)
    $path = Join-Path $OutDir ($p.Name + ".wav")
    [IO.File]::WriteAllBytes($path, $bytes)
    Write-Output "$path ($([math]::Round($bytes.Length / 1KB)) KB)"
}
