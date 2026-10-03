param([string]$InputDocx = 'output/AI_Risk_Sentiment_2026_Report.docx', [string]$OutputPdf = 'output/AI_Risk_Sentiment_2026_Report.pdf')
$inputAbsolute = (Resolve-Path -LiteralPath $InputDocx).Path
$outputAbsolute = [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $OutputPdf))
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    if ($word.Documents.Count -ne 0) { throw 'The automation instance contains existing documents; stopping to preserve them.' }
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($inputAbsolute, $false, $true, $false)
    $document.Repaginate()
    $pageCount = $document.ComputeStatistics(2)
    $document.ExportAsFixedFormat($outputAbsolute, 17)
    Write-Output "Exported PDF; Word page count: $pageCount"
} finally {
    if ($null -ne $document) { $document.Close(0) }
    if ($null -ne $word -and $word.Documents.Count -eq 0) { $word.Quit() }
}
