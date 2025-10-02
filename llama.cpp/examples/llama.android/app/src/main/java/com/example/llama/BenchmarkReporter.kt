package com.example.llama

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

class BenchmarkLogger(private val context: Context) {
    private val tag = "BenchmarkLogger"
    private val logFile = File(context.getExternalFilesDir(null), "llm_benchmark.log")
    
    init {
        // Create header if file doesn't exist
        if (!logFile.exists()) {
            logFile.writeText("# LLM Benchmark Log - ${Build.MODEL}\n")
            logFile.appendText("# Format: timestamp,ttft_ms,tokens,tps,peak_mem_mb,avg_mem_mb,prompt_chars,response_chars\n")
        }
    }
    
    fun logRun(
        prompt: String,
        response: String,
        ttftMs: Long,
        tokensGenerated: Int,
        decodeTPS: Double,
        memoryStats: Pair<Long, Long>
    ) {
        val timestamp = SimpleDateFormat("yyyy-MM-dd_HH:mm:ss", Locale.US).format(Date())
        
        // Structured log for easy parsing
        val csvLine = "$timestamp,$ttftMs,$tokensGenerated,${"%.2f".format(decodeTPS)},${memoryStats.first},${memoryStats.second},${prompt.length},${response.length}"
        logFile.appendText(csvLine + "\n")
        
        // Also log to console for real-time monitoring
        Log.i(tag, "PERF_CSV|$csvLine")
        Log.i(tag, "RUN_DETAIL|Prompt: ${prompt.take(50)}...|Response: ${response.take(50)}...")
        
        Log.i(tag, "Benchmark logged to: ${logFile.absolutePath}")
    }
    
    fun getLogPath(): String = logFile.absolutePath
}
