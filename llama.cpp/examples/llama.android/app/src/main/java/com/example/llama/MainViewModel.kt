package com.example.llama

import android.content.Context
import android.llama.cpp.LLamaAndroid
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import java.io.File
import kotlin.math.max

class MainViewModel(
    private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance(),
    private val context: Context? = null
): ViewModel() {
    companion object {
        @JvmStatic
        private val NanosPerSecond = 1_000_000_000.0

        // Special marker for horizontal separators
        const val SEPARATOR = "--------------------------------------------------------------------"
    }

    private val tag: String? = this::class.simpleName

    var messages by mutableStateOf(listOf("Welcome!"))
        private set

    var message by mutableStateOf("")
        private set

    // Simple logging and monitoring
    private var benchmarkLogger: BenchmarkLogger? = null
    private val memoryMonitor = MemoryMonitor()
    private var currentModelFile: File? = null

    init {
        context?.let { ctx ->
            benchmarkLogger = BenchmarkLogger(ctx)
            memoryMonitor.initialize(ctx)
        }
    }

    override fun onCleared() {
        super.onCleared()
        memoryMonitor.stopMonitoring()
        viewModelScope.launch {
            try {
                llamaAndroid.unload()
            } catch (exc: IllegalStateException) {
                messages += (exc.message ?: "Unload failed")
            }
        }
    }

    private fun cleanQwenResponse(response: String): String {
        return response
            .replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .trim()
    }

    fun send() {
        val userText = message
        if (userText.isBlank()) return
        message = ""

        messages += "**User**: $userText"
        messages += "**AI**:"

        viewModelScope.launch {
            val startTime = System.currentTimeMillis()
            
            // Start memory monitoring
            memoryMonitor.startMonitoring()
            
            try {
                val startNs = System.nanoTime()
                var firstTokenNs = 0L
                var tokenCount = 0
                var ttftMs = 0L

                val prompt = ChatTemplates.qwen25(userText)

                Log.i(tag, "=== EXACT PROMPT START ===")
                Log.i(tag, prompt)
                Log.i(tag, "=== EXACT PROMPT END ===")

                Log.i(tag, "Prompt length: ${prompt.length} chars, ${prompt.toByteArray().size} bytes")

                var responseText = ""
                var chunkCount = 0
                var hasContent = false

                // Per-token latency stats
                val tokenLatenciesNs = mutableListOf<Long>()
                var lastChunkNs: Long? = null

                llamaAndroid
                    // IMPORTANT: formatChat = true to tokenize special tokens (ChatML)
                    .send(prompt, true)
                    .catch {
                        Log.e(tag, "send() failed", it)
                        messages += (it.message ?: "send failed")
                    }
                    .collect { chunk ->
                        val now = System.nanoTime()

                        if (tokenCount == 0) {
                            firstTokenNs = now
                            ttftMs = (firstTokenNs - startNs) / 1_000_000
                            Log.i(tag, "BENCH_TTFT_ms: $ttftMs")
                            lastChunkNs = now
                        } else {
                            // Per-token latency from previous emission
                            lastChunkNs?.let { prev ->
                                tokenLatenciesNs += (now - prev)
                            }
                            lastChunkNs = now
                        }

                        chunkCount++
                        tokenCount++

                        // Check for stop token BEFORE adding to response
                        val stopTokenIndex = chunk.indexOf("<|im_end|>")
                        if (stopTokenIndex != -1) {
                            Log.i(tag, "Stopping: Found Qwen end token at position $stopTokenIndex in chunk: '$chunk'")
                            val validContent = chunk.substring(0, stopTokenIndex)
                            responseText += validContent

                            val cleanedResponse = cleanQwenResponse(responseText)
                            messages = messages.dropLast(1) + cleanedResponse
                            return@collect
                        }

                        // No stop token found - add full chunk
                        responseText += chunk

                        // Other stopping conditions (relaxed)
                        when {
                            responseText.length > 2000 -> {
                                Log.i(tag, "Stopping: Max response length reached")
                                val cleanedResponse = cleanQwenResponse(responseText)
                                messages = messages.dropLast(1) + cleanedResponse
                                return@collect
                            }
                            hasRepeatedPhrases(responseText) -> {
                                Log.i(tag, "Stopping: Repetition detected")
                                val cleanedResponse = cleanQwenResponse(responseText)
                                messages = messages.dropLast(1) + cleanedResponse
                                return@collect
                            }
                        }

                        // Check if we're getting actual content vs control tokens
                        if (chunk.trim().isNotEmpty() && !chunk.startsWith("<|") && !chunk.startsWith("##")) {
                            hasContent = true
                        }

                        // Log first few chunks for analysis
                        if (chunkCount <= 5) {
                            Log.i(tag, "Chunk $chunkCount: '$chunk' (${chunk.length} chars)")
                        }

                        // Regular display update
                        val cleanedResponse = cleanQwenResponse(responseText)
                        messages = messages.dropLast(1) + cleanedResponse
                    }

                // Final analysis (LOG ONLY - not shown in GUI)
                val finalCleanedResponse = cleanQwenResponse(responseText)
                Log.i(tag, "=== RESPONSE ANALYSIS ===")
                Log.i(tag, "Raw response length: ${responseText.length}")
                Log.i(tag, "Cleaned response length: ${finalCleanedResponse.length}")
                Log.i(tag, "Total chunks: $chunkCount")
                Log.i(tag, "Has actual content: $hasContent")
                Log.i(tag, "Cleaned response: '${finalCleanedResponse.take(200)}'")

                // Check for common problematic patterns (LOG ONLY)
                when {
                    finalCleanedResponse.contains("I can't") || finalCleanedResponse.contains("I cannot") -> {
                        Log.w(tag, "DETECTED: Refusal pattern in response")
                    }
                    finalCleanedResponse.contains("policy") || finalCleanedResponse.contains("guidelines") -> {
                        Log.w(tag, "DETECTED: Policy/guidelines mention")
                    }
                    finalCleanedResponse.trim().isEmpty() -> {
                        Log.w(tag, "DETECTED: Empty response")
                    }
                    !hasContent -> {
                        Log.w(tag, "DETECTED: Only control tokens, no actual content")
                    }
                    hasRepeatedPhrases(finalCleanedResponse) -> {
                        Log.w(tag, "DETECTED: Repetition pattern")
                    }
                }

                // Prefill metrics (from native call around completion_init)
                llamaAndroid.getLastMeta()?.let { meta ->
                    Log.i(tag, "BENCH_PREFILL: prompt_tokens=${meta.nPrompt}, prefill_ms=${meta.prefillMs}, nlen=${meta.nLen}")
                    messages += "Prefill: ${meta.nPrompt} tokens in ${meta.prefillMs} ms"
                }

                // Benchmark calculations (decode TPS)
                var tps = 0.0
                var avgLatency = 0.0
                var p95Latency = 0.0
                
                if (firstTokenNs != 0L) {
                    val endNs = System.nanoTime()
                    val secs = (endNs - firstTokenNs).toDouble() / NanosPerSecond
                    val toks = tokenCount.coerceAtLeast(1)
                    tps = toks / max(secs, 1e-6)
                    Log.i(tag, "BENCH_TPS: tokens=$toks time_s=${"%.3f".format(secs)} tps=${"%.2f".format(tps)}")

                    // Per-token latency summary (ms)
                    if (tokenLatenciesNs.isNotEmpty()) {
                        val latsMs = tokenLatenciesNs.map { it / 1_000_000.0 }.sorted()
                        avgLatency = latsMs.average()
                        val p50 = latsMs[(latsMs.size * 0.50).toInt().coerceIn(0, latsMs.lastIndex)]
                        p95Latency = latsMs[(latsMs.size * 0.95).toInt().coerceIn(0, latsMs.lastIndex)]
                        Log.i(tag, "BENCH_LAT_MS: avg=${"%.2f".format(avgLatency)} p50=${"%.2f".format(p50)} p95=${"%.2f".format(p95Latency)}")
                    }

                    val perfMsg = "Performance: ${toks} tokens, ${"%.3f".format(secs)}s, ${"%.2f".format(tps)} tok/s"
                    messages += perfMsg
                } else {
                    Log.w(tag, "No tokens produced")
                }

                // Simple benchmark logging
                benchmarkLogger?.let { logger ->
                    val memoryStats = memoryMonitor.getStats()
                    logger.logRun(
                        prompt = userText,
                        response = finalCleanedResponse,
                        ttftMs = ttftMs,
                        tokensGenerated = tokenCount,
                        decodeTPS = tps,
                        memoryStats = memoryStats
                    )
                    
                    messages += "📊 Benchmark logged"
                }
                
                messages += SEPARATOR

            } catch (exc: IllegalStateException) {
                Log.e(tag, "send() failed", exc)
                messages += (exc.message ?: "send failed")
            } finally {
                memoryMonitor.stopMonitoring()
            }
        }
    }

    private fun hasRepeatedPhrases(text: String): Boolean {
        val words = text.split(" ")
        if (words.size < 10) return false

        // Look for sequences of 3+ words that repeat
        for (i in 0..words.size - 6) {
            val phrase = words.subList(i, i + 3).joinToString(" ")
            val remaining = words.subList(i + 3, words.size).joinToString(" ")
            if (remaining.contains(phrase)) {
                return true
            }
        }
        return false
    }

    fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) {
        viewModelScope.launch {
            try {
                val start = System.nanoTime()
                val warmupResult = llamaAndroid.bench(pp, tg, pl, nr)
                val end = System.nanoTime()

                messages += warmupResult

                val warmup = (end - start).toDouble() / NanosPerSecond
                messages += "Warm up time: $warmup seconds, please wait..."

                if (warmup > 5.0) {
                    messages += "Warm up took too long, aborting benchmark"
                    return@launch
                }

                messages += llamaAndroid.bench(512, 128, 1, 3)
            } catch (exc: IllegalStateException) {
                Log.e(tag, "bench() failed", exc)
                messages += (exc.message ?: "bench failed")
            }
        }
    }

    fun load(pathToModel: String) {
        viewModelScope.launch {
            try {
                Log.i(tag, "Loading model: $pathToModel")

                // Log file information
                val file = java.io.File(pathToModel)
                if (file.exists()) {
                    currentModelFile = file
                    
                    val sizeMB = file.length() / (1024 * 1024)
                    Log.i(tag, "Model file size: ${sizeMB} MB")
                    messages += "Model file: ${file.name} (${sizeMB} MB)"

                    // Validate it's a Qwen model
                    if (file.name.lowercase().contains("qwen")) {
                        messages += "✓ Detected Qwen model, using ChatML format"
                    } else {
                        messages += "[WARNING] Model name doesn't contain 'qwen' - may not be compatible"
                    }

                    // Basic file validation
                    if (!file.name.lowercase().endsWith(".gguf")) {
                        messages += "[WARNING] File doesn't end with .gguf - may not be a valid model"
                    }

                    if (sizeMB < 100) {
                        messages += "[WARNING] Model file seems small (${sizeMB}MB) - verify it's complete"
                    }
                } else {
                    messages += "[ERROR] Model file does not exist: $pathToModel"
                    return@launch
                }

                messages += "⏳ Model is loading..."
                llamaAndroid.load(pathToModel)
                messages += "✓ Loaded: ${file.name}"
                Log.i(tag, "Model loaded successfully")

            } catch (exc: IllegalStateException) {
                Log.e(tag, "load() failed", exc)
                messages += "Load failed: ${exc.message ?: "Unknown error"}"
            }
        }
    }

    fun updateMessage(newMessage: String) {
        message = newMessage
    }

    fun log(message: String) {
        messages += message
    }

    fun clear() {
        messages = listOf()
        viewModelScope.launch {
            try {
                llamaAndroid.clearConversation()
            } catch (exc: Exception) {
                Log.e(tag, "Failed to clear conversation", exc)
            }
        }
    }
}

// package com.example.llama

// import android.content.Context
// import android.llama.cpp.LLamaAndroid
// import android.util.Log
// import androidx.compose.runtime.getValue
// import androidx.compose.runtime.mutableStateOf
// import androidx.compose.runtime.setValue
// import androidx.lifecycle.ViewModel
// import androidx.lifecycle.viewModelScope
// import kotlinx.coroutines.flow.catch
// import kotlinx.coroutines.launch
// import java.io.File
// import kotlin.math.max

// class MainViewModel(
//     private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance(),
//     private val context: Context? = null
// ): ViewModel() {
//     companion object {
//         @JvmStatic
//         private val NanosPerSecond = 1_000_000_000.0

//         // Special marker for horizontal separators
//         const val SEPARATOR = "--------------------------------------------------------------------"
//     }

//     private val tag: String? = this::class.simpleName

//     var messages by mutableStateOf(listOf("Welcome!"))
//         private set

//     var message by mutableStateOf("")
//         private set

//     // Benchmarking and monitoring
//     private var benchmarkReporter: BenchmarkReporter? = null
//     private val memoryMonitor = MemoryMonitor()
//     private var currentModelFile: File? = null
    
//     // Current configuration tracking
//     private var currentThreads: Int = 4
//     private var currentNCtx: Int = 4096
//     private val nlen: Int = 512

//     init {
//         context?.let { ctx ->
//             benchmarkReporter = BenchmarkReporter(ctx)
//             memoryMonitor.initialize(ctx)
//         }
//     }

//     override fun onCleared() {
//         super.onCleared()
//         memoryMonitor.stopMonitoring()
//         viewModelScope.launch {
//             try {
//                 llamaAndroid.unload()
//             } catch (exc: IllegalStateException) {
//                 messages += (exc.message ?: "Unload failed")
//             }
//         }
//     }

//     private fun cleanQwenResponse(response: String): String {
//         return response
//             .replace("<|im_end|>", "")
//             .replace("<|im_start|>", "")
//             .trim()
//     }

//     fun send() {
//         val userText = message
//         if (userText.isBlank()) return
//         message = ""

//         messages += "**User**: $userText"
//         messages += "**AI**:"

//         viewModelScope.launch {
//             val startTime = System.currentTimeMillis()
//             var success = false
//             var errorMsg: String? = null
            
//             // Start comprehensive monitoring
//             memoryMonitor.startMonitoring()
            
//             try {
//                 val startNs = System.nanoTime()
//                 var firstTokenNs = 0L
//                 var tokenCount = 0
//                 var ttftMs = 0L

//                 val prompt = ChatTemplates.qwen25(userText)

//                 Log.i(tag, "=== EXACT PROMPT START ===")
//                 Log.i(tag, prompt)
//                 Log.i(tag, "=== EXACT PROMPT END ===")

//                 Log.i(tag, "Prompt length: ${prompt.length} chars, ${prompt.toByteArray().size} bytes")

//                 var responseText = ""
//                 var chunkCount = 0
//                 var hasContent = false

//                 // Per-token latency stats
//                 val tokenLatenciesNs = mutableListOf<Long>()
//                 var lastChunkNs: Long? = null

//                 llamaAndroid
//                     // IMPORTANT: formatChat = true to tokenize special tokens (ChatML)
//                     .send(prompt, true)
//                     .catch {
//                         Log.e(tag, "send() failed", it)
//                         errorMsg = it.message ?: "send failed"
//                         messages += errorMsg!!
//                     }
//                     .collect { chunk ->
//                         val now = System.nanoTime()

//                         if (tokenCount == 0) {
//                             firstTokenNs = now
//                             ttftMs = (firstTokenNs - startNs) / 1_000_000
//                             Log.i(tag, "BENCH_TTFT_ms: $ttftMs")
//                             lastChunkNs = now
//                         } else {
//                             // Per-token latency from previous emission
//                             lastChunkNs?.let { prev ->
//                                 tokenLatenciesNs += (now - prev)
//                             }
//                             lastChunkNs = now
//                         }

//                         chunkCount++
//                         tokenCount++

//                         // Check for stop token BEFORE adding to response
//                         val stopTokenIndex = chunk.indexOf("<|im_end|>")
//                         if (stopTokenIndex != -1) {
//                             Log.i(tag, "Stopping: Found Qwen end token at position $stopTokenIndex in chunk: '$chunk'")
//                             val validContent = chunk.substring(0, stopTokenIndex)
//                             responseText += validContent

//                             val cleanedResponse = cleanQwenResponse(responseText)
//                             messages = messages.dropLast(1) + cleanedResponse
//                             return@collect
//                         }

//                         // No stop token found - add full chunk
//                         responseText += chunk

//                         // Other stopping conditions (relaxed)
//                         when {
//                             responseText.length > 2000 -> {
//                                 Log.i(tag, "Stopping: Max response length reached")
//                                 val cleanedResponse = cleanQwenResponse(responseText)
//                                 messages = messages.dropLast(1) + cleanedResponse
//                                 return@collect
//                             }
//                             hasRepeatedPhrases(responseText) -> {
//                                 Log.i(tag, "Stopping: Repetition detected")
//                                 val cleanedResponse = cleanQwenResponse(responseText)
//                                 messages = messages.dropLast(1) + cleanedResponse
//                                 return@collect
//                             }
//                         }

//                         // Check if we're getting actual content vs control tokens
//                         if (chunk.trim().isNotEmpty() && !chunk.startsWith("<|") && !chunk.startsWith("##")) {
//                             hasContent = true
//                         }

//                         // Log first few chunks for analysis
//                         if (chunkCount <= 5) {
//                             Log.i(tag, "Chunk $chunkCount: '$chunk' (${chunk.length} chars)")
//                         }

//                         // Regular display update
//                         val cleanedResponse = cleanQwenResponse(responseText)
//                         messages = messages.dropLast(1) + cleanedResponse
//                     }

//                 // Final analysis (LOG ONLY - not shown in GUI)
//                 val finalCleanedResponse = cleanQwenResponse(responseText)
//                 Log.i(tag, "=== RESPONSE ANALYSIS ===")
//                 Log.i(tag, "Raw response length: ${responseText.length}")
//                 Log.i(tag, "Cleaned response length: ${finalCleanedResponse.length}")
//                 Log.i(tag, "Total chunks: $chunkCount")
//                 Log.i(tag, "Has actual content: $hasContent")
//                 Log.i(tag, "Cleaned response: '${finalCleanedResponse.take(200)}'")

//                 // Check for common problematic patterns (LOG ONLY)
//                 when {
//                     finalCleanedResponse.contains("I can't") || finalCleanedResponse.contains("I cannot") -> {
//                         Log.w(tag, "DETECTED: Refusal pattern in response")
//                     }
//                     finalCleanedResponse.contains("policy") || finalCleanedResponse.contains("guidelines") -> {
//                         Log.w(tag, "DETECTED: Policy/guidelines mention")
//                     }
//                     finalCleanedResponse.trim().isEmpty() -> {
//                         Log.w(tag, "DETECTED: Empty response")
//                     }
//                     !hasContent -> {
//                         Log.w(tag, "DETECTED: Only control tokens, no actual content")
//                     }
//                     hasRepeatedPhrases(finalCleanedResponse) -> {
//                         Log.w(tag, "DETECTED: Repetition pattern")
//                     }
//                 }

//                 // Calculate comprehensive metrics
//                 var prefillMs = 0L
//                 var prefillTokens = 0
//                 var prefillTPS = 0.0
                
//                 // Prefill metrics (from native call around completion_init)
//                 llamaAndroid.getLastMeta()?.let { meta ->
//                     prefillMs = meta.prefillMs
//                     prefillTokens = meta.nPrompt
//                     prefillTPS = if (prefillMs > 0) prefillTokens.toDouble() / (prefillMs / 1000.0) else 0.0
                    
//                     Log.i(tag, "BENCH_PREFILL: prompt_tokens=${meta.nPrompt}, prefill_ms=${meta.prefillMs}, nlen=${meta.nLen}")
//                     messages += "Prefill: ${meta.nPrompt} tokens in ${meta.prefillMs} ms"
//                 }

//                 // Benchmark calculations (decode TPS)
//                 var tps = 0.0
//                 var avgLatency = 0.0
//                 var p95Latency = 0.0
                
//                 if (firstTokenNs != 0L) {
//                     val endNs = System.nanoTime()
//                     val secs = (endNs - firstTokenNs).toDouble() / NanosPerSecond
//                     val toks = tokenCount.coerceAtLeast(1)
//                     tps = toks / max(secs, 1e-6)
//                     Log.i(tag, "BENCH_TPS: tokens=$toks time_s=${"%.3f".format(secs)} tps=${"%.2f".format(tps)}")

//                     // Per-token latency summary (ms)
//                     if (tokenLatenciesNs.isNotEmpty()) {
//                         val latsMs = tokenLatenciesNs.map { it / 1_000_000.0 }.sorted()
//                         avgLatency = latsMs.average()
//                         val p50 = latsMs[(latsMs.size * 0.50).toInt().coerceIn(0, latsMs.lastIndex)]
//                         p95Latency = latsMs[(latsMs.size * 0.95).toInt().coerceIn(0, latsMs.lastIndex)]
//                         Log.i(tag, "BENCH_LAT_MS: avg=${"%.2f".format(avgLatency)} p50=${"%.2f".format(p50)} p95=${"%.2f".format(p95Latency)}")
//                     }

//                     val perfMsg = "Performance: ${toks} tokens, ${"%.3f".format(secs)}s, ${"%.2f".format(tps)} tok/s"
//                     messages += perfMsg
//                 } else {
//                     Log.w(tag, "No tokens produced")
//                 }

//                 success = true
                
//                 // Generate comprehensive benchmark report
//                 benchmarkReporter?.let { reporter ->
//                     currentModelFile?.let { modelFile ->
//                         try {
//                             val endTime = System.currentTimeMillis()
//                             val memoryStats = memoryMonitor.getStats()
                            
//                             val reportFile = reporter.generateReport(
//                                 modelFile = modelFile,
//                                 prompt = userText,
//                                 response = finalCleanedResponse,
//                                 metrics = PerformanceMetrics(
//                                     ttftMs = ttftMs,
//                                     prefillMs = prefillMs,
//                                     prefillTokens = prefillTokens,
//                                     prefillTPS = prefillTPS,
//                                     decodeTPS = tps,
//                                     tokensGenerated = tokenCount,
//                                     totalDurationMs = endTime - startTime,
//                                     avgTokenLatencyMs = avgLatency,
//                                     p95TokenLatencyMs = p95Latency
//                                 ),
//                                 params = InferenceParams(
//                                     threads = currentThreads,
//                                     nCtx = currentNCtx,
//                                     maxTokens = nlen
//                                 ),
//                                 memoryStats = memoryStats,
//                                 success = success,
//                                 errorMessage = errorMsg
//                             )
                            
//                             messages += "📊 Benchmark report: ${reportFile.name}"
//                             Log.i(tag, "Benchmark report generated: ${reportFile.absolutePath}")
                            
//                         } catch (e: Exception) {
//                             Log.e(tag, "Failed to generate benchmark report", e)
//                         }
//                     }
//                 }
                
//                 messages += SEPARATOR

//             } catch (exc: IllegalStateException) {
//                 success = false
//                 errorMsg = exc.message ?: "send failed"
//                 Log.e(tag, "send() failed", exc)
//                 messages += errorMsg!!
                
//                 // Still generate report for failed runs
//                 benchmarkReporter?.let { reporter ->
//                     currentModelFile?.let { modelFile ->
//                         try {
//                             val endTime = System.currentTimeMillis()
//                             val memoryStats = memoryMonitor.getStats()
                            
//                             reporter.generateReport(
//                                 modelFile = modelFile,
//                                 prompt = userText,
//                                 response = "",
//                                 metrics = PerformanceMetrics(0, 0, 0, 0.0, 0.0, 0, endTime - startTime, 0.0, 0.0),
//                                 params = InferenceParams(currentThreads, currentNCtx, maxTokens = nlen),
//                                 memoryStats = memoryStats,
//                                 success = false,
//                                 errorMessage = errorMsg
//                             )
//                         } catch (e: Exception) {
//                             Log.e(tag, "Failed to generate error report", e)
//                         }
//                     }
//                 }
//             } finally {
//                 memoryMonitor.stopMonitoring()
//             }
//         }
//     }

//     private fun hasRepeatedPhrases(text: String): Boolean {
//         val words = text.split(" ")
//         if (words.size < 10) return false

//         // Look for sequences of 3+ words that repeat
//         for (i in 0..words.size - 6) {
//             val phrase = words.subList(i, i + 3).joinToString(" ")
//             val remaining = words.subList(i + 3, words.size).joinToString(" ")
//             if (remaining.contains(phrase)) {
//                 return true
//             }
//         }
//         return false
//     }

//     fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) {
//         viewModelScope.launch {
//             try {
//                 val start = System.nanoTime()
//                 val warmupResult = llamaAndroid.bench(pp, tg, pl, nr)
//                 val end = System.nanoTime()

//                 messages += warmupResult

//                 val warmup = (end - start).toDouble() / NanosPerSecond
//                 messages += "Warm up time: $warmup seconds, please wait..."

//                 if (warmup > 5.0) {
//                     messages += "Warm up took too long, aborting benchmark"
//                     return@launch
//                 }

//                 messages += llamaAndroid.bench(512, 128, 1, 3)
//             } catch (exc: IllegalStateException) {
//                 Log.e(tag, "bench() failed", exc)
//                 messages += (exc.message ?: "bench failed")
//             }
//         }
//     }

//     fun load(pathToModel: String) {
//         viewModelScope.launch {
//             try {
//                 Log.i(tag, "Loading model: $pathToModel")

//                 // Log file information
//                 val file = java.io.File(pathToModel)
//                 if (file.exists()) {
//                     currentModelFile = file
                    
//                     val sizeMB = file.length() / (1024 * 1024)
//                     Log.i(tag, "Model file size: ${sizeMB} MB")
//                     messages += "Model file: ${file.name} (${sizeMB} MB)"

//                     // Validate it's a Qwen model
//                     if (file.name.lowercase().contains("qwen")) {
//                         messages += "✓ Detected Qwen model, using ChatML format"
//                     } else {
//                         messages += "[WARNING] Model name doesn't contain 'qwen' - may not be compatible"
//                     }

//                     // Basic file validation
//                     if (!file.name.lowercase().endsWith(".gguf")) {
//                         messages += "[WARNING] File doesn't end with .gguf - may not be a valid model"
//                     }

//                     if (sizeMB < 100) {
//                         messages += "[WARNING] Model file seems small (${sizeMB}MB) - verify it's complete"
//                     }
//                 } else {
//                     messages += "[ERROR] Model file does not exist: $pathToModel"
//                     return@launch
//                 }

//                 messages += "⏳ Model is loading..."
//                 llamaAndroid.load(pathToModel)
//                 messages += "✓ Loaded: ${file.name}"
//                 Log.i(tag, "Model loaded successfully")

//             } catch (exc: IllegalStateException) {
//                 Log.e(tag, "load() failed", exc)
//                 messages += "Load failed: ${exc.message ?: "Unknown error"}"
//             }
//         }
//     }

//     fun updateMessage(newMessage: String) {
//         message = newMessage
//     }

//     fun log(message: String) {
//         messages += message
//     }

//     fun clear() {
//         messages = listOf()
//         viewModelScope.launch {
//             try {
//                 llamaAndroid.clearConversation()
//             } catch (exc: Exception) {
//                 Log.e(tag, "Failed to clear conversation", exc)
//             }
//         }
//     }

//     fun exportLatestReport() {
//         benchmarkReporter?.getLatestReport()?.let { reportFile ->
//             benchmarkReporter?.shareReport(reportFile)
//             messages += "📤 Shared report: ${reportFile.name}"
//         } ?: run {
//             messages += "❌ No benchmark reports found"
//         }
//     }
// }

// // package com.example.llama

// // import android.llama.cpp.LLamaAndroid
// // import android.util.Log
// // import androidx.compose.runtime.getValue
// // import androidx.compose.runtime.mutableStateOf
// // import androidx.compose.runtime.setValue
// // import androidx.lifecycle.ViewModel
// // import androidx.lifecycle.viewModelScope
// // import kotlinx.coroutines.flow.catch
// // import kotlinx.coroutines.launch
// // import kotlin.math.max

// // class MainViewModel(private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance()): ViewModel() {
// //     companion object {
// //         @JvmStatic
// //         private val NanosPerSecond = 1_000_000_000.0

// //         // Special marker for horizontal separators
// //         const val SEPARATOR = "--------------------------------------------------------------------"
// //     }

// //     private val tag: String? = this::class.simpleName

// //     var messages by mutableStateOf(listOf("Welcome!"))
// //         private set

// //     var message by mutableStateOf("")
// //         private set

// //     override fun onCleared() {
// //         super.onCleared()
// //         viewModelScope.launch {
// //             try {
// //                 llamaAndroid.unload()
// //             } catch (exc: IllegalStateException) {
// //                 messages += (exc.message ?: "Unload failed")
// //             }
// //         }
// //     }

// //     private fun cleanQwenResponse(response: String): String {
// //         return response
// //             .replace("<|im_end|>", "")
// //             .replace("<|im_start|>", "")
// //             .trim()
// //     }

// //     private val benchmarkReporter = BenchmarkReporter(/* context needed */)
// //     private var memoryMonitor = MemoryMonitor()

// //     fun send() {
// //         val userText = message
// //         if (userText.isBlank()) return
// //         message = ""

// //         messages += "**User**: $userText"
// //         messages += "**AI**:"

// //         viewModelScope.launch {
// //             try {
// //                 val startNs = System.nanoTime()
// //                 var firstTokenNs = 0L
// //                 var tokenCount = 0

// //                 val prompt = ChatTemplates.qwen25(userText)

// //                 Log.i(tag, "=== EXACT PROMPT START ===")
// //                 Log.i(tag, prompt)
// //                 Log.i(tag, "=== EXACT PROMPT END ===")

// //                 Log.i(tag, "Prompt length: ${prompt.length} chars, ${prompt.toByteArray().size} bytes")

// //                 var responseText = ""
// //                 var chunkCount = 0
// //                 var hasContent = false

// //                 // Per-token latency stats
// //                 val tokenLatenciesNs = mutableListOf<Long>()
// //                 var lastChunkNs: Long? = null

// //                 llamaAndroid
// //                     // IMPORTANT: formatChat = true to tokenize special tokens (ChatML)
// //                     .send(prompt, true)
// //                     .catch {
// //                         Log.e(tag, "send() failed", it)
// //                         messages += (it.message ?: "send failed")
// //                     }
// //                     .collect { chunk ->
// //                         val now = System.nanoTime()

// //                         if (tokenCount == 0) {
// //                             firstTokenNs = now
// //                             val ttftMs = (firstTokenNs - startNs) / 1_000_000
// //                             Log.i(tag, "BENCH_TTFT_ms: $ttftMs")
// //                             lastChunkNs = now
// //                         } else {
// //                             // Per-token latency from previous emission
// //                             lastChunkNs?.let { prev ->
// //                                 tokenLatenciesNs += (now - prev)
// //                             }
// //                             lastChunkNs = now
// //                         }

// //                         chunkCount++
// //                         tokenCount++

// //                         // Check for stop token BEFORE adding to response
// //                         val stopTokenIndex = chunk.indexOf("<|im_end|>")
// //                         if (stopTokenIndex != -1) {
// //                             Log.i(tag, "Stopping: Found Qwen end token at position $stopTokenIndex in chunk: '$chunk'")
// //                             val validContent = chunk.substring(0, stopTokenIndex)
// //                             responseText += validContent

// //                             val cleanedResponse = cleanQwenResponse(responseText)
// //                             messages = messages.dropLast(1) + cleanedResponse
// //                             return@collect
// //                         }

// //                         // No stop token found - add full chunk
// //                         responseText += chunk

// //                         // Other stopping conditions (relaxed)
// //                         when {
// //                             responseText.length > 2000 -> {
// //                                 Log.i(tag, "Stopping: Max response length reached")
// //                                 val cleanedResponse = cleanQwenResponse(responseText)
// //                                 messages = messages.dropLast(1) + cleanedResponse
// //                                 return@collect
// //                             }
// //                             hasRepeatedPhrases(responseText) -> {
// //                                 Log.i(tag, "Stopping: Repetition detected")
// //                                 val cleanedResponse = cleanQwenResponse(responseText)
// //                                 messages = messages.dropLast(1) + cleanedResponse
// //                                 return@collect
// //                             }
// //                         }

// //                         // Check if we're getting actual content vs control tokens
// //                         if (chunk.trim().isNotEmpty() && !chunk.startsWith("<|") && !chunk.startsWith("##")) {
// //                             hasContent = true
// //                         }

// //                         // Log first few chunks for analysis
// //                         if (chunkCount <= 5) {
// //                             Log.i(tag, "Chunk $chunkCount: '$chunk' (${chunk.length} chars)")
// //                         }

// //                         // Regular display update
// //                         val cleanedResponse = cleanQwenResponse(responseText)
// //                         messages = messages.dropLast(1) + cleanedResponse
// //                     }

// //                 // Final analysis (LOG ONLY - not shown in GUI)
// //                 val finalCleanedResponse = cleanQwenResponse(responseText)
// //                 Log.i(tag, "=== RESPONSE ANALYSIS ===")
// //                 Log.i(tag, "Raw response length: ${responseText.length}")
// //                 Log.i(tag, "Cleaned response length: ${finalCleanedResponse.length}")
// //                 Log.i(tag, "Total chunks: $chunkCount")
// //                 Log.i(tag, "Has actual content: $hasContent")
// //                 Log.i(tag, "Cleaned response: '${finalCleanedResponse.take(200)}'")

// //                 // Check for common problematic patterns (LOG ONLY)
// //                 when {
// //                     finalCleanedResponse.contains("I can't") || finalCleanedResponse.contains("I cannot") -> {
// //                         Log.w(tag, "DETECTED: Refusal pattern in response")
// //                     }
// //                     finalCleanedResponse.contains("policy") || finalCleanedResponse.contains("guidelines") -> {
// //                         Log.w(tag, "DETECTED: Policy/guidelines mention")
// //                     }
// //                     finalCleanedResponse.trim().isEmpty() -> {
// //                         Log.w(tag, "DETECTED: Empty response")
// //                     }
// //                     !hasContent -> {
// //                         Log.w(tag, "DETECTED: Only control tokens, no actual content")
// //                     }
// //                     hasRepeatedPhrases(finalCleanedResponse) -> {
// //                         Log.w(tag, "DETECTED: Repetition pattern")
// //                     }
// //                 }

// //                 // Prefill metrics (from native call around completion_init)
// //                 llamaAndroid.getLastMeta()?.let { meta ->
// //                     Log.i(tag, "BENCH_PREFILL: prompt_tokens=${meta.nPrompt}, prefill_ms=${meta.prefillMs}, nlen=${meta.nLen}")
// //                     messages += "Prefill: ${meta.nPrompt} tokens in ${meta.prefillMs} ms"
// //                 }

// //                 // Benchmark calculations (decode TPS)
// //                 if (firstTokenNs != 0L) {
// //                     val endNs = System.nanoTime()
// //                     val secs = (endNs - firstTokenNs).toDouble() / NanosPerSecond
// //                     val toks = tokenCount.coerceAtLeast(1)
// //                     val tps = toks / max(secs, 1e-6)
// //                     Log.i(tag, "BENCH_TPS: tokens=$toks time_s=${"%.3f".format(secs)} tps=${"%.2f".format(tps)}")

// //                     // Per-token latency summary (ms)
// //                     if (tokenLatenciesNs.isNotEmpty()) {
// //                         val latsMs = tokenLatenciesNs.map { it / 1_000_000.0 }.sorted()
// //                         val avg = latsMs.average()
// //                         val p50 = latsMs[(latsMs.size * 0.50).toInt().coerceIn(0, latsMs.lastIndex)]
// //                         val p95 = latsMs[(latsMs.size * 0.95).toInt().coerceIn(0, latsMs.lastIndex)]
// //                         Log.i(tag, "BENCH_LAT_MS: avg=${"%.2f".format(avg)} p50=${"%.2f".format(p50)} p95=${"%.2f".format(p95)}")
// //                     }

// //                     val perfMsg = "Performance: ${toks} tokens, ${"%.3f".format(secs)}s, ${"%.2f".format(tps)} tok/s"
// //                     messages += perfMsg
// //                     messages += SEPARATOR
// //                 } else {
// //                     Log.w(tag, "No tokens produced")
// //                 }

// //             } catch (exc: IllegalStateException) {
// //                 Log.e(tag, "send() failed", exc)
// //                 messages += (exc.message ?: "send failed")
// //             }
// //         }
// //     }

// //     private fun hasRepeatedPhrases(text: String): Boolean {
// //         val words = text.split(" ")
// //         if (words.size < 10) return false

// //         // Look for sequences of 3+ words that repeat
// //         for (i in 0..words.size - 6) {
// //             val phrase = words.subList(i, i + 3).joinToString(" ")
// //             val remaining = words.subList(i + 3, words.size).joinToString(" ")
// //             if (remaining.contains(phrase)) {
// //                 return true
// //             }
// //         }
// //         return false
// //     }

// //     fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) {
// //         viewModelScope.launch {
// //             try {
// //                 val start = System.nanoTime()
// //                 val warmupResult = llamaAndroid.bench(pp, tg, pl, nr)
// //                 val end = System.nanoTime()

// //                 messages += warmupResult

// //                 val warmup = (end - start).toDouble() / NanosPerSecond
// //                 messages += "Warm up time: $warmup seconds, please wait..."

// //                 if (warmup > 5.0) {
// //                     messages += "Warm up took too long, aborting benchmark"
// //                     return@launch
// //                 }

// //                 messages += llamaAndroid.bench(512, 128, 1, 3)
// //             } catch (exc: IllegalStateException) {
// //                 Log.e(tag, "bench() failed", exc)
// //                 messages += (exc.message ?: "bench failed")
// //             }
// //         }
// //     }

// //     fun load(pathToModel: String) {
// //         viewModelScope.launch {
// //             try {
// //                 Log.i(tag, "Loading model: $pathToModel")

// //                 // Log file information
// //                 val file = java.io.File(pathToModel)
// //                 if (file.exists()) {
// //                     val sizeMB = file.length() / (1024 * 1024)
// //                     Log.i(tag, "Model file size: ${sizeMB} MB")
// //                     messages += "Model file: ${file.name} (${sizeMB} MB)"

// //                     // Validate it's a Qwen model
// //                     if (file.name.lowercase().contains("qwen")) {
// //                         messages += "✓ Detected Qwen model, using ChatML format"
// //                     } else {
// //                         messages += "[WARNING] Model name doesn't contain 'qwen' - may not be compatible"
// //                     }

// //                     // Basic file validation
// //                     if (!file.name.lowercase().endsWith(".gguf")) {
// //                         messages += "[WARNING] File doesn't end with .gguf - may not be a valid model"
// //                     }

// //                     if (sizeMB < 100) {
// //                         messages += "[WARNING] Model file seems small (${sizeMB}MB) - verify it's complete"
// //                     }
// //                 } else {
// //                     messages += "[ERROR] Model file does not exist: $pathToModel"
// //                     return@launch
// //                 }

// //                 messages += "⏳ Model is loading..."
// //                 llamaAndroid.load(pathToModel)
// //                 messages += "✓ Loaded: ${file.name}"
// //                 Log.i(tag, "Model loaded successfully")

// //             } catch (exc: IllegalStateException) {
// //                 Log.e(tag, "load() failed", exc)
// //                 messages += "Load failed: ${exc.message ?: "Unknown error"}"
// //             }
// //         }
// //     }

// //     fun updateMessage(newMessage: String) {
// //         message = newMessage
// //     }

// //     fun log(message: String) {
// //         messages += message
// //     }

// //     fun clear() {
// //         messages = listOf()
// //         viewModelScope.launch {
// //             try {
// //                 llamaAndroid.clearConversation()
// //             } catch (exc: Exception) {
// //                 Log.e(tag, "Failed to clear conversation", exc)
// //             }
// //         }
// //     }
// // }
