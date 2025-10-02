package android.llama.cpp

import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class LLamaAndroid {
    private val tag: String? = this::class.simpleName

    private val threadLocalState: ThreadLocal<State> = ThreadLocal.withInitial { State.Idle }

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "Llm-RunLoop") {
            Log.d(tag, "Dedicated thread for native code: ${Thread.currentThread().name}")

            // No-op if called more than once.
            System.loadLibrary("llama-android")

            // Set llama log handler to Android
            log_to_android()
            backend_init(false)

            Log.d(tag, system_info())

            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(tag, "Unhandled exception", exception)
            }
        }
    }.asCoroutineDispatcher()

    private val nlen: Int = 512

    // Meta from last completion_init (prefill)
    data class LastRunMeta(
        val nPrompt: Int,
        val prefillMs: Long,
        val nLen: Int
    )

    @Volatile
    private var lastMeta: LastRunMeta? = null
    fun getLastMeta(): LastRunMeta? = lastMeta

    // Track conversation state to build proper ChatML history
    private var conversationHistory = mutableListOf<Pair<String, String>>() // (role, content)

    private external fun log_to_android()
    private external fun load_model(filename: String): Long
    private external fun free_model(model: Long)
    private external fun new_context(model: Long): Long
    private external fun new_context_with_opts(model: Long, nThreads: Int, nCtx: Int): Long
    private external fun free_context(context: Long)
    private external fun backend_init(numa: Boolean)
    private external fun backend_free()
    private external fun new_batch(nTokens: Int, embd: Int, nSeqMax: Int): Long
    private external fun free_batch(batch: Long)
    private external fun new_sampler(): Long
    private external fun free_sampler(sampler: Long)
    private external fun bench_model(
        context: Long,
        model: Long,
        batch: Long,
        pp: Int,
        tg: Int,
        pl: Int,
        nr: Int
    ): String

    private external fun system_info(): String

    private external fun completion_init(
        context: Long,
        batch: Long,
        text: String,
        formatChat: Boolean,
        nLen: Int
    ): Int

    private external fun completion_loop(
        context: Long,
        batch: Long,
        sampler: Long,
        nLen: Int,
        ncur: IntVar
    ): String?

    private external fun kv_cache_clear(context: Long)

    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String {
        return withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> {
                    Log.d(tag, "bench(): $state")
                    bench_model(state.context, state.model, state.batch, pp, tg, pl, nr)
                }

                else -> throw IllegalStateException("No model loaded")
            }
        }
    }

    suspend fun load(pathToModel: String) {
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.Idle -> {
                    val model = load_model(pathToModel)
                    if (model == 0L)  throw IllegalStateException("load_model() failed")

                    val context = new_context(model)
                    if (context == 0L) throw IllegalStateException("new_context() failed")

                    val batch = new_batch(512, 0, 1)
                    if (batch == 0L) throw IllegalStateException("new_batch() failed")

                    val sampler = new_sampler()
                    if (sampler == 0L) throw IllegalStateException("new_sampler() failed")

                    Log.i(tag, "Loaded model $pathToModel")
                    threadLocalState.set(State.Loaded(model, context, batch, sampler))
                    
                    // Clear conversation history when loading new model
                    conversationHistory.clear()
                }
                else -> throw IllegalStateException("Model already loaded")
            }
        }
    }

    // Allow reloading context with custom threads and n_ctx (Step 7; no UI wires here)
    suspend fun reloadContext(threads: Int, nCtx: Int) {
        withContext(runLoop) {
            val state = threadLocalState.get()
            if (state !is State.Loaded) throw IllegalStateException("No model loaded")
            free_context(state.context)
            val ctx = new_context_with_opts(state.model, threads, nCtx)
            if (ctx == 0L) throw IllegalStateException("new_context_with_opts() failed")
            threadLocalState.set(State.Loaded(state.model, ctx, state.batch, state.sampler))
            
            // Clear conversation history when reloading context
            conversationHistory.clear()
        }
    }

    // Build full conversation prompt with history
    private fun buildConversationPrompt(newUserMessage: String): String {
        val systemPrompt = "You are a helpful assistant."
        
        return buildString {
            // System message
            append("<|im_start|>system\n")
            append(systemPrompt.trim())
            append("<|im_end|>\n")
            
            // Previous conversation history
            for ((role, content) in conversationHistory) {
                append("<|im_start|>$role\n")
                append(content.trim())
                append("<|im_end|>\n")
            }
            
            // New user message
            append("<|im_start|>user\n")
            append(newUserMessage.trim())
            append("<|im_end|>\n")
            append("<|im_start|>assistant\n")
        }
    }

    // Clear conversation history (called when user clicks "Clear")
    suspend fun clearConversation() {
        withContext(runLoop) {
            val state = threadLocalState.get()
            if (state is State.Loaded) {
                kv_cache_clear(state.context)
                conversationHistory.clear()
                Log.d(tag, "Conversation history and KV cache cleared")
            }
        }
    }

    // Modified send function - NO automatic KV cache clearing
    fun send(message: String, formatChat: Boolean = false): Flow<String> = flow {
        when (val state = threadLocalState.get()) {
            is State.Loaded -> {
                // Build full conversation prompt including history
                val fullPrompt = buildConversationPrompt(message)
                
                Log.d(tag, "Full conversation prompt:")
                Log.d(tag, fullPrompt)
                
                var assistantResponse = ""
                
                try {
                    // Clear KV cache and start fresh with full conversation
                    kv_cache_clear(state.context)
                    
                    // Prefill timing around completion_init
                    val tPrefillStart = System.nanoTime()
                    val nInit = completion_init(state.context, state.batch, fullPrompt, formatChat, nlen)
                    val tPrefillEnd = System.nanoTime()
                    lastMeta = LastRunMeta(
                        nPrompt = nInit,
                        prefillMs = ((tPrefillEnd - tPrefillStart) / 1_000_000L),
                        nLen = nlen
                    )

                    val ncur = IntVar(nInit)
                    while (ncur.value <= nlen) {
                        val str = completion_loop(state.context, state.batch, state.sampler, nlen, ncur)
                        if (str == null) {
                            break
                        }
                        assistantResponse += str
                        emit(str)
                    }
                    
                    // Add this conversation turn to history AFTER successful completion
                    conversationHistory.add("user" to message)
                    conversationHistory.add("assistant" to assistantResponse.replace("<|im_end|>", "").trim())
                    
                    // Log conversation state
                    Log.d(tag, "Conversation history now has ${conversationHistory.size} messages")
                    
                } catch (e: Exception) {
                    Log.e(tag, "Error during generation", e)
                    throw e
                }
                // ❌ REMOVED: No automatic KV cache clearing here!
                // The context now retains the full conversation
            }
            else -> {}
        }
    }.flowOn(runLoop)

    /**
     * Unloads the model and frees resources.
     *
     * This is a no-op if there's no model loaded.
     */
    suspend fun unload() {
        withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.Loaded -> {
                    free_context(state.context)
                    free_model(state.model)
                    free_batch(state.batch)
                    free_sampler(state.sampler);

                    threadLocalState.set(State.Idle)
                    conversationHistory.clear()
                }
                else -> {}
            }
        }
    }

    companion object {
        private class IntVar(value: Int) {
            @Volatile
            var value: Int = value
                private set

            fun inc() {
                synchronized(this) {
                    value += 1
                }
            }
        }

        private sealed interface State {
            data object Idle: State
            data class Loaded(val model: Long, val context: Long, val batch: Long, val sampler: Long): State
        }

        // Enforce only one instance of Llm.
        private val _instance: LLamaAndroid = LLamaAndroid()

        fun instance(): LLamaAndroid = _instance
    }
}

// package android.llama.cpp

// import android.util.Log
// import kotlinx.coroutines.CoroutineDispatcher
// import kotlinx.coroutines.asCoroutineDispatcher
// import kotlinx.coroutines.flow.Flow
// import kotlinx.coroutines.flow.flow
// import kotlinx.coroutines.flow.flowOn
// import kotlinx.coroutines.withContext
// import java.util.concurrent.Executors
// import kotlin.concurrent.thread

// class LLamaAndroid {
//     private val tag: String? = this::class.simpleName

//     private val threadLocalState: ThreadLocal<State> = ThreadLocal.withInitial { State.Idle }

//     private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
//         thread(start = false, name = "Llm-RunLoop") {
//             Log.d(tag, "Dedicated thread for native code: ${Thread.currentThread().name}")

//             // No-op if called more than once.
//             System.loadLibrary("llama-android")

//             // Set llama log handler to Android
//             log_to_android()
//             backend_init(false)

//             Log.d(tag, system_info())

//             it.run()
//         }.apply {
//             uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
//                 Log.e(tag, "Unhandled exception", exception)
//             }
//         }
//     }.asCoroutineDispatcher()

//     private val nlen: Int = 512

//     // Meta from last completion_init (prefill)
//     data class LastRunMeta(
//         val nPrompt: Int,
//         val prefillMs: Long,
//         val nLen: Int
//     )

//     @Volatile
//     private var lastMeta: LastRunMeta? = null
//     fun getLastMeta(): LastRunMeta? = lastMeta

//     private external fun log_to_android()
//     private external fun load_model(filename: String): Long
//     private external fun free_model(model: Long)
//     private external fun new_context(model: Long): Long
//     private external fun new_context_with_opts(model: Long, nThreads: Int, nCtx: Int): Long
//     private external fun free_context(context: Long)
//     private external fun backend_init(numa: Boolean)
//     private external fun backend_free()
//     private external fun new_batch(nTokens: Int, embd: Int, nSeqMax: Int): Long
//     private external fun free_batch(batch: Long)
//     private external fun new_sampler(): Long
//     private external fun free_sampler(sampler: Long)
//     private external fun bench_model(
//         context: Long,
//         model: Long,
//         batch: Long,
//         pp: Int,
//         tg: Int,
//         pl: Int,
//         nr: Int
//     ): String

//     private external fun system_info(): String

//     private external fun completion_init(
//         context: Long,
//         batch: Long,
//         text: String,
//         formatChat: Boolean,
//         nLen: Int
//     ): Int

//     private external fun completion_loop(
//         context: Long,
//         batch: Long,
//         sampler: Long,
//         nLen: Int,
//         ncur: IntVar
//     ): String?

//     private external fun kv_cache_clear(context: Long)

//     suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String {
//         return withContext(runLoop) {
//             when (val state = threadLocalState.get()) {
//                 is State.Loaded -> {
//                     Log.d(tag, "bench(): $state")
//                     bench_model(state.context, state.model, state.batch, pp, tg, pl, nr)
//                 }

//                 else -> throw IllegalStateException("No model loaded")
//             }
//         }
//     }

//     suspend fun load(pathToModel: String) {
//         withContext(runLoop) {
//             when (threadLocalState.get()) {
//                 is State.Idle -> {
//                     val model = load_model(pathToModel)
//                     if (model == 0L)  throw IllegalStateException("load_model() failed")

//                     val context = new_context(model)
//                     if (context == 0L) throw IllegalStateException("new_context() failed")

//                     val batch = new_batch(512, 0, 1)
//                     if (batch == 0L) throw IllegalStateException("new_batch() failed")

//                     val sampler = new_sampler()
//                     if (sampler == 0L) throw IllegalStateException("new_sampler() failed")

//                     Log.i(tag, "Loaded model $pathToModel")
//                     threadLocalState.set(State.Loaded(model, context, batch, sampler))
//                 }
//                 else -> throw IllegalStateException("Model already loaded")
//             }
//         }
//     }

//     // Allow reloading context with custom threads and n_ctx (Step 7; no UI wires here)
//     suspend fun reloadContext(threads: Int, nCtx: Int) {
//         withContext(runLoop) {
//             val state = threadLocalState.get()
//             if (state !is State.Loaded) throw IllegalStateException("No model loaded")
//             free_context(state.context)
//             val ctx = new_context_with_opts(state.model, threads, nCtx)
//             if (ctx == 0L) throw IllegalStateException("new_context_with_opts() failed")
//             threadLocalState.set(State.Loaded(state.model, ctx, state.batch, state.sampler))
//         }
//     }

//     // Build full conversation prompt with history
//     private fun buildConversationPrompt(newUserMessage: String): String {
//         val systemPrompt = "You are a helpful assistant."
        
//         return buildString {
//             // System message
//             append("<|im_start|>system\n")
//             append(systemPrompt.trim())
//             append("<|im_end|>\n")
            
//             // Previous conversation history
//             for ((role, content) in conversationHistory) {
//                 append("<|im_start|>$role\n")
//                 append(content.trim())
//                 append("<|im_end|>\n")
//             }
            
//             // New user message
//             append("<|im_start|>user\n")
//             append(newUserMessage.trim())
//             append("<|im_end|>\n")
//             append("<|im_start|>assistant\n")
//         }
//     }

//     // Clear conversation history (called when user clicks "Clear")
//     suspend fun clearConversation() {
//         withContext(runLoop) {
//             val state = threadLocalState.get()
//             if (state is State.Loaded) {
//                 kv_cache_clear(state.context)
//                 conversationHistory.clear()
//                 Log.d(tag, "Conversation history and KV cache cleared")
//             }
//         }
//     }

//     // Added proper KV cache clearing and prefill metrics capture
//     fun send(message: String, formatChat: Boolean = false): Flow<String> = flow {
//         when (val state = threadLocalState.get()) {
//             is State.Loaded -> {
//                 try {
//                     // Prefill timing around completion_init
//                     val tPrefillStart = System.nanoTime()
//                     val nInit = completion_init(state.context, state.batch, message, formatChat, nlen)
//                     val tPrefillEnd = System.nanoTime()
//                     lastMeta = LastRunMeta(
//                         nPrompt = nInit,
//                         prefillMs = ((tPrefillEnd - tPrefillStart) / 1_000_000L),
//                         nLen = nlen
//                     )

//                     val ncur = IntVar(nInit)
//                     while (ncur.value <= nlen) {
//                         val str = completion_loop(state.context, state.batch, state.sampler, nlen, ncur)
//                         if (str == null) {
//                             break
//                         }
//                         emit(str)
//                     }
//                 } finally {
//                     try {
//                         kv_cache_clear(state.context)
//                         Log.d(tag, "KV cache cleared")
//                     } catch (e: Exception) {
//                         Log.e(tag, "Failed to clear KV cache", e)
//                     }
//                 }
//             }
//             else -> {}
//         }
//     }.flowOn(runLoop)

//     /**
//      * Unloads the model and frees resources.
//      *
//      * This is a no-op if there's no model loaded.
//      */
//     suspend fun unload() {
//         withContext(runLoop) {
//             when (val state = threadLocalState.get()) {
//                 is State.Loaded -> {
//                     free_context(state.context)
//                     free_model(state.model)
//                     free_batch(state.batch)
//                     free_sampler(state.sampler);

//                     threadLocalState.set(State.Idle)
//                 }
//                 else -> {}
//             }
//         }
//     }

//     companion object {
//         private class IntVar(value: Int) {
//             @Volatile
//             var value: Int = value
//                 private set

//             fun inc() {
//                 synchronized(this) {
//                     value += 1
//                 }
//             }
//         }

//         private sealed interface State {
//             data object Idle: State
//             data class Loaded(val model: Long, val context: Long, val batch: Long, val sampler: Long): State
//         }

//         // Enforce only one instance of Llm.
//         private val _instance: LLamaAndroid = LLamaAndroid()

//         fun instance(): LLamaAndroid = _instance
//     }
// }
