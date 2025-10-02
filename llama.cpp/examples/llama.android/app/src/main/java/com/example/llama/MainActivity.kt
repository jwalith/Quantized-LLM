package com.example.llama

import androidx.lifecycle.ViewModelProvider
import android.content.Context
import androidx.lifecycle.ViewModel

import android.app.ActivityManager
import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.os.StrictMode
import android.os.StrictMode.VmPolicy
import android.text.format.Formatter
import android.util.Log // Import Log for logging
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.core.content.getSystemService
import com.example.llama.ui.theme.LlamaAndroidTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions

// Top-level UI state type (must not be local)
sealed interface LoadUi {
    object Idle : LoadUi
    object Copying : LoadUi
    object Loading : LoadUi
    data class Loaded(val name: String) : LoadUi
    data class Error(val message: String) : LoadUi
}

class MainActivity(
    activityManager: ActivityManager? = null,
    clipboardManager: ClipboardManager? = null,
) : ComponentActivity() {

    // Added tag for logging
    private val tag: String? = this::class.simpleName

    private val activityManager by lazy { activityManager ?: getSystemService<ActivityManager>()!! }
    private val clipboardManager by lazy { clipboardManager ?: getSystemService<ClipboardManager>()!! }

    private val viewModel: MainViewModel by viewModels {
        MainViewModelFactory(this)
    }

    // Device memory info helper
    private fun availableMemory(): ActivityManager.MemoryInfo {
        return ActivityManager.MemoryInfo().also { memoryInfo ->
            activityManager.getMemoryInfo(memoryInfo)
        }
    }

    // Copy all .gguf files from assets/models to app external files dir; return destination files
    // Copy ALL .gguf files from assets/models to app external files dir
    private fun copyAllGgufAssetsToExternalFiles(): List<File> {
        val destDir = getExternalFilesDir(null) ?: filesDir
        val copied = mutableListOf<File>()

        Log.i(tag, "Scanning assets/models for .gguf files...")

        try {
            val assetsList = assets.list("models") ?: arrayOf()
            Log.i(tag, "Found ${assetsList.size} total files in assets/models")
            
            for (name in assetsList) {
                Log.i(tag, "Checking file: $name")
                
                if (!name.endsWith(".gguf", ignoreCase = true)) {
                    Log.i(tag, "Skipping non-GGUF file: $name")
                    continue
                }

                val dest = File(destDir, name)
                if (dest.exists()) {
                    val sizeMB = dest.length() / (1024 * 1024)
                    Log.i(tag, "Model '$name' already exists ($sizeMB MB). Skipping copy.")
                    copied.add(dest)
                    continue
                }

                Log.i(tag, "Copying '$name' from assets to $dest...")
                assets.open("models/$name").use { input ->
                    FileOutputStream(dest).use { out ->
                        val buffer = ByteArray(1024 * 1024) // 1 MB buffer
                        var read: Int
                        while (input.read(buffer).also { read = it } >= 0) {
                            out.write(buffer, 0, read)
                        }
                        out.flush()
                    }
                }
                
                val sizeMB = dest.length() / (1024 * 1024)
                copied.add(dest)
                Log.i(tag, "Successfully copied '$name' ($sizeMB MB)")
            }
            
            Log.i(tag, "Copying complete. Total .gguf files copied: ${copied.size}")
            copied.forEach { file ->
                val sizeMB = file.length() / (1024 * 1024)
                Log.i(tag, "  - ${file.name} ($sizeMB MB)")
            }
            
        } catch (e: IOException) {
            Log.e(tag, "Error copying assets: ${e.message}", e)
        }

        return copied
    }


    // Suspended helper to copy then select which model file to load
    private suspend fun prepareBundledModel(preferredName: String? = null): File? {
        val copied = withContext(Dispatchers.IO) { copyAllGgufAssetsToExternalFiles() }
        if (copied.isEmpty()) {
            Log.w(tag, "No .gguf files found or copied from assets/models.") // Log if no models
            return null
        }

        // Look for Qwen model first, then any model
        val selectedFile = copied.find { it.name.lowercase().contains("qwen") }
            ?: copied.find { it.name == preferredName }
            ?: copied.first()

        Log.i(tag, "Selected bundled model: ${selectedFile.name}") // Log selected model
        return selectedFile
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        StrictMode.setVmPolicy(
            VmPolicy.Builder(StrictMode.getVmPolicy())
                .detectLeakedClosableObjects()
                .build()
        )

        val free = Formatter.formatFileSize(this, availableMemory().availMem)
        val total = Formatter.formatFileSize(this, availableMemory().totalMem)
        viewModel.log("Current memory: $free / $total")
        viewModel.log("App files directory: ${getExternalFilesDir(null)}")
        Log.i(tag, "MainActivity created. Available memory: $free / $total") // Log onCreate event

        setContent {
            LlamaAndroidTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainCompose(
                        viewModel = viewModel,
                        clipboard = clipboardManager,
                        onLoadBundled = {
                            prepareBundledModel()
                        }
                    )
                }
            }
        }
    }
}

class MainViewModelFactory(private val context: Context) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return MainViewModel(context = context) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}

@Composable
fun MinimalMarkdownText(
    text: String,
    modifier: Modifier = Modifier
) {
    val onBg = MaterialTheme.colorScheme.onBackground
    val codeBg = MaterialTheme.colorScheme.surfaceVariant

    val styled: AnnotatedString = remember(text, onBg, codeBg) {
        val b = AnnotatedString.Builder()
        var i = 0
        var boldOpen = false
        var italicOpen = false
        var codeOpen = false

        fun pushBold() { b.pushStyle(SpanStyle(fontWeight = FontWeight.Bold, color = onBg)); boldOpen = true }
        fun popBold() { if (boldOpen) { b.pop(); boldOpen = false } }
        fun pushItalic() { b.pushStyle(SpanStyle(fontStyle = FontStyle.Italic, color = onBg)); italicOpen = true }
        fun popItalic() { if (italicOpen) { b.pop(); italicOpen = false } }
        fun pushCode() {
            b.pushStyle(
                SpanStyle(
                    fontFamily = FontFamily.Monospace,
                    background = codeBg.copy(alpha = 0.35f),
                    color = onBg
                )
            )
            codeOpen = true
        }
        fun popCode() { if (codeOpen) { b.pop(); codeOpen = false } }

        while (i < text.length) {
            // **bold**
            if (!codeOpen && i + 1 < text.length && text[i] == '*' && text[i + 1] == '*') {
                if (boldOpen) popBold() else pushBold()
                i += 2
                continue
            }
            // *italic* (avoid bullets "* ")
            if (!codeOpen && text[i] == '*' && (i + 1 >= text.length || text[i + 1] != ' ')) {
                if (italicOpen) popItalic() else pushItalic()
                i += 1
                continue
            }
            // `code`
            if (text[i] == '`') {
                if (codeOpen) popCode() else pushCode()
                i += 1
                continue
            }

            b.append(text[i])
            i += 1
        }

        popCode(); popItalic(); popBold()
        b.toAnnotatedString()
    }

    Text(text = styled, modifier = modifier.padding(16.dp), style = MaterialTheme.typography.bodyLarge)
}

@Composable
fun MainCompose(
    viewModel: MainViewModel,
    clipboard: ClipboardManager,
    onLoadBundled: suspend () -> File?,
) {
    var loadUi by remember { mutableStateOf<LoadUi>(LoadUi.Idle) }
    val scope = rememberCoroutineScope()

    Column {
        val scrollState = rememberLazyListState()

        // Show everything as-is: no filtering, no sanitization, no extraction
        val chatMessages = viewModel.messages

        // Auto-scroll when new messages arrive
        LaunchedEffect(chatMessages.size) {
            if (chatMessages.isNotEmpty()) {
                scrollState.animateScrollToItem(chatMessages.lastIndex)
            }
        }

        Box(modifier = Modifier.weight(1f)) {
            LazyColumn(
                state = scrollState,
                contentPadding = PaddingValues(bottom = 8.dp)
            ) {
                items(chatMessages) { msg ->
                    MinimalMarkdownText(msg)
                }
            }
        }

        OutlinedTextField(
            value = viewModel.message,
            onValueChange = { viewModel.updateMessage(it) },
            label = { Text("Message") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(
                capitalization = KeyboardCapitalization.Sentences,
                imeAction = ImeAction.Send
            ),
            keyboardActions = KeyboardActions(
                onSend = { viewModel.send() }
            )
        )

        // Main action buttons
        Row {
            Button(
                onClick = { viewModel.send() },
                enabled = viewModel.message.isNotBlank()
            ) { Text("Send") }
            Button(onClick = { viewModel.bench(8, 4, 1) }) { Text("Bench") }
            Button(onClick = { viewModel.clear() }) { Text("Clear") }
            Button(onClick = {
                chatMessages.joinToString("\n").let {
                    clipboard.setPrimaryClip(ClipData.newPlainText("", it))
                }
            }) { Text("Copy") }
        }

        // Row for loading bundled model(s)
        Row(modifier = Modifier.padding(16.dp)) {
            when (val s = loadUi) {
                LoadUi.Copying -> {
                    CircularProgressIndicator(modifier = Modifier.padding(end = 12.dp))
                    Text("Copying model from assets…")
                }
                LoadUi.Loading -> {
                    CircularProgressIndicator(modifier = Modifier.padding(end = 12.dp))
                    Text("Loading model…")
                }
                is LoadUi.Loaded -> {
                    Text("Loaded: ${s.name}")
                }
                is LoadUi.Error -> {
                    Text("Error: ${s.message}")
                }
                LoadUi.Idle -> {
                    Button(onClick = {
                        scope.launch {
                            try {
                                loadUi = LoadUi.Copying
                                val file = onLoadBundled()
                                if (file == null) {
                                    loadUi = LoadUi.Error("No .gguf files in assets/models")
                                    return@launch
                                }
                                loadUi = LoadUi.Loading
                                viewModel.load(file.absolutePath)
                                loadUi = LoadUi.Loaded(file.name)
                            } catch (e: Exception) {
                                loadUi = LoadUi.Error(e.message ?: "Load failed")
                            }
                        }
                    }) { Text("Load bundled model") }
                }
            }
        }
    }
}
