package com.example.llama

object ChatTemplates {
    // Simple, neutral system prompt
    // private const val DEFAULT_SYSTEM = "You are a helpful assistant. Be concise and direct. Answer in 1-3 sentences, unless it is very important to be longer for the given user input."
    private const val DEFAULT_SYSTEM = """You are a helpful technical assistant. 
    Follow these guidelines:
    - Give complete but concise answers
    - Aim for 2-4 sentences for explanations
    - Only provide detailed explanations when explicitly requested
    - Focus on the most important points first
    - Use clear, direct language without unnecessary elaboration"""

    // Qwen2.5 ChatML format (recommended for new model)
    fun qwen25(userMessage: String, systemPrompt: String = DEFAULT_SYSTEM): String {
        return buildString {
            append("<|im_start|>system\n")
            append(systemPrompt.trim())
            append("<|im_end|>\n")
            append("<|im_start|>user\n")
            append(userMessage.trim())
            append("<|im_end|>\n")
            append("<|im_start|>assistant\n")
        }
    }
}
