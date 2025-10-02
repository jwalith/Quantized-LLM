package com.example.llama

import android.app.ActivityManager
import android.content.Context
import android.util.Log
import kotlinx.coroutines.*

class MemoryMonitor {
    private val tag = "MemoryMonitor"
    private var monitoringJob: Job? = null
    private val memoryReadings = mutableListOf<Long>()
    private var activityManager: ActivityManager? = null
    
    fun initialize(context: Context) {
        activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    }
    
    fun startMonitoring() {
        monitoringJob?.cancel()
        memoryReadings.clear()
        
        monitoringJob = CoroutineScope(Dispatchers.IO).launch {
            while (isActive) {
                try {
                    val memInfo = ActivityManager.MemoryInfo()
                    activityManager?.getMemoryInfo(memInfo)
                    val usedMemoryMB = (memInfo.totalMem - memInfo.availMem) / (1024 * 1024)
                    
                    synchronized(memoryReadings) {
                        memoryReadings.add(usedMemoryMB)
                    }
                    
                    delay(100) // Sample every 100ms
                } catch (e: Exception) {
                    Log.e(tag, "Error monitoring memory", e)
                }
            }
        }
    }
    
    fun stopMonitoring() {
        monitoringJob?.cancel()
        monitoringJob = null
    }
    
    fun getStats(): Pair<Long, Long> {
        synchronized(memoryReadings) {
            if (memoryReadings.isEmpty()) return Pair(0L, 0L)
            
            val peak = memoryReadings.maxOrNull() ?: 0L
            val avg = memoryReadings.average().toLong()
            
            Log.i(tag, "Memory stats - Peak: ${peak}MB, Avg: ${avg}MB, Samples: ${memoryReadings.size}")
            return Pair(peak, avg)
        }
    }
}
