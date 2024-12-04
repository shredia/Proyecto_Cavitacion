package github.umer0586.sensorserver.service

import android.Manifest
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.IBinder
import androidx.core.app.ActivityCompat
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

class UdpAudioService : Service() {

    private var isStreaming = false
    private var streamingThread: Thread? = null
    internal lateinit var audioRecord: AudioRecord

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val ipAddress = intent?.getStringExtra("IP_ADDRESS") ?: "127.0.0.1"
        val port = intent?.getIntExtra("PORT", 12345) ?: 12345
        startStreaming(ipAddress, port)
        return START_STICKY
    }

    private fun startStreaming(ipAddress: String, port: Int) {
        isStreaming = true

        streamingThread = Thread {
            val minBufferSize = AudioRecord.getMinBufferSize(
                44100,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )

            val audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                44100,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                minBufferSize
            )

            if (ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.RECORD_AUDIO
                ) != PackageManager.PERMISSION_GRANTED) {
                return@Thread
            }

            val socket = DatagramSocket()
            val targetAddress = InetAddress.getByName(ipAddress)
            val buffer = ByteArray(minBufferSize)

            try {
                audioRecord.startRecording()
                while (isStreaming) {
                    val read = audioRecord.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        val packet = DatagramPacket(buffer, read, targetAddress, port)
                        socket.send(packet)
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                audioRecord.stop()
                audioRecord.release()
                socket.close()
            }
        }

        streamingThread?.start()  // Iniciar el hilo de transmisión de audio
    }

    override fun onDestroy() {
        super.onDestroy()
        isStreaming = false
        streamingThread?.join()
    }
}
