package github.umer0586.sensorserver.fragments

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.net.nsd.NsdServiceInfo
import android.net.wifi.WifiManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.snackbar.Snackbar
import com.permissionx.guolindev.PermissionX
import github.umer0586.sensorserver.R
import github.umer0586.sensorserver.databinding.FragmentServerBinding
import github.umer0586.sensorserver.service.ServerStateListener
import github.umer0586.sensorserver.service.ServiceBindHelper
import github.umer0586.sensorserver.service.ServiceRegistrationState
import github.umer0586.sensorserver.service.UdpAudioService
import github.umer0586.sensorserver.service.WebsocketService
import github.umer0586.sensorserver.service.WebsocketService.LocalBinder
import github.umer0586.sensorserver.setting.AppSettings
import github.umer0586.sensorserver.websocketserver.ServerInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.net.BindException
import java.net.UnknownHostException


interface UdpAudioStreamer {
    fun startAudioStreaming(ip: String, port: Int)
    fun stopAudioStreaming()
}

class ServerFragment : Fragment(), ServerStateListener,  UdpAudioStreamer
{

    private var websocketService: WebsocketService? = null
    private lateinit var serviceBindHelper: ServiceBindHelper
    private lateinit var appSettings: AppSettings

    private lateinit var bottomSheetDialog: BottomSheetDialog
    private lateinit var bottomSheetContent: View

    private var _binding : FragmentServerBinding? = null
    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun startAudioStreaming(ip: String, port: Int) {
        val intent = Intent(requireContext(), UdpAudioService::class.java).apply {
            putExtra("IP_ADDRESS", ip)
            putExtra("PORT", port)
        }
        requireContext().startService(intent)  // Inicia el servicio
    }

    override fun stopAudioStreaming() {
        val intent = Intent(requireContext(), UdpAudioService::class.java)
        requireContext().stopService(intent)  // Detiene el servicio
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View?
    {
        _binding = FragmentServerBinding.inflate(inflater, container, false)
        val view = binding.root
        return view
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?)
    {
        super.onViewCreated(view, savedInstanceState)
        Log.i(TAG, "onViewCreated: ")
        val buttonAudio = view.findViewById<View>(R.id.buttonAudio)
        bottomSheetDialog = BottomSheetDialog(requireContext())
        bottomSheetContent = layoutInflater.inflate(R.layout.bottom_sheet_dialog, null)
        bottomSheetDialog.setContentView(bottomSheetContent)


        appSettings = AppSettings(requireContext())


        serviceBindHelper = ServiceBindHelper(
            context = requireContext(),
            service = WebsocketService::class.java,
            componentLifecycle = lifecycle
        )

        serviceBindHelper.onServiceConnected { binder ->

            val localBinder = binder as LocalBinder
            websocketService = localBinder.service

            websocketService?.setServerStateListener(this)
            websocketService?.setServiceRegistrationCallBack(this::onServiceRegistrationStateChanged)
            websocketService?.checkState() // this callbacks onServerAlreadyRunning()
        }

        hidePulseAnimation()
        hideServerAddress()

        // we will use tag to determine last state of button
        binding.startButton.setOnClickListener { v ->
            if (v.tag == "stopped")
                startServer()
            else if (v.tag == "started")
                stopServer()
        }
        binding.buttonAudio.setOnClickListener { v ->
            if (v is Button) { // Check if v is a Button
                val isStarted = v.tag == "started"
                val ipAddress = binding.direccionIp.text.toString()

                if (isStarted) {
                    v.setText(R.string.start_server) // Change to "START"
                    v.tag = "stopped"
                    stopUdpAudioService()
                } else {
                    v.setText(R.string.stop_server) // Change to "STOP"
                    v.tag = "started"
                    startUdpAudioService(ipAddress) // Pass IP address
                }
            }
        }

    }


    private fun startUdpAudioService(ipAddress: String) {
        val intent = Intent(requireContext(), UdpAudioService::class.java)
        intent.putExtra("IP_ADDRESS", ipAddress) // Pass IP address
        intent.putExtra("PORT", 5000) // Example port
        requireContext().startService(intent) // Start the service in background
    }

    private fun stopUdpAudioService() {
        val intent = Intent(requireContext(), UdpAudioService::class.java)
        requireContext().stopService(intent) // Stop the service
    }


    private fun showServerAddress(address: String)
    {
        binding.cardView.visibility = View.VISIBLE
        binding.serverAddress.visibility = View.VISIBLE
        binding.serverAddress.text = address
        showPulseAnimation()
    }

    private fun startServer()
    {
        Log.d(TAG, "startServer() called")

        // Android 13 introduced a runtime permission for posting notifications,
        // requiring that apps ask for this permission and users have to explicitly grant it, otherwise notifications will not be visible.
        //
        // Whether user grant this permission or not we will start service anyway
        // If permission is not granted foreground notification will not be shown
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            PermissionX.init(this)
                    .permissions(android.Manifest.permission.POST_NOTIFICATIONS)
                    .request{_,_,_ -> }
        }

        val wifiManager = requireContext().applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager

        val noOptionEnabled = with(appSettings){
            !(isLocalHostOptionEnable() || isAllInterfaceOptionEnabled() || isHotspotOptionEnabled())
        }

        // if a User has selected no option then we will get Ip of wifi (in service)
        // wifi must be enabled to obtain its ip
        if (noOptionEnabled && !wifiManager.isWifiEnabled)
            showMessage("Please Enable Wi-Fi")
        else
        {
            // IP address will be obtained in service
            val intent = Intent(context, WebsocketService::class.java)
            ContextCompat.startForegroundService(requireContext(), intent)
        }

    }

    private fun stopServer()
    {
        Log.d(TAG, "stopServer()")

        // We are using the RECEIVER_NOT_EXPORTED flag in BroadcastMessageReceiver.
        // Therefore, it is mandatory to set the package name of this app with the custom action in the intent.
        // Not setting the package name will prevent the broadcast receiver from being triggered in Android 14 or later.
        // For more information, see https://stackoverflow.com/questions/76919130/android-14-context-registered-broadcast-receivers-not-working
        val intent = Intent(WebsocketService.ACTION_STOP_SERVER).apply {
            setPackage(requireContext().packageName)
        }
        requireContext().sendBroadcast(intent)
    }

    override fun onPause()
    {
        super.onPause()
        Log.d(TAG, "onPause()")

        // To prevent memory leak
        websocketService?.setServerStateListener(null)
        websocketService?.setServiceRegistrationCallBack(null)
    }

    override fun onServerStarted(serverInfo: ServerInfo)
    {
        Log.d(TAG, "onServerStarted() called")
        lifecycleScope.launch(Dispatchers.Main) {

            showServerAddress("ws://" + serverInfo.ipAddress + ":" + serverInfo.port)
            showPulseAnimation()

            binding.startButton.tag = "started"
            binding.startButton.text = "STOP"

            showMessage("Server started")

        }
    }

    override fun onServerStopped()
    {
        Log.d(TAG, "onServerStopped() called ")

       lifecycleScope.launch(Dispatchers.Main) {

           hideServerAddress()
           hidePulseAnimation()

           binding.startButton.tag = "stopped"
           binding.startButton.text = "START"

           showMessage("Server Stopped")
       }
    }

    override fun onServerError(ex: Exception?)
    {
        lifecycleScope.launch(Dispatchers.Main) {
            if (ex is BindException)
                showMessage("Port already in use")
            else if (ex is UnknownHostException)
                showMessage("Unable to obtain IP")

            else showMessage("Failed to start server")
            Log.w(TAG, "onServerError() called")

            binding.startButton.tag = "stopped"
            binding.startButton.text = "START"

            hideServerAddress()
            hidePulseAnimation()
        }
    }

    override fun onServerAlreadyRunning(serverInfo: ServerInfo)
    {
        Log.d(TAG, "onServerAlreadyRunning() called")
        lifecycleScope.launch(Dispatchers.Main) {
            showServerAddress("ws://" + serverInfo.ipAddress + ":" + serverInfo.port)
            Toast.makeText(context, "service running", Toast.LENGTH_SHORT).show()
            binding.startButton.tag = "started"
            binding.startButton.text = "STOP"
        }
    }

    override fun UdpAudioStreamer() {
        TODO("Not yet implemented")
    }

    private fun showPulseAnimation()
    {
        binding.pulseAnimation.visibility = View.VISIBLE

    }

    private fun hidePulseAnimation()
    {
        binding.pulseAnimation.visibility = View.INVISIBLE
    }

    private fun hideServerAddress()
    {
        binding.cardView.visibility = View.GONE
        binding.serverAddress.visibility = View.GONE
    }


    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun showMessage(message: String)
    {

        view?.let{

            Snackbar.make(it, message, Snackbar.LENGTH_SHORT).apply{
                //R.id.nav_bar is in FragmentNavigationActivity
                setAnchorView(R.id.bottom_nav_view)

            }.show()

        }

    }

    @SuppressLint("SetTextI18n")
    private fun onServiceRegistrationStateChanged(
        serviceRegistrationState: ServiceRegistrationState,
        serivceInfo: NsdServiceInfo?,
        errorCode: Int?
    ) {

        lifecycleScope.launch(Dispatchers.Main) {

            val messageText = bottomSheetContent.findViewById<TextView>(R.id.message)
            val progressIndicator =
                bottomSheetContent.findViewById<LinearProgressIndicator>(R.id.progress_indicator)


            when (serviceRegistrationState) {
                ServiceRegistrationState.REGISTERING -> {
                    messageText.text = "Registering..."
                    bottomSheetDialog.show()
                    progressIndicator.show()

                }

                ServiceRegistrationState.REGISTRATION_SUCCESS -> {

                    messageText.text = "Successfully registered"
                    delay(2000L)
                    showToastMessage("Service Discoverable !")
                    bottomSheetDialog.dismiss()
                    progressIndicator.hide()

                }

                ServiceRegistrationState.UNREGISTERING -> {

                    messageText.text = "Unregistering..."
                    bottomSheetDialog.show()
                    progressIndicator.show()

                }

                ServiceRegistrationState.REGISTRATION_FAIL -> {

                    messageText.text = "Registration Failed"
                    delay(2000L)
                    showToastMessage("Service Not Discoverable")
                    bottomSheetDialog.show()
                    progressIndicator.hide()

                }

                ServiceRegistrationState.UNREGISTRATION_SUCCESS -> {

                    messageText.text = "Successfully unregistered"
                    delay(2000L)
                    showToastMessage("Service Not Discoverable")
                    bottomSheetDialog.dismiss()
                    progressIndicator.hide()


                }

                ServiceRegistrationState.UNREGISTRATION_FAIL -> {

                    messageText.text = "Unregistration Failed"
                    delay(2000L)
                    showToastMessage("Failed to unregister service")
                    bottomSheetDialog.show()
                    progressIndicator.hide()

                }
            }
        }

    }

    private fun showToastMessage(message: String){
        lifecycleScope.launch(Dispatchers.Main) {
            Toast.makeText(
                requireContext(),
                message,
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    companion object
    {
        private val TAG: String = ServerFragment::class.java.simpleName
    }
}