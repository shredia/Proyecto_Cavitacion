package github.umer0586.sensorserver

import android.app.Application
import android.content.Context
import androidx.appcompat.app.AppCompatDelegate
import org.acra.config.*
import org.acra.data.StringFormat
import org.acra.ktx.initAcra


class MyApplication : Application()
{

    override fun attachBaseContext(base: Context) {
        super.attachBaseContext(base)

        // TODO: Dark theme support will be added later.
        // For now, when a device is is dark mode we still use light theme (i-e no use of themes.xml night values)
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO)

        initAcra {
            //core configuration:
            buildConfigClass = BuildConfig::class.java
            reportFormat = StringFormat.JSON

            //each plugin you chose above can be configured in a block like this:
            mailSender {
                //required
                mailTo = "umerfarooq.phone@gmail.com"
                //defaults to true
                reportAsFile = false
                //defaults to ACRA-report.stacktrace
                //reportFileName = "Crash.txt"
                //defaults to "<applicationId> Crash Report"
                subject = "SensorServer Crash report"
                //defaults to empty
                //body = "Crash report attached"
            }


        }
    }

}

