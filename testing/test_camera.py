
from ids_peak import ids_peak
ids_peak.Library.Initialize()
try:
    device_manager = ids_peak.DeviceManager.Instance()
    device_manager.Update()
    devices = device_manager.Devices()
    print(devices.size())
    for device in devices:
        print("Camera: ", device.ModelName())
finally:

    ids_peak.Library.Close()

#Code written by AI