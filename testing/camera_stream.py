import cv2 as cv
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl

ids_peak.Library.Initialize()

manager = ids_peak.DeviceManager.Instance()
manager.Update()

device = manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
data_stream = device.DataStreams()[0].OpenDataStream()
node_map = device.RemoteDevice().NodeMaps()[0]

node_map.FindNode("AcquisitionMode").SetCurrentEntry("Continuous")

payload_size = node_map.FindNode("PayloadSize").Value()
for _ in range(data_stream.NumBuffersAnnouncedMinRequired()):
    data_stream.QueueBuffer(data_stream.AllocAndAnnounceBuffer(payload_size))

data_stream.StartAcquisition()
node_map.FindNode("AcquisitionStart").Execute()

while True:
    buffer = data_stream.WaitForFinishedBuffer(5000)
    image = ids_peak_ipl.Image.CreateFromSizeAndBuffer(
        buffer.PixelFormat(), buffer.BasePtr(), buffer.Size(), buffer.Width(), buffer.Height()
    ).ConvertTo(ids_peak_ipl.PixelFormatName_BGRa8)
    frame = cv.cvtColor(image.get_numpy_3D().copy(), cv.COLOR_BGRA2BGR)
    data_stream.QueueBuffer(buffer)

    cv.imshow("IDS Camera", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

node_map.FindNode("AcquisitionStop").Execute()
data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
cv.destroyAllWindows()
ids_peak.Library.Close()

#Code written by AI